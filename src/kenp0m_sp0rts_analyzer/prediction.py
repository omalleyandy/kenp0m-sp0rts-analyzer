"""Machine learning-based game prediction with confidence intervals."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class PredictionResult(BaseModel):
    """Game prediction with confidence intervals.

    Attributes:
        predicted_margin: Predicted point margin (team1 - team2).
        predicted_total: Predicted total combined score.
        confidence_interval: 50% confidence interval for margin (25th-75th percentile).
        team1_score: Predicted score for team 1.
        team2_score: Predicted score for team 2.
        team1_win_prob: Probability of team 1 winning (0-1).
        confidence_level: Confidence level of the interval (default 0.5 = 50%).
    """

    predicted_margin: float
    predicted_total: float
    confidence_interval: tuple[float, float]
    team1_score: float
    team2_score: float
    team1_win_prob: float
    confidence_level: float = 0.5

    @field_validator("confidence_interval")
    @classmethod
    def validate_confidence_interval(
        cls, v: tuple[float, float]
    ) -> tuple[float, float]:
        """Ensure confidence interval is ordered correctly."""
        if v[0] > v[1]:
            raise ValueError(f"Lower bound {v[0]} must be <= upper bound {v[1]}")
        return v

    @field_validator("team1_win_prob")
    @classmethod
    def validate_win_prob(cls, v: float) -> float:
        """Ensure win probability is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"Win probability must be in [0,1], got {v}")
        return v


@dataclass
class BacktestMetrics:
    """Backtesting performance metrics.

    Attributes:
        mae_margin: Mean absolute error for margin predictions.
        rmse_margin: Root mean squared error for margin predictions.
        mae_total: Mean absolute error for total predictions.
        rmse_total: Root mean squared error for total predictions.
        accuracy: Fraction of games where winner was predicted correctly.
        brier_score: Brier score for probability calibration (lower is better).
        r2_margin: R² score for margin predictions.
        ats_record: Against-the-spread record (wins, losses).
        ats_percentage: ATS win percentage.
    """

    mae_margin: float
    rmse_margin: float
    mae_total: float
    rmse_total: float
    accuracy: float
    brier_score: float
    r2_margin: float
    ats_record: tuple[int, int]
    ats_percentage: float


class FeatureEngineer:
    """Transform KenPom stats into ML features."""

    FEATURE_NAMES = [
        "em_diff",
        "tempo_avg",
        "tempo_diff",
        "oe_diff",
        "de_diff",
        "pythag_diff",
        "sos_diff",
        "home_advantage",
        "em_tempo_interaction",
        # APL (Average Possession Length) features
        "apl_off_diff",
        "apl_def_diff",
        "apl_off_mismatch_team1",
        "apl_off_mismatch_team2",
        "tempo_control_factor",
    ]

    @staticmethod
    def create_features(
        team1_stats: dict[str, Any],
        team2_stats: dict[str, Any],
        neutral_site: bool = True,
        home_team1: bool = False,
    ) -> dict[str, float]:
        """Create feature vector for prediction.

        Args:
            team1_stats: Team 1 KenPom statistics (AdjEM, AdjO, AdjD, AdjT, etc.).
            team2_stats: Team 2 KenPom statistics.
            neutral_site: Whether game is at neutral site.
            home_team1: If not neutral, whether team 1 is home team.

        Returns:
            Dictionary with engineered features.
        """
        features: dict[str, float] = {}

        # Efficiency margin difference
        features["em_diff"] = float(team1_stats["AdjEM"] - team2_stats["AdjEM"])

        # Tempo features
        team1_tempo = float(team1_stats["AdjT"])
        team2_tempo = float(team2_stats["AdjT"])
        features["tempo_avg"] = (team1_tempo + team2_tempo) / 2
        features["tempo_diff"] = team1_tempo - team2_tempo

        # Offensive/Defensive splits
        features["oe_diff"] = float(team1_stats["AdjO"] - team2_stats["AdjO"])
        features["de_diff"] = float(team1_stats["AdjD"] - team2_stats["AdjD"])

        # Strength metrics (use .get() with default 0 for optional fields)
        features["pythag_diff"] = float(
            team1_stats.get("Pythag", 0) - team2_stats.get("Pythag", 0)
        )
        features["sos_diff"] = float(
            team1_stats.get("SOS", 0) - team2_stats.get("SOS", 0)
        )

        # Home advantage
        if neutral_site:
            features["home_advantage"] = 0.0
        else:
            features["home_advantage"] = 1.0 if home_team1 else -1.0

        # Interaction terms (high tempo favors better team)
        features["em_tempo_interaction"] = features["em_diff"] * features["tempo_avg"]

        # APL (Average Possession Length) features
        # Use .get() with defaults for APL fields (may not be available in all data)
        team1_apl_off = float(team1_stats.get("APL_Off", 17.5))
        team2_apl_off = float(team2_stats.get("APL_Off", 17.5))
        team1_apl_def = float(team1_stats.get("APL_Def", 17.5))
        team2_apl_def = float(team2_stats.get("APL_Def", 17.5))

        features["apl_off_diff"] = team1_apl_off - team2_apl_off
        features["apl_def_diff"] = team1_apl_def - team2_apl_def

        # APL mismatch: Team's offensive pace vs opponent's defensive pace allowed
        features["apl_off_mismatch_team1"] = team1_apl_off - team2_apl_def
        features["apl_off_mismatch_team2"] = team2_apl_off - team1_apl_def

        # Tempo control factor using tempo_analysis module
        # Only calculate if APL data is available
        if all(k in team1_stats for k in ["APL_Off", "APL_Def", "AdjEM"]) and all(
            k in team2_stats for k in ["APL_Off", "APL_Def", "AdjEM"]
        ):
            from .tempo_analysis import TempoMatchupAnalyzer

            analyzer = TempoMatchupAnalyzer()
            control = analyzer.calculate_tempo_control(team1_stats, team2_stats)
            features["tempo_control_factor"] = control
        else:
            features["tempo_control_factor"] = 0.0

        return features


class GamePredictor:
    """Machine learning-based game prediction with confidence intervals.

    Uses Gradient Boosting for margin prediction and quantile regression
    for confidence intervals. Ridge regression predicts total score.

    Example:
        >>> predictor = GamePredictor()
        >>> predictor.fit(games_df, margins, totals)
        >>> result = predictor.predict_with_confidence(team1_stats, team2_stats)
        >>> print(f"Margin: {result.predicted_margin} ({result.confidence_interval})")
    """

    def __init__(self) -> None:
        """Initialize predictor with default hyperparameters."""
        # Margin prediction (point estimate)
        self.margin_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_split=10,
            random_state=42,
        )

        # Confidence interval models (quantile regression)
        self.margin_upper = GradientBoostingRegressor(
            loss="quantile",
            alpha=0.75,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )

        self.margin_lower = GradientBoostingRegressor(
            loss="quantile",
            alpha=0.25,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )

        # Total prediction (Ridge regression is fast and stable)
        self.total_model = Ridge(alpha=1.0)

        self.feature_engineer = FeatureEngineer()
        self.is_fitted = False

    def fit(
        self,
        games_df: pd.DataFrame,
        margins: pd.Series,
        totals: pd.Series,
    ) -> None:
        """Train models on historical game data.

        Args:
            games_df: DataFrame with feature columns (em_diff, tempo_avg, etc.).
            margins: Series of actual margins (team1_score - team2_score).
            totals: Series of actual totals (team1_score + team2_score).

        Raises:
            ValueError: If feature columns are missing from games_df.
        """
        # Validate feature columns
        missing_features = set(FeatureEngineer.FEATURE_NAMES) - set(games_df.columns)
        if missing_features:
            raise ValueError(
                f"Missing required feature columns: {missing_features}. "
                f"Expected: {FeatureEngineer.FEATURE_NAMES}"
            )

        # Train margin models
        self.margin_model.fit(games_df, margins)
        self.margin_upper.fit(games_df, margins)
        self.margin_lower.fit(games_df, margins)

        # Train total model
        self.total_model.fit(games_df, totals)

        self.is_fitted = True

    def predict_with_confidence(
        self,
        team1_stats: dict[str, Any],
        team2_stats: dict[str, Any],
        neutral_site: bool = True,
        home_team1: bool = False,
    ) -> PredictionResult:
        """Predict game outcome with confidence intervals.

        Args:
            team1_stats: Team 1 KenPom statistics.
            team2_stats: Team 2 KenPom statistics.
            neutral_site: Whether game is at neutral site.
            home_team1: If not neutral, whether team 1 is home team.

        Returns:
            PredictionResult with margin, total, scores, and confidence interval.

        Raises:
            ValueError: If model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )

        # Create features
        features = self.feature_engineer.create_features(
            team1_stats, team2_stats, neutral_site, home_team1
        )
        feature_array = pd.DataFrame([features])

        # Predict margin and bounds
        predicted_margin = float(self.margin_model.predict(feature_array)[0])
        upper_margin = float(self.margin_upper.predict(feature_array)[0])
        lower_margin = float(self.margin_lower.predict(feature_array)[0])

        # Ensure confidence interval is properly ordered (handle quantile crossing)
        if lower_margin > upper_margin:
            lower_margin, upper_margin = upper_margin, lower_margin

        # Apply tempo-based confidence adjustment
        # Slow games have higher variance (fewer possessions = more randomness)
        if all(k in team1_stats for k in ["APL_Off", "APL_Def", "AdjTempo"]) and all(
            k in team2_stats for k in ["APL_Off", "APL_Def", "AdjTempo"]
        ):
            from .tempo_analysis import TempoMatchupAnalyzer

            tempo_analyzer = TempoMatchupAnalyzer()
            tempo_analysis = tempo_analyzer.analyze_pace_matchup(
                team1_stats, team2_stats
            )

            # Adjust confidence interval width based on tempo
            # confidence_adjustment > 1 means wider CI (slower game)
            # confidence_adjustment < 1 means narrower CI (faster game)
            base_interval_width = upper_margin - lower_margin
            adjusted_width = base_interval_width * tempo_analysis.confidence_adjustment

            # Recalculate bounds
            lower_margin = predicted_margin - (adjusted_width / 2)
            upper_margin = predicted_margin + (adjusted_width / 2)

        # Predict total
        predicted_total = float(self.total_model.predict(feature_array)[0])

        # Calculate team scores
        team1_score = (predicted_total + predicted_margin) / 2
        team2_score = (predicted_total - predicted_margin) / 2

        # Estimate win probability (assuming normal distribution)
        # IQR (interquartile range) to standard deviation conversion: σ ≈ IQR / 1.35
        margin_std = max((upper_margin - lower_margin) / 1.35, 1.0)
        z_score = predicted_margin / margin_std
        # Sigmoid approximation: Φ(z) ≈ 0.5 * (1 + tanh(z/2))
        win_prob = 0.5 * (1.0 + float(np.tanh(z_score / 2)))

        return PredictionResult(
            predicted_margin=round(predicted_margin, 1),
            predicted_total=round(predicted_total, 1),
            confidence_interval=(round(lower_margin, 1), round(upper_margin, 1)),
            team1_score=round(team1_score, 1),
            team2_score=round(team2_score, 1),
            team1_win_prob=round(win_prob, 3),
            confidence_level=0.5,
        )

    def predict_with_injuries(
        self,
        team1_stats: dict[str, Any],
        team2_stats: dict[str, Any],
        team1_injuries: list[Any] | None = None,
        team2_injuries: list[Any] | None = None,
        neutral_site: bool = True,
        home_team1: bool = False,
    ) -> PredictionResult:
        """Predict game outcome with injury adjustments.

        This method adjusts team ratings for injured players before making
        predictions. It uses the InjuryImpact objects to modify AdjEM, AdjOE,
        and AdjDE values, then calls the standard prediction method.

        Args:
            team1_stats: Team 1 KenPom statistics
            team2_stats: Team 2 KenPom statistics
            team1_injuries: List of InjuryImpact objects for team 1
            team2_injuries: List of InjuryImpact objects for team 2
            neutral_site: Whether game is at neutral site
            home_team1: If not neutral, whether team 1 is home team

        Returns:
            PredictionResult with margin, total, scores, and confidence interval
            based on injury-adjusted team ratings

        Example:
            >>> from kenp0m_sp0rts_analyzer.player_impact import PlayerImpactModel
            >>> from kenp0m_sp0rts_analyzer.client import KenPomClient
            >>>
            >>> # Get player and team data
            >>> client = KenPomClient()
            >>> player_stats = client.get_playerstats(season=2025)
            >>> duke_players = player_stats[player_stats["Team"] == "Duke"]
            >>>
            >>> # Calculate injury impact
            >>> impact_model = PlayerImpactModel()
            >>> star_value = impact_model.calculate_player_value(
            ...     duke_players.iloc[0].to_dict(), duke_stats
            ... )
            >>> injury = impact_model.estimate_injury_impact(
            ...     star_value, duke_stats, "out"
            ... )
            >>>
            >>> # Predict with injury
            >>> predictor = GamePredictor()
            >>> result = predictor.predict_with_injuries(
            ...     duke_stats, unc_stats, team1_injuries=[injury]
            ... )
        """
        # Copy stats to avoid modifying originals
        team1_adjusted = team1_stats.copy()
        team2_adjusted = team2_stats.copy()

        # Apply injury adjustments to team 1
        if team1_injuries:
            for injury in team1_injuries:
                # Use the most severe injury's adjustments
                # (For multiple injuries, this is simplified - in reality,
                # we'd want to aggregate impacts more carefully)
                team1_adjusted["AdjEM"] = injury.adjusted_adj_em
                team1_adjusted["AdjOE"] = injury.adjusted_adj_oe
                team1_adjusted["AdjDE"] = injury.adjusted_adj_de

        # Apply injury adjustments to team 2
        if team2_injuries:
            for injury in team2_injuries:
                team2_adjusted["AdjEM"] = injury.adjusted_adj_em
                team2_adjusted["AdjOE"] = injury.adjusted_adj_oe
                team2_adjusted["AdjDE"] = injury.adjusted_adj_de

        # Make prediction with adjusted stats
        return self.predict_with_confidence(
            team1_adjusted, team2_adjusted, neutral_site, home_team1
        )


class BacktestingFramework:
    """Validate predictions against historical outcomes.

    Provides train/test split validation and comprehensive performance metrics
    including MAE, RMSE, accuracy, Brier score, and ATS record.

    Example:
        >>> framework = BacktestingFramework()
        >>> metrics = framework.run_backtest(games_df, train_split=0.8)
        >>> print(f"MAE: {metrics.mae_margin} points, Accuracy: {metrics.accuracy:.1%}")
    """

    def run_backtest(
        self,
        games_df: pd.DataFrame,
        train_split: float = 0.8,
    ) -> BacktestMetrics:
        """Run train/test split backtest.

        Args:
            games_df: Historical games with features and outcomes.
                Must have columns: feature columns + actual_margin + actual_total.
            train_split: Fraction of data for training (default 0.8 = 80%).

        Returns:
            BacktestMetrics with performance statistics.

        Raises:
            ValueError: If required columns are missing.
        """
        # Validate required columns
        required_cols = set(FeatureEngineer.FEATURE_NAMES) | {
            "actual_margin",
            "actual_total",
        }
        missing_cols = required_cols - set(games_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Split chronologically (earlier games for training)
        split_idx = int(len(games_df) * train_split)
        train_df = games_df.iloc[:split_idx].copy()
        test_df = games_df.iloc[split_idx:].copy()

        if len(train_df) < 10:
            raise ValueError(
                f"Insufficient training data: {len(train_df)} games. Need at least 10."
            )
        if len(test_df) < 1:
            raise ValueError("No test data available.")

        # Train predictor
        predictor = GamePredictor()
        predictor.fit(
            games_df=train_df[FeatureEngineer.FEATURE_NAMES],
            margins=train_df["actual_margin"],
            totals=train_df["actual_total"],
        )

        # Make predictions on test set
        # We need to reconstruct team stats from features for each game
        # For simplicity, we'll predict directly from test features
        pred_margins = predictor.margin_model.predict(
            test_df[FeatureEngineer.FEATURE_NAMES]
        )
        pred_totals = predictor.total_model.predict(
            test_df[FeatureEngineer.FEATURE_NAMES]
        )

        # For win probability, we need quantile predictions
        pred_upper = predictor.margin_upper.predict(
            test_df[FeatureEngineer.FEATURE_NAMES]
        )
        pred_lower = predictor.margin_lower.predict(
            test_df[FeatureEngineer.FEATURE_NAMES]
        )

        # Calculate win probabilities
        margin_stds = np.maximum((pred_upper - pred_lower) / 1.35, 1.0)
        z_scores = pred_margins / margin_stds
        win_probs = 0.5 * (1.0 + np.tanh(z_scores / 2))

        # Calculate metrics
        return self._calculate_metrics(
            pred_margins=pred_margins,
            pred_totals=pred_totals,
            win_probs=win_probs,
            test_df=test_df,
        )

    def _calculate_metrics(
        self,
        pred_margins: np.ndarray,
        pred_totals: np.ndarray,
        win_probs: np.ndarray,
        test_df: pd.DataFrame,
    ) -> BacktestMetrics:
        """Compute performance metrics.

        Args:
            pred_margins: Predicted margins.
            pred_totals: Predicted totals.
            win_probs: Predicted win probabilities.
            test_df: Test data with actual outcomes.

        Returns:
            BacktestMetrics object.
        """
        actual_margins = test_df["actual_margin"].values
        actual_totals = test_df["actual_total"].values

        # Regression metrics
        mae_margin = float(mean_absolute_error(actual_margins, pred_margins))
        rmse_margin = float(np.sqrt(mean_squared_error(actual_margins, pred_margins)))
        mae_total = float(mean_absolute_error(actual_totals, pred_totals))
        rmse_total = float(np.sqrt(mean_squared_error(actual_totals, pred_totals)))
        r2_margin = float(r2_score(actual_margins, pred_margins))

        # Classification accuracy (correct winner)
        pred_winners = pred_margins > 0
        actual_winners = actual_margins > 0
        accuracy = float((pred_winners == actual_winners).mean())

        # Brier score (probability calibration)
        brier_score = float(np.mean((win_probs - actual_winners.astype(float)) ** 2))

        # ATS record (if line data available)
        if "spread" in test_df.columns:
            # Team 1 covers if: (actual_margin + spread) > 0
            # We predicted correctly if: sign(pred_margin + spread) == sign(actual_margin)
            ats_correct = (pred_margins + test_df["spread"].values) * actual_margins > 0
            ats_wins = int(ats_correct.sum())
            ats_total = len(test_df)
            ats_record = (ats_wins, ats_total - ats_wins)
            ats_percentage = float(ats_wins / ats_total)
        else:
            ats_record = (0, 0)
            ats_percentage = 0.0

        return BacktestMetrics(
            mae_margin=round(mae_margin, 2),
            rmse_margin=round(rmse_margin, 2),
            mae_total=round(mae_total, 2),
            rmse_total=round(rmse_total, 2),
            accuracy=round(accuracy, 3),
            brier_score=round(brier_score, 3),
            r2_margin=round(r2_margin, 3),
            ats_record=ats_record,
            ats_percentage=round(ats_percentage, 3),
        )

    def cross_validate(
        self,
        games_df: pd.DataFrame,
        n_folds: int = 5,
    ) -> list[BacktestMetrics]:
        """Run k-fold cross-validation.

        Args:
            games_df: Historical games with features and outcomes.
            n_folds: Number of folds for cross-validation (default 5).

        Returns:
            List of BacktestMetrics, one per fold.

        Raises:
            ValueError: If n_folds < 2 or insufficient data.
        """
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")

        if len(games_df) < n_folds:
            raise ValueError(
                f"Insufficient data for {n_folds}-fold CV: {len(games_df)} games"
            )

        fold_size = len(games_df) // n_folds
        metrics_list = []

        for i in range(n_folds):
            # Create fold indices
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < n_folds - 1 else len(games_df)

            # Split data
            test_df = games_df.iloc[test_start:test_end].copy()
            train_df = pd.concat(
                [games_df.iloc[:test_start], games_df.iloc[test_end:]]
            ).copy()

            # Train and evaluate
            predictor = GamePredictor()
            predictor.fit(
                games_df=train_df[FeatureEngineer.FEATURE_NAMES],
                margins=train_df["actual_margin"],
                totals=train_df["actual_total"],
            )

            # Predict
            pred_margins = predictor.margin_model.predict(
                test_df[FeatureEngineer.FEATURE_NAMES]
            )
            pred_totals = predictor.total_model.predict(
                test_df[FeatureEngineer.FEATURE_NAMES]
            )
            pred_upper = predictor.margin_upper.predict(
                test_df[FeatureEngineer.FEATURE_NAMES]
            )
            pred_lower = predictor.margin_lower.predict(
                test_df[FeatureEngineer.FEATURE_NAMES]
            )

            margin_stds = np.maximum((pred_upper - pred_lower) / 1.35, 1.0)
            z_scores = pred_margins / margin_stds
            win_probs = 0.5 * (1.0 + np.tanh(z_scores / 2))

            # Calculate metrics
            metrics = self._calculate_metrics(
                pred_margins=pred_margins,
                pred_totals=pred_totals,
                win_probs=win_probs,
                test_df=test_df,
            )
            metrics_list.append(metrics)

        return metrics_list

    def calibration_plot_data(
        self,
        predictions: list[PredictionResult],
        actuals: pd.DataFrame,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """Generate calibration curve data for plotting.

        Args:
            predictions: List of prediction results.
            actuals: DataFrame with actual outcomes (must have 'actual_margin' column).
            n_bins: Number of bins for calibration curve (default 10).

        Returns:
            DataFrame with columns: bin_center, actual_win_rate, count.
        """
        if len(predictions) != len(actuals):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs {len(actuals)} actuals"
            )

        # Extract probabilities and outcomes
        win_probs = np.array([p.team1_win_prob for p in predictions])
        actual_wins = (actuals["actual_margin"].values > 0).astype(float)

        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(win_probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Calculate actual win rate in each bin
        calibration_data = []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_center = (bins[i] + bins[i + 1]) / 2
                actual_rate = float(actual_wins[mask].mean())
                count = int(mask.sum())
                calibration_data.append(
                    {
                        "bin_center": round(bin_center, 2),
                        "actual_win_rate": round(actual_rate, 3),
                        "count": count,
                    }
                )

        return pd.DataFrame(calibration_data)
