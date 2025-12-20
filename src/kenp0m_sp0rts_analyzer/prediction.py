"""Machine learning-based game prediction with confidence intervals."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


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
            raise ValueError(
                f"Lower bound {v[0]} must be <= upper bound {v[1]}"
            )
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
        features["em_diff"] = float(
            team1_stats["AdjEM"] - team2_stats["AdjEM"]
        )

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
        features["em_tempo_interaction"] = (
            features["em_diff"] * features["tempo_avg"]
        )

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

        # Tempo control factor (simplified - advanced tempo analysis removed)
        # TODO: Restore tempo_analysis module for advanced tempo control features
        features["tempo_control_factor"] = 0.0

        return features


class XGBoostFeatureEngineer(FeatureEngineer):
    """Enhanced feature engineering for XGBoost with Phase 2 features.

    Adds three critical feature categories for betting edge detection:
    1. Luck regression - Identify teams due for regression to mean
    2. Point distribution - Detect high-variance 3PT-dependent teams
    3. Momentum tracking - Track team trajectories (hot/cold streaks)

    Example:
        >>> engineer = XGBoostFeatureEngineer()
        >>> features = engineer.create_enhanced_features(
        ...     team1_stats, team2_stats,
        ...     team1_history=[...],  # Last 4 weeks of ratings
        ...     point_dist_team1={...},  # Point distribution stats
        ...     point_dist_team2={...}
        ... )
        >>> print(features['luck_regression_team1'])  # Expected regression
        >>> print(features['three_point_dependency_avg'])  # Variance indicator
    """

    # Extended feature list with Phase 2 enhancements
    ENHANCED_FEATURE_NAMES = [
        # Base features (14)
        *FeatureEngineer.FEATURE_NAMES,
        # Luck regression features (4)
        "luck_team1",
        "luck_team2",
        "luck_diff",
        "luck_regression_expected",
        # Point distribution features (6)
        "three_point_pct_team1",
        "three_point_pct_team2",
        "three_point_dependency_avg",
        "scoring_variance_score",
        "ft_reliance_diff",
        "two_point_reliance_diff",
        # Momentum features (3)
        "momentum_score_team1",
        "momentum_score_team2",
        "momentum_diff",
        # PHASE 1.6: Misc Stats features (8)
        "fg3_pct_diff",
        "fg2_pct_diff",
        "ft_pct_diff",
        "assist_rate_diff",
        "steal_rate_diff",
        "block_pct_diff",
        "shooting_quality_team1",
        "shooting_quality_team2",
        # PHASE 2.2: Height/Experience features (4)
        "height_diff",
        "experience_diff",
        "continuity_diff",
        "bench_minutes_diff",
        # PHASE 2.2: Archive momentum features (3) - Total: 42 features
        "rating_momentum_4wk",
        "rating_acceleration",
        "preseason_deviation",
    ]

    @staticmethod
    def create_enhanced_features(
        team1_stats: dict[str, Any],
        team2_stats: dict[str, Any],
        neutral_site: bool = True,
        home_team1: bool = False,
        team1_history: list[dict] | None = None,
        team2_history: list[dict] | None = None,
        point_dist_team1: dict | None = None,
        point_dist_team2: dict | None = None,
        misc_stats_team1: dict | None = None,
        misc_stats_team2: dict | None = None,
        height_team1: dict | None = None,
        height_team2: dict | None = None,
    ) -> dict[str, float]:
        """Create enhanced feature vector with Phase 2.2 features.

        Args:
            team1_stats: Team 1 KenPom statistics (must include Luck field)
            team2_stats: Team 2 KenPom statistics
            neutral_site: Whether game is at neutral site
            home_team1: If not neutral, whether team 1 is home team
            team1_history: Historical ratings for momentum (list of dicts with AdjEM)
            team2_history: Historical ratings for momentum
            point_dist_team1: Point distribution stats (FT_Pct, TwoP_Pct, ThreeP_Pct)
            point_dist_team2: Point distribution stats
            misc_stats_team1: Misc stats (FG3Pct, FG2Pct, FTPct, ARate, etc.)
            misc_stats_team2: Misc stats
            height_team1: Height/experience stats (EffHeight, Experience, Continuity, Bench)
            height_team2: Height/experience stats

        Returns:
            Dictionary with 42 engineered features (14 base + 28 enhanced)
        """
        # Start with base features
        features = FeatureEngineer.create_features(
            team1_stats, team2_stats, neutral_site, home_team1
        )

        # --- TIER 0: Luck Regression Features ---
        luck1 = float(team1_stats.get("Luck", 0))
        luck2 = float(team2_stats.get("Luck", 0))

        features["luck_team1"] = luck1
        features["luck_team2"] = luck2
        features["luck_diff"] = luck1 - luck2

        # Expected regression: Lucky teams (Luck > 0.03) regress downward
        # Unlucky teams (Luck < -0.03) regress upward
        # Regression factor: -0.5 * Luck (50% regression to mean)
        if abs(luck1 - luck2) > 0.05:  # Significant luck differential
            features["luck_regression_expected"] = -(luck1 - luck2) * 0.5
        else:
            features["luck_regression_expected"] = 0.0

        # --- TIER 0: Point Distribution Features ---
        if point_dist_team1 and point_dist_team2:
            # Extract point distribution percentages
            three_pct1 = float(point_dist_team1.get("ThreeP_Pct", 33.0))
            three_pct2 = float(point_dist_team2.get("ThreeP_Pct", 33.0))
            two_pct1 = float(point_dist_team1.get("TwoP_Pct", 50.0))
            two_pct2 = float(point_dist_team2.get("TwoP_Pct", 50.0))
            ft_pct1 = float(point_dist_team1.get("FT_Pct", 17.0))
            ft_pct2 = float(point_dist_team2.get("FT_Pct", 17.0))

            features["three_point_pct_team1"] = three_pct1
            features["three_point_pct_team2"] = three_pct2

            # Average 3PT dependency (variance indicator)
            # High 3PT% = high variance = good for underdogs
            three_pt_avg = (three_pct1 + three_pct2) / 2
            features["three_point_dependency_avg"] = three_pt_avg

            # Scoring variance score (0.0 = low variance, 1.0 = high variance)
            # Games with >35% 3PT scoring have high variance
            if three_pt_avg > 35:
                features["scoring_variance_score"] = 1.0
            elif three_pt_avg > 33:
                features["scoring_variance_score"] = 0.5
            else:
                features["scoring_variance_score"] = 0.0

            # FT and 2PT reliance differences
            features["ft_reliance_diff"] = ft_pct1 - ft_pct2
            features["two_point_reliance_diff"] = two_pct1 - two_pct2

        else:
            # Default values if point distribution not available
            features["three_point_pct_team1"] = 33.0
            features["three_point_pct_team2"] = 33.0
            features["three_point_dependency_avg"] = 33.0
            features["scoring_variance_score"] = 0.5
            features["ft_reliance_diff"] = 0.0
            features["two_point_reliance_diff"] = 0.0

        # --- TIER 1: Momentum Features ---
        if team1_history and len(team1_history) >= 2:
            features["momentum_score_team1"] = (
                XGBoostFeatureEngineer._calculate_momentum(team1_history)
            )
        else:
            features["momentum_score_team1"] = 0.0

        if team2_history and len(team2_history) >= 2:
            features["momentum_score_team2"] = (
                XGBoostFeatureEngineer._calculate_momentum(team2_history)
            )
        else:
            features["momentum_score_team2"] = 0.0

        features["momentum_diff"] = (
            features["momentum_score_team1"] - features["momentum_score_team2"]
        )

        # --- PHASE 1.6: Misc Stats Features (8 features: 27 → 35) ---
        if misc_stats_team1 and misc_stats_team2:
            # Shooting percentage differentials
            features["fg3_pct_diff"] = float(
                misc_stats_team1.get("FG3Pct", 35.0)
            ) - float(misc_stats_team2.get("FG3Pct", 35.0))

            features["fg2_pct_diff"] = float(
                misc_stats_team1.get("FG2Pct", 50.0)
            ) - float(misc_stats_team2.get("FG2Pct", 50.0))

            features["ft_pct_diff"] = float(
                misc_stats_team1.get("FTPct", 70.0)
            ) - float(misc_stats_team2.get("FTPct", 70.0))

            # Advanced metrics differentials
            features["assist_rate_diff"] = float(
                misc_stats_team1.get("ARate", 50.0)
            ) - float(misc_stats_team2.get("ARate", 50.0))

            features["steal_rate_diff"] = float(
                misc_stats_team1.get("StlRate", 10.0)
            ) - float(misc_stats_team2.get("StlRate", 10.0))

            features["block_pct_diff"] = float(
                misc_stats_team1.get("BlockPct", 10.0)
            ) - float(misc_stats_team2.get("BlockPct", 10.0))

            # Composite shooting quality (weighted: 2PT 40%, 3PT 35%, FT 25%)
            # Higher = better overall shooting team
            t1_quality = (
                0.40 * float(misc_stats_team1.get("FG2Pct", 50.0))
                + 0.35 * float(misc_stats_team1.get("FG3Pct", 35.0))
                + 0.25 * float(misc_stats_team1.get("FTPct", 70.0))
            )
            t2_quality = (
                0.40 * float(misc_stats_team2.get("FG2Pct", 50.0))
                + 0.35 * float(misc_stats_team2.get("FG3Pct", 35.0))
                + 0.25 * float(misc_stats_team2.get("FTPct", 70.0))
            )
            features["shooting_quality_team1"] = t1_quality
            features["shooting_quality_team2"] = t2_quality

        else:
            # Default values when misc stats not available
            features["fg3_pct_diff"] = 0.0
            features["fg2_pct_diff"] = 0.0
            features["ft_pct_diff"] = 0.0
            features["assist_rate_diff"] = 0.0
            features["steal_rate_diff"] = 0.0
            features["block_pct_diff"] = 0.0
            features["shooting_quality_team1"] = (
                52.5  # Weighted avg of defaults
            )
            features["shooting_quality_team2"] = 52.5

        # --- PHASE 2.2: Height/Experience Features (4 features: 35 → 39) ---
        if height_team1 and height_team2:
            # Effective height differential (inches)
            features["height_diff"] = float(
                height_team1.get("EffHeight", 78.0)
            ) - float(height_team2.get("EffHeight", 78.0))

            # Experience differential (years)
            features["experience_diff"] = float(
                height_team1.get("Experience", 2.0)
            ) - float(height_team2.get("Experience", 2.0))

            # Continuity differential (percentage of minutes returning)
            features["continuity_diff"] = float(
                height_team1.get("Continuity", 50.0)
            ) - float(height_team2.get("Continuity", 50.0))

            # Bench minutes differential (depth)
            features["bench_minutes_diff"] = float(
                height_team1.get("Bench", 30.0)
            ) - float(height_team2.get("Bench", 30.0))
        else:
            # Defaults when height/experience not available
            features["height_diff"] = 0.0
            features["experience_diff"] = 0.0
            features["continuity_diff"] = 0.0
            features["bench_minutes_diff"] = 0.0

        # --- PHASE 2.2: Archive Momentum Features (3 features: 39 → 42) ---
        # Calculate 4-week rating momentum using historical snapshots
        if team1_history and len(team1_history) >= 4:
            em_recent = float(team1_history[0].get("AdjEM", 0))
            em_4wk = float(team1_history[3].get("AdjEM", em_recent))
            t1_momentum = em_recent - em_4wk

            # Acceleration: recent change vs earlier change
            if len(team1_history) >= 2:
                em_1wk = float(team1_history[1].get("AdjEM", em_recent))
                t1_accel = (em_recent - em_1wk) - (em_1wk - em_4wk)
            else:
                t1_accel = 0.0

            # Preseason deviation: how much has team changed from preseason?
            preseason_em = float(team1_history[-1].get("AdjEM", em_recent))
            t1_preseason = em_recent - preseason_em
        else:
            t1_momentum = t1_accel = t1_preseason = 0.0

        # Same for team2
        if team2_history and len(team2_history) >= 4:
            em_recent = float(team2_history[0].get("AdjEM", 0))
            em_4wk = float(team2_history[3].get("AdjEM", em_recent))
            t2_momentum = em_recent - em_4wk

            if len(team2_history) >= 2:
                em_1wk = float(team2_history[1].get("AdjEM", em_recent))
                t2_accel = (em_recent - em_1wk) - (em_1wk - em_4wk)
            else:
                t2_accel = 0.0

            preseason_em = float(team2_history[-1].get("AdjEM", em_recent))
            t2_preseason = em_recent - preseason_em
        else:
            t2_momentum = t2_accel = t2_preseason = 0.0

        # Differential features
        features["rating_momentum_4wk"] = t1_momentum - t2_momentum
        features["rating_acceleration"] = t1_accel - t2_accel
        features["preseason_deviation"] = t1_preseason - t2_preseason

        return features

    @staticmethod
    def _calculate_momentum(history: list[dict]) -> float:
        """Calculate momentum score from historical ratings.

        Momentum is calculated as the slope of AdjEM over recent games.
        Positive = improving (surging), Negative = declining (slumping)

        Args:
            history: List of historical ratings (newest to oldest)
                     Each dict must have 'AdjEM' key
                     Example: [
                         {'date': '2025-01-15', 'AdjEM': 20.5},
                         {'date': '2025-01-08', 'AdjEM': 19.2},
                         ...
                     ]

        Returns:
            Momentum score (points per week):
                > +1.0 = surging (back this team)
                > +0.5 = improving
                -0.5 to +0.5 = stable
                < -0.5 = declining
                < -1.0 = slumping (fade this team)
        """
        if len(history) < 2:
            return 0.0

        # Extract AdjEM values (newest to oldest)
        em_values = [float(h.get("AdjEM", 0)) for h in history]

        # Calculate simple slope: (newest - oldest) / num_periods
        # This gives points per week if history is weekly snapshots
        slope = (em_values[0] - em_values[-1]) / (len(em_values) - 1)

        return slope


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
        missing_features = set(FeatureEngineer.FEATURE_NAMES) - set(
            games_df.columns
        )
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
        team1_history: list[dict] | None = None,
        team2_history: list[dict] | None = None,
        point_dist_team1: dict | None = None,
        point_dist_team2: dict | None = None,
        misc_stats_team1: dict | None = None,
        misc_stats_team2: dict | None = None,
        height_team1: dict | None = None,
        height_team2: dict | None = None,
    ) -> PredictionResult:
        """Predict game outcome with confidence intervals.

        Args:
            team1_stats: Team 1 KenPom statistics.
            team2_stats: Team 2 KenPom statistics.
            neutral_site: Whether game is at neutral site.
            home_team1: If not neutral, whether team 1 is home team.
            team1_history: Optional historical ratings for momentum.
            team2_history: Optional historical ratings for momentum.
            point_dist_team1: Optional point distribution stats.
            point_dist_team2: Optional point distribution stats.
            misc_stats_team1: Optional misc stats for team 1 (Phase 1.6).
            misc_stats_team2: Optional misc stats for team 2 (Phase 1.6).
            height_team1: Optional height/experience stats for team 1 (Phase 2.2).
            height_team2: Optional height/experience stats for team 2 (Phase 2.2).

        Returns:
            PredictionResult with margin, total, scores, and confidence interval.

        Raises:
            ValueError: If model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )

        # Create features using appropriate method based on mode
        if self.use_enhanced_features:
            features = XGBoostFeatureEngineer.create_enhanced_features(
                team1_stats,
                team2_stats,
                neutral_site,
                home_team1,
                team1_history,
                team2_history,
                point_dist_team1,
                point_dist_team2,
                misc_stats_team1,
                misc_stats_team2,
                height_team1,
                height_team2,
            )
        else:
            features = FeatureEngineer.create_features(
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

        # Note: Advanced tempo-based confidence adjustment removed
        # (requires tempo_analysis module restoration)
        # TODO: Add back tempo confidence adjustment when tempo_analysis is restored

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
            confidence_interval=(
                round(lower_margin, 1),
                round(upper_margin, 1),
            ),
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


class XGBoostGamePredictor:
    """XGBoost-based game predictor (drop-in replacement for GamePredictor).

    Uses XGBoost for better performance, regularization, and feature importance.
    Maintains same interface as GamePredictor for backward compatibility.

    Benefits over sklearn GradientBoosting:
    - Better regularization (L1/L2) to prevent overfitting
    - 10-20% faster training with parallel tree construction
    - Built-in feature importance (weight, gain, cover)
    - Early stopping support
    - Native handling of missing values

    Example:
        >>> # Base features (14)
        >>> predictor = XGBoostGamePredictor()
        >>> predictor.fit(games_df, margins, totals)
        >>>
        >>> # Enhanced features (27) for betting edge detection
        >>> predictor = XGBoostGamePredictor(use_enhanced_features=True)
        >>> predictor.fit(enhanced_games_df, margins, totals)
        >>> result = predictor.predict_with_confidence(team1_stats, team2_stats)
        >>> print(f"Margin: {result.predicted_margin} ({result.confidence_interval})")
        >>> importance = predictor.get_feature_importance(importance_type='gain')
    """

    def __init__(self, use_enhanced_features: bool = False) -> None:
        """Initialize XGBoost models with optimized hyperparameters.

        Args:
            use_enhanced_features: If True, use 27 enhanced features including
                luck regression, point distribution, and momentum. If False,
                use 14 base features only.
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Install with: uv add xgboost"
            )

        # Track which feature set to use
        self.use_enhanced_features = use_enhanced_features

        # Margin prediction (point estimate)
        self.margin_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            reg_alpha=0.1,  # L1 regularization (feature selection)
            reg_lambda=1.0,  # L2 regularization (prevent overfitting)
            gamma=0.1,  # Minimum loss reduction for split
            subsample=0.8,  # Row sampling (80% per tree)
            colsample_bytree=0.8,  # Column sampling (80% features per tree)
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
        )

        # Quantile regression models (confidence intervals)
        quantile_params = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }

        self.margin_upper = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=0.75,
            **quantile_params,
        )

        self.margin_lower = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=0.25,
            **quantile_params,
        )

        # Total prediction
        self.total_model = xgb.XGBRegressor(**quantile_params)

        # Use appropriate feature engineer based on mode
        self.feature_engineer = (
            XGBoostFeatureEngineer()
            if use_enhanced_features
            else FeatureEngineer()
        )
        self.is_fitted = False

    def fit(
        self,
        games_df: pd.DataFrame,
        margins: pd.Series,
        totals: pd.Series,
        eval_set: tuple | None = None,
        early_stopping_rounds: int = 20,
        verbose: bool = False,
    ) -> dict[str, float]:
        """Train XGBoost models on historical game data.

        Args:
            games_df: DataFrame with feature columns.
            margins: Series of actual margins (team1_score - team2_score).
            totals: Series of actual totals (team1_score + team2_score).
            eval_set: Optional (X_val, y_margin_val, y_total_val) for early stopping.
            early_stopping_rounds: Stop if no improvement for N rounds.
            verbose: Print training progress.

        Returns:
            Dictionary with best scores for each model.

        Raises:
            ValueError: If feature columns are missing from games_df.
        """
        # Validate feature columns based on mode
        expected_features = (
            XGBoostFeatureEngineer.ENHANCED_FEATURE_NAMES
            if self.use_enhanced_features
            else FeatureEngineer.FEATURE_NAMES
        )
        missing_features = set(expected_features) - set(games_df.columns)
        if missing_features:
            raise ValueError(
                f"Missing required feature columns: {missing_features}. "
                f"Expected: {expected_features}"
            )

        # Train models (simple approach for Phase 1)
        # Note: Early stopping will be added in Phase 3 during hyperparameter optimization
        self.margin_model.fit(games_df, margins)
        self.margin_upper.fit(games_df, margins)
        self.margin_lower.fit(games_df, margins)
        self.total_model.fit(games_df, totals)

        self.is_fitted = True

        # Return empty results dict (early stopping metrics will be added in Phase 3)
        return {}

    def predict_with_confidence(
        self,
        team1_stats: dict[str, Any],
        team2_stats: dict[str, Any],
        neutral_site: bool = True,
        home_team1: bool = False,
        team1_history: list[dict] | None = None,
        team2_history: list[dict] | None = None,
        point_dist_team1: dict | None = None,
        point_dist_team2: dict | None = None,
        misc_stats_team1: dict | None = None,
        misc_stats_team2: dict | None = None,
        height_team1: dict | None = None,
        height_team2: dict | None = None,
    ) -> PredictionResult:
        """Predict game outcome with confidence intervals.

        Args:
            team1_stats: Team 1 KenPom statistics.
            team2_stats: Team 2 KenPom statistics.
            neutral_site: Whether game is at neutral site.
            home_team1: If not neutral, whether team 1 is home team.
            team1_history: Optional historical ratings for momentum.
            team2_history: Optional historical ratings for momentum.
            point_dist_team1: Optional point distribution stats.
            point_dist_team2: Optional point distribution stats.
            misc_stats_team1: Optional misc stats for team 1 (Phase 1.6).
            misc_stats_team2: Optional misc stats for team 2 (Phase 1.6).
            height_team1: Optional height/experience stats for team 1 (Phase 2.2).
            height_team2: Optional height/experience stats for team 2 (Phase 2.2).

        Returns:
            PredictionResult with margin, total, scores, and confidence interval.

        Raises:
            ValueError: If model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )

        # Create features (use enhanced if configured)
        if self.use_enhanced_features:
            features = self.feature_engineer.create_enhanced_features(
                team1_stats,
                team2_stats,
                neutral_site,
                home_team1,
                team1_history,
                team2_history,
                point_dist_team1,
                point_dist_team2,
                misc_stats_team1,
                misc_stats_team2,
                height_team1,
                height_team2,
            )
        else:
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

        # Note: Advanced tempo-based confidence adjustment removed
        # (requires tempo_analysis module restoration)
        # TODO: Add back tempo confidence adjustment when tempo_analysis is restored

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
            confidence_interval=(
                round(lower_margin, 1),
                round(upper_margin, 1),
            ),
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
        predictions. Maintains same interface as GamePredictor.

        Args:
            team1_stats: Team 1 KenPom statistics
            team2_stats: Team 2 KenPom statistics
            team1_injuries: List of InjuryImpact objects for team 1
            team2_injuries: List of InjuryImpact objects for team 2
            neutral_site: Whether game is at neutral site
            home_team1: If not neutral, whether team 1 is home team

        Returns:
            PredictionResult with injury-adjusted predictions
        """
        # Copy stats to avoid modifying originals
        team1_adjusted = team1_stats.copy()
        team2_adjusted = team2_stats.copy()

        # Apply injury adjustments to team 1
        if team1_injuries:
            for injury in team1_injuries:
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

    def get_feature_importance(
        self, importance_type: str = "gain", top_n: int = 15
    ) -> pd.DataFrame:
        """Get feature importance from trained model.

        XGBoost provides three types of feature importance:
        - 'weight': Number of times a feature is used for splits
        - 'gain': Average gain (loss reduction) when using this feature
        - 'cover': Average number of samples affected by splits on this feature

        Args:
            importance_type: Type of importance ('weight', 'gain', or 'cover')
            top_n: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores, sorted by importance

        Raises:
            ValueError: If model has not been fitted yet or invalid importance_type

        Example:
            >>> predictor.fit(games_df, margins, totals)
            >>> importance = predictor.get_feature_importance(importance_type='gain')
            >>> print(importance.head(10))
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting importance")

        if importance_type not in ["weight", "gain", "cover"]:
            raise ValueError(
                f"Invalid importance_type: {importance_type}. "
                "Must be 'weight', 'gain', or 'cover'"
            )

        # Get importance dictionary from XGBoost model
        importance = self.margin_model.get_booster().get_score(
            importance_type=importance_type
        )

        # Convert to DataFrame
        df = pd.DataFrame(
            [{"feature": k, "importance": v} for k, v in importance.items()]
        )

        # Sort and return top N
        return df.sort_values("importance", ascending=False).head(top_n)

    def load_model(
        self, margin_path: str, total_path: str | None = None
    ) -> None:
        """Load trained XGBoost models from disk.

        Args:
            margin_path: Path to margin model JSON file
            total_path: Path to total model JSON file (optional, will infer from margin_path)

        Raises:
            FileNotFoundError: If model files don't exist
            ValueError: If models can't be loaded

        Example:
            >>> predictor = XGBoostGamePredictor(use_enhanced_features=True)
            >>> predictor.load_model('data/xgboost_models/margin_model_2025_enhanced.json')
        """
        from pathlib import Path

        margin_path = Path(margin_path)
        if not margin_path.exists():
            raise FileNotFoundError(f"Margin model not found: {margin_path}")

        # Infer total model path if not provided
        if total_path is None:
            total_path = margin_path.parent / margin_path.name.replace(
                "margin_model", "total_model"
            )
        else:
            total_path = Path(total_path)

        if not total_path.exists():
            raise FileNotFoundError(f"Total model not found: {total_path}")

        # Infer quantile model paths
        margin_upper_path = margin_path.parent / margin_path.name.replace(
            "margin_model", "margin_upper"
        )
        margin_lower_path = margin_path.parent / margin_path.name.replace(
            "margin_model", "margin_lower"
        )

        # Validate quantile model files exist
        if not margin_upper_path.exists():
            raise FileNotFoundError(
                f"Upper quantile model not found: {margin_upper_path}"
            )
        if not margin_lower_path.exists():
            raise FileNotFoundError(
                f"Lower quantile model not found: {margin_lower_path}"
            )

        # Load models into booster objects (each from separate file)
        self.margin_model.get_booster().load_model(str(margin_path))
        self.margin_upper.get_booster().load_model(str(margin_upper_path))
        self.margin_lower.get_booster().load_model(str(margin_lower_path))
        self.total_model.get_booster().load_model(str(total_path))

        self.is_fitted = True
        logger.info(f"Loaded models from {margin_path.parent}/")


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
        rmse_margin = float(
            np.sqrt(mean_squared_error(actual_margins, pred_margins))
        )
        mae_total = float(mean_absolute_error(actual_totals, pred_totals))
        rmse_total = float(
            np.sqrt(mean_squared_error(actual_totals, pred_totals))
        )
        r2_margin = float(r2_score(actual_margins, pred_margins))

        # Classification accuracy (correct winner)
        pred_winners = pred_margins > 0
        actual_winners = actual_margins > 0
        accuracy = float((pred_winners == actual_winners).mean())

        # Brier score (probability calibration)
        brier_score = float(
            np.mean((win_probs - actual_winners.astype(float)) ** 2)
        )

        # ATS record (if line data available)
        if "spread" in test_df.columns:
            # Team 1 covers if: (actual_margin + spread) > 0
            # We predicted correctly if: sign(pred_margin + spread) == sign(actual_margin)
            ats_correct = (
                pred_margins + test_df["spread"].values
            ) * actual_margins > 0
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
            test_end = (
                (i + 1) * fold_size if i < n_folds - 1 else len(games_df)
            )

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
