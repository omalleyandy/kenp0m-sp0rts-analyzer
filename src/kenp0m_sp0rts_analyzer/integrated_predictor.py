"""Integrated prediction system combining KenPom data with XGBoost ML.

This module provides a unified interface for:
- KenPom data retrieval and management
- XGBoost model training and prediction
- Vegas line comparison and edge detection
- Accuracy tracking and reporting

Example:
    >>> from kenp0m_sp0rts_analyzer import IntegratedPredictor
    >>>
    >>> predictor = IntegratedPredictor()
    >>>
    >>> # Get prediction for a game
    >>> result = predictor.predict_game(
    ...     home_team="Duke",
    ...     away_team="North Carolina"
    ... )
    >>> print(f"Predicted margin: {result.margin:+.1f}")
    >>> print(f"Win probability: {result.win_prob:.1%}")
"""

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import DatabaseFeatureEngineer, GameFeatures
from .kenpom import (
    AccuracyReport,
    BatchScheduler,
    KenPomService,
    TeamRating,
)
from .models import XGBoostModelWrapper
from .prediction import (
    XGBoostFeatureEngineer,
    XGBoostGamePredictor,
)

logger = logging.getLogger(__name__)


@dataclass
class GameAnalysis:
    """Complete analysis for a single game."""

    home_team: str
    away_team: str
    home_id: int
    away_id: int

    # KenPom ratings
    home_rating: TeamRating
    away_rating: TeamRating

    # Prediction results
    predicted_margin: float
    predicted_total: float
    win_probability: float
    confidence_interval: tuple[float, float]

    # Vegas comparison (if available)
    vegas_spread: float | None = None
    vegas_total: float | None = None
    edge_vs_spread: float | None = None
    edge_vs_total: float | None = None

    # Metadata
    prediction_date: date = None
    model_version: str = "xgboost-v2.0"

    def __post_init__(self):
        if self.prediction_date is None:
            self.prediction_date = date.today()

        # Calculate edges if Vegas lines available
        if self.vegas_spread is not None:
            self.edge_vs_spread = self.predicted_margin - (-self.vegas_spread)
        if self.vegas_total is not None:
            self.edge_vs_total = self.predicted_total - self.vegas_total

    @property
    def has_spread_edge(self) -> bool:
        """Check if there's meaningful edge vs spread (>2 points)."""
        return (
            self.edge_vs_spread is not None and abs(self.edge_vs_spread) >= 2.0
        )

    @property
    def has_total_edge(self) -> bool:
        """Check if there's meaningful edge vs total (>3 points)."""
        return (
            self.edge_vs_total is not None and abs(self.edge_vs_total) >= 3.0
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "predicted_margin": round(self.predicted_margin, 1),
            "predicted_total": round(self.predicted_total, 1),
            "win_probability": round(self.win_probability, 3),
            "confidence_lower": round(self.confidence_interval[0], 1),
            "confidence_upper": round(self.confidence_interval[1], 1),
            "vegas_spread": self.vegas_spread,
            "vegas_total": self.vegas_total,
            "edge_vs_spread": (
                round(self.edge_vs_spread, 1) if self.edge_vs_spread else None
            ),
            "edge_vs_total": (
                round(self.edge_vs_total, 1) if self.edge_vs_total else None
            ),
            "home_adj_em": round(self.home_rating.adj_em, 1),
            "away_adj_em": round(self.away_rating.adj_em, 1),
            "prediction_date": str(self.prediction_date),
        }


class IntegratedPredictor:
    """Unified prediction system combining KenPom + XGBoost.

    This class provides a single entry point for all prediction operations,
    abstracting away the complexity of data retrieval, feature engineering,
    and model inference.

    Example:
        >>> predictor = IntegratedPredictor()
        >>>
        >>> # Simple prediction
        >>> result = predictor.predict_game("Duke", "UNC")
        >>>
        >>> # With Vegas lines for edge detection
        >>> result = predictor.predict_game(
        ...     "Duke", "UNC",
        ...     vegas_spread=-3.5,
        ...     vegas_total=145.0
        ... )
        >>>
        >>> # Batch predictions
        >>> games = [("Duke", "UNC"), ("Kansas", "Kentucky")]
        >>> results = predictor.predict_batch(games)
    """

    def __init__(
        self,
        db_path: str = "data/kenpom.db",
        model_path: str | None = None,
        use_enhanced_features: bool = True,
        use_ensemble: bool = True,
        ensemble_weights: dict[str, float] | None = None,
        use_xgb_wrapper: bool = False,
    ):
        """Initialize the integrated predictor.

        Args:
            db_path: Path to KenPom SQLite database.
            model_path: Path to trained XGBoost model (optional).
            use_enhanced_features: Use 42-feature enhanced model.
            use_ensemble: Enable ensemble blending (XGBoost + FanMatch).
            ensemble_weights: Custom weights dict
                (default: {'xgboost': 0.70, 'fanmatch': 0.30}).
            use_xgb_wrapper: Use new XGBoostModelWrapper with
                DatabaseFeatureEngineer.
        """
        self.kenpom = KenPomService(db_path=db_path)
        self.use_enhanced_features = use_enhanced_features
        self.use_ensemble = use_ensemble
        self.use_xgb_wrapper = use_xgb_wrapper

        # Set ensemble weights (default 70/30 XGBoost/FanMatch)
        if ensemble_weights is None:
            self.ensemble_weights = {"xgboost": 0.70, "fanmatch": 0.30}
        else:
            self.ensemble_weights = ensemble_weights

        # Initialize XGBoost predictor (legacy)
        self.predictor = XGBoostGamePredictor(
            use_enhanced_features=use_enhanced_features
        )
        self.feature_engineer = XGBoostFeatureEngineer()

        # Initialize new XGBoost components
        self.db_feature_engineer = DatabaseFeatureEngineer(self.kenpom)
        self.xgb_wrapper: XGBoostModelWrapper | None = None

        # Load model if path provided
        if model_path and Path(model_path).exists():
            model_dir = Path(model_path).parent
            if use_xgb_wrapper:
                # Load using new wrapper
                self.xgb_wrapper = XGBoostModelWrapper(
                    model_name="ncaab_xgb", model_type="margin"
                )
                self.xgb_wrapper.load(model_dir)
                logger.info(f"Loaded XGBoost wrapper model from {model_dir}")
            else:
                # Load using legacy predictor
                self.predictor.load_model(model_path)
                logger.info(f"Loaded legacy model from {model_path}")

        # Team name cache for fast lookup
        self._team_cache: dict[str, int] = {}

    def _resolve_team_id(self, team: str | int) -> tuple[int, str]:
        """Resolve team identifier to (team_id, team_name).

        Args:
            team: Team name (string) or team ID (int).

        Returns:
            Tuple of (team_id, team_name).

        Raises:
            ValueError: If team cannot be found.
        """
        if isinstance(team, int):
            # Already an ID, get the name
            rating = self.kenpom.get_team_rating(team_id=team)
            if rating is None:
                raise ValueError(f"Team ID {team} not found")
            return team, rating.team_name

        # Check cache first
        if team.lower() in self._team_cache:
            team_id = self._team_cache[team.lower()]
            return team_id, team

        # Search by name
        teams = self.kenpom.search_teams(team, limit=5)
        if not teams:
            raise ValueError(f"Team '{team}' not found")

        # Exact match preferred
        for t in teams:
            if t.team_name.lower() == team.lower():
                self._team_cache[team.lower()] = t.team_id
                return t.team_id, t.team_name

        # Otherwise use first result
        self._team_cache[team.lower()] = teams[0].team_id
        return teams[0].team_id, teams[0].team_name

    def predict_game(
        self,
        home_team: str | int,
        away_team: str | int,
        neutral_site: bool = False,
        vegas_spread: float | None = None,
        vegas_total: float | None = None,
        snapshot_date: date | None = None,
    ) -> GameAnalysis:
        """Generate prediction for a single game.

        Args:
            home_team: Home team name or ID.
            away_team: Away team name or ID.
            neutral_site: Whether game is at neutral site.
            vegas_spread: Vegas spread (home team perspective).
            vegas_total: Vegas over/under total.
            snapshot_date: Date for ratings (defaults to latest).

        Returns:
            GameAnalysis with complete prediction details.

        Example:
            >>> result = predictor.predict_game("Duke", "UNC")
            >>> print(f"Duke by {result.predicted_margin:+.1f}")
        """
        # Resolve team IDs
        home_id, home_name = self._resolve_team_id(home_team)
        away_id, away_name = self._resolve_team_id(away_team)

        # Get ratings
        home_rating = self.kenpom.get_team_rating(
            team_id=home_id, snapshot_date=snapshot_date
        )
        away_rating = self.kenpom.get_team_rating(
            team_id=away_id, snapshot_date=snapshot_date
        )

        if home_rating is None or away_rating is None:
            raise ValueError(
                f"Could not get ratings for {home_name} vs {away_name}"
            )

        # Get features from KenPom module
        features = self.kenpom.get_features_for_game(
            team1_id=home_id,
            team2_id=away_id,
            home_team_id=None if neutral_site else home_id,
            neutral_site=neutral_site,
            snapshot_date=snapshot_date,
        )

        # Get XGBoost prediction
        if self.predictor.is_fitted:
            # Use trained model
            result = self.predictor.predict(features)
            xgb_margin = result.margin
            xgb_total = result.total
            xgb_win_prob = result.win_probability
            ci = (result.confidence_lower, result.confidence_upper)
        else:
            # Fallback to KenPom-based calculation
            xgb_margin = self._kenpom_margin(
                home_rating, away_rating, neutral_site
            )
            xgb_total = self._kenpom_total(home_rating, away_rating)
            xgb_win_prob = self._margin_to_probability(xgb_margin)
            ci = (xgb_margin - 10, xgb_margin + 10)

        # Ensemble blending (XGBoost + FanMatch)
        if self.use_ensemble:
            # Try to get FanMatch prediction
            fanmatch = self.kenpom.repository.get_fanmatch_for_game(
                home_team_id=home_id,
                away_team_id=away_id,
                snapshot_date=snapshot_date,
            )

            if fanmatch:
                # Blend predictions using configured weights
                w_xgb = self.ensemble_weights["xgboost"]
                w_fm = self.ensemble_weights["fanmatch"]

                margin = w_xgb * xgb_margin + w_fm * fanmatch.pred_margin
                total = w_xgb * xgb_total + w_fm * (
                    fanmatch.pred_home_score + fanmatch.pred_visitor_score
                )
                win_prob = w_xgb * xgb_win_prob + w_fm * fanmatch.home_win_prob

                logger.info(
                    f"Ensemble blend: XGB({xgb_margin:+.1f}) + "
                    f"FM({fanmatch.pred_margin:+.1f}) = {margin:+.1f}"
                )
            else:
                # FanMatch unavailable, use XGBoost only
                margin, total, win_prob = xgb_margin, xgb_total, xgb_win_prob
                logger.debug(
                    f"FanMatch unavailable for {home_name} vs {away_name}, "
                    "using XGBoost only"
                )
        else:
            # Ensemble disabled, use XGBoost only
            margin, total, win_prob = xgb_margin, xgb_total, xgb_win_prob

        return GameAnalysis(
            home_team=home_name,
            away_team=away_name,
            home_id=home_id,
            away_id=away_id,
            home_rating=home_rating,
            away_rating=away_rating,
            predicted_margin=margin,
            predicted_total=total,
            win_probability=win_prob,
            confidence_interval=ci,
            vegas_spread=vegas_spread,
            vegas_total=vegas_total,
        )

    def predict_batch(
        self,
        games: list[tuple[str | int, str | int]],
        neutral_site: bool = False,
    ) -> list[GameAnalysis]:
        """Generate predictions for multiple games.

        Args:
            games: List of (home_team, away_team) tuples.
            neutral_site: Whether all games are at neutral site.

        Returns:
            List of GameAnalysis objects.
        """
        results = []
        for home, away in games:
            try:
                result = self.predict_game(
                    home_team=home,
                    away_team=away,
                    neutral_site=neutral_site,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to predict {home} vs {away}: {e}")
        return results

    def _get_team_stats_for_features(
        self,
        team_id: int,
        snapshot_date: date | None = None,
    ) -> dict[str, Any]:
        """Get team stats formatted for XGBoostFeatureEngineer.

        Fetches data from the database and converts it to the dictionary
        format expected by XGBoostFeatureEngineer.create_enhanced_features().

        Args:
            team_id: Team ID.
            snapshot_date: Optional snapshot date.

        Returns:
            Dictionary with team stats (AdjEM, AdjO, AdjD, AdjT, Luck, etc.)
        """
        rating = self.kenpom.get_team_rating(
            team_id=team_id, snapshot_date=snapshot_date
        )
        if rating is None:
            raise ValueError(f"No rating found for team ID {team_id}")

        # Core stats from ratings
        stats: dict[str, Any] = {
            "TeamName": rating.team_name,
            "TeamID": team_id,
            "AdjEM": rating.adj_em,
            "AdjO": rating.adj_oe,
            "AdjD": rating.adj_de,
            "AdjOE": rating.adj_oe,
            "AdjDE": rating.adj_de,
            "AdjT": rating.adj_tempo,
            "AdjTempo": rating.adj_tempo,
            "Luck": rating.luck if rating.luck is not None else 0.0,
            "Pythag": 0.5 + (rating.adj_em / 40),  # Estimate from AdjEM
            "SOS": rating.sos if hasattr(rating, "sos") else 5.0,
            "SOSO": rating.soso if hasattr(rating, "soso") else 5.0,
            "SOSD": rating.sosd if hasattr(rating, "sosd") else 5.0,
        }

        # Try to get APL from tempo analysis (estimate if not available)
        # Average possession length in seconds (~18s average)
        if rating.adj_tempo > 0:
            # Faster tempo = shorter possessions
            stats["APL_Off"] = (
                2400 / rating.adj_tempo if rating.adj_tempo else 18.0
            )
            stats["APL_Def"] = (
                2400 / rating.adj_tempo if rating.adj_tempo else 18.0
            )
        else:
            stats["APL_Off"] = 18.0
            stats["APL_Def"] = 18.0

        return stats

    def predict_with_enhanced_features(
        self,
        home_team: str | int,
        away_team: str | int,
        neutral_site: bool = False,
        vegas_spread: float | None = None,
        vegas_total: float | None = None,
        snapshot_date: date | None = None,
    ) -> dict[str, Any]:
        """Predict game using trained XGBoost model with 42 enhanced features.

        This method uses XGBoostFeatureEngineer.create_enhanced_features() to
        generate the exact 42 features the trained model expects, then runs
        inference using the loaded model.

        Args:
            home_team: Home team name or ID.
            away_team: Away team name or ID.
            neutral_site: Whether game is at neutral site.
            vegas_spread: Current Vegas spread (negative = home favored).
            vegas_total: Current Vegas over/under total.
            snapshot_date: Date for ratings snapshot.

        Returns:
            Dictionary with prediction results:
                - predicted_margin: XGBoost predicted margin (home - away)
                - predicted_total: XGBoost predicted total points
                - confidence_interval: (lower, upper) bounds
                - win_probability: Home team win probability
                - vegas_spread: Input Vegas spread
                - edge: Model edge vs Vegas (if spread provided)
                - features: All 42 feature values
        """
        # Resolve team IDs and names
        home_id, home_name = self._resolve_team_id(home_team)
        away_id, away_name = self._resolve_team_id(away_team)

        # Get team stats in format for XGBoostFeatureEngineer
        home_stats = self._get_team_stats_for_features(home_id, snapshot_date)
        away_stats = self._get_team_stats_for_features(away_id, snapshot_date)

        # Get additional data for enhanced features (optional)
        # These can be None and create_enhanced_features will use defaults
        misc_stats_home = None
        misc_stats_away = None
        height_home = None
        height_away = None
        point_dist_home = None
        point_dist_away = None

        # Try to get misc stats from database
        try:
            with self.kenpom.repository.db.connection() as conn:
                # Get misc stats (columns use _off/_def suffixes)
                misc_home = conn.execute(
                    """
                    SELECT fg3_pct_off, fg2_pct_off, ft_pct_off,
                           assist_rate, steal_rate, block_pct_off
                    FROM misc_stats
                    WHERE team_id = ?
                    ORDER BY snapshot_date DESC LIMIT 1
                    """,
                    (home_id,),
                ).fetchone()
                if misc_home:
                    misc_stats_home = {
                        "FG3Pct": misc_home[0] or 35.0,
                        "FG2Pct": misc_home[1] or 50.0,
                        "FTPct": misc_home[2] or 70.0,
                        "ARate": misc_home[3] or 50.0,
                        "StlRate": misc_home[4] or 10.0,
                        "BlkPct": misc_home[5] or 10.0,
                    }

                misc_away = conn.execute(
                    """
                    SELECT fg3_pct_off, fg2_pct_off, ft_pct_off,
                           assist_rate, steal_rate, block_pct_off
                    FROM misc_stats
                    WHERE team_id = ?
                    ORDER BY snapshot_date DESC LIMIT 1
                    """,
                    (away_id,),
                ).fetchone()
                if misc_away:
                    misc_stats_away = {
                        "FG3Pct": misc_away[0] or 35.0,
                        "FG2Pct": misc_away[1] or 50.0,
                        "FTPct": misc_away[2] or 70.0,
                        "ARate": misc_away[3] or 50.0,
                        "StlRate": misc_away[4] or 10.0,
                        "BlkPct": misc_away[5] or 10.0,
                    }

                # Get height/experience data
                height_h = conn.execute(
                    """SELECT effective_height, experience,
                    continuity, bench_minutes FROM height
                    WHERE team_id = ?
                    ORDER BY snapshot_date DESC LIMIT 1""",
                    (home_id,),
                ).fetchone()
                if height_h:
                    height_home = {
                        "EffHeight": height_h[0] or 0.0,
                        "Experience": height_h[1] or 2.0,
                        "Continuity": height_h[2] or 0.5,
                        "Bench": height_h[3] or 30.0,
                    }

                height_a = conn.execute(
                    """SELECT effective_height, experience,
                    continuity, bench_minutes FROM height
                    WHERE team_id = ?
                    ORDER BY snapshot_date DESC LIMIT 1""",
                    (away_id,),
                ).fetchone()
                if height_a:
                    height_away = {
                        "EffHeight": height_a[0] or 0.0,
                        "Experience": height_a[1] or 2.0,
                        "Continuity": height_a[2] or 0.5,
                        "Bench": height_a[3] or 30.0,
                    }

                # Get point distribution (columns: ft_pct, two_pct, three_pct)
                pd_home = conn.execute(
                    """
                    SELECT ft_pct, two_pct, three_pct
                    FROM point_distribution
                    WHERE team_id = ?
                    ORDER BY snapshot_date DESC LIMIT 1
                    """,
                    (home_id,),
                ).fetchone()
                if pd_home:
                    point_dist_home = {
                        "FT_Pct": pd_home[0] or 20.0,
                        "TwoP_Pct": pd_home[1] or 50.0,
                        "ThreeP_Pct": pd_home[2] or 30.0,
                    }

                pd_away = conn.execute(
                    """
                    SELECT ft_pct, two_pct, three_pct
                    FROM point_distribution
                    WHERE team_id = ?
                    ORDER BY snapshot_date DESC LIMIT 1
                    """,
                    (away_id,),
                ).fetchone()
                if pd_away:
                    point_dist_away = {
                        "FT_Pct": pd_away[0] or 20.0,
                        "TwoP_Pct": pd_away[1] or 50.0,
                        "ThreeP_Pct": pd_away[2] or 30.0,
                    }
        except Exception as e:
            logger.warning(f"Could not fetch enhanced stats: {e}")

        # Create enhanced features (42 total)
        features = XGBoostFeatureEngineer.create_enhanced_features(
            team1_stats=home_stats,
            team2_stats=away_stats,
            neutral_site=neutral_site,
            home_team1=not neutral_site,  # home_team is team1
            misc_stats_team1=misc_stats_home,
            misc_stats_team2=misc_stats_away,
            height_team1=height_home,
            height_team2=height_away,
            point_dist_team1=point_dist_home,
            point_dist_team2=point_dist_away,
        )

        # Convert to DataFrame for XGBoost
        feature_names = XGBoostFeatureEngineer.ENHANCED_FEATURE_NAMES
        features_df = pd.DataFrame([features])[feature_names]

        # Run prediction using trained predictor
        if self.predictor.is_fitted:
            # Use the internal models directly
            margin_pred = float(
                self.predictor.margin_model.predict(features_df)[0]
            )
            total_pred = float(
                self.predictor.total_model.predict(features_df)[0]
            )
            margin_upper = float(
                self.predictor.margin_upper.predict(features_df)[0]
            )
            margin_lower = float(
                self.predictor.margin_lower.predict(features_df)[0]
            )
        else:
            # Fallback to KenPom formula
            margin_pred = self._kenpom_margin(
                self.kenpom.get_team_rating(team_id=home_id),
                self.kenpom.get_team_rating(team_id=away_id),
                neutral_site,
            )
            total_pred = self._kenpom_total(
                self.kenpom.get_team_rating(team_id=home_id),
                self.kenpom.get_team_rating(team_id=away_id),
            )
            margin_upper = margin_pred + 10
            margin_lower = margin_pred - 10

        # Calculate win probability
        win_prob = self._margin_to_probability(margin_pred)

        # Calculate edge if Vegas spread provided
        edge = None
        if vegas_spread is not None:
            vegas_implied_margin = -vegas_spread
            edge = margin_pred - vegas_implied_margin

        return {
            "home_team": home_name,
            "away_team": away_name,
            "predicted_margin": round(margin_pred, 1),
            "predicted_total": round(total_pred, 1),
            "confidence_interval": (
                round(margin_lower, 1),
                round(margin_upper, 1),
            ),
            "win_probability": round(win_prob, 3),
            "vegas_spread": vegas_spread,
            "vegas_total": vegas_total,
            "edge": round(edge, 1) if edge is not None else None,
            "has_edge": edge is not None and abs(edge) >= 2.0,
            "features": features,
            "feature_count": len(features),
        }

    def predict_with_xgb_wrapper(
        self,
        home_team: str,
        away_team: str,
        vegas_spread: float,
        vegas_total: float,
        game_date: date | None = None,
        opening_spread: float | None = None,
        opening_total: float | None = None,
    ) -> dict[str, Any]:
        """Predict game using XGBoostModelWrapper with 40 database features.

        NOTE: This method uses DatabaseFeatureEngineer which produces different
        features than the enhanced model. For the trained 42-feature model,
        use predict_with_enhanced_features() instead.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            vegas_spread: Current Vegas spread (negative = home favored).
            vegas_total: Current Vegas over/under total.
            game_date: Game date (defaults to today).
            opening_spread: Opening line spread for line movement analysis.
            opening_total: Opening line total.

        Returns:
            Dictionary with prediction results.
        """
        if self.xgb_wrapper is None:
            raise ValueError(
                "XGBoostModelWrapper not loaded. "
                "Initialize with use_xgb_wrapper=True "
                "and provide a valid model_path."
            )

        # Generate features from database
        game_features = self.db_feature_engineer.create_features(
            home_team=home_team,
            away_team=away_team,
            vegas_spread=vegas_spread,
            vegas_total=vegas_total,
            game_date=game_date,
            opening_spread=opening_spread,
            opening_total=opening_total,
        )

        # Convert to array for prediction
        feature_array = game_features.to_array().reshape(1, -1)

        # Get prediction from XGBoost wrapper (with feature names)
        prediction = self.xgb_wrapper.predict(
            feature_array,
            feature_names=game_features.feature_names,
        )[0]

        # Calculate edge vs Vegas (implied margin from spread)
        # Vegas spread is from home perspective: negative = home favored
        vegas_implied_margin = -vegas_spread
        edge = prediction - vegas_implied_margin

        return {
            "game_id": game_features.game_id,
            "home_team": home_team,
            "away_team": away_team,
            "predicted_margin": float(prediction),
            "vegas_spread": vegas_spread,
            "vegas_total": vegas_total,
            "vegas_implied_margin": vegas_implied_margin,
            "edge": float(edge),
            "has_edge": abs(edge) >= 2.0,
            "features": game_features.features,
            "feature_names": game_features.feature_names,
            "model_mae": self.xgb_wrapper.metadata.get("mae"),
            "timestamp": game_features.timestamp,
        }

    def get_game_features(
        self,
        home_team: str,
        away_team: str,
        vegas_spread: float,
        vegas_total: float,
        game_date: date | None = None,
    ) -> GameFeatures:
        """Get all 40 features for a game matchup from database.

        This is useful for debugging, analysis, or external model usage.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            vegas_spread: Vegas spread.
            vegas_total: Vegas total.
            game_date: Game date.

        Returns:
            GameFeatures object with all 40 features.
        """
        return self.db_feature_engineer.create_features(
            home_team=home_team,
            away_team=away_team,
            vegas_spread=vegas_spread,
            vegas_total=vegas_total,
            game_date=game_date,
        )

    def find_edges(
        self,
        games_with_lines: list[dict],
        min_spread_edge: float = 2.0,
        min_total_edge: float = 3.0,
    ) -> list[GameAnalysis]:
        """Find games with betting edges vs Vegas lines.

        Args:
            games_with_lines: List of dicts with:
                - home_team, away_team
                - vegas_spread, vegas_total (optional)
            min_spread_edge: Minimum spread edge to flag.
            min_total_edge: Minimum total edge to flag.

        Returns:
            List of GameAnalysis objects that have edges.

        Example:
            >>> games = [
            ...     {"home_team": "Duke", "away_team": "UNC",
            ...      "vegas_spread": -3.5, "vegas_total": 145.0}
            ... ]
            >>> edges = predictor.find_edges(games)
        """
        edges = []

        for game in games_with_lines:
            try:
                result = self.predict_game(
                    home_team=game["home_team"],
                    away_team=game["away_team"],
                    vegas_spread=game.get("vegas_spread"),
                    vegas_total=game.get("vegas_total"),
                )

                # Check for edges
                has_edge = False
                if result.edge_vs_spread is not None:
                    if abs(result.edge_vs_spread) >= min_spread_edge:
                        has_edge = True
                if result.edge_vs_total is not None:
                    if abs(result.edge_vs_total) >= min_total_edge:
                        has_edge = True

                if has_edge:
                    edges.append(result)

            except Exception as e:
                logger.warning(
                    f"Error analyzing {game.get('home_team')} vs "
                    f"{game.get('away_team')}: {e}"
                )

        return edges

    def _kenpom_margin(
        self,
        home: TeamRating,
        away: TeamRating,
        neutral: bool,
    ) -> float:
        """Calculate expected margin using KenPom efficiency.

        Formula: (HomeAdjEM - AwayAdjEM) * TempoFactor + HCA
        """
        em_diff = home.adj_em - away.adj_em

        # Tempo adjustment (games played at average of both tempos)
        avg_tempo = (home.adj_tempo + away.adj_tempo) / 2
        league_avg_tempo = 67.5
        tempo_factor = avg_tempo / league_avg_tempo

        # Home court advantage (typically 3.5-4 points)
        hca = 0 if neutral else 3.75

        return em_diff * tempo_factor + hca

    def _kenpom_total(
        self,
        home: TeamRating,
        away: TeamRating,
    ) -> float:
        """Calculate expected total using KenPom efficiency."""
        # Average efficiency
        avg_oe = (home.adj_oe + away.adj_oe) / 2
        avg_de = (home.adj_de + away.adj_de) / 2

        # Average tempo
        avg_tempo = (home.adj_tempo + away.adj_tempo) / 2

        # Expected points = tempo * (offensive efficiency / 100)
        possessions = avg_tempo * 2  # Total possessions in game
        expected_ppp = (avg_oe + (200 - avg_de)) / 200  # Points per possession

        return possessions * expected_ppp

    def _margin_to_probability(self, margin: float) -> float:
        """Convert point margin to win probability.

        Uses logistic function calibrated to college basketball.
        """
        # Calibrated from historical data: ~4 point margin = 65% win prob
        k = 0.15  # Steepness parameter
        return 1 / (1 + np.exp(-k * margin))

    def sync_data(self, year: int | None = None) -> dict:
        """Sync KenPom data from API.

        Args:
            year: Season year (defaults to current).

        Returns:
            Dictionary with sync results.
        """
        scheduler = BatchScheduler(db_path=self.kenpom.db_path)
        result = scheduler.run_daily_workflow(year=year)
        return {
            "success": result.success,
            "records_synced": result.total_records,
            "duration": result.duration_seconds,
        }

    def train_model(
        self,
        historical_data: pd.DataFrame | str,
        save_path: str = "models/xgboost_model.json",
    ) -> dict:
        """Train XGBoost model on historical data.

        Args:
            historical_data: DataFrame or path to CSV with historical games.
            save_path: Where to save trained model.

        Returns:
            Training metrics.
        """
        if isinstance(historical_data, str):
            historical_data = pd.read_csv(historical_data)

        # Prepare features
        X, y = self.feature_engineer.prepare_training_data(
            historical_data,
            use_enhanced=self.use_enhanced_features,
        )

        # Train
        metrics = self.predictor.train(X, y)

        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.predictor.save_model(save_path)
        logger.info(f"Model saved to {save_path}")

        return metrics

    def get_accuracy_report(self, days: int = 30) -> AccuracyReport:
        """Get prediction accuracy report.

        Args:
            days: Number of days to include.

        Returns:
            AccuracyReport with performance metrics.
        """
        return self.kenpom.get_accuracy_report(days=days)

    def get_top_teams(self, n: int = 25) -> list[TeamRating]:
        """Get top N teams by AdjEM.

        Args:
            n: Number of teams to return.

        Returns:
            List of TeamRating objects.
        """
        ratings = self.kenpom.get_latest_ratings()
        return ratings[:n]

    def compare_teams(
        self,
        team1: str | int,
        team2: str | int,
    ) -> dict:
        """Compare two teams side by side.

        Args:
            team1: First team name or ID.
            team2: Second team name or ID.

        Returns:
            Dictionary with comparison data.
        """
        id1, name1 = self._resolve_team_id(team1)
        id2, name2 = self._resolve_team_id(team2)

        r1 = self.kenpom.get_team_rating(team_id=id1)
        r2 = self.kenpom.get_team_rating(team_id=id2)

        return {
            "team1": {
                "name": name1,
                "adj_em": r1.adj_em,
                "adj_oe": r1.adj_oe,
                "adj_de": r1.adj_de,
                "adj_tempo": r1.adj_tempo,
                "rank": r1.rank_adj_em,
            },
            "team2": {
                "name": name2,
                "adj_em": r2.adj_em,
                "adj_oe": r2.adj_oe,
                "adj_de": r2.adj_de,
                "adj_tempo": r2.adj_tempo,
                "rank": r2.rank_adj_em,
            },
            "differential": {
                "adj_em": round(r1.adj_em - r2.adj_em, 2),
                "adj_oe": round(r1.adj_oe - r2.adj_oe, 2),
                "adj_de": round(r1.adj_de - r2.adj_de, 2),
                "rank": r1.rank_adj_em - r2.rank_adj_em,
            },
        }
