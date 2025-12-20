"""
Advanced feature engineering with 40+ features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

from ..config import config
from ..utils.exceptions import FeatureEngineeringError
from ..utils.logging import logger


@dataclass
class GameFeatures:
    """Container for game features"""

    game_id: str
    features: Dict[str, float]
    feature_names: List[str]
    timestamp: str

    def to_array(self) -> np.ndarray:
        """Convert to numpy array in feature order"""
        return np.array([self.features[name] for name in self.feature_names])


class AdvancedFeatureEngineer:
    """
    Create 40+ engineered features for NCAA basketball predictions
    """

    # Core feature names (in order)
    FEATURE_NAMES = [
        # Efficiency Metrics (4)
        "adj_em_diff",
        "adj_oe_diff",
        "adj_de_diff",
        "four_factors_index",
        # Four Factors (4)
        "fg_pct_diff",
        "three_pt_diff",
        "ft_rate_diff",
        "to_rate_diff",
        # Tempo & Rhythm (5)
        "tempo_avg",
        "tempo_diff",
        "pace_consistency",
        "home_recent_pace",
        "away_recent_pace",
        # Recent Form (5)
        "home_recent_oe",
        "away_recent_oe",
        "home_recent_de",
        "away_recent_de",
        "form_momentum",
        # Contextual Factors (6)
        "rest_advantage",
        "back_to_back_home",
        "back_to_back_away",
        "travel_distance",
        "home_arena_elevation",
        "is_neutral_site",
        # Advanced Metrics (4)
        "pythag_diff",
        "luck_diff",
        "sos_diff",
        "season_progression",
        # Vegas Market Signals (6)
        "vegas_spread",
        "vegas_total",
        "line_movement_pct",
        "spread_direction",
        "sharp_indicator",
        "consensus_diff",
        # Interaction Terms (6)
        "em_pace_interaction",
        "em_rest_interaction",
        "oe_pace_interaction",
        "recent_form_mult",
        "market_em_interaction",
        "form_market_interaction",
    ]

    def __init__(self, kenpom_data: Dict, game_history: Dict):
        """
        Initialize feature engineer

        Args:
            kenpom_data: Dictionary of team stats (name -> stats dict)
            game_history: Dictionary of historical game results
        """
        self.kenpom_data = kenpom_data
        self.game_history = game_history
        self.logger = logger

    def create_features(
        self,
        home_team: str,
        away_team: str,
        game_date: str,
        vegas_spread: float,
        vegas_total: float,
        opening_spread: float = None,
        opening_total: float = None,
    ) -> GameFeatures:
        """
        Create comprehensive feature set for a game

        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Game date (YYYY-MM-DD)
            vegas_spread: Current Vegas spread
            vegas_total: Current Vegas total
            opening_spread: Opening spread (for line movement)
            opening_total: Opening total

        Returns:
            GameFeatures object with all features
        """
        try:
            features = {}

            # Validate input teams exist
            if home_team not in self.kenpom_data:
                raise FeatureEngineeringError(f"Unknown team: {home_team}")
            if away_team not in self.kenpom_data:
                raise FeatureEngineeringError(f"Unknown team: {away_team}")

            # Extract team stats
            home_stats = self.kenpom_data[home_team]
            away_stats = self.kenpom_data[away_team]

            # 1. Efficiency Metrics (4 features)
            features["adj_em_diff"] = (
                home_stats["adj_em"] - away_stats["adj_em"]
            )
            features["adj_oe_diff"] = (
                home_stats["adj_oe"] - away_stats["adj_oe"]
            )
            features["adj_de_diff"] = (
                home_stats["adj_de"] - away_stats["adj_de"]
            )
            features["four_factors_index"] = (
                self._calculate_four_factors_index(home_stats, away_stats)
            )

            # 2. Four Factors (4 features)
            features["fg_pct_diff"] = home_stats.get(
                "fg_pct", 0
            ) - away_stats.get("fg_pct", 0)
            features["three_pt_diff"] = home_stats.get(
                "three_pct", 0
            ) - away_stats.get("three_pct", 0)
            features["ft_rate_diff"] = home_stats.get(
                "ft_rate", 0
            ) - away_stats.get("ft_rate", 0)
            features["to_rate_diff"] = away_stats.get(
                "to_rate", 0
            ) - home_stats.get("to_rate", 0)

            # 3. Tempo & Rhythm (5 features)
            features["tempo_avg"] = (
                home_stats["pace"] + away_stats["pace"]
            ) / 2
            features["tempo_diff"] = home_stats["pace"] - away_stats["pace"]
            features["pace_consistency"] = self._calculate_pace_consistency(
                home_team, away_team
            )
            features["home_recent_pace"] = self._get_recent_pace(
                home_team, days=5
            )
            features["away_recent_pace"] = self._get_recent_pace(
                away_team, days=5
            )

            # 4. Recent Form (5 features)
            features["home_recent_oe"] = self._get_recent_metric(
                home_team, "oe", days=10
            )
            features["away_recent_oe"] = self._get_recent_metric(
                away_team, "oe", days=10
            )
            features["home_recent_de"] = self._get_recent_metric(
                home_team, "de", days=10
            )
            features["away_recent_de"] = self._get_recent_metric(
                away_team, "de", days=10
            )
            features["form_momentum"] = self._calculate_momentum(
                home_team, away_team, days=10
            )

            # 5. Contextual Factors (6 features)
            features["rest_advantage"] = self._calculate_rest_advantage(
                home_team, away_team, game_date
            )
            features["back_to_back_home"] = (
                1.0 if self._is_back_to_back(home_team, game_date) else 0.0
            )
            features["back_to_back_away"] = (
                1.0 if self._is_back_to_back(away_team, game_date) else 0.0
            )
            features["travel_distance"] = self._calculate_travel_distance(
                home_team, away_team
            )
            features["home_arena_elevation"] = home_stats.get("elevation", 500)
            features["is_neutral_site"] = (
                1.0 if self._is_neutral_site(home_team, away_team) else 0.0
            )

            # 6. Advanced Metrics (4 features)
            features["pythag_diff"] = home_stats.get(
                "pythag", 0.5
            ) - away_stats.get("pythag", 0.5)
            features["luck_diff"] = home_stats.get("luck", 0) - away_stats.get(
                "luck", 0
            )
            features["sos_diff"] = home_stats.get("sos", 0) - away_stats.get(
                "sos", 0
            )
            features["season_progression"] = (
                self._calculate_season_progression(game_date)
            )

            # 7. Vegas Market Signals (6 features)
            features["vegas_spread"] = vegas_spread
            features["vegas_total"] = vegas_total
            features["line_movement_pct"] = (
                self._calculate_line_movement(opening_spread, vegas_spread)
                if opening_spread
                else 0.0
            )
            features["spread_direction"] = 1.0 if vegas_spread > 0 else -1.0
            features["sharp_indicator"] = self._detect_sharp_signal(
                opening_spread, vegas_spread, opening_total, vegas_total
            )
            features["consensus_diff"] = self._calculate_consensus_diff(
                home_team, away_team, vegas_spread
            )

            # 8. Interaction Terms (6 features) - CRITICAL FOR XGBOOST
            features["em_pace_interaction"] = (
                features["adj_em_diff"] * features["tempo_avg"]
            )
            features["em_rest_interaction"] = (
                features["adj_em_diff"] * features["rest_advantage"]
            )
            features["oe_pace_interaction"] = (
                features["adj_oe_diff"] * features["tempo_avg"]
            )
            features["recent_form_mult"] = (
                features["home_recent_oe"] - features["away_recent_oe"]
            ) * (features["home_recent_de"] - features["away_recent_de"])
            features["market_em_interaction"] = (
                features["vegas_spread"] * features["adj_em_diff"]
            )
            features["form_market_interaction"] = (
                features["form_momentum"] * features["line_movement_pct"]
            )

            # Validate all features present
            missing_features = set(self.FEATURE_NAMES) - set(features.keys())
            if missing_features:
                raise FeatureEngineeringError(
                    f"Missing features: {missing_features}"
                )

            # Create GameFeatures object
            game_id = f"{game_date}_{away_team}_{home_team}"

            return GameFeatures(
                game_id=game_id,
                features=features,
                feature_names=self.FEATURE_NAMES,
                timestamp=str(pd.Timestamp.now()),
            )

        except Exception as e:
            self.logger.error(
                f"Feature engineering failed for {home_team} vs {away_team}: {str(e)}"
            )
            raise FeatureEngineeringError(
                f"Failed to create features: {str(e)}"
            )

    # Helper methods (implementations below)
    def _calculate_four_factors_index(self, home: Dict, away: Dict) -> float:
        """Composite four factors index"""
        home_ff = (
            home.get("fg_pct", 0)
            + home.get("three_pct", 0)
            + home.get("ft_rate", 0)
            - home.get("to_rate", 0)
        ) / 4
        away_ff = (
            away.get("fg_pct", 0)
            + away.get("three_pct", 0)
            + away.get("ft_rate", 0)
            - away.get("to_rate", 0)
        ) / 4
        return home_ff - away_ff

    def _calculate_pace_consistency(
        self, home_team: str, away_team: str
    ) -> float:
        """Calculate consistency of pace (lower is more consistent)"""
        # Would use game_history to calculate standard deviation
        return 0.0  # Placeholder

    def _get_recent_pace(self, team: str, days: int = 5) -> float:
        """Get average pace for last N games"""
        return 0.0  # Placeholder

    def _get_recent_metric(
        self, team: str, metric: str, days: int = 10
    ) -> float:
        """Get average metric for last N games"""
        return 0.0  # Placeholder

    def _calculate_momentum(
        self, home: str, away: str, days: int = 10
    ) -> float:
        """Calculate momentum (trend) for both teams"""
        return 0.0  # Placeholder

    def _calculate_rest_advantage(
        self, home: str, away: str, game_date: str
    ) -> float:
        """Calculate rest advantage (days difference)"""
        return 0.0  # Placeholder

    def _is_back_to_back(self, team: str, game_date: str) -> bool:
        """Check if team is on back-to-back"""
        return False  # Placeholder

    def _calculate_travel_distance(self, home: str, away: str) -> float:
        """Calculate travel distance for away team"""
        return 0.0  # Placeholder

    def _is_neutral_site(self, home: str, away: str) -> bool:
        """Check if game is at neutral site"""
        return False  # Placeholder

    def _calculate_season_progression(self, game_date: str) -> float:
        """Normalize season progression (0 = start, 1 = end)"""
        return 0.5  # Placeholder

    def _calculate_line_movement(
        self, opening: float, current: float
    ) -> float:
        """Calculate percentage line movement"""
        if opening == 0:
            return 0.0
        return abs(current - opening) / abs(opening)

    def _detect_sharp_signal(
        self, opening_spread, current_spread, opening_total, current_total
    ) -> float:
        """Detect sharp money signal (1 = sharp backed home, -1 = backed away)"""
        if opening_spread is None:
            return 0.0
        # Sharp money typically moves against public
        return 0.0  # Placeholder

    def _calculate_consensus_diff(
        self, home: str, away: str, spread: float
    ) -> float:
        """Difference between consensus and sharp lines"""
        return 0.0  # Placeholder
