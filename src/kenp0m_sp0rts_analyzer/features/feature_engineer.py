"""
Advanced feature engineering with 40+ features.

Includes:
- AdvancedFeatureEngineer: Base class using in-memory data dictionaries
- DatabaseFeatureEngineer: Database-backed version using KenPom SQLite DB
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..utils.exceptions import FeatureEngineeringError
from ..utils.logging import logger

if TYPE_CHECKING:
    from ..kenpom import KenPomService


@dataclass
class GameFeatures:
    """Container for game features"""

    game_id: str
    features: dict[str, float]
    feature_names: list[str]
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

    def __init__(self, kenpom_data: dict, game_history: dict):
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
                f"Feature engineering failed for {home_team} vs "
                f"{away_team}: {e}"
            )
            raise FeatureEngineeringError(
                f"Failed to create features: {e}"
            ) from e

    # Helper methods (implementations below)
    def _calculate_four_factors_index(self, home: dict, away: dict) -> float:
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


class DatabaseFeatureEngineer:
    """Database-backed feature engineer using KenPom SQLite database.

    Pulls real team statistics from the database and creates features
    for XGBoost predictions.

    Usage:
        from kenp0m_sp0rts_analyzer.kenpom import KenPomService
        from kenp0m_sp0rts_analyzer.features import DatabaseFeatureEngineer

        service = KenPomService()
        engineer = DatabaseFeatureEngineer(service)

        features = engineer.create_features(
            home_team="Duke",
            away_team="North Carolina",
            vegas_spread=-3.5,
            vegas_total=152.5,
        )
    """

    # Use same feature names as AdvancedFeatureEngineer for consistency
    FEATURE_NAMES = AdvancedFeatureEngineer.FEATURE_NAMES

    def __init__(self, kenpom_service: KenPomService):
        """Initialize with a KenPomService instance.

        Args:
            kenpom_service: Initialized KenPomService with database connection.
        """
        self.service = kenpom_service
        self.repository = kenpom_service.repository
        self._team_cache: dict[str, dict[str, Any]] = {}
        self._cache_date: date | None = None

    def _load_team_data(
        self, snapshot_date: date | None = None
    ) -> dict[str, dict[str, Any]]:
        """Load all team data from database for a given date.

        Args:
            snapshot_date: Date to load data for (defaults to latest).

        Returns:
            Dictionary mapping team name -> stats dict.
        """
        snapshot_date = snapshot_date or date.today()

        # Use cache if available and date matches
        if self._cache_date == snapshot_date and self._team_cache:
            return self._team_cache

        team_data: dict[str, dict[str, Any]] = {}

        with self.repository.db.connection() as conn:
            # Get latest snapshot date if today has no data
            cursor = conn.execute(
                """
                SELECT MAX(snapshot_date) FROM ratings
                WHERE snapshot_date <= ?
                """,
                (str(snapshot_date),),
            )
            actual_date = cursor.fetchone()[0]
            if not actual_date:
                raise FeatureEngineeringError(
                    f"No ratings data found for {snapshot_date}"
                )

            # Load ratings (core efficiency metrics)
            cursor = conn.execute(
                """
                SELECT
                    t.team_name,
                    r.adj_em, r.adj_oe, r.adj_de, r.adj_tempo,
                    r.pythag, r.luck, r.sos, r.ncsos,
                    r.wins, r.losses
                FROM ratings r
                JOIN teams t ON r.team_id = t.team_id
                WHERE r.snapshot_date = ?
                """,
                (actual_date,),
            )
            for row in cursor.fetchall():
                team_name = row[0]
                team_data[team_name] = {
                    "adj_em": row[1] or 0.0,
                    "adj_oe": row[2] or 0.0,
                    "adj_de": row[3] or 0.0,
                    "pace": row[4] or 67.0,  # Map adj_tempo to pace
                    "pythag": row[5] or 0.5,
                    "luck": row[6] or 0.0,
                    "sos": row[7] or 0.0,
                    "ncsos": row[8] or 0.0,
                    "wins": row[9] or 0,
                    "losses": row[10] or 0,
                }

            # Load four factors
            cursor = conn.execute(
                """
                SELECT
                    t.team_name,
                    f.efg_pct_off, f.to_pct_off, f.or_pct_off, f.ft_rate_off,
                    f.efg_pct_def, f.to_pct_def, f.or_pct_def, f.ft_rate_def
                FROM four_factors f
                JOIN teams t ON f.team_id = t.team_id
                WHERE f.snapshot_date = ?
                """,
                (actual_date,),
            )
            for row in cursor.fetchall():
                team_name = row[0]
                if team_name in team_data:
                    team_data[team_name].update(
                        {
                            "efg_pct": row[1] or 0.0,
                            "to_rate": row[2] or 0.0,
                            "or_pct": row[3] or 0.0,
                            "ft_rate": row[4] or 0.0,
                            "efg_pct_def": row[5] or 0.0,
                            "to_rate_def": row[6] or 0.0,
                            "or_pct_def": row[7] or 0.0,
                            "ft_rate_def": row[8] or 0.0,
                            # Map to expected names
                            "fg_pct": row[1] or 0.0,  # Use eFG% as FG proxy
                        }
                    )

            # Load misc stats (shooting percentages)
            cursor = conn.execute(
                """
                SELECT
                    t.team_name,
                    m.fg3_pct_off, m.fg2_pct_off, m.ft_pct_off,
                    m.fg3_pct_def, m.fg2_pct_def
                FROM misc_stats m
                JOIN teams t ON m.team_id = t.team_id
                WHERE m.snapshot_date = ?
                """,
                (actual_date,),
            )
            for row in cursor.fetchall():
                team_name = row[0]
                if team_name in team_data:
                    team_data[team_name].update(
                        {
                            "three_pct": row[1] or 0.0,
                            "two_pct": row[2] or 0.0,
                            "ft_pct": row[3] or 0.0,
                            "three_pct_def": row[4] or 0.0,
                            "two_pct_def": row[5] or 0.0,
                        }
                    )

            # Add default elevation (could be enhanced with arena data)
            for team_name in team_data:
                team_data[team_name].setdefault("elevation", 500)

        # Cache the data
        self._team_cache = team_data
        self._cache_date = snapshot_date

        logger.info(
            f"Loaded {len(team_data)} teams from database ({actual_date})"
        )
        return team_data

    def get_team_stats(
        self, team_name: str, snapshot_date: date | None = None
    ) -> dict[str, Any]:
        """Get stats for a specific team.

        Args:
            team_name: Team name (must match KenPom format).
            snapshot_date: Date to get stats for.

        Returns:
            Dictionary of team statistics.

        Raises:
            FeatureEngineeringError: If team not found.
        """
        team_data = self._load_team_data(snapshot_date)

        if team_name not in team_data:
            # Try normalized lookup
            from ..helpers import normalize_team_name

            normalized = normalize_team_name(team_name)
            if normalized in team_data:
                return team_data[normalized]
            raise FeatureEngineeringError(
                f"Team not found: {team_name} (tried: {normalized})"
            )

        return team_data[team_name]

    def create_features(
        self,
        home_team: str,
        away_team: str,
        vegas_spread: float,
        vegas_total: float,
        game_date: date | str | None = None,
        opening_spread: float | None = None,
        opening_total: float | None = None,
    ) -> GameFeatures:
        """Create features for a game matchup from database.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            vegas_spread: Current Vegas spread (negative = home favored).
            vegas_total: Current Vegas total.
            game_date: Game date (defaults to today).
            opening_spread: Opening line spread (for line movement).
            opening_total: Opening line total.

        Returns:
            GameFeatures object with 40 features.
        """
        # Parse game date
        if game_date is None:
            game_date = date.today()
        elif isinstance(game_date, str):
            game_date = datetime.strptime(game_date, "%Y-%m-%d").date()

        # Load team data
        team_data = self._load_team_data(game_date)

        # Normalize team names
        from ..helpers import normalize_team_name

        home_team_norm = normalize_team_name(home_team)
        away_team_norm = normalize_team_name(away_team)

        if home_team_norm not in team_data:
            raise FeatureEngineeringError(f"Home team not found: {home_team}")
        if away_team_norm not in team_data:
            raise FeatureEngineeringError(f"Away team not found: {away_team}")

        home_stats = team_data[home_team_norm]
        away_stats = team_data[away_team_norm]

        features: dict[str, float] = {}

        # 1. Efficiency Metrics (4 features)
        features["adj_em_diff"] = home_stats["adj_em"] - away_stats["adj_em"]
        features["adj_oe_diff"] = home_stats["adj_oe"] - away_stats["adj_oe"]
        features["adj_de_diff"] = home_stats["adj_de"] - away_stats["adj_de"]
        features["four_factors_index"] = self._calc_four_factors_index(
            home_stats, away_stats
        )

        # 2. Four Factors (4 features)
        features["fg_pct_diff"] = home_stats.get(
            "efg_pct", 0
        ) - away_stats.get("efg_pct", 0)
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
        home_pace = home_stats.get("pace", 67.0)
        away_pace = away_stats.get("pace", 67.0)
        features["tempo_avg"] = (home_pace + away_pace) / 2
        features["tempo_diff"] = home_pace - away_pace
        features["pace_consistency"] = 0.0  # TODO: Calculate from history
        features["home_recent_pace"] = home_pace  # Use current as proxy
        features["away_recent_pace"] = away_pace

        # 4. Recent Form (5 features)
        features["home_recent_oe"] = home_stats.get("adj_oe", 100.0)
        features["away_recent_oe"] = away_stats.get("adj_oe", 100.0)
        features["home_recent_de"] = home_stats.get("adj_de", 100.0)
        features["away_recent_de"] = away_stats.get("adj_de", 100.0)
        features["form_momentum"] = 0.0  # TODO: Calculate from game results

        # 5. Contextual Factors (6 features)
        features["rest_advantage"] = 0.0  # TODO: Calculate from schedule
        features["back_to_back_home"] = 0.0
        features["back_to_back_away"] = 0.0
        features["travel_distance"] = 0.0  # TODO: Add arena locations
        features["home_arena_elevation"] = home_stats.get("elevation", 500)
        features["is_neutral_site"] = 0.0

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
        features["season_progression"] = self._calc_season_progression(
            game_date
        )

        # 7. Vegas Market Signals (6 features)
        features["vegas_spread"] = vegas_spread
        features["vegas_total"] = vegas_total
        if opening_spread is not None and opening_spread != 0:
            features["line_movement_pct"] = abs(
                vegas_spread - opening_spread
            ) / abs(opening_spread)
        else:
            features["line_movement_pct"] = 0.0
        features["spread_direction"] = -1.0 if vegas_spread < 0 else 1.0
        features["sharp_indicator"] = 0.0  # TODO: Detect reverse line movement
        features["consensus_diff"] = 0.0  # TODO: Add consensus tracking

        # 8. Interaction Terms (6 features)
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

        # Create game ID
        game_id = f"{game_date}_{away_team_norm}@{home_team_norm}"

        return GameFeatures(
            game_id=game_id,
            features=features,
            feature_names=self.FEATURE_NAMES,
            timestamp=str(pd.Timestamp.now()),
        )

    def _calc_four_factors_index(
        self, home: dict[str, Any], away: dict[str, Any]
    ) -> float:
        """Calculate composite four factors index."""
        home_ff = (
            home.get("efg_pct", 0)
            + (1 - home.get("to_rate", 0) / 100)  # Lower TO% is better
            + home.get("or_pct", 0) / 100
            + home.get("ft_rate", 0)
        ) / 4

        away_ff = (
            away.get("efg_pct", 0)
            + (1 - away.get("to_rate", 0) / 100)
            + away.get("or_pct", 0) / 100
            + away.get("ft_rate", 0)
        ) / 4

        return home_ff - away_ff

    def _calc_season_progression(self, game_date: date) -> float:
        """Calculate season progression (0 = start, 1 = tournament)."""
        # Season typically runs Nov 4 to April 7 (Selection Sunday)
        season_start = date(game_date.year, 11, 4)
        if game_date.month < 7:
            season_start = date(game_date.year - 1, 11, 4)

        season_end = date(season_start.year + 1, 4, 7)
        total_days = (season_end - season_start).days
        elapsed_days = (game_date - season_start).days

        return max(0.0, min(1.0, elapsed_days / total_days))

    def clear_cache(self) -> None:
        """Clear the team data cache."""
        self._team_cache.clear()
        self._cache_date = None
