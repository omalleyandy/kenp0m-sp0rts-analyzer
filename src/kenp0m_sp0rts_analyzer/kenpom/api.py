"""Unified KenPom service interface.

This module provides the main entry point for all KenPom data operations,
integrating API access, storage, validation, and ML feature generation.
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .database import DatabaseManager
from .exceptions import (
    ConfigurationError,
    KenPomError,
    TeamNotFoundError,
)
from .models import (
    AccuracyReport,
    FourFactors,
    GamePrediction,
    GameResult,
    MatchupData,
    PointDistribution,
    SyncResult,
    TeamRating,
)
from .repository import KenPomRepository
from .validators import DataValidator

logger = logging.getLogger(__name__)


class KenPomService:
    """Unified service for KenPom data operations.

    This is the main entry point for the KenPom module, providing:
    - Data retrieval (current and historical)
    - Data storage with validation
    - Matchup analysis
    - ML feature generation
    - Accuracy tracking

    Example:
        >>> from kenp0m_sp0rts_analyzer.kenpom import KenPomService
        >>> service = KenPomService()
        >>>
        >>> # Get current ratings
        >>> ratings = service.get_latest_ratings()
        >>>
        >>> # Sync from API
        >>> result = service.sync_ratings()
        >>>
        >>> # Get matchup data for ML
        >>> features = service.get_features_for_game(team1_id=73, team2_id=152)
    """

    def __init__(
        self,
        db_path: str = "data/kenpom.db",
        api_client: Any = None,
    ):
        """Initialize the KenPom service.

        Args:
            db_path: Path to SQLite database file.
            api_client: Optional KenPomAPI instance. If not provided,
                will create one using environment variable KENPOM_API_KEY.
        """
        self.db_path = db_path
        self.repository = KenPomRepository(db_path)
        self.validator = DataValidator()
        self._api_client = api_client
        self._api_initialized = False

    @property
    def api(self):
        """Lazy-load the API client."""
        if not self._api_initialized:
            if self._api_client is None:
                try:
                    from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
                    self._api_client = KenPomAPI()
                except Exception as e:
                    raise ConfigurationError(
                        f"Failed to initialize KenPom API client: {e}"
                    ) from e
            self._api_initialized = True
        return self._api_client

    # ==================== Data Retrieval ====================

    def get_latest_ratings(self) -> list[TeamRating]:
        """Get the most recent ratings for all teams.

        Returns:
            List of TeamRating models sorted by rank.
        """
        return self.repository.get_latest_ratings()

    def get_team_rating(
        self,
        team_id: int | None = None,
        team_name: str | None = None,
        snapshot_date: date | None = None,
    ) -> TeamRating | None:
        """Get a team's rating.

        Args:
            team_id: Team ID (preferred).
            team_name: Team name (case-insensitive search).
            snapshot_date: Optional date (defaults to latest).

        Returns:
            TeamRating model or None.

        Raises:
            ValueError: If neither team_id nor team_name provided.
        """
        if team_id is None and team_name is None:
            raise ValueError("Must provide either team_id or team_name")

        if team_id is None:
            team = self.repository.get_team_by_name(team_name)
            if team is None:
                return None
            team_id = team.team_id

        return self.repository.get_team_rating(team_id, snapshot_date)

    def get_ratings_on_date(self, snapshot_date: date) -> list[TeamRating]:
        """Get all team ratings for a specific date.

        Args:
            snapshot_date: Date to query.

        Returns:
            List of TeamRating models.
        """
        return self.repository.get_ratings_on_date(snapshot_date)

    def get_team_history(
        self,
        team_id: int,
        days: int = 30,
        season: int | None = None,
    ) -> list[TeamRating]:
        """Get historical ratings for a team.

        Args:
            team_id: Team ID.
            days: Number of days of history (default 30).
            season: Optional season year filter.

        Returns:
            List of TeamRating models, newest first.
        """
        return self.repository.get_team_rating_history(team_id, days, season)

    def get_four_factors(
        self,
        team_id: int,
        snapshot_date: date | None = None,
    ) -> FourFactors | None:
        """Get team's Four Factors data.

        Args:
            team_id: Team ID.
            snapshot_date: Optional date (defaults to latest).

        Returns:
            FourFactors model or None.
        """
        return self.repository.get_four_factors(team_id, snapshot_date)

    def get_point_distribution(
        self,
        team_id: int,
        snapshot_date: date | None = None,
    ) -> PointDistribution | None:
        """Get team's point distribution data.

        Args:
            team_id: Team ID.
            snapshot_date: Optional date (defaults to latest).

        Returns:
            PointDistribution model or None.
        """
        return self.repository.get_point_distribution(team_id, snapshot_date)

    def search_teams(self, query: str, limit: int = 10) -> list:
        """Search for teams by name.

        Args:
            query: Search string.
            limit: Maximum results.

        Returns:
            List of matching Team models.
        """
        return self.repository.search_teams(query, limit)

    # ==================== Data Sync ====================

    def sync_ratings(
        self,
        year: int | None = None,
        snapshot_date: date | None = None,
    ) -> SyncResult:
        """Sync team ratings from the API.

        Args:
            year: Season year (defaults to current season).
            snapshot_date: Date to record snapshot (defaults to today).

        Returns:
            SyncResult with operation details.
        """
        from datetime import datetime
        start_time = datetime.now()
        errors = []

        try:
            # Get current year if not specified
            if year is None:
                today = date.today()
                year = today.year if today.month >= 11 else today.year

            # Fetch from API
            response = self.api.get_ratings(year=year)
            data = list(response.data)

            # Validate responses
            sanitized = self.validator.sanitize_response(data)

            # Update teams table FIRST (ratings_snapshots has FK to teams)
            self.repository.upsert_teams(sanitized)

            # Store ratings in database
            snapshot_date = snapshot_date or date.today()
            count = self.repository.save_ratings_snapshot(
                snapshot_date=snapshot_date,
                season=year,
                ratings=sanitized,
            )

            # Record sync
            self.repository.db.record_sync(
                endpoint="ratings",
                sync_type="full",
                status="success",
                records_synced=count,
                started_at=start_time,
            )

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Synced {count} ratings in {duration:.2f}s")

            return SyncResult(
                success=True,
                endpoint="ratings",
                records_synced=count,
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Ratings sync failed: {e}")
            errors.append(str(e))

            self.repository.db.record_sync(
                endpoint="ratings",
                sync_type="full",
                status="failed",
                error_message=str(e),
                started_at=start_time,
            )

            return SyncResult(
                success=False,
                endpoint="ratings",
                records_synced=0,
                errors=errors,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

    def sync_four_factors(
        self,
        year: int | None = None,
        snapshot_date: date | None = None,
    ) -> SyncResult:
        """Sync Four Factors data from the API.

        Args:
            year: Season year.
            snapshot_date: Date to record snapshot.

        Returns:
            SyncResult with operation details.
        """
        from datetime import datetime
        start_time = datetime.now()
        errors = []

        try:
            if year is None:
                today = date.today()
                year = today.year if today.month >= 11 else today.year

            response = self.api.get_four_factors(year=year)
            data = list(response.data)

            snapshot_date = snapshot_date or date.today()
            count = self.repository.save_four_factors(snapshot_date, data)

            self.repository.db.record_sync(
                endpoint="four-factors",
                sync_type="full",
                status="success",
                records_synced=count,
                started_at=start_time,
            )

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Synced {count} four factors in {duration:.2f}s")

            return SyncResult(
                success=True,
                endpoint="four-factors",
                records_synced=count,
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Four factors sync failed: {e}")
            errors.append(str(e))
            return SyncResult(
                success=False,
                endpoint="four-factors",
                records_synced=0,
                errors=errors,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

    def sync_point_distribution(
        self,
        year: int | None = None,
        snapshot_date: date | None = None,
    ) -> SyncResult:
        """Sync point distribution data from the API.

        Args:
            year: Season year.
            snapshot_date: Date to record snapshot.

        Returns:
            SyncResult with operation details.
        """
        from datetime import datetime
        start_time = datetime.now()
        errors = []

        try:
            if year is None:
                today = date.today()
                year = today.year if today.month >= 11 else today.year

            response = self.api.get_point_distribution(year=year)
            data = list(response.data)

            snapshot_date = snapshot_date or date.today()
            count = self.repository.save_point_distribution(snapshot_date, data)

            self.repository.db.record_sync(
                endpoint="pointdist",
                sync_type="full",
                status="success",
                records_synced=count,
                started_at=start_time,
            )

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Synced {count} point distributions in {duration:.2f}s")

            return SyncResult(
                success=True,
                endpoint="pointdist",
                records_synced=count,
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Point distribution sync failed: {e}")
            errors.append(str(e))
            return SyncResult(
                success=False,
                endpoint="pointdist",
                records_synced=0,
                errors=errors,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

    def sync_all(self, year: int | None = None) -> dict[str, SyncResult]:
        """Sync all data types from the API.

        Args:
            year: Season year.

        Returns:
            Dictionary mapping endpoint names to SyncResults.
        """
        results = {}

        results["ratings"] = self.sync_ratings(year=year)
        results["four_factors"] = self.sync_four_factors(year=year)
        results["point_distribution"] = self.sync_point_distribution(year=year)

        # Log summary
        total_synced = sum(r.records_synced for r in results.values())
        success_count = sum(1 for r in results.values() if r.success)
        logger.info(
            f"Sync complete: {success_count}/{len(results)} endpoints, "
            f"{total_synced} total records"
        )

        return results

    # ==================== Matchup Analysis ====================

    def get_matchup(
        self,
        team1_id: int,
        team2_id: int,
        snapshot_date: date | None = None,
    ) -> MatchupData:
        """Get comprehensive matchup data for two teams.

        Args:
            team1_id: First team ID.
            team2_id: Second team ID.
            snapshot_date: Optional date (defaults to latest).

        Returns:
            MatchupData model with all relevant data.

        Raises:
            TeamNotFoundError: If either team not found.
        """
        data = self.repository.get_matchup_data(team1_id, team2_id, snapshot_date)

        return MatchupData(
            team1=data["team1"],
            team2=data["team2"],
            team1_four_factors=data.get("team1_four_factors"),
            team2_four_factors=data.get("team2_four_factors"),
            team1_point_dist=data.get("team1_point_dist"),
            team2_point_dist=data.get("team2_point_dist"),
        )

    def get_features_for_game(
        self,
        team1_id: int,
        team2_id: int,
        home_team_id: int | None = None,
        neutral_site: bool = False,
        snapshot_date: date | None = None,
    ) -> dict[str, float]:
        """Generate ML features for a game prediction.

        This produces features compatible with the XGBoost enhanced model
        (27 features) used in prediction.py.

        Args:
            team1_id: First team ID (home unless neutral).
            team2_id: Second team ID (away unless neutral).
            home_team_id: Optional explicit home team.
            neutral_site: Whether game is at neutral site.
            snapshot_date: Optional date for ratings snapshot.

        Returns:
            Dictionary of feature names to values.

        Example:
            >>> features = service.get_features_for_game(73, 152)
            >>> features.keys()
            dict_keys(['adj_em_diff', 'adj_oe_diff', 'adj_de_diff', ...])
        """
        matchup = self.get_matchup(team1_id, team2_id, snapshot_date)
        t1 = matchup.team1
        t2 = matchup.team2

        # Determine home court advantage
        home_advantage = 0.0
        if not neutral_site:
            if home_team_id == team1_id or home_team_id is None:
                home_advantage = 3.75  # Team1 is home
            elif home_team_id == team2_id:
                home_advantage = -3.75  # Team2 is home

        # Build feature dictionary - matches XGBoostFeatureEngineer
        features = {
            # Core efficiency differentials
            "adj_em_diff": t1.adj_em - t2.adj_em,
            "adj_oe_diff": t1.adj_oe - t2.adj_oe,
            "adj_de_diff": t1.adj_de - t2.adj_de,
            "adj_tempo_diff": t1.adj_tempo - t2.adj_tempo,
            # Pythagorean expectation
            "pythag_diff": t1.pythag - t2.pythag,
            # Home court
            "home_advantage": home_advantage,
            # Luck differential (regression target)
            "luck_diff": t1.luck - t2.luck,
            # Strength of schedule
            "sos_diff": t1.sos - t2.sos,
            "ncsos_diff": t1.ncsos - t2.ncsos,
            # Average possession length
            "apl_off_diff": (t1.apl_off or 16.5) - (t2.apl_off or 16.5),
            "apl_def_diff": (t1.apl_def or 16.5) - (t2.apl_def or 16.5),
            # Ranking differentials (lower is better)
            "rank_diff": (t1.rank_adj_em or 200) - (t2.rank_adj_em or 200),
            "rank_oe_diff": (t1.rank_adj_oe or 200) - (t2.rank_adj_oe or 200),
            "rank_de_diff": (t1.rank_adj_de or 200) - (t2.rank_adj_de or 200),
            # Record-based features
            "win_pct_diff": (t1.wins / max(1, t1.wins + t1.losses)) -
                           (t2.wins / max(1, t2.wins + t2.losses)),
        }

        # Add Four Factors if available
        ff1 = matchup.team1_four_factors
        ff2 = matchup.team2_four_factors

        if ff1 and ff2:
            features.update({
                "efg_diff": ff1.efg_pct_off - ff2.efg_pct_off,
                "to_diff": ff1.to_pct_off - ff2.to_pct_off,
                "or_diff": ff1.or_pct_off - ff2.or_pct_off,
                "ft_rate_diff": ff1.ft_rate_off - ff2.ft_rate_off,
                # Offensive vs opposing defense
                "efg_adv_t1": ff1.efg_pct_off - ff2.efg_pct_def,
                "efg_adv_t2": ff2.efg_pct_off - ff1.efg_pct_def,
            })
        else:
            # Default values when Four Factors not available
            features.update({
                "efg_diff": 0.0,
                "to_diff": 0.0,
                "or_diff": 0.0,
                "ft_rate_diff": 0.0,
                "efg_adv_t1": 0.0,
                "efg_adv_t2": 0.0,
            })

        # Add Point Distribution if available
        pd1 = matchup.team1_point_dist
        pd2 = matchup.team2_point_dist

        if pd1 and pd2:
            features.update({
                "three_pct_diff": pd1.three_pct - pd2.three_pct,
                "two_pct_diff": pd1.two_pct - pd2.two_pct,
                "ft_pct_diff": pd1.ft_pct - pd2.ft_pct,
                # Variance proxy (3-point reliance indicates volatility)
                "three_reliance_diff": pd1.three_pct - pd2.three_pct,
            })
        else:
            features.update({
                "three_pct_diff": 0.0,
                "two_pct_diff": 0.0,
                "ft_pct_diff": 0.0,
                "three_reliance_diff": 0.0,
            })

        # Combined tempo for total prediction
        features["avg_tempo"] = (t1.adj_tempo + t2.adj_tempo) / 2

        return features

    # ==================== Predictions & Accuracy ====================

    def save_prediction(self, prediction: GamePrediction) -> int:
        """Save a game prediction for tracking.

        Args:
            prediction: GamePrediction model.

        Returns:
            ID of saved prediction.
        """
        return self.repository.save_prediction(prediction)

    def get_pending_predictions(self) -> list[GamePrediction]:
        """Get predictions awaiting results.

        Returns:
            List of unresolved GamePrediction models.
        """
        return self.repository.get_pending_predictions()

    def update_prediction_result(
        self,
        prediction_id: int,
        result: GameResult,
    ) -> None:
        """Update a prediction with actual game results.

        Args:
            prediction_id: ID of prediction to update.
            result: GameResult with actual outcome.
        """
        self.repository.update_prediction_result(prediction_id, result)

    def get_accuracy_report(
        self,
        days: int = 30,
        model_version: str | None = None,
    ) -> AccuracyReport:
        """Generate prediction accuracy report.

        Args:
            days: Number of days to include.
            model_version: Optional filter by model version.

        Returns:
            AccuracyReport with performance metrics.
        """
        return self.repository.get_accuracy_report(days, model_version)

    # ==================== Database Management ====================

    def get_database_stats(self) -> dict:
        """Get database statistics.

        Returns:
            Dictionary with table counts and database info.
        """
        return self.repository.db.get_stats()

    def backup_database(self, backup_path: str | None = None) -> Path:
        """Create a database backup.

        Args:
            backup_path: Optional custom backup path.

        Returns:
            Path to backup file.
        """
        return self.repository.db.backup(backup_path)

    def check_data_freshness(
        self,
        endpoint: str = "ratings",
        max_age_hours: int = 24,
    ) -> tuple[bool, str]:
        """Check if data needs refresh.

        Args:
            endpoint: Endpoint to check.
            max_age_hours: Maximum acceptable age.

        Returns:
            Tuple of (is_fresh, message).
        """
        sync = self.repository.db.get_latest_sync(endpoint)
        if sync is None:
            return False, f"No sync history for {endpoint}"

        return self.validator.check_data_freshness(
            sync["completed_at"],
            max_age_hours,
        )
