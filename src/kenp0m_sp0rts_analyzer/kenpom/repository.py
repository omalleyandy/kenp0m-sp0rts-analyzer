"""Repository layer for KenPom data access.

This module provides a clean interface for all database operations,
abstracting away SQL queries and transaction management.
"""

import logging
from datetime import date, timedelta
from typing import Any

from .database import DatabaseManager
from .exceptions import DatabaseError, TeamNotFoundError
from .models import (
    AccuracyReport,
    FanMatchPrediction,
    FourFactors,
    GamePrediction,
    GameResult,
    HeightExperience,
    MiscStats,
    PointDistribution,
    Team,
    TeamRating,
)
from .validators import DataValidator

logger = logging.getLogger(__name__)


class KenPomRepository:
    """Repository for KenPom data with full CRUD operations.

    Provides clean interface for:
    - Team management
    - Ratings snapshots
    - Four Factors data
    - Point distribution data
    - Predictions and accuracy tracking
    """

    def __init__(self, db_path: str = "data/kenpom.db"):
        """Initialize repository.

        Args:
            db_path: Path to SQLite database.
        """
        self.db = DatabaseManager(db_path)
        self.validator = DataValidator()

    # ==================== Teams ====================

    def upsert_teams(self, teams: list[dict[str, Any]]) -> int:
        """Insert or update team records.

        Args:
            teams: List of team dictionaries with team_id, team_name, etc.

        Returns:
            Number of teams upserted.
        """
        count = 0
        with self.db.transaction() as conn:
            for team in teams:
                # Generate team_id from RankAdjEM if not provided
                # API doesn't return TeamID, so we use rank as stable identifier
                team_id = (
                    team.get("team_id")
                    or team.get("TeamID")
                    or team.get("RankAdjEM")
                )

                if team_id is None:
                    logger.warning(
                        f"Skipping team without ID: {team.get('TeamName')}"
                    )
                    continue

                conn.execute(
                    """
                    INSERT INTO teams (team_id, team_name, conference, coach, arena)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(team_id) DO UPDATE SET
                        team_name = excluded.team_name,
                        conference = excluded.conference,
                        coach = excluded.coach,
                        arena = excluded.arena,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        team_id,
                        team.get("team_name") or team.get("TeamName"),
                        team.get("conference") or team.get("ConfShort"),
                        team.get("coach") or team.get("Coach"),
                        team.get("arena") or team.get("Arena"),
                    ),
                )
                count += 1
        logger.debug(f"Upserted {count} teams")
        return count

    def get_team_by_id(self, team_id: int) -> Team | None:
        """Get team by ID.

        Args:
            team_id: KenPom team ID.

        Returns:
            Team model or None if not found.
        """
        with self.db.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM teams WHERE team_id = ?", (team_id,)
            )
            row = cursor.fetchone()
            return Team(**dict(row)) if row else None

    def get_team_by_name(self, name: str) -> Team | None:
        """Get team by name (case-insensitive).

        Args:
            name: Team name.

        Returns:
            Team model or None if not found.
        """
        with self.db.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM teams WHERE LOWER(team_name) = LOWER(?)",
                (name,),
            )
            row = cursor.fetchone()
            return Team(**dict(row)) if row else None

    def get_all_teams(self) -> list[Team]:
        """Get all teams.

        Returns:
            List of all Team models.
        """
        with self.db.connection() as conn:
            cursor = conn.execute("SELECT * FROM teams ORDER BY team_name")
            return [Team(**dict(row)) for row in cursor.fetchall()]

    def search_teams(self, query: str, limit: int = 10) -> list[Team]:
        """Search teams by name.

        Args:
            query: Search string.
            limit: Maximum results.

        Returns:
            List of matching teams.
        """
        with self.db.connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM teams
                WHERE team_name LIKE ?
                ORDER BY team_name
                LIMIT ?
                """,
                (f"%{query}%", limit),
            )
            return [Team(**dict(row)) for row in cursor.fetchall()]

    # ==================== Ratings Snapshots ====================

    def save_ratings_snapshot(
        self,
        snapshot_date: date,
        season: int,
        ratings: list[dict[str, Any]],
    ) -> int:
        """Save a daily ratings snapshot.

        Args:
            snapshot_date: Date of the snapshot.
            season: Season year.
            ratings: List of rating dictionaries.

        Returns:
            Number of ratings saved.
        """
        count = 0
        with self.db.transaction() as conn:
            for rating in ratings:
                # Validate before saving
                result = self.validator.validate_rating(rating)
                if not result.valid:
                    logger.warning(
                        f"Skipping invalid rating for {rating.get('TeamName')}: "
                        f"{result.errors}"
                    )
                    continue

                # Map both API and internal field names
                team_id = rating.get("team_id") or rating.get("TeamID")
                team_name = rating.get("team_name") or rating.get("TeamName")

                # If team_id not in response, look it up by name
                if not team_id and team_name:
                    team = self.get_team_by_name(team_name)
                    if team:
                        team_id = team.team_id
                    else:
                        logger.warning(
                            f"Team '{team_name}' not found in database. Skipping."
                        )
                        continue

                conn.execute(
                    """
                    INSERT INTO ratings (
                        snapshot_date, season, team_id, team_name, conference,
                        adj_em, adj_oe, adj_de, adj_tempo,
                        luck, sos, soso, sosd, ncsos, pythag,
                        rank_adj_em, rank_adj_oe, rank_adj_de, rank_tempo,
                        rank_sos, rank_luck,
                        wins, losses, apl_off, apl_def
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(snapshot_date, team_id) DO UPDATE SET
                        adj_em = excluded.adj_em,
                        adj_oe = excluded.adj_oe,
                        adj_de = excluded.adj_de,
                        adj_tempo = excluded.adj_tempo,
                        luck = excluded.luck,
                        sos = excluded.sos,
                        pythag = excluded.pythag,
                        rank_adj_em = excluded.rank_adj_em,
                        wins = excluded.wins,
                        losses = excluded.losses
                    """,
                    (
                        snapshot_date,
                        season,
                        team_id,
                        team_name,
                        rating.get("conference") or rating.get("ConfShort"),
                        rating.get("adj_em") or rating.get("AdjEM"),
                        rating.get("adj_oe") or rating.get("AdjOE"),
                        rating.get("adj_de") or rating.get("AdjDE"),
                        rating.get("adj_tempo") or rating.get("AdjTempo"),
                        rating.get("luck") or rating.get("Luck", 0),
                        rating.get("sos") or rating.get("SOS", 0),
                        rating.get("soso") or rating.get("SOSO", 0),
                        rating.get("sosd") or rating.get("SOSD", 0),
                        rating.get("ncsos") or rating.get("NCSOS", 0),
                        rating.get("pythag") or rating.get("Pythag", 0.5),
                        rating.get("rank_adj_em") or rating.get("RankAdjEM"),
                        rating.get("rank_adj_oe") or rating.get("RankAdjOE"),
                        rating.get("rank_adj_de") or rating.get("RankAdjDE"),
                        rating.get("rank_tempo") or rating.get("RankAdjTempo"),
                        rating.get("rank_sos") or rating.get("RankSOS"),
                        rating.get("rank_luck") or rating.get("RankLuck"),
                        rating.get("wins") or rating.get("Wins", 0),
                        rating.get("losses") or rating.get("Losses", 0),
                        rating.get("apl_off") or rating.get("APL_Off"),
                        rating.get("apl_def") or rating.get("APL_Def"),
                    ),
                )
                count += 1

        logger.info(f"Saved {count} ratings for {snapshot_date}")
        return count

    def get_latest_ratings(self) -> list[TeamRating]:
        """Get the most recent ratings for all teams.

        Returns:
            List of TeamRating models.
        """
        with self.db.connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM ratings r
                WHERE snapshot_date = (
                    SELECT MAX(snapshot_date) FROM ratings
                )
                ORDER BY rank_adj_em
                """
            )
            return [TeamRating(**dict(row)) for row in cursor.fetchall()]

    def get_ratings_on_date(self, snapshot_date: date) -> list[TeamRating]:
        """Get ratings for a specific date.

        Args:
            snapshot_date: Date to query.

        Returns:
            List of TeamRating models.
        """
        with self.db.connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM ratings
                WHERE snapshot_date = ?
                ORDER BY rank_adj_em
                """,
                (snapshot_date,),
            )
            return [TeamRating(**dict(row)) for row in cursor.fetchall()]

    def get_team_rating(
        self,
        team_id: int,
        snapshot_date: date | None = None,
    ) -> TeamRating | None:
        """Get a team's rating for a specific date.

        Args:
            team_id: Team ID.
            snapshot_date: Date to query (defaults to latest).

        Returns:
            TeamRating model or None.
        """
        with self.db.connection() as conn:
            if snapshot_date:
                cursor = conn.execute(
                    """
                    SELECT * FROM ratings
                    WHERE team_id = ? AND snapshot_date = ?
                    """,
                    (team_id, snapshot_date),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM ratings
                    WHERE team_id = ?
                    ORDER BY snapshot_date DESC
                    LIMIT 1
                    """,
                    (team_id,),
                )
            row = cursor.fetchone()
            return TeamRating(**dict(row)) if row else None

    def get_team_rating_history(
        self,
        team_id: int,
        days: int = 30,
        season: int | None = None,
    ) -> list[TeamRating]:
        """Get historical ratings for a team.

        Args:
            team_id: Team ID.
            days: Number of days of history.
            season: Optional season filter.

        Returns:
            List of TeamRating models, newest first.
        """
        with self.db.connection() as conn:
            if season:
                cursor = conn.execute(
                    """
                    SELECT * FROM ratings
                    WHERE team_id = ? AND season = ?
                    ORDER BY snapshot_date DESC
                    """,
                    (team_id, season),
                )
            else:
                cutoff = date.today() - timedelta(days=days)
                cursor = conn.execute(
                    """
                    SELECT * FROM ratings
                    WHERE team_id = ? AND snapshot_date >= ?
                    ORDER BY snapshot_date DESC
                    """,
                    (team_id, cutoff),
                )
            return [TeamRating(**dict(row)) for row in cursor.fetchall()]

    def get_available_dates(self, season: int | None = None) -> list[date]:
        """Get list of dates with rating snapshots.

        Args:
            season: Optional season filter.

        Returns:
            List of dates with data.
        """
        with self.db.connection() as conn:
            if season:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT snapshot_date FROM ratings
                    WHERE season = ?
                    ORDER BY snapshot_date
                    """,
                    (season,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT snapshot_date FROM ratings
                    ORDER BY snapshot_date
                    """
                )
            return [row["snapshot_date"] for row in cursor.fetchall()]

    # ==================== Archive ====================

    def save_archive(
        self,
        archive_date: date,
        season: int,
        is_preseason: bool,
        data: list[dict[str, Any]],
    ) -> int:
        """Save archive ratings data from the archive API.

        The archive API provides historical snapshots with comparison to final
        season values (AdjEMFinal, RankChg, etc.).

        Args:
            archive_date: Date of the archived ratings.
            season: Season year (e.g., 2025 for 2024-25).
            is_preseason: Whether this is preseason data.
            data: List of archive rating dictionaries from API.

        Returns:
            Number of records saved.
        """
        count = 0
        with self.db.transaction() as conn:
            for record in data:
                team_id = record.get("TeamID")
                team_name = record.get("TeamName")

                # Look up team_id if not provided
                if not team_id and team_name:
                    team = self.get_team_by_name(team_name)
                    if team:
                        team_id = team.team_id
                    else:
                        logger.warning(
                            f"Team '{team_name}' not found for archive. Skipping."
                        )
                        continue

                conn.execute(
                    """
                    INSERT INTO archive (
                        archive_date, season, is_preseason, team_id, team_name,
                        conference, seed, event,
                        adj_em, adj_oe, adj_de, adj_tempo,
                        rank_adj_em, rank_adj_oe, rank_adj_de, rank_adj_tempo,
                        adj_em_final, adj_oe_final, adj_de_final, adj_tempo_final,
                        rank_adj_em_final, rank_adj_oe_final,
                        rank_adj_de_final, rank_adj_tempo_final,
                        rank_change, adj_em_change, adj_tempo_change
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    ON CONFLICT(archive_date, team_id, is_preseason) DO UPDATE SET
                        season = excluded.season,
                        is_preseason = excluded.is_preseason,
                        team_name = excluded.team_name,
                        conference = excluded.conference,
                        seed = excluded.seed,
                        event = excluded.event,
                        adj_em = excluded.adj_em,
                        adj_oe = excluded.adj_oe,
                        adj_de = excluded.adj_de,
                        adj_tempo = excluded.adj_tempo,
                        rank_adj_em = excluded.rank_adj_em,
                        rank_adj_oe = excluded.rank_adj_oe,
                        rank_adj_de = excluded.rank_adj_de,
                        rank_adj_tempo = excluded.rank_adj_tempo,
                        adj_em_final = excluded.adj_em_final,
                        adj_oe_final = excluded.adj_oe_final,
                        adj_de_final = excluded.adj_de_final,
                        adj_tempo_final = excluded.adj_tempo_final,
                        rank_adj_em_final = excluded.rank_adj_em_final,
                        rank_adj_oe_final = excluded.rank_adj_oe_final,
                        rank_adj_de_final = excluded.rank_adj_de_final,
                        rank_adj_tempo_final = excluded.rank_adj_tempo_final,
                        rank_change = excluded.rank_change,
                        adj_em_change = excluded.adj_em_change,
                        adj_tempo_change = excluded.adj_tempo_change
                    """,
                    (
                        archive_date,
                        season,
                        is_preseason,
                        team_id,
                        team_name,
                        record.get("ConfShort"),
                        record.get("Seed"),
                        record.get("Event"),
                        # Current archive date values
                        record.get("AdjEM"),
                        record.get("AdjOE"),
                        record.get("AdjDE"),
                        record.get("AdjTempo"),
                        record.get("RankAdjEM"),
                        record.get("RankAdjOE"),
                        record.get("RankAdjDE"),
                        record.get("RankAdjTempo"),
                        # Final season values
                        record.get("AdjEMFinal"),
                        record.get("AdjOEFinal"),
                        record.get("AdjDEFinal"),
                        record.get("AdjTempoFinal"),
                        record.get("RankAdjEMFinal"),
                        record.get("RankAdjOEFinal"),
                        record.get("RankAdjDEFinal"),
                        record.get("RankAdjTempoFinal"),
                        # Change values
                        record.get("RankChg"),
                        record.get("AdjEMChg"),
                        record.get("AdjTChg"),
                    ),
                )
                count += 1

        logger.info(
            f"Saved {count} archive ratings for {archive_date} "
            f"(season={season}, preseason={is_preseason})"
        )
        return count

    def get_archive(
        self,
        archive_date: date | None = None,
        season: int | None = None,
        is_preseason: bool | None = None,
        team_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get archive ratings with optional filters.

        Args:
            archive_date: Filter by specific date.
            season: Filter by season.
            is_preseason: Filter by preseason flag.
            team_id: Filter by team.

        Returns:
            List of archive rating records.
        """
        conditions = []
        params: list[Any] = []

        if archive_date:
            conditions.append("archive_date = ?")
            params.append(archive_date)
        if season:
            conditions.append("season = ?")
            params.append(season)
        if is_preseason is not None:
            conditions.append("is_preseason = ?")
            params.append(is_preseason)
        if team_id:
            conditions.append("team_id = ?")
            params.append(team_id)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self.db.connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT * FROM archive
                WHERE {where_clause}
                ORDER BY archive_date DESC, rank_adj_em
                """,
                params,
            )
            return [dict(row) for row in cursor.fetchall()]

    # ==================== Four Factors ====================

    def save_four_factors(
        self,
        snapshot_date: date,
        data: list[dict[str, Any]],
    ) -> int:
        """Save four factors data.

        Args:
            snapshot_date: Date of the data.
            data: List of four factors dictionaries.

        Returns:
            Number of records saved.
        """
        count = 0
        with self.db.transaction() as conn:
            for record in data:
                team_id = record.get("team_id") or record.get("TeamID")

                # If no team_id, look up by team name
                if not team_id and record.get("TeamName"):
                    team = self.get_team_by_name(record["TeamName"])
                    if team:
                        team_id = team.team_id

                conn.execute(
                    """
                    INSERT INTO four_factors (
                        snapshot_date, team_id,
                        efg_pct_off, to_pct_off, or_pct_off, ft_rate_off,
                        efg_pct_def, to_pct_def, or_pct_def, ft_rate_def,
                        rank_efg_off, rank_efg_def, rank_to_off, rank_to_def,
                        rank_or_off, rank_or_def, rank_ft_rate_off, rank_ft_rate_def
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(snapshot_date, team_id) DO UPDATE SET
                        efg_pct_off = excluded.efg_pct_off,
                        to_pct_off = excluded.to_pct_off,
                        or_pct_off = excluded.or_pct_off,
                        ft_rate_off = excluded.ft_rate_off,
                        efg_pct_def = excluded.efg_pct_def,
                        to_pct_def = excluded.to_pct_def,
                        or_pct_def = excluded.or_pct_def,
                        ft_rate_def = excluded.ft_rate_def
                    """,
                    (
                        snapshot_date,
                        team_id,
                        record.get("efg_pct_off") or record.get("eFG_Pct"),
                        record.get("to_pct_off") or record.get("TO_Pct"),
                        record.get("or_pct_off") or record.get("OR_Pct"),
                        record.get("ft_rate_off") or record.get("FT_Rate"),
                        record.get("efg_pct_def") or record.get("DeFG_Pct"),
                        record.get("to_pct_def") or record.get("DTO_Pct"),
                        record.get("or_pct_def") or record.get("DOR_Pct"),
                        record.get("ft_rate_def") or record.get("DFT_Rate"),
                        record.get("rank_efg_off")
                        or record.get("RankeFG_Pct"),
                        record.get("rank_efg_def")
                        or record.get("RankDeFG_Pct"),
                        record.get("rank_to_off") or record.get("RankTO_Pct"),
                        record.get("rank_to_def") or record.get("RankDTO_Pct"),
                        record.get("rank_or_off") or record.get("RankOR_Pct"),
                        record.get("rank_or_def") or record.get("RankDOR_Pct"),
                        record.get("rank_ft_rate_off")
                        or record.get("RankFT_Rate"),
                        record.get("rank_ft_rate_def")
                        or record.get("RankDFT_Rate"),
                    ),
                )
                count += 1

        logger.debug(f"Saved {count} four factors records for {snapshot_date}")
        return count

    def get_four_factors(
        self,
        team_id: int,
        snapshot_date: date | None = None,
    ) -> FourFactors | None:
        """Get four factors for a team.

        Args:
            team_id: Team ID.
            snapshot_date: Optional date (defaults to latest).

        Returns:
            FourFactors model or None.
        """
        with self.db.connection() as conn:
            if snapshot_date:
                cursor = conn.execute(
                    """
                    SELECT * FROM four_factors
                    WHERE team_id = ? AND snapshot_date = ?
                    """,
                    (team_id, snapshot_date),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM four_factors
                    WHERE team_id = ?
                    ORDER BY snapshot_date DESC
                    LIMIT 1
                    """,
                    (team_id,),
                )
            row = cursor.fetchone()
            return FourFactors(**dict(row)) if row else None

    # ==================== Point Distribution ====================

    def save_point_distribution(
        self,
        snapshot_date: date,
        data: list[dict[str, Any]],
    ) -> int:
        """Save point distribution data.

        Args:
            snapshot_date: Date of the data.
            data: List of point distribution dictionaries.

        Returns:
            Number of records saved.
        """
        count = 0
        with self.db.transaction() as conn:
            for record in data:
                team_id = record.get("team_id") or record.get("TeamID")

                # If no team_id, look up by team name
                if not team_id and record.get("TeamName"):
                    team = self.get_team_by_name(record["TeamName"])
                    if team:
                        team_id = team.team_id

                conn.execute(
                    """
                    INSERT INTO point_distribution (
                        snapshot_date, team_id,
                        ft_pct, two_pct, three_pct,
                        ft_pct_def, two_pct_def, three_pct_def,
                        rank_three_pct, rank_two_pct, rank_ft_pct
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(snapshot_date, team_id) DO UPDATE SET
                        ft_pct = excluded.ft_pct,
                        two_pct = excluded.two_pct,
                        three_pct = excluded.three_pct,
                        ft_pct_def = excluded.ft_pct_def,
                        two_pct_def = excluded.two_pct_def,
                        three_pct_def = excluded.three_pct_def
                    """,
                    (
                        snapshot_date,
                        team_id,
                        record.get("ft_pct") or record.get("OffFt"),
                        record.get("two_pct") or record.get("OffFg2"),
                        record.get("three_pct") or record.get("OffFg3"),
                        record.get("ft_pct_def") or record.get("DefFt"),
                        record.get("two_pct_def") or record.get("DefFg2"),
                        record.get("three_pct_def") or record.get("DefFg3"),
                        record.get("rank_three_pct")
                        or record.get("RankOffFg3"),
                        record.get("rank_two_pct") or record.get("RankOffFg2"),
                        record.get("rank_ft_pct") or record.get("RankOffFt"),
                    ),
                )
                count += 1

        logger.debug(
            f"Saved {count} point distribution records for {snapshot_date}"
        )
        return count

    def get_point_distribution(
        self,
        team_id: int,
        snapshot_date: date | None = None,
    ) -> PointDistribution | None:
        """Get point distribution for a team.

        Args:
            team_id: Team ID.
            snapshot_date: Optional date (defaults to latest).

        Returns:
            PointDistribution model or None.
        """
        with self.db.connection() as conn:
            if snapshot_date:
                cursor = conn.execute(
                    """
                    SELECT * FROM point_distribution
                    WHERE team_id = ? AND snapshot_date = ?
                    """,
                    (team_id, snapshot_date),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM point_distribution
                    WHERE team_id = ?
                    ORDER BY snapshot_date DESC
                    LIMIT 1
                    """,
                    (team_id,),
                )
            row = cursor.fetchone()
            return PointDistribution(**dict(row)) if row else None

    # ==================== FanMatch Predictions ====================

    def save_fanmatch_predictions(
        self,
        snapshot_date: date,
        data: list[dict[str, Any]],
    ) -> int:
        """Save KenPom FanMatch predictions.

        Args:
            snapshot_date: Date of the data.
            data: List of fanmatch prediction dictionaries.

        Returns:
            Number of records saved.
        """
        count = 0
        with self.db.transaction() as conn:
            for record in data:
                home_id = record.get("HomeTeamID") or record.get(
                    "home_team_id"
                )
                visitor_id = record.get("VisitorTeamID") or record.get(
                    "visitor_team_id"
                )

                # Generate game_id from teams and date
                game_date = record.get(
                    "GameDate", snapshot_date.strftime("%Y%m%d")
                )
                game_id = f"{home_id}-{visitor_id}-{game_date}"

                # Extract predictions
                home_pred = record.get("HomePred") or record.get(
                    "pred_home_score", 0
                )
                visitor_pred = record.get("VisitorPred") or record.get(
                    "pred_visitor_score", 0
                )
                margin = record.get("pred_margin") or (
                    home_pred - visitor_pred
                )

                # Win probability (convert from percentage if needed)
                home_wp = record.get("HomeWP") or record.get(
                    "home_win_prob", 50
                )
                if home_wp > 1:  # Assume it's a percentage
                    home_wp = home_wp / 100

                conn.execute(
                    """
                    INSERT INTO fanmatch_predictions (
                        snapshot_date, game_id, home_team_id, visitor_team_id,
                        home_team_name, visitor_team_name,
                        pred_home_score, pred_visitor_score, pred_margin,
                        home_win_prob, pred_tempo, thrill_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(snapshot_date, game_id) DO UPDATE SET
                        pred_home_score = excluded.pred_home_score,
                        pred_visitor_score = excluded.pred_visitor_score,
                        pred_margin = excluded.pred_margin,
                        home_win_prob = excluded.home_win_prob,
                        pred_tempo = excluded.pred_tempo,
                        thrill_score = excluded.thrill_score
                    """,
                    (
                        snapshot_date,
                        game_id,
                        home_id,
                        visitor_id,
                        record.get("Home") or record.get("home_team_name"),
                        record.get("Visitor")
                        or record.get("visitor_team_name"),
                        home_pred,
                        visitor_pred,
                        margin,
                        home_wp,
                        record.get("PredTempo") or record.get("pred_tempo"),
                        record.get("ThrillScore")
                        or record.get("thrill_score"),
                    ),
                )
                count += 1

        logger.debug(f"Saved {count} fanmatch predictions for {snapshot_date}")
        return count

    def get_fanmatch_for_game(
        self,
        home_team_id: int,
        away_team_id: int,
        snapshot_date: date | None = None,
    ) -> FanMatchPrediction | None:
        """Get KenPom prediction for specific game.

        Args:
            home_team_id: Home team ID.
            away_team_id: Away team ID.
            snapshot_date: Optional date (defaults to latest).

        Returns:
            FanMatchPrediction if found, None otherwise.
        """
        with self.db.connection() as conn:
            if snapshot_date:
                cursor = conn.execute(
                    """
                    SELECT * FROM fanmatch_predictions
                    WHERE home_team_id = ? AND visitor_team_id = ?
                        AND snapshot_date = ?
                    """,
                    (home_team_id, away_team_id, snapshot_date),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM fanmatch_predictions
                    WHERE home_team_id = ? AND visitor_team_id = ?
                    ORDER BY snapshot_date DESC
                    LIMIT 1
                    """,
                    (home_team_id, away_team_id),
                )
            row = cursor.fetchone()
            return FanMatchPrediction(**dict(row)) if row else None

    # ==================== Misc Stats ====================

    def save_misc_stats(
        self,
        snapshot_date: date,
        data: list[dict[str, Any]],
    ) -> int:
        """Save misc stats data.

        Args:
            snapshot_date: Date of the data.
            data: List of misc stats dictionaries.

        Returns:
            Number of records saved.
        """
        count = 0
        with self.db.transaction() as conn:
            for record in data:
                team_id = record.get("team_id") or record.get("TeamID")

                # If no team_id, look up by team name
                if not team_id and record.get("TeamName"):
                    team = self.get_team_by_name(record["TeamName"])
                    if team:
                        team_id = team.team_id

                if not team_id:
                    continue

                conn.execute(
                    """
                    INSERT INTO misc_stats (
                        snapshot_date, team_id,
                        fg3_pct_off, fg3_pct_def, fg2_pct_off, fg2_pct_def,
                        ft_pct_off, ft_pct_def, assist_rate, assist_rate_def,
                        steal_rate, steal_rate_def, block_pct_off, block_pct_def,
                        rank_fg3_pct, rank_fg2_pct, rank_ft_pct, rank_assist_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(snapshot_date, team_id) DO UPDATE SET
                        fg3_pct_off = excluded.fg3_pct_off,
                        fg3_pct_def = excluded.fg3_pct_def,
                        fg2_pct_off = excluded.fg2_pct_off,
                        fg2_pct_def = excluded.fg2_pct_def,
                        ft_pct_off = excluded.ft_pct_off,
                        ft_pct_def = excluded.ft_pct_def,
                        assist_rate = excluded.assist_rate,
                        assist_rate_def = excluded.assist_rate_def,
                        steal_rate = excluded.steal_rate,
                        steal_rate_def = excluded.steal_rate_def,
                        block_pct_off = excluded.block_pct_off,
                        block_pct_def = excluded.block_pct_def
                    """,
                    (
                        snapshot_date,
                        team_id,
                        record.get("FG3Pct") or record.get("fg3_pct_off"),
                        record.get("OppFG3Pct") or record.get("fg3_pct_def"),
                        record.get("FG2Pct") or record.get("fg2_pct_off"),
                        record.get("OppFG2Pct") or record.get("fg2_pct_def"),
                        record.get("FTPct") or record.get("ft_pct_off"),
                        record.get("OppFTPct") or record.get("ft_pct_def"),
                        record.get("ARate") or record.get("assist_rate"),
                        record.get("OppARate")
                        or record.get("assist_rate_def"),
                        record.get("StlRate") or record.get("steal_rate"),
                        record.get("OppStlRate")
                        or record.get("steal_rate_def"),
                        record.get("BlockPct") or record.get("block_pct_off"),
                        record.get("OppBlockPct")
                        or record.get("block_pct_def"),
                        record.get("RankFG3Pct") or record.get("rank_fg3_pct"),
                        record.get("RankFG2Pct") or record.get("rank_fg2_pct"),
                        record.get("RankFTPct") or record.get("rank_ft_pct"),
                        record.get("RankARate")
                        or record.get("rank_assist_rate"),
                    ),
                )
                count += 1

        logger.debug(f"Saved {count} misc stats records for {snapshot_date}")
        return count

    def get_misc_stats(
        self,
        team_id: int,
        snapshot_date: date | None = None,
    ) -> MiscStats | None:
        """Get misc stats for a team.

        Args:
            team_id: Team ID.
            snapshot_date: Optional date (defaults to latest).

        Returns:
            MiscStats if found, None otherwise.
        """
        with self.db.connection() as conn:
            if snapshot_date:
                cursor = conn.execute(
                    """
                    SELECT * FROM misc_stats
                    WHERE team_id = ? AND snapshot_date = ?
                    """,
                    (team_id, snapshot_date),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM misc_stats
                    WHERE team_id = ?
                    ORDER BY snapshot_date DESC
                    LIMIT 1
                    """,
                    (team_id,),
                )
            row = cursor.fetchone()
            return MiscStats(**dict(row)) if row else None

    def get_height_experience(
        self,
        team_id: int,
        snapshot_date: date | None = None,
    ) -> HeightExperience | None:
        """Get height/experience data for a team.

        Args:
            team_id: Team ID.
            snapshot_date: Optional date (defaults to latest).

        Returns:
            HeightExperience if found, None otherwise.
        """
        with self.db.connection() as conn:
            if snapshot_date:
                cursor = conn.execute(
                    """
                    SELECT * FROM height_experience
                    WHERE team_id = ? AND snapshot_date = ?
                    """,
                    (team_id, snapshot_date),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM height_experience
                    WHERE team_id = ?
                    ORDER BY snapshot_date DESC
                    LIMIT 1
                    """,
                    (team_id,),
                )
            row = cursor.fetchone()
            return HeightExperience(**dict(row)) if row else None

    # ==================== Predictions ====================

    def save_prediction(self, prediction: GamePrediction) -> int:
        """Save a game prediction.

        Args:
            prediction: GamePrediction model.

        Returns:
            ID of saved prediction.
        """
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO game_predictions (
                    game_date, team1_id, team2_id, team1_name, team2_name,
                    predicted_margin, predicted_total, win_probability,
                    confidence_lower, confidence_upper,
                    vegas_spread, vegas_total,
                    neutral_site, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction.game_date,
                    prediction.team1_id,
                    prediction.team2_id,
                    prediction.team1_name,
                    prediction.team2_name,
                    prediction.predicted_margin,
                    prediction.predicted_total,
                    prediction.win_probability,
                    prediction.confidence_lower,
                    prediction.confidence_upper,
                    prediction.vegas_spread,
                    prediction.vegas_total,
                    1 if prediction.neutral_site else 0,
                    prediction.model_version,
                ),
            )
            return cursor.lastrowid

    def get_pending_predictions(self) -> list[GamePrediction]:
        """Get predictions that haven't been resolved.

        Returns:
            List of unresolved GamePrediction models.
        """
        with self.db.connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM game_predictions
                WHERE resolved_at IS NULL
                ORDER BY game_date
                """
            )
            return [GamePrediction(**dict(row)) for row in cursor.fetchall()]

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
        with self.db.transaction() as conn:
            # Get the prediction to calculate metrics
            cursor = conn.execute(
                "SELECT * FROM game_predictions WHERE id = ?",
                (prediction_id,),
            )
            row = cursor.fetchone()
            if not row:
                raise DatabaseError(f"Prediction {prediction_id} not found")

            pred = dict(row)

            # Calculate error and ATS result
            prediction_error = result.actual_margin - pred["predicted_margin"]

            beat_spread = None
            if pred["vegas_spread"] is not None:
                spread_diff = result.actual_margin - pred["vegas_spread"]
                beat_spread = 1 if spread_diff > 0 else 0

            # Calculate CLV if closing line differs from prediction time line
            clv = None

            conn.execute(
                """
                UPDATE game_predictions SET
                    actual_margin = ?,
                    actual_total = ?,
                    team1_score = ?,
                    team2_score = ?,
                    prediction_error = ?,
                    beat_spread = ?,
                    clv = ?,
                    resolved_at = ?
                WHERE id = ?
                """,
                (
                    result.actual_margin,
                    result.actual_total,
                    result.team1_score,
                    result.team2_score,
                    prediction_error,
                    beat_spread,
                    clv,
                    result.resolved_at,
                    prediction_id,
                ),
            )

    def get_accuracy_report(
        self,
        days: int = 30,
        model_version: str | None = None,
    ) -> AccuracyReport:
        """Generate accuracy report for recent predictions.

        Args:
            days: Number of days to include.
            model_version: Optional filter by model version.

        Returns:
            AccuracyReport with performance metrics.
        """
        import statistics

        start_date = date.today() - timedelta(days=days)
        end_date = date.today()

        with self.db.connection() as conn:
            if model_version:
                cursor = conn.execute(
                    """
                    SELECT * FROM game_predictions
                    WHERE game_date >= ? AND resolved_at IS NOT NULL
                    AND model_version = ?
                    """,
                    (start_date, model_version),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM game_predictions
                    WHERE game_date >= ? AND resolved_at IS NOT NULL
                    """,
                    (start_date,),
                )

            predictions = [dict(row) for row in cursor.fetchall()]

        if not predictions:
            return AccuracyReport(
                start_date=start_date,
                end_date=end_date,
                games_predicted=0,
                games_resolved=0,
                mae_margin=0.0,
                rmse_margin=0.0,
                r2_margin=0.0,
                win_accuracy=0.0,
                brier_score=0.0,
            )

        # Calculate metrics
        errors = [
            p["prediction_error"] for p in predictions if p["prediction_error"]
        ]
        mae = statistics.mean([abs(e) for e in errors]) if errors else 0
        rmse = statistics.mean([e**2 for e in errors]) ** 0.5 if errors else 0

        # Win accuracy
        correct_wins = sum(
            1
            for p in predictions
            if (p["predicted_margin"] > 0 and p["actual_margin"] > 0)
            or (p["predicted_margin"] < 0 and p["actual_margin"] < 0)
        )
        win_accuracy = correct_wins / len(predictions) if predictions else 0

        # ATS record
        ats_wins = sum(1 for p in predictions if p["beat_spread"] == 1)
        ats_losses = sum(1 for p in predictions if p["beat_spread"] == 0)
        total_ats = ats_wins + ats_losses
        ats_pct = ats_wins / total_ats if total_ats > 0 else 0

        # Brier score
        brier_scores = []
        for p in predictions:
            if (
                p["win_probability"] is not None
                and p["actual_margin"] is not None
            ):
                actual_win = 1 if p["actual_margin"] > 0 else 0
                brier_scores.append((p["win_probability"] - actual_win) ** 2)
        brier = statistics.mean(brier_scores) if brier_scores else 0

        return AccuracyReport(
            start_date=start_date,
            end_date=end_date,
            games_predicted=len(predictions),
            games_resolved=len(predictions),
            mae_margin=round(mae, 2),
            rmse_margin=round(rmse, 2),
            r2_margin=0.0,
            win_accuracy=round(win_accuracy, 3),
            brier_score=round(brier, 3),
            ats_wins=ats_wins,
            ats_losses=ats_losses,
            ats_percentage=round(ats_pct, 3),
        )

    # ==================== Convenience Methods ====================

    def get_matchup_data(
        self,
        team1_id: int,
        team2_id: int,
        snapshot_date: date | None = None,
    ) -> dict:
        """Get comprehensive matchup data for two teams.

        Args:
            team1_id: First team ID.
            team2_id: Second team ID.
            snapshot_date: Optional date (defaults to latest).

        Returns:
            Dictionary with all relevant matchup data.
        """
        team1_rating = self.get_team_rating(team1_id, snapshot_date)
        team2_rating = self.get_team_rating(team2_id, snapshot_date)

        if not team1_rating or not team2_rating:
            raise TeamNotFoundError(
                f"Rating not found for team(s): {team1_id}, {team2_id}"
            )

        return {
            "team1": team1_rating,
            "team2": team2_rating,
            "team1_four_factors": self.get_four_factors(
                team1_id, snapshot_date
            ),
            "team2_four_factors": self.get_four_factors(
                team2_id, snapshot_date
            ),
            "team1_point_dist": self.get_point_distribution(
                team1_id, snapshot_date
            ),
            "team2_point_dist": self.get_point_distribution(
                team2_id, snapshot_date
            ),
            "team1_misc_stats": self.get_misc_stats(team1_id, snapshot_date),
            "team2_misc_stats": self.get_misc_stats(team2_id, snapshot_date),
            "team1_height": self.get_height_experience(
                team1_id, snapshot_date
            ),
            "team2_height": self.get_height_experience(
                team2_id, snapshot_date
            ),
            "team1_history": self.get_team_rating_history(team1_id, days=28),
            "team2_history": self.get_team_rating_history(team2_id, days=28),
        }

    def get_conference_standings(
        self,
        conference: str,
        snapshot_date: date | None = None,
    ) -> list[TeamRating]:
        """Get ratings for all teams in a conference.

        Args:
            conference: Conference abbreviation (e.g., 'B12', 'SEC').
            snapshot_date: Optional date (defaults to latest).

        Returns:
            List of TeamRating models sorted by rank.
        """
        with self.db.connection() as conn:
            if snapshot_date:
                cursor = conn.execute(
                    """
                    SELECT * FROM ratings
                    WHERE conference = ? AND snapshot_date = ?
                    ORDER BY rank_adj_em
                    """,
                    (conference, snapshot_date),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM ratings r
                    WHERE conference = ?
                    AND snapshot_date = (
                        SELECT MAX(snapshot_date) FROM ratings
                    )
                    ORDER BY rank_adj_em
                    """,
                    (conference,),
                )
            return [TeamRating(**dict(row)) for row in cursor.fetchall()]
