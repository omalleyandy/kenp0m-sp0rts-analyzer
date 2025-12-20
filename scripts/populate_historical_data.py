#!/usr/bin/env python3
"""Populate KenPom database with historical data (2020-2025).

This script fetches historical data from the KenPom API and populates
all database tables with 5+ years of data for XGBoost training.

Rate limiting: 6 second delay between requests (~10 req/min)

Usage:
    # Populate all data types for all seasons
    python scripts/populate_historical_data.py

    # Populate specific seasons only
    python scripts/populate_historical_data.py --seasons 2020 2021 2022

    # Populate specific data type only
    python scripts/populate_historical_data.py --type teams

    # Dry run (show what would be fetched)
    python scripts/populate_historical_data.py --dry-run

Data types:
    - teams: Team roster with IDs, coaches, arenas
    - conferences: Conference list
    - ratings: Team ratings (AdjEM, AdjO, AdjD, etc.)
    - four_factors: Dean Oliver Four Factors
    - misc_stats: Miscellaneous statistics
    - height: Height and experience data
    - point_distribution: Point distribution data
"""

import argparse
import logging
import sqlite3
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kenp0m_sp0rts_analyzer.api_client import (  # noqa: E402
    KenPomAPI,
    KenPomAPIError,
)

logger = logging.getLogger(__name__)

# Rate limiting configuration
REQUEST_DELAY = 6.0  # seconds between requests
MAX_RETRIES = 3
RETRY_DELAY = 30  # seconds


class HistoricalDataLoader:
    """Load historical KenPom data into SQLite database."""

    def __init__(self, db_path: str = "data/kenpom.db", dry_run: bool = False):
        """Initialize loader.

        Args:
            db_path: Path to SQLite database.
            dry_run: If True, don't actually write to database.
        """
        self.db_path = Path(db_path)
        self.dry_run = dry_run
        self.api = KenPomAPI()
        self.request_count = 0
        self.start_time = datetime.now()
        # Cache for team name -> team_id mapping per season
        self._team_cache: dict[int, dict[str, int]] = {}

    def _rate_limit(self) -> None:
        """Apply rate limiting delay."""
        self.request_count += 1
        if self.request_count > 1:
            logger.debug(f"Rate limiting: waiting {REQUEST_DELAY}s...")
            time.sleep(REQUEST_DELAY)

    def _retry_request(self, func, *args, **kwargs) -> Any:
        """Execute request with retry logic."""
        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                return func(*args, **kwargs)
            except KenPomAPIError as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/"
                        f"{MAX_RETRIES}): {e}"
                    )
                    logger.info(f"Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_team_id(self, season: int, team_name: str) -> int | None:
        """Get team_id for a team name in a season.

        First checks cache, then fetches from API if needed.

        Args:
            season: Season year.
            team_name: Team name to look up.

        Returns:
            Team ID or None if not found.
        """
        # Check cache first
        if season in self._team_cache:
            return self._team_cache[season].get(team_name)

        # Fetch teams for this season and cache
        if self.dry_run:
            return None

        try:
            response = self._retry_request(self.api.get_teams, year=season)
            teams = list(response.data)
            self._team_cache[season] = {
                t.get("TeamName"): t.get("TeamID") for t in teams
            }
            return self._team_cache[season].get(team_name)
        except KenPomAPIError:
            return None

    def populate_teams(self, seasons: list[int]) -> int:
        """Populate teams table for specified seasons.

        Args:
            seasons: List of season years to populate.

        Returns:
            Number of teams populated.
        """
        total = 0
        for season in seasons:
            logger.info(f"Fetching teams for {season} season...")

            if self.dry_run:
                logger.info(f"  [DRY RUN] Would fetch teams for {season}")
                continue

            try:
                response = self._retry_request(self.api.get_teams, year=season)
                teams = list(response.data)
                logger.info(f"  Fetched {len(teams)} teams")

                # Save to database
                conn = self._get_connection()
                try:
                    for team in teams:
                        conn.execute(
                            """
                            INSERT INTO teams (
                                team_id, team_name, conference, coach, arena,
                                season, arena_city, arena_state
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(team_id) DO UPDATE SET
                                team_name = excluded.team_name,
                                conference = excluded.conference,
                                coach = excluded.coach,
                                arena = excluded.arena,
                                season = excluded.season,
                                arena_city = excluded.arena_city,
                                arena_state = excluded.arena_state,
                                updated_at = CURRENT_TIMESTAMP
                            """,
                            (
                                team.get("TeamID"),
                                team.get("TeamName"),
                                team.get("ConfShort"),
                                team.get("Coach"),
                                team.get("Arena"),
                                season,
                                team.get("ArenaCity"),
                                team.get("ArenaState"),
                            ),
                        )
                    conn.commit()
                    total += len(teams)
                    logger.info(f"  Saved {len(teams)} teams for {season}")
                finally:
                    conn.close()

            except KenPomAPIError as e:
                logger.error(f"  Failed to fetch teams for {season}: {e}")

        return total

    def populate_conferences(self, seasons: list[int]) -> int:
        """Populate conferences table for specified seasons.

        Args:
            seasons: List of season years to populate.

        Returns:
            Number of conferences populated.
        """
        total = 0
        for season in seasons:
            logger.info(f"Fetching conferences for {season} season...")

            if self.dry_run:
                logger.info(
                    f"  [DRY RUN] Would fetch conferences for {season}"
                )
                continue

            try:
                response = self._retry_request(
                    self.api.get_conferences, year=season
                )
                conferences = list(response.data)
                logger.info(f"  Fetched {len(conferences)} conferences")

                # Save to database
                conn = self._get_connection()
                try:
                    for conf in conferences:
                        conn.execute(
                            """
                            INSERT INTO conferences (
                                season, conf_id, conf_short, conf_long
                            )
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(season, conf_id) DO UPDATE SET
                                conf_short = excluded.conf_short,
                                conf_long = excluded.conf_long
                            """,
                            (
                                season,
                                conf.get("ConfID"),
                                conf.get("ConfShort"),
                                conf.get("ConfLong"),
                            ),
                        )
                    conn.commit()
                    total += len(conferences)
                    logger.info(
                        f"  Saved {len(conferences)} conferences for {season}"
                    )
                finally:
                    conn.close()

            except KenPomAPIError as e:
                logger.error(
                    f"  Failed to fetch conferences for {season}: {e}"
                )

        return total

    def populate_ratings(self, seasons: list[int]) -> int:
        """Populate ratings_snapshots table for specified seasons.

        Note: This fetches current/end-of-season ratings. For daily snapshots,
        use the archive endpoint or batch_scheduler.

        Args:
            seasons: List of season years to populate.

        Returns:
            Number of ratings populated.
        """
        total = 0
        today = date.today()

        for season in seasons:
            logger.info(f"Fetching ratings for {season} season...")

            if self.dry_run:
                logger.info(f"  [DRY RUN] Would fetch ratings for {season}")
                continue

            try:
                response = self._retry_request(
                    self.api.get_ratings, year=season
                )
                ratings = list(response.data)
                logger.info(f"  Fetched {len(ratings)} team ratings")

                # Determine snapshot date
                # For past seasons, use end of season (April 8)
                # For current season, use today
                if season < today.year or (
                    season == today.year and today.month > 4
                ):
                    snapshot_date = date(season, 4, 8)  # End of season
                else:
                    snapshot_date = today

                # Save to database
                conn = self._get_connection()
                try:
                    for rating in ratings:
                        team_id = rating.get("TeamID")
                        team_name = rating.get("TeamName")
                        # Look up team_id if not in response
                        if team_id is None and team_name:
                            team_id = self._get_team_id(season, team_name)
                        if team_id is None:
                            continue

                        conn.execute(
                            """
                            INSERT INTO ratings_snapshots (
                                snapshot_date, season, team_id,
                                team_name, conference,
                                adj_em, adj_oe, adj_de, adj_tempo,
                                rank_adj_em, rank_adj_oe,
                                rank_adj_de, rank_tempo,
                                sos, ncsos, luck, pythag
                            )
                            VALUES (
                                ?, ?, ?, ?, ?, ?, ?, ?,
                                ?, ?, ?, ?, ?, ?, ?, ?, ?
                            )
                            ON CONFLICT(snapshot_date, team_id) DO UPDATE SET
                                team_name = excluded.team_name,
                                conference = excluded.conference,
                                adj_em = excluded.adj_em,
                                adj_oe = excluded.adj_oe,
                                adj_de = excluded.adj_de,
                                adj_tempo = excluded.adj_tempo,
                                rank_adj_em = excluded.rank_adj_em,
                                rank_adj_oe = excluded.rank_adj_oe,
                                rank_adj_de = excluded.rank_adj_de,
                                rank_tempo = excluded.rank_tempo,
                                sos = excluded.sos,
                                ncsos = excluded.ncsos,
                                luck = excluded.luck,
                                pythag = excluded.pythag
                            """,
                            (
                                snapshot_date.isoformat(),
                                season,
                                team_id,
                                rating.get("TeamName"),
                                rating.get("ConfShort"),
                                rating.get("AdjEM"),
                                rating.get("AdjOE"),
                                rating.get("AdjDE"),
                                rating.get("AdjTempo"),
                                rating.get("RankAdjEM"),
                                rating.get("RankAdjOE"),
                                rating.get("RankAdjDE"),
                                rating.get("RankAdjTempo"),
                                rating.get("SOS"),
                                rating.get("NCSOS"),
                                rating.get("Luck"),
                                rating.get("Pythag"),
                            ),
                        )
                    conn.commit()
                    total += len(ratings)
                    logger.info(
                        f"  Saved {len(ratings)} ratings for {season} "
                        f"(snapshot: {snapshot_date})"
                    )
                finally:
                    conn.close()

            except KenPomAPIError as e:
                logger.error(f"  Failed to fetch ratings for {season}: {e}")

        return total

    def populate_four_factors(self, seasons: list[int]) -> int:
        """Populate four_factors table for specified seasons.

        Args:
            seasons: List of season years to populate.

        Returns:
            Number of four_factors records populated.
        """
        total = 0
        today = date.today()

        for season in seasons:
            logger.info(f"Fetching four factors for {season} season...")

            if self.dry_run:
                logger.info(
                    f"  [DRY RUN] Would fetch four factors for {season}"
                )
                continue

            try:
                response = self._retry_request(
                    self.api.get_four_factors, year=season
                )
                factors = list(response.data)
                logger.info(f"  Fetched {len(factors)} team records")

                # Determine snapshot date
                if season < today.year or (
                    season == today.year and today.month > 4
                ):
                    snapshot_date = date(season, 4, 8)
                else:
                    snapshot_date = today

                # Save to database
                conn = self._get_connection()
                try:
                    for ff in factors:
                        team_id = ff.get("TeamID")
                        team_name = ff.get("TeamName")
                        if team_id is None and team_name:
                            team_id = self._get_team_id(season, team_name)
                        if team_id is None:
                            continue

                        conn.execute(
                            """
                            INSERT INTO four_factors (
                                snapshot_date, team_id,
                                efg_pct_off, to_pct_off,
                                or_pct_off, ft_rate_off,
                                efg_pct_def, to_pct_def,
                                or_pct_def, ft_rate_def,
                                rank_efg_off, rank_efg_def,
                                rank_to_off, rank_to_def,
                                rank_or_off, rank_or_def,
                                rank_ft_rate_off, rank_ft_rate_def
                            )
                            VALUES (
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                ?, ?, ?, ?, ?, ?, ?, ?
                            )
                            ON CONFLICT(snapshot_date, team_id) DO UPDATE SET
                                efg_pct_off = excluded.efg_pct_off,
                                to_pct_off = excluded.to_pct_off,
                                or_pct_off = excluded.or_pct_off,
                                ft_rate_off = excluded.ft_rate_off,
                                efg_pct_def = excluded.efg_pct_def,
                                to_pct_def = excluded.to_pct_def,
                                or_pct_def = excluded.or_pct_def,
                                ft_rate_def = excluded.ft_rate_def,
                                rank_efg_off = excluded.rank_efg_off,
                                rank_efg_def = excluded.rank_efg_def,
                                rank_to_off = excluded.rank_to_off,
                                rank_to_def = excluded.rank_to_def,
                                rank_or_off = excluded.rank_or_off,
                                rank_or_def = excluded.rank_or_def,
                                rank_ft_rate_off = excluded.rank_ft_rate_off,
                                rank_ft_rate_def = excluded.rank_ft_rate_def
                            """,
                            (
                                snapshot_date.isoformat(),
                                team_id,
                                ff.get("eFG_Pct"),
                                ff.get("TO_Pct"),
                                ff.get("OR_Pct"),
                                ff.get("FT_Rate"),
                                ff.get("DeFG_Pct"),
                                ff.get("DTO_Pct"),
                                ff.get("DOR_Pct"),
                                ff.get("DFT_Rate"),
                                ff.get("RankeFG_Pct"),
                                ff.get("RankDeFG_Pct"),
                                ff.get("RankTO_Pct"),
                                ff.get("RankDTO_Pct"),
                                ff.get("RankOR_Pct"),
                                ff.get("RankDOR_Pct"),
                                ff.get("RankFT_Rate"),
                                ff.get("RankDFT_Rate"),
                            ),
                        )
                    conn.commit()
                    total += len(factors)
                    logger.info(
                        f"  Saved {len(factors)} four factors for {season} "
                        f"(snapshot: {snapshot_date})"
                    )
                finally:
                    conn.close()

            except KenPomAPIError as e:
                logger.error(
                    f"  Failed to fetch four factors for {season}: {e}"
                )

        return total

    def populate_misc_stats(self, seasons: list[int]) -> int:
        """Populate misc_stats table for specified seasons.

        Args:
            seasons: List of season years to populate.

        Returns:
            Number of misc_stats records populated.
        """
        total = 0
        today = date.today()

        for season in seasons:
            logger.info(f"Fetching misc stats for {season} season...")

            if self.dry_run:
                logger.info(f"  [DRY RUN] Would fetch misc stats for {season}")
                continue

            try:
                response = self._retry_request(
                    self.api.get_misc_stats, year=season
                )
                stats = list(response.data)
                logger.info(f"  Fetched {len(stats)} team records")

                # Determine snapshot date
                if season < today.year or (
                    season == today.year and today.month > 4
                ):
                    snapshot_date = date(season, 4, 8)
                else:
                    snapshot_date = today

                # Save to database
                conn = self._get_connection()
                try:
                    for s in stats:
                        team_id = s.get("TeamID")
                        team_name = s.get("TeamName")
                        if team_id is None and team_name:
                            team_id = self._get_team_id(season, team_name)
                        if team_id is None:
                            continue

                        conn.execute(
                            """
                            INSERT INTO misc_stats (
                                snapshot_date, team_id,
                                fg3_pct_off, fg2_pct_off, ft_pct_off,
                                fg3_pct_def, fg2_pct_def, ft_pct_def,
                                assist_rate, assist_rate_def,
                                steal_rate, steal_rate_def,
                                block_pct_off, block_pct_def
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(snapshot_date, team_id) DO UPDATE SET
                                fg3_pct_off = excluded.fg3_pct_off,
                                fg2_pct_off = excluded.fg2_pct_off,
                                ft_pct_off = excluded.ft_pct_off,
                                fg3_pct_def = excluded.fg3_pct_def,
                                fg2_pct_def = excluded.fg2_pct_def,
                                ft_pct_def = excluded.ft_pct_def,
                                assist_rate = excluded.assist_rate,
                                assist_rate_def = excluded.assist_rate_def,
                                steal_rate = excluded.steal_rate,
                                steal_rate_def = excluded.steal_rate_def,
                                block_pct_off = excluded.block_pct_off,
                                block_pct_def = excluded.block_pct_def
                            """,
                            (
                                snapshot_date.isoformat(),
                                team_id,
                                s.get("FG3Pct"),
                                s.get("FG2Pct"),
                                s.get("FTPct"),
                                s.get("OppFG3Pct"),
                                s.get("OppFG2Pct"),
                                s.get("OppFTPct"),
                                s.get("ARate"),
                                s.get("OppARate"),
                                s.get("StlRate"),
                                s.get("OppStlRate"),
                                s.get("BlockPct"),
                                s.get("OppBlockPct"),
                            ),
                        )
                    conn.commit()
                    total += len(stats)
                    logger.info(
                        f"  Saved {len(stats)} misc stats for {season} "
                        f"(snapshot: {snapshot_date})"
                    )
                finally:
                    conn.close()

            except KenPomAPIError as e:
                logger.error(f"  Failed to fetch misc stats for {season}: {e}")

        return total

    def populate_height(self, seasons: list[int]) -> int:
        """Populate height_experience table for specified seasons.

        Args:
            seasons: List of season years to populate.

        Returns:
            Number of height records populated.
        """
        total = 0
        today = date.today()

        for season in seasons:
            logger.info(f"Fetching height/experience for {season} season...")

            if self.dry_run:
                logger.info(f"  [DRY RUN] Would fetch height for {season}")
                continue

            try:
                response = self._retry_request(
                    self.api.get_height, year=season
                )
                heights = list(response.data)
                logger.info(f"  Fetched {len(heights)} team records")

                # Determine snapshot date
                if season < today.year or (
                    season == today.year and today.month > 4
                ):
                    snapshot_date = date(season, 4, 8)
                else:
                    snapshot_date = today

                # Save to database
                conn = self._get_connection()
                try:
                    for h in heights:
                        team_id = h.get("TeamID")
                        team_name = h.get("TeamName")
                        if team_id is None and team_name:
                            team_id = self._get_team_id(season, team_name)
                        if team_id is None:
                            continue

                        conn.execute(
                            """
                            INSERT INTO height_experience (
                                snapshot_date, team_id,
                                avg_height, effective_height, experience,
                                bench_minutes, continuity,
                                rank_height, rank_experience, rank_continuity
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(snapshot_date, team_id) DO UPDATE SET
                                avg_height = excluded.avg_height,
                                effective_height = excluded.effective_height,
                                experience = excluded.experience,
                                bench_minutes = excluded.bench_minutes,
                                continuity = excluded.continuity,
                                rank_height = excluded.rank_height,
                                rank_experience = excluded.rank_experience,
                                rank_continuity = excluded.rank_continuity
                            """,
                            (
                                snapshot_date.isoformat(),
                                team_id,
                                h.get("AvgHgt"),
                                h.get("HgtEff"),
                                h.get("Exp"),
                                h.get("Bench"),
                                h.get("Continuity"),
                                h.get("AvgHgtRank"),
                                h.get("ExpRank"),
                                h.get("RankContinuity"),
                            ),
                        )
                    conn.commit()
                    total += len(heights)
                    logger.info(
                        f"  Saved {len(heights)} height records for {season} "
                        f"(snapshot: {snapshot_date})"
                    )
                finally:
                    conn.close()

            except KenPomAPIError as e:
                logger.error(f"  Failed to fetch height for {season}: {e}")

        return total

    def populate_point_distribution(self, seasons: list[int]) -> int:
        """Populate point_distribution table for specified seasons.

        Args:
            seasons: List of season years to populate.

        Returns:
            Number of point_distribution records populated.
        """
        total = 0
        today = date.today()

        for season in seasons:
            logger.info(f"Fetching point distribution for {season} season...")

            if self.dry_run:
                logger.info(f"  [DRY RUN] Would fetch point dist for {season}")
                continue

            try:
                response = self._retry_request(
                    self.api.get_point_distribution, year=season
                )
                points = list(response.data)
                logger.info(f"  Fetched {len(points)} team records")

                # Determine snapshot date
                if season < today.year or (
                    season == today.year and today.month > 4
                ):
                    snapshot_date = date(season, 4, 8)
                else:
                    snapshot_date = today

                # Save to database
                conn = self._get_connection()
                try:
                    for p in points:
                        team_id = p.get("TeamID")
                        team_name = p.get("TeamName")
                        if team_id is None and team_name:
                            team_id = self._get_team_id(season, team_name)
                        if team_id is None:
                            continue

                        conn.execute(
                            """
                            INSERT INTO point_distribution (
                                snapshot_date, team_id,
                                ft_pct, two_pct, three_pct,
                                ft_pct_def, two_pct_def, three_pct_def
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(snapshot_date, team_id) DO UPDATE SET
                                ft_pct = excluded.ft_pct,
                                two_pct = excluded.two_pct,
                                three_pct = excluded.three_pct,
                                ft_pct_def = excluded.ft_pct_def,
                                two_pct_def = excluded.two_pct_def,
                                three_pct_def = excluded.three_pct_def
                            """,
                            (
                                snapshot_date.isoformat(),
                                team_id,
                                p.get("OffFt"),
                                p.get("OffFg2"),
                                p.get("OffFg3"),
                                p.get("DefFt"),
                                p.get("DefFg2"),
                                p.get("DefFg3"),
                            ),
                        )
                    conn.commit()
                    total += len(points)
                    logger.info(
                        f"  Saved {len(points)} point dist for {season} "
                        f"(snapshot: {snapshot_date})"
                    )
                finally:
                    conn.close()

            except KenPomAPIError as e:
                logger.error(f"  Failed to fetch point dist for {season}: {e}")

        return total

    def populate_all(self, seasons: list[int]) -> dict[str, int]:
        """Populate all tables for specified seasons.

        Args:
            seasons: List of season years to populate.

        Returns:
            Dictionary with counts per data type.
        """
        results = {}

        logger.info(
            f"Starting historical data population for seasons: {seasons}"
        )
        logger.info(f"Rate limit: {REQUEST_DELAY}s delay between requests")
        logger.info("")

        # Order matters for foreign key relationships
        data_types = [
            ("teams", self.populate_teams),
            ("conferences", self.populate_conferences),
            ("ratings", self.populate_ratings),
            ("four_factors", self.populate_four_factors),
            ("misc_stats", self.populate_misc_stats),
            ("height", self.populate_height),
            ("point_distribution", self.populate_point_distribution),
        ]

        for name, populate_func in data_types:
            logger.info(f"{'=' * 50}")
            logger.info(f"Populating {name}...")
            logger.info(f"{'=' * 50}")

            try:
                count = populate_func(seasons)
                results[name] = count
                logger.info(f"[OK] {name}: {count} records")
            except Exception as e:
                logger.error(f"[ERROR] {name}: {e}")
                results[name] = 0

            logger.info("")

        # Print summary
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info("=" * 50)
        logger.info("SUMMARY")
        logger.info("=" * 50)
        for name, count in results.items():
            logger.info(f"  {name}: {count:,} records")
        logger.info(f"  Total requests: {self.request_count}")
        logger.info(f"  Elapsed time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

        return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Populate KenPom database with historical data"
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[2020, 2021, 2022, 2023, 2024, 2025],
        help="Seasons to populate (default: 2020-2025)",
    )
    parser.add_argument(
        "--type",
        choices=[
            "teams",
            "conferences",
            "ratings",
            "four_factors",
            "misc_stats",
            "height",
            "point_distribution",
            "all",
        ],
        default="all",
        help="Data type to populate (default: all)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/kenpom.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without making changes",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    loader = HistoricalDataLoader(db_path=args.db, dry_run=args.dry_run)

    if args.dry_run:
        logger.info("[DRY RUN MODE] No changes will be made")
        logger.info("")

    if args.type == "all":
        loader.populate_all(args.seasons)
    else:
        # Map type name to method
        methods = {
            "teams": loader.populate_teams,
            "conferences": loader.populate_conferences,
            "ratings": loader.populate_ratings,
            "four_factors": loader.populate_four_factors,
            "misc_stats": loader.populate_misc_stats,
            "height": loader.populate_height,
            "point_distribution": loader.populate_point_distribution,
        }
        count = methods[args.type](args.seasons)
        logger.info(f"[OK] Populated {count} {args.type} records")


if __name__ == "__main__":
    main()
