#!/usr/bin/env python3
"""Populate player statistics database from scraped KenPom data.

This script:
1. Loads scraped player data from CSV
2. Creates/updates database schema
3. Populates raw KenPom player stats table
4. Maps data to player impact tracking tables
5. Calculates tier classifications for impact analysis
"""

import argparse
import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class PlayerStatsPopulator:
    """Populate player statistics database from scraped data."""

    def __init__(self, db_path: str = "data/kenpom.db"):
        """Initialize populator with database path.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None

    def __enter__(self):
        """Context manager entry."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.conn:
            self.conn.close()

    def create_schemas(self):
        """Create database schemas if they don't exist."""
        logger.info("Creating database schemas...")

        # Create raw KenPom player stats schema
        schema_file = Path(__file__).parent / "kenpom_player_stats_schema.sql"
        if schema_file.exists():
            with open(schema_file) as f:
                self.conn.executescript(f.read())
            logger.info("  [OK] Raw KenPom player stats schema created")

        # Create player impact schema
        impact_schema_file = Path(__file__).parent / "player_impact_schema.sql"
        if impact_schema_file.exists():
            with open(impact_schema_file) as f:
                self.conn.executescript(f.read())
            logger.info("  [OK] Player impact schema created")

        self.conn.commit()

    def load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """Load scraped player data from CSV.

        Args:
            csv_path: Path to CSV file

        Returns:
            DataFrame with player statistics
        """
        logger.info(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        logger.info(f"  [OK] Loaded {len(df)} players")
        return df

    def populate_raw_stats(self, df: pd.DataFrame):
        """Populate kenpom_player_stats table with raw data.

        Args:
            df: DataFrame with scraped player statistics
        """
        logger.info("Populating raw KenPom player stats...")

        cursor = self.conn.cursor()
        inserted = 0
        updated = 0

        for _, row in df.iterrows():
            # Convert height to inches
            height_inches = self._height_to_inches(row.get("height", ""))

            # Check if player already exists
            cursor.execute(
                """
                SELECT player_stat_id FROM kenpom_player_stats
                WHERE player_name = ? AND team_name = ? AND season = ?
                """,
                (row["name"], row["team"], row["season"]),
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing record
                cursor.execute(
                    """
                    UPDATE kenpom_player_stats SET
                        class_year = ?,
                        height = ?,
                        height_inches = ?,
                        rank_overall = ?,
                        offensive_rating = ?,
                        pct_possessions = ?,
                        efg_pct = ?,
                        ts_pct = ?,
                        three_pt_rate = ?,
                        ft_rate = ?,
                        ppg = ?,
                        rpg = ?,
                        apg = ?,
                        assist_rate = ?,
                        turnover_rate = ?,
                        offensive_reb_pct = ?,
                        defensive_reb_pct = ?,
                        fouls_committed_per_40 = ?,
                        minutes_pct = ?,
                        scraped_at = CURRENT_TIMESTAMP
                    WHERE player_stat_id = ?
                    """,
                    (
                        row.get("class_year"),
                        row.get("height"),
                        height_inches,
                        row.get("rank"),
                        row.get("ortg"),
                        row.get("pct_poss"),
                        row.get("efg_pct"),
                        row.get("ts_pct"),
                        row.get("three_pt_rate"),
                        row.get("ft_rate"),
                        row.get("ppg"),
                        row.get("rpg"),
                        row.get("apg"),
                        row.get("ast_rate"),
                        row.get("to_rate"),
                        row.get("or_pct"),
                        row.get("dr_pct"),
                        row.get("fc_per_40"),
                        row.get("minutes_pct"),
                        existing[0],
                    ),
                )
                updated += 1
            else:
                # Insert new record
                cursor.execute(
                    """
                    INSERT INTO kenpom_player_stats (
                        player_name, team_name, conference, position,
                        height, height_inches, class_year, season,
                        rank_overall, offensive_rating, pct_possessions,
                        efg_pct, ts_pct, three_pt_rate, ft_rate,
                        ppg, rpg, apg,
                        assist_rate, turnover_rate,
                        offensive_reb_pct, defensive_reb_pct,
                        fouls_committed_per_40, minutes_pct
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["name"],
                        row["team"],
                        row.get("conference", ""),
                        row.get("position", ""),
                        row.get("height"),
                        height_inches,
                        row.get("class_year"),
                        row["season"],
                        row.get("rank"),
                        row.get("ortg"),
                        row.get("pct_poss"),
                        row.get("efg_pct"),
                        row.get("ts_pct"),
                        row.get("three_pt_rate"),
                        row.get("ft_rate"),
                        row.get("ppg"),
                        row.get("rpg"),
                        row.get("apg"),
                        row.get("ast_rate"),
                        row.get("to_rate"),
                        row.get("or_pct"),
                        row.get("dr_pct"),
                        row.get("fc_per_40"),
                        row.get("minutes_pct"),
                    ),
                )
                inserted += 1

        self.conn.commit()
        logger.info(f"  [OK] Inserted {inserted} players, updated {updated} players")

    def populate_player_impact(self, df: pd.DataFrame):
        """Populate player impact tables from raw data.

        Args:
            df: DataFrame with scraped player statistics
        """
        logger.info("Populating player impact tables...")

        cursor = self.conn.cursor()

        # First, ensure teams exist in the teams table
        self._ensure_teams_exist(df)

        inserted = 0
        updated = 0

        for _, row in df.iterrows():
            # Get team_id (teams table is season-agnostic)
            cursor.execute(
                "SELECT team_id FROM teams WHERE team_name = ?",
                (row["team"],),
            )
            team_row = cursor.fetchone()

            if not team_row:
                logger.warning(
                    f"Team not found: {row['team']}, skipping player {row['name']}"
                )
                continue

            team_id = team_row[0]

            # Convert height to inches
            height_inches = self._height_to_inches(row.get("height", ""))

            # Convert class year to uppercase
            class_year = str(row.get("class_year", "")).upper()

            # Convert percentages from 0-100 to 0-1 scale for database constraints
            ts_pct = row.get("ts_pct", 0.0) / 100.0 if row.get("ts_pct", 0.0) > 1 else row.get("ts_pct", 0.0)
            efg_pct = row.get("efg_pct", 0.0) / 100.0 if row.get("efg_pct", 0.0) > 1 else row.get("efg_pct", 0.0)

            # Calculate tier classification
            tier = self._calculate_tier(row)
            tier_justification = self._get_tier_justification(row, tier)

            # Estimate spread impact (simplified model)
            spread_impact = self._estimate_spread_impact(row, tier)

            # Check if player already exists
            cursor.execute(
                """
                SELECT player_id FROM players
                WHERE name = ? AND team_id = ? AND season = ?
                """,
                (row["name"], team_id, row["season"]),
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing player
                cursor.execute(
                    """
                    UPDATE players SET
                        position = ?,
                        class_year = ?,
                        height_inches = ?,
                        season = ?,
                        games_played = ?,
                        ppg = ?,
                        rpg = ?,
                        apg = ?,
                        ts_pct = ?,
                        efg_pct = ?,
                        usage_rate = ?,
                        offensive_rating = ?,
                        tier = ?,
                        tier_justification = ?,
                        spread_impact = ?,
                        assist_rate = ?,
                        turnover_rate = ?,
                        rebound_rate = ?,
                        is_active = 1,
                        injury_status = 'Healthy',
                        last_updated = CURRENT_TIMESTAMP
                    WHERE player_id = ?
                    """,
                    (
                        row.get("position", ""),
                        class_year,
                        height_inches,
                        row["season"],
                        row.get("games_played", 0),
                        row.get("ppg", 0.0),
                        row.get("rpg", 0.0),
                        row.get("apg", 0.0),
                        ts_pct,
                        efg_pct,
                        row.get("pct_poss", 0.0),  # Usage rate
                        row.get("ortg", 0.0),
                        tier,
                        tier_justification,
                        spread_impact,
                        row.get("ast_rate", 0.0),
                        row.get("to_rate", 0.0),
                        (
                            row.get("or_pct", 0.0) + row.get("dr_pct", 0.0)
                        ),  # Total rebound rate
                        existing[0],
                    ),
                )
                updated += 1
            else:
                # Insert new player
                cursor.execute(
                    """
                    INSERT INTO players (
                        name, team_id, position, class_year, height_inches,
                        season, games_played,
                        ppg, rpg, apg,
                        ts_pct, efg_pct,
                        usage_rate, offensive_rating,
                        tier, tier_justification, spread_impact,
                        assist_rate, turnover_rate, rebound_rate,
                        is_active, injury_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'Healthy')
                    """,
                    (
                        row["name"],
                        team_id,
                        row.get("position", ""),
                        class_year,
                        height_inches,
                        row["season"],
                        row.get("games_played", 0),
                        row.get("ppg", 0.0),
                        row.get("rpg", 0.0),
                        row.get("apg", 0.0),
                        ts_pct,
                        efg_pct,
                        row.get("pct_poss", 0.0),
                        row.get("ortg", 0.0),
                        tier,
                        tier_justification,
                        spread_impact,
                        row.get("ast_rate", 0.0),
                        row.get("to_rate", 0.0),
                        (row.get("or_pct", 0.0) + row.get("dr_pct", 0.0)),
                    ),
                )
                inserted += 1

        self.conn.commit()
        logger.info(
            f"  [OK] Inserted {inserted} players, updated {updated} players in impact tables"
        )

    def _ensure_teams_exist(self, df: pd.DataFrame):
        """Ensure all teams exist in teams table.

        Args:
            df: DataFrame with player data
        """
        cursor = self.conn.cursor()
        teams = df["team"].unique()

        for team_name in teams:
            cursor.execute(
                "SELECT team_id FROM teams WHERE team_name = ?",
                (team_name,),
            )

            if not cursor.fetchone():
                # Insert placeholder team (teams table is season-agnostic)
                cursor.execute(
                    """
                    INSERT INTO teams (team_name, conference)
                    VALUES (?, '')
                    """,
                    (team_name,),
                )

        self.conn.commit()

    def _height_to_inches(self, height_str: str) -> int:
        """Convert height string to inches.

        Args:
            height_str: Height in format "6-11"

        Returns:
            Total height in inches
        """
        if not height_str or "-" not in height_str:
            return 0

        try:
            feet, inches = height_str.split("-")
            return int(feet) * 12 + int(inches)
        except (ValueError, AttributeError):
            return 0

    def _calculate_tier(self, row: pd.Series) -> int:
        """Calculate player tier classification (1-5).

        Tier 1: Elite Game-Changer (7-12 pt impact)
        Tier 2: All-Conference Star (4-7 pt impact)
        Tier 3: Key Rotation Player (2-4 pt impact)
        Tier 4: Role Player/Specialist (1-2 pt impact)
        Tier 5: Bench Depth (<1 pt impact)

        Args:
            row: Player data row

        Returns:
            Tier classification (1-5)
        """
        ortg = row.get("ortg", 0.0)
        usage = row.get("pct_poss", 0.0)
        minutes = row.get("minutes_pct", 0.0)

        # Elite tier: High ORtg + High Usage + High Minutes
        if ortg >= 120.0 and usage >= 25.0 and minutes >= 70.0:
            return 1

        # All-Conference: Very good ORtg + Good usage + Significant minutes
        if ortg >= 115.0 and usage >= 22.0 and minutes >= 60.0:
            return 2

        # Key rotation: Solid ORtg + Moderate usage + Regular minutes
        if ortg >= 108.0 and usage >= 18.0 and minutes >= 50.0:
            return 3

        # Role player: Contributing in some capacity
        if ortg >= 100.0 and minutes >= 30.0:
            return 4

        # Bench depth
        return 5

    def _get_tier_justification(self, row: pd.Series, tier: int) -> str:
        """Get tier classification justification.

        Args:
            row: Player data row
            tier: Calculated tier

        Returns:
            Justification string
        """
        ortg = row.get("ortg", 0.0)
        usage = row.get("pct_poss", 0.0)
        minutes = row.get("minutes_pct", 0.0)

        descriptions = {
            1: "Elite Game-Changer",
            2: "All-Conference Star",
            3: "Key Rotation Player",
            4: "Role Player/Specialist",
            5: "Bench Depth",
        }

        return (
            f"{descriptions.get(tier, 'Unknown')}: "
            f"ORtg={ortg:.1f}, Usage={usage:.1f}%, Minutes={minutes:.1f}%"
        )

    def _estimate_spread_impact(self, row: pd.Series, tier: int) -> float:
        """Estimate player's spread impact in points.

        Uses a simplified model based on tier and offensive rating.

        Args:
            row: Player data row
            tier: Player tier

        Returns:
            Estimated spread impact in points
        """
        # Base impact by tier
        tier_impact = {
            1: 9.0,  # 7-12 pt range, use midpoint
            2: 5.5,  # 4-7 pt range
            3: 3.0,  # 2-4 pt range
            4: 1.5,  # 1-2 pt range
            5: 0.5,  # <1 pt
        }

        base = tier_impact.get(tier, 0.0)

        # Adjust based on offensive rating deviation from 110 (average)
        ortg = row.get("ortg", 110.0)
        ortg_adjustment = (ortg - 110.0) * 0.05  # 0.05 pts per ORtg point

        # Adjust based on usage
        usage = row.get("pct_poss", 20.0)
        usage_adjustment = (usage - 20.0) * 0.05  # 0.05 pts per usage%

        total_impact = base + ortg_adjustment + usage_adjustment

        # Clamp to reasonable range
        return max(0.5, min(12.0, total_impact))

    def generate_summary(self):
        """Generate summary statistics of populated data."""
        logger.info("\n" + "=" * 80)
        logger.info("DATABASE POPULATION SUMMARY")
        logger.info("=" * 80)

        cursor = self.conn.cursor()

        # Raw stats summary
        cursor.execute(
            "SELECT COUNT(*), season FROM kenpom_player_stats GROUP BY season"
        )
        for row in cursor.fetchall():
            logger.info(f"Raw KenPom Stats - Season {row[1]}: {row[0]} players")

        # Player impact summary
        cursor.execute("SELECT COUNT(*), season FROM players GROUP BY season")
        for row in cursor.fetchall():
            logger.info(f"Player Impact Tables - Season {row[1]}: {row[0]} players")

        # Tier distribution
        cursor.execute(
            """
            SELECT tier, COUNT(*) as count
            FROM players
            WHERE season = (SELECT MAX(season) FROM players)
            GROUP BY tier
            ORDER BY tier
            """
        )
        logger.info("\nTier Distribution (Current Season):")
        tier_names = {
            1: "Elite Game-Changer",
            2: "All-Conference Star",
            3: "Key Rotation Player",
            4: "Role Player",
            5: "Bench Depth",
        }
        for row in cursor.fetchall():
            tier, count = row
            logger.info(f"  Tier {tier} ({tier_names.get(tier, 'Unknown')}): {count}")

        # Top 10 players by spread impact
        cursor.execute(
            """
            SELECT name, team_id, tier, spread_impact, offensive_rating
            FROM players
            WHERE season = (SELECT MAX(season) FROM players)
            ORDER BY spread_impact DESC
            LIMIT 10
            """
        )
        logger.info("\nTop 10 Players by Spread Impact:")
        for row in cursor.fetchall():
            # Get team name
            cursor.execute("SELECT team_name FROM teams WHERE team_id = ?", (row[1],))
            team_row = cursor.fetchone()
            team_name = team_row[0] if team_row else "Unknown"

            logger.info(
                f"  {row[0]} ({team_name}): Tier {row[2]}, Impact={row[3]:.1f} pts, ORtg={row[4]:.1f}"
            )

        logger.info("=" * 80 + "\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Populate player statistics database from scraped data"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/kenpom_players.csv",
        help="Path to scraped player CSV",
    )
    parser.add_argument(
        "--db", type=str, default="data/kenpom.db", help="Path to SQLite database"
    )
    parser.add_argument(
        "--skip-impact",
        action="store_true",
        help="Skip populating player impact tables",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger.info("Starting player statistics database population...")
    logger.info(f"CSV: {args.csv}")
    logger.info(f"Database: {args.db}")

    try:
        with PlayerStatsPopulator(args.db) as populator:
            # Create schemas
            populator.create_schemas()

            # Load CSV data
            df = populator.load_csv_data(args.csv)

            # Populate raw stats
            populator.populate_raw_stats(df)

            # Populate player impact tables
            if not args.skip_impact:
                populator.populate_player_impact(df)

            # Generate summary
            populator.generate_summary()

        logger.info("Population complete!")

    except Exception as e:
        logger.error(f"Error during population: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
