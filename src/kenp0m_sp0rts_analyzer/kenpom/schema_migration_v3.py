"""Database schema migration v3 - Align with KenPom API documentation.

This migration adds missing columns and renames fields to match the official
KenPom API field names as documented in kenpom_api_docs_20251216.json.

Key changes:
1. ratings: Add raw efficiency/tempo, rename rank_tempo to rank_adj_tempo
2. point_distribution: Add defensive rank columns
3. height: Add position heights and missing ranks
4. misc_stats: Add missing rate and rank columns
5. fanmatch_predictions: Add season and rank columns

Run once to upgrade existing database.
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


def add_column_if_missing(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    col_type: str,
    default: str | None = None,
) -> bool:
    """Add a column to a table if it doesn't exist.

    Args:
        conn: Database connection.
        table: Table name.
        column: Column name.
        col_type: Column type (e.g., 'REAL', 'INTEGER', 'TEXT').
        default: Optional default value.

    Returns:
        True if column was added, False if it already existed.
    """
    if column_exists(conn, table, column):
        return False

    default_clause = f" DEFAULT {default}" if default else ""
    sql = f"ALTER TABLE {table} ADD COLUMN {column} {col_type}{default_clause}"
    conn.execute(sql)
    logger.info(f"  Added column {table}.{column}")
    return True


def apply_migration(db_path: str = "data/kenpom.db") -> None:
    """Apply schema migration v3 to align with KenPom API.

    Args:
        db_path: Path to SQLite database file.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    logger.info(f"Applying schema migration v3 to {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("BEGIN")

        # ============================================================
        # 1. ratings - Add missing columns from ratings API
        # ============================================================
        logger.info("Updating ratings table...")

        # Raw (non-adjusted) efficiency metrics
        add_column_if_missing(conn, "ratings", "oe", "REAL")
        add_column_if_missing(conn, "ratings", "de", "REAL")
        add_column_if_missing(conn, "ratings", "tempo", "REAL")

        # Raw efficiency ranks
        add_column_if_missing(conn, "ratings", "rank_oe", "INTEGER")
        add_column_if_missing(conn, "ratings", "rank_de", "INTEGER")
        add_column_if_missing(conn, "ratings", "rank_tempo_raw", "INTEGER")

        # Additional ranks
        add_column_if_missing(conn, "ratings", "rank_pythag", "INTEGER")
        add_column_if_missing(conn, "ratings", "rank_ncsos", "INTEGER")
        add_column_if_missing(conn, "ratings", "rank_soso", "INTEGER")
        add_column_if_missing(conn, "ratings", "rank_sosd", "INTEGER")

        # APL ranks (APL columns already exist)
        add_column_if_missing(conn, "ratings", "rank_apl_off", "INTEGER")
        add_column_if_missing(conn, "ratings", "rank_apl_def", "INTEGER")

        # Conference APL
        add_column_if_missing(conn, "ratings", "conf_apl_off", "REAL")
        add_column_if_missing(conn, "ratings", "conf_apl_def", "REAL")
        add_column_if_missing(conn, "ratings", "rank_conf_apl_off", "INTEGER")
        add_column_if_missing(conn, "ratings", "rank_conf_apl_def", "INTEGER")

        # Rename rank_tempo to rank_adj_tempo (if not already renamed)
        # SQLite doesn't support RENAME COLUMN in older versions, so we need
        # to check if the new column exists and migrate data if needed
        if column_exists(conn, "ratings", "rank_tempo"):
            if not column_exists(conn, "ratings", "rank_adj_tempo"):
                add_column_if_missing(
                    conn, "ratings", "rank_adj_tempo", "INTEGER"
                )
                conn.execute(
                    """
                    UPDATE ratings
                    SET rank_adj_tempo = rank_tempo
                    WHERE rank_adj_tempo IS NULL
                    """
                )
                logger.info("  Migrated rank_tempo -> rank_adj_tempo")

        # ============================================================
        # 2. point_distribution - Add defensive rank columns
        # ============================================================
        logger.info("Updating point_distribution table...")

        # Defensive ranks (API: RankDefFt, RankDefFg2, RankDefFg3)
        add_column_if_missing(
            conn, "point_distribution", "rank_ft_pct_def", "INTEGER"
        )
        add_column_if_missing(
            conn, "point_distribution", "rank_two_pct_def", "INTEGER"
        )
        add_column_if_missing(
            conn, "point_distribution", "rank_three_pct_def", "INTEGER"
        )

        # ============================================================
        # 3. height - Add position heights and ranks
        # ============================================================
        logger.info("Updating height table...")

        # Position-specific heights (API: Hgt5, Hgt4, Hgt3, Hgt2, Hgt1)
        add_column_if_missing(conn, "height", "hgt_c", "REAL")  # Center
        add_column_if_missing(conn, "height", "hgt_pf", "REAL")  # PF
        add_column_if_missing(conn, "height", "hgt_sf", "REAL")  # SF
        add_column_if_missing(conn, "height", "hgt_sg", "REAL")  # SG
        add_column_if_missing(conn, "height", "hgt_pg", "REAL")  # PG

        # Position height ranks
        add_column_if_missing(conn, "height", "rank_hgt_c", "INTEGER")
        add_column_if_missing(conn, "height", "rank_hgt_pf", "INTEGER")
        add_column_if_missing(conn, "height", "rank_hgt_sf", "INTEGER")
        add_column_if_missing(conn, "height", "rank_hgt_sg", "INTEGER")
        add_column_if_missing(conn, "height", "rank_hgt_pg", "INTEGER")

        # Additional ranks
        add_column_if_missing(
            conn, "height", "rank_effective_height", "INTEGER"
        )
        add_column_if_missing(conn, "height", "rank_bench", "INTEGER")

        # ============================================================
        # 4. misc_stats - Add missing rate and rank columns
        # ============================================================
        logger.info("Updating misc_stats table...")

        # Non-steal turnover rate (API: NSTRate, OppNSTRate)
        add_column_if_missing(conn, "misc_stats", "nst_rate_off", "REAL")
        add_column_if_missing(conn, "misc_stats", "nst_rate_def", "REAL")

        # 3-point attempt rate (API: F3GRate, OppF3GRate)
        add_column_if_missing(conn, "misc_stats", "fg3_rate_off", "REAL")
        add_column_if_missing(conn, "misc_stats", "fg3_rate_def", "REAL")

        # Adjusted efficiencies (already in ratings, but also in misc-stats)
        add_column_if_missing(conn, "misc_stats", "adj_oe", "REAL")
        add_column_if_missing(conn, "misc_stats", "adj_de", "REAL")

        # All rank columns for misc stats
        add_column_if_missing(
            conn, "misc_stats", "rank_fg3_pct_def", "INTEGER"
        )
        add_column_if_missing(
            conn, "misc_stats", "rank_fg2_pct_def", "INTEGER"
        )
        add_column_if_missing(conn, "misc_stats", "rank_ft_pct_def", "INTEGER")
        add_column_if_missing(
            conn, "misc_stats", "rank_block_pct_off", "INTEGER"
        )
        add_column_if_missing(
            conn, "misc_stats", "rank_block_pct_def", "INTEGER"
        )
        add_column_if_missing(conn, "misc_stats", "rank_steal_rate", "INTEGER")
        add_column_if_missing(
            conn, "misc_stats", "rank_steal_rate_def", "INTEGER"
        )
        add_column_if_missing(
            conn, "misc_stats", "rank_nst_rate_off", "INTEGER"
        )
        add_column_if_missing(
            conn, "misc_stats", "rank_nst_rate_def", "INTEGER"
        )
        add_column_if_missing(
            conn, "misc_stats", "rank_assist_rate_def", "INTEGER"
        )
        add_column_if_missing(
            conn, "misc_stats", "rank_fg3_rate_off", "INTEGER"
        )
        add_column_if_missing(
            conn, "misc_stats", "rank_fg3_rate_def", "INTEGER"
        )
        add_column_if_missing(conn, "misc_stats", "rank_adj_oe", "INTEGER")
        add_column_if_missing(conn, "misc_stats", "rank_adj_de", "INTEGER")

        # ============================================================
        # 5. fanmatch_predictions - Add season and rank columns
        # ============================================================
        logger.info("Updating fanmatch_predictions table...")

        add_column_if_missing(
            conn, "fanmatch_predictions", "season", "INTEGER"
        )
        add_column_if_missing(
            conn, "fanmatch_predictions", "home_rank", "INTEGER"
        )
        add_column_if_missing(
            conn, "fanmatch_predictions", "visitor_rank", "INTEGER"
        )

        # ============================================================
        # 6. Update schema version
        # ============================================================
        conn.execute(
            "INSERT OR REPLACE INTO schema_version (version) VALUES (3)"
        )

        conn.commit()
        logger.info("Schema migration v3 completed successfully!")

        # Print column count summary
        tables = [
            "ratings",
            "four_factors",
            "point_distribution",
            "height",
            "misc_stats",
            "fanmatch_predictions",
        ]
        logger.info("\nColumn counts after migration:")
        for table in tables:
            cursor = conn.execute(f"PRAGMA table_info({table})")
            col_count = len(cursor.fetchall())
            logger.info(f"  {table}: {col_count} columns")

    except Exception as e:
        conn.rollback()
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    apply_migration()
