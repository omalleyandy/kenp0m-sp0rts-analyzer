"""Database schema migration - Add missing KenPom API tables.

This migration adds:
1. conferences table (from conferences endpoint)
2. archive table (from archive endpoint)
3. Updates to teams table (arena fields)
4. Removes incomplete efficiency_tempo table

Run once to upgrade existing database to match API documentation.
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# SQL for new tables and indexes
CONFERENCES_TABLE = """
CREATE TABLE IF NOT EXISTS conferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    conf_id INTEGER NOT NULL,
    conf_short TEXT NOT NULL,
    conf_long TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(season, conf_id)
);

CREATE INDEX IF NOT EXISTS idx_conferences_season ON conferences(season);
CREATE INDEX IF NOT EXISTS idx_conferences_short ON conferences(conf_short);
"""

ARCHIVE_TABLE = """
CREATE TABLE IF NOT EXISTS archive (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    archive_date DATE NOT NULL,
    season INTEGER NOT NULL,
    is_preseason BOOLEAN DEFAULT 0,
    team_id INTEGER NOT NULL,
    team_name TEXT NOT NULL,
    conference TEXT,
    seed INTEGER,
    event TEXT,

    -- Archive date metrics
    adj_em REAL NOT NULL,
    adj_oe REAL NOT NULL,
    adj_de REAL NOT NULL,
    adj_tempo REAL NOT NULL,
    rank_adj_em INTEGER,
    rank_adj_oe INTEGER,
    rank_adj_de INTEGER,
    rank_adj_tempo INTEGER,

    -- Final season metrics (for comparison)
    adj_em_final REAL,
    adj_oe_final REAL,
    adj_de_final REAL,
    adj_tempo_final REAL,
    rank_adj_em_final INTEGER,
    rank_adj_oe_final INTEGER,
    rank_adj_de_final INTEGER,
    rank_adj_tempo_final INTEGER,

    -- Changes (archive to final)
    rank_change INTEGER,
    adj_em_change REAL,
    adj_tempo_change REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    UNIQUE(archive_date, team_id, is_preseason)
);

CREATE INDEX IF NOT EXISTS idx_archive_date ON archive(archive_date);
CREATE INDEX IF NOT EXISTS idx_archive_team ON archive(team_id);
CREATE INDEX IF NOT EXISTS idx_archive_season ON archive(season);
CREATE INDEX IF NOT EXISTS idx_archive_preseason ON archive(is_preseason);
"""


# Check if columns exist before adding
def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


def apply_migration(db_path: str = "data/kenpom.db") -> None:
    """Apply schema migration to existing database.

    Args:
        db_path: Path to SQLite database file.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    logger.info(f"Applying schema migration to {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        # Start transaction
        conn.execute("BEGIN")

        # 1. Add conferences table
        logger.info("Creating conferences table...")
        conn.executescript(CONFERENCES_TABLE)

        # 2. Add archive table
        logger.info("Creating archive table...")
        conn.executescript(ARCHIVE_TABLE)

        # 3. Update teams table with new columns
        logger.info("Updating teams table...")

        # Check and add missing columns to teams table
        if not column_exists(conn, "teams", "season"):
            conn.execute("ALTER TABLE teams ADD COLUMN season INTEGER")
            logger.info("  Added 'season' column to teams")

        if not column_exists(conn, "teams", "arena_city"):
            conn.execute("ALTER TABLE teams ADD COLUMN arena_city TEXT")
            logger.info("  Added 'arena_city' column to teams")

        if not column_exists(conn, "teams", "arena_state"):
            conn.execute("ALTER TABLE teams ADD COLUMN arena_state TEXT")
            logger.info("  Added 'arena_state' column to teams")

        # Note: 'arena' column already exists in teams table

        # Add index for season-based queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_teams_season ON teams(season)"
        )

        # 4. Drop incomplete efficiency_tempo table if it exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='efficiency_tempo'"
        )
        if cursor.fetchone():
            logger.info("Dropping incomplete efficiency_tempo table...")
            conn.execute("DROP TABLE efficiency_tempo")

        # 5. Update schema version
        conn.execute(
            "INSERT OR REPLACE INTO schema_version (version) VALUES (2)"
        )

        # Commit transaction
        conn.commit()
        logger.info("Schema migration completed successfully!")

        # Print summary
        cursor = conn.execute("SELECT COUNT(*) FROM conferences")
        conf_count = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM archive")
        archive_count = cursor.fetchone()[0]

        logger.info(f"  Conferences: {conf_count} records")
        logger.info(f"  Archive: {archive_count} records")

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
