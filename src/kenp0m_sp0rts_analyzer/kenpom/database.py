"""SQLite database management for KenPom data.

This module handles database initialization, schema management,
migrations, and backup operations.
"""

import logging
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

from .exceptions import DatabaseError

logger = logging.getLogger(__name__)

# Current schema version for migrations
SCHEMA_VERSION = 1

SCHEMA = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Core team information
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    team_name TEXT NOT NULL UNIQUE,
    conference TEXT,
    coach TEXT,
    arena TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Daily ratings snapshots
CREATE TABLE IF NOT EXISTS ratings_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date DATE NOT NULL,
    season INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    team_name TEXT NOT NULL,
    conference TEXT,
    -- Core metrics
    adj_em REAL NOT NULL,
    adj_oe REAL NOT NULL,
    adj_de REAL NOT NULL,
    adj_tempo REAL NOT NULL,
    -- Luck and strength
    luck REAL DEFAULT 0.0,
    sos REAL DEFAULT 0.0,
    soso REAL DEFAULT 0.0,
    sosd REAL DEFAULT 0.0,
    ncsos REAL DEFAULT 0.0,
    pythag REAL DEFAULT 0.5,
    -- Rankings
    rank_adj_em INTEGER,
    rank_adj_oe INTEGER,
    rank_adj_de INTEGER,
    rank_tempo INTEGER,
    rank_sos INTEGER,
    rank_luck INTEGER,
    -- Record
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    -- APL
    apl_off REAL,
    apl_def REAL,
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    UNIQUE(snapshot_date, team_id)
);

-- Four Factors data
CREATE TABLE IF NOT EXISTS four_factors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date DATE NOT NULL,
    team_id INTEGER NOT NULL,
    -- Offense
    efg_pct_off REAL NOT NULL,
    to_pct_off REAL NOT NULL,
    or_pct_off REAL NOT NULL,
    ft_rate_off REAL NOT NULL,
    -- Defense
    efg_pct_def REAL NOT NULL,
    to_pct_def REAL NOT NULL,
    or_pct_def REAL NOT NULL,
    ft_rate_def REAL NOT NULL,
    -- Rankings
    rank_efg_off INTEGER,
    rank_efg_def INTEGER,
    rank_to_off INTEGER,
    rank_to_def INTEGER,
    rank_or_off INTEGER,
    rank_or_def INTEGER,
    rank_ft_rate_off INTEGER,
    rank_ft_rate_def INTEGER,
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    UNIQUE(snapshot_date, team_id)
);

-- Point distribution data
CREATE TABLE IF NOT EXISTS point_distribution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date DATE NOT NULL,
    team_id INTEGER NOT NULL,
    -- Offense (% of points from each source)
    ft_pct REAL NOT NULL,
    two_pct REAL NOT NULL,
    three_pct REAL NOT NULL,
    -- Defense (opponent)
    ft_pct_def REAL NOT NULL,
    two_pct_def REAL NOT NULL,
    three_pct_def REAL NOT NULL,
    -- Rankings
    rank_three_pct INTEGER,
    rank_two_pct INTEGER,
    rank_ft_pct INTEGER,
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    UNIQUE(snapshot_date, team_id)
);

-- Height and experience data
CREATE TABLE IF NOT EXISTS height_experience (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date DATE NOT NULL,
    team_id INTEGER NOT NULL,
    avg_height REAL NOT NULL,
    effective_height REAL NOT NULL,
    experience REAL NOT NULL,
    bench_minutes REAL,
    continuity REAL,
    -- Rankings
    rank_height INTEGER,
    rank_experience INTEGER,
    rank_continuity INTEGER,
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    UNIQUE(snapshot_date, team_id)
);

-- Game predictions for tracking accuracy
CREATE TABLE IF NOT EXISTS game_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_date DATE NOT NULL,
    team1_id INTEGER NOT NULL,
    team2_id INTEGER NOT NULL,
    team1_name TEXT,
    team2_name TEXT,
    -- Predictions
    predicted_margin REAL NOT NULL,
    predicted_total REAL NOT NULL,
    win_probability REAL NOT NULL,
    confidence_lower REAL NOT NULL,
    confidence_upper REAL NOT NULL,
    -- Vegas lines at prediction time
    vegas_spread REAL,
    vegas_total REAL,
    -- Actual results (filled after game)
    actual_margin REAL,
    actual_total REAL,
    team1_score INTEGER,
    team2_score INTEGER,
    -- Calculated after game
    prediction_error REAL,
    beat_spread INTEGER,  -- SQLite doesn't have BOOLEAN, use INTEGER
    clv REAL,  -- Closing Line Value
    -- Context
    neutral_site INTEGER DEFAULT 0,
    home_team_id INTEGER,
    -- Metadata
    model_version TEXT DEFAULT 'v1.0',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    FOREIGN KEY (team1_id) REFERENCES teams(team_id),
    FOREIGN KEY (team2_id) REFERENCES teams(team_id)
);

-- Daily accuracy metrics
CREATE TABLE IF NOT EXISTS accuracy_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_date DATE NOT NULL,
    model_version TEXT NOT NULL,
    -- Counts
    games_predicted INTEGER NOT NULL,
    games_resolved INTEGER NOT NULL,
    -- Margin accuracy
    mae_margin REAL,
    rmse_margin REAL,
    r2_margin REAL,
    -- Win accuracy
    win_accuracy REAL,
    brier_score REAL,
    -- ATS performance
    ats_wins INTEGER DEFAULT 0,
    ats_losses INTEGER DEFAULT 0,
    ats_pushes INTEGER DEFAULT 0,
    ats_percentage REAL,
    -- CLV metrics
    avg_clv REAL DEFAULT 0.0,
    positive_clv_rate REAL DEFAULT 0.0,
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(metric_date, model_version)
);

-- Sync history for tracking data freshness
CREATE TABLE IF NOT EXISTS sync_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint TEXT NOT NULL,
    sync_type TEXT NOT NULL,  -- 'full', 'incremental', 'backfill'
    status TEXT NOT NULL,  -- 'success', 'partial', 'failed'
    records_synced INTEGER DEFAULT 0,
    records_skipped INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    duration_seconds REAL
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_ratings_date ON ratings_snapshots(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_ratings_team ON ratings_snapshots(team_id);
CREATE INDEX IF NOT EXISTS idx_ratings_season ON ratings_snapshots(season);
CREATE INDEX IF NOT EXISTS idx_ff_date ON four_factors(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_ff_team ON four_factors(team_id);
CREATE INDEX IF NOT EXISTS idx_pd_date ON point_distribution(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_pd_team ON point_distribution(team_id);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON game_predictions(game_date);
CREATE INDEX IF NOT EXISTS idx_predictions_resolved ON game_predictions(resolved_at);
CREATE INDEX IF NOT EXISTS idx_predictions_pending 
    ON game_predictions(game_date) WHERE resolved_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_sync_endpoint ON sync_history(endpoint, sync_type);
CREATE INDEX IF NOT EXISTS idx_teams_name ON teams(team_name);
"""


class DatabaseManager:
    """Manages SQLite database operations for KenPom data."""

    def __init__(self, db_path: str = "data/kenpom.db"):
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self._ensure_directory()
        self._initialize_schema()

    def _ensure_directory(self) -> None:
        """Create database directory if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _initialize_schema(self) -> None:
        """Initialize database schema if needed."""
        with self.connection() as conn:
            conn.executescript(SCHEMA)
            self._apply_migrations(conn)
            conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    def _apply_migrations(self, conn: sqlite3.Connection) -> None:
        """Apply any pending schema migrations.

        Args:
            conn: Active database connection.
        """
        # Get current schema version
        cursor = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        )
        result = cursor.fetchone()
        current_version = result[0] if result[0] else 0

        if current_version < SCHEMA_VERSION:
            # Apply migrations
            for version in range(current_version + 1, SCHEMA_VERSION + 1):
                migration_func = getattr(self, f"_migrate_v{version}", None)
                if migration_func:
                    logger.info(f"Applying migration v{version}")
                    migration_func(conn)

            # Record new version
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with automatic cleanup.

        Yields:
            sqlite3.Connection with Row factory enabled.

        Example:
            >>> with db.connection() as conn:
            ...     cursor = conn.execute("SELECT * FROM teams")
            ...     teams = cursor.fetchall()
        """
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            raise DatabaseError(f"Database operation failed: {e}") from e
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Execute operations within a transaction.

        Automatically commits on success, rolls back on failure.

        Yields:
            sqlite3.Connection within a transaction.

        Example:
            >>> with db.transaction() as conn:
            ...     conn.execute("INSERT INTO teams ...")
            ...     conn.execute("INSERT INTO ratings ...")
            ...     # Commits automatically
        """
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise DatabaseError(f"Transaction failed: {e}") from e
        finally:
            conn.close()

    def backup(self, backup_path: str | None = None) -> Path:
        """Create a backup of the database.

        Args:
            backup_path: Optional custom backup path.

        Returns:
            Path to the backup file.
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.db_path.parent / f"kenpom_backup_{timestamp}.db"
        else:
            backup_path = Path(backup_path)

        shutil.copy2(self.db_path, backup_path)
        logger.info(f"Database backed up to {backup_path}")
        return backup_path

    def vacuum(self) -> None:
        """Optimize database by running VACUUM."""
        with self.connection() as conn:
            conn.execute("VACUUM")
        logger.info("Database vacuumed")

    def get_stats(self) -> dict:
        """Get database statistics.

        Returns:
            Dictionary with table counts and database size.
        """
        stats = {"db_size_mb": self.db_path.stat().st_size / (1024 * 1024)}

        tables = [
            "teams",
            "ratings_snapshots",
            "four_factors",
            "point_distribution",
            "game_predictions",
            "sync_history",
        ]

        with self.connection() as conn:
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]

            # Get date range of ratings
            cursor = conn.execute(
                """
                SELECT MIN(snapshot_date) as min_date, 
                       MAX(snapshot_date) as max_date
                FROM ratings_snapshots
                """
            )
            row = cursor.fetchone()
            stats["ratings_start_date"] = row["min_date"]
            stats["ratings_end_date"] = row["max_date"]

        return stats

    def get_latest_sync(self, endpoint: str) -> dict | None:
        """Get the most recent successful sync for an endpoint.

        Args:
            endpoint: API endpoint name.

        Returns:
            Sync record dict or None if no successful syncs.
        """
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM sync_history 
                WHERE endpoint = ? AND status = 'success'
                ORDER BY completed_at DESC
                LIMIT 1
                """,
                (endpoint,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def record_sync(
        self,
        endpoint: str,
        sync_type: str,
        status: str,
        records_synced: int = 0,
        records_skipped: int = 0,
        error_message: str | None = None,
        started_at: datetime = None,
        completed_at: datetime = None,
    ) -> int:
        """Record a sync operation in history.

        Args:
            endpoint: API endpoint name.
            sync_type: Type of sync ('full', 'incremental', 'backfill').
            status: Sync status ('success', 'partial', 'failed').
            records_synced: Number of records synced.
            records_skipped: Number of records skipped.
            error_message: Error message if failed.
            started_at: When sync started.
            completed_at: When sync completed.

        Returns:
            ID of the sync record.
        """
        started_at = started_at or datetime.now()
        completed_at = completed_at or datetime.now()
        duration = (completed_at - started_at).total_seconds()

        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sync_history 
                (endpoint, sync_type, status, records_synced, records_skipped,
                 error_message, started_at, completed_at, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    endpoint,
                    sync_type,
                    status,
                    records_synced,
                    records_skipped,
                    error_message,
                    started_at,
                    completed_at,
                    duration,
                ),
            )
            return cursor.lastrowid

    def clear_table(self, table_name: str) -> int:
        """Clear all records from a table.

        Args:
            table_name: Name of table to clear.

        Returns:
            Number of records deleted.
        """
        valid_tables = [
            "teams",
            "ratings_snapshots",
            "four_factors",
            "point_distribution",
            "height_experience",
            "game_predictions",
            "accuracy_metrics",
            "sync_history",
        ]

        if table_name not in valid_tables:
            raise ValueError(f"Invalid table name: {table_name}")

        with self.transaction() as conn:
            cursor = conn.execute(f"DELETE FROM {table_name}")
            count = cursor.rowcount
            logger.warning(f"Cleared {count} records from {table_name}")
            return count

    def reset_database(self) -> None:
        """Reset database to empty state (DESTRUCTIVE!)."""
        logger.warning("Resetting database - all data will be lost!")

        with self.transaction() as conn:
            tables = [
                "accuracy_metrics",
                "game_predictions",
                "height_experience",
                "point_distribution",
                "four_factors",
                "ratings_snapshots",
                "sync_history",
                "teams",
            ]
            for table in tables:
                conn.execute(f"DELETE FROM {table}")

        self.vacuum()
        logger.info("Database reset complete")
