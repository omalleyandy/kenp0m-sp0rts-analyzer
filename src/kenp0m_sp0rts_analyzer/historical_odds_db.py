"""Historical odds and results database for sports betting analysis.

This module provides a SQLite database for storing:
- Historical Vegas lines (opening, closing, line movements)
- Game results (final scores, covers, ATS records)
- Model predictions for backtesting
- Performance tracking and analytics

Schema designed for:
- Multi-sport support (CBB, NFL, NBA, etc.)
- Time-series line movement tracking
- Prediction vs actual comparison
- ROI and CLV analysis
"""

import sqlite3
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Default database location
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "historical_odds.db"


@dataclass
class GameRecord:
    """A game with odds and results."""
    game_id: str
    sport: str
    league: str
    game_date: str
    game_time: str
    away_team: str
    home_team: str
    # Odds
    spread_open: float | None = None
    spread_close: float | None = None
    total_open: float | None = None
    total_close: float | None = None
    away_ml_open: int | None = None
    away_ml_close: int | None = None
    home_ml_open: int | None = None
    home_ml_close: int | None = None
    # Results
    away_score: int | None = None
    home_score: int | None = None
    # Calculated fields
    actual_spread: float | None = None  # home_score - away_score
    actual_total: int | None = None
    home_covered: bool | None = None
    over_hit: bool | None = None
    # Metadata
    source: str = "overtime.ag"
    created_at: str = ""
    updated_at: str = ""


class HistoricalOddsDB:
    """SQLite database for historical odds and results.
    
    Example:
        ```python
        db = HistoricalOddsDB()
        db.initialize()
        
        # Store today's odds
        db.store_odds_snapshot(
            sport="Basketball",
            league="College Basketball", 
            game_date="2025-12-17",
            games=[...],
            snapshot_type="close"
        )
        
        # Store results
        db.update_game_result(
            game_id="cbb_2025-12-17_duke_unc",
            away_score=72,
            home_score=85
        )
        
        # Analyze performance
        stats = db.get_ats_record(league="College Basketball", days=30)
        ```
    """
    
    def __init__(self, db_path: Path | str | None = None):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        
    def connect(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn
        
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            
    def initialize(self) -> None:
        """Create database tables if they don't exist."""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Sports table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sports (
                sport_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport_name TEXT UNIQUE NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Leagues table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS leagues (
                league_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport_id INTEGER NOT NULL,
                league_name TEXT NOT NULL,
                league_code TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sport_id) REFERENCES sports(sport_id),
                UNIQUE(sport_id, league_name)
            )
        ''')
        
        # Teams table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS teams (
                team_id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name TEXT NOT NULL,
                team_name_normalized TEXT NOT NULL,
                league_id INTEGER,
                kenpom_name TEXT,
                conference TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (league_id) REFERENCES leagues(league_id)
            )
        ''')
        
        # Games table - core game information
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                sport TEXT NOT NULL,
                league TEXT NOT NULL,
                game_date TEXT NOT NULL,
                game_time TEXT,
                away_team TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team_normalized TEXT,
                home_team_normalized TEXT,
                neutral_site INTEGER DEFAULT 0,
                -- Results
                away_score INTEGER,
                home_score INTEGER,
                status TEXT DEFAULT 'scheduled',  -- scheduled, final, postponed, cancelled
                -- Calculated from results
                actual_spread REAL,  -- home - away (negative = home won by more)
                actual_total INTEGER,
                -- Metadata
                source TEXT DEFAULT 'overtime.ag',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Odds snapshots - track line movements over time
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                snapshot_time TEXT NOT NULL,
                snapshot_type TEXT NOT NULL,  -- 'open', 'current', 'close'
                -- Spread (from home team perspective, negative = home favored)
                spread REAL,
                spread_away_odds INTEGER,
                spread_home_odds INTEGER,
                -- Total
                total REAL,
                over_odds INTEGER,
                under_odds INTEGER,
                -- Moneyline
                away_ml INTEGER,
                home_ml INTEGER,
                -- Source
                source TEXT DEFAULT 'overtime.ag',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            )
        ''')
        
        # Predictions table - store our model predictions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                prediction_time TEXT NOT NULL,
                model_name TEXT NOT NULL,  -- 'kenpom_basic', 'comprehensive', etc.
                model_version TEXT,
                -- Predictions
                predicted_spread REAL,  -- Our predicted spread
                predicted_total REAL,   -- Our predicted total
                home_win_prob REAL,     -- Win probability (0-100)
                confidence REAL,        -- Model confidence
                -- Edge calculations (vs closing line)
                spread_edge REAL,       -- Our spread - Vegas spread
                total_edge REAL,        -- Our total - Vegas total
                -- Picks
                spread_pick TEXT,       -- 'home', 'away', 'no_play'
                total_pick TEXT,        -- 'over', 'under', 'no_play'
                -- Metadata
                features_json TEXT,     -- JSON of key features used
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            )
        ''')
        
        # Prediction results - track outcomes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                game_id TEXT NOT NULL,
                -- Outcomes
                spread_result TEXT,     -- 'win', 'loss', 'push'
                total_result TEXT,      -- 'win', 'loss', 'push'
                ml_result TEXT,         -- 'win', 'loss'
                -- CLV (Closing Line Value)
                clv_spread REAL,        -- Predicted spread - Closing spread
                clv_total REAL,
                -- Actual values for analysis
                actual_spread REAL,
                actual_total INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id),
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            )
        ''')
        
        # Create indexes for common queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_league ON games(league)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_sport_date ON games(sport, game_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_odds_game ON odds_snapshots(game_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_odds_type ON odds_snapshots(snapshot_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_game ON predictions(game_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name)')
        
        conn.commit()
        logger.info(f"Database initialized at {self.db_path}")
        
    def _normalize_team_name(self, name: str) -> str:
        """Normalize team name for matching."""
        import re
        # Lowercase, remove punctuation, normalize spaces
        normalized = name.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
        
    def _generate_game_id(self, sport: str, league: str, game_date: str, 
                          away_team: str, home_team: str) -> str:
        """Generate a unique game ID."""
        away_norm = self._normalize_team_name(away_team).replace(' ', '_')[:20]
        home_norm = self._normalize_team_name(home_team).replace(' ', '_')[:20]
        league_code = league.lower().replace(' ', '_')[:10]
        return f"{league_code}_{game_date}_{away_norm}_{home_norm}"
        
    def store_odds_snapshot(
        self,
        sport: str,
        league: str,
        game_date: str,
        games: list[dict],
        snapshot_type: str = "current",
        source: str = "overtime.ag",
    ) -> int:
        """Store a snapshot of odds for multiple games.
        
        Args:
            sport: Sport name (e.g., "Basketball")
            league: League name (e.g., "College Basketball")
            game_date: Date string (YYYY-MM-DD)
            games: List of game dictionaries with odds
            snapshot_type: Type of snapshot ('open', 'current', 'close')
            source: Odds source
            
        Returns:
            Number of games stored
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        snapshot_time = datetime.now().isoformat()
        stored_count = 0
        
        for game in games:
            away_team = game.get('away_team', '')
            home_team = game.get('home_team', '')
            
            if not away_team or not home_team:
                continue
                
            game_id = self._generate_game_id(sport, league, game_date, away_team, home_team)
            
            # Insert or update game record
            cursor.execute('''
                INSERT INTO games (
                    game_id, sport, league, game_date, game_time,
                    away_team, home_team, away_team_normalized, home_team_normalized,
                    source, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    updated_at = excluded.updated_at
            ''', (
                game_id, sport, league, game_date, game.get('time', ''),
                away_team, home_team,
                self._normalize_team_name(away_team),
                self._normalize_team_name(home_team),
                source, snapshot_time
            ))
            
            # Insert odds snapshot
            cursor.execute('''
                INSERT INTO odds_snapshots (
                    game_id, snapshot_time, snapshot_type,
                    spread, spread_away_odds, spread_home_odds,
                    total, over_odds, under_odds,
                    away_ml, home_ml, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                game_id, snapshot_time, snapshot_type,
                game.get('spread'), game.get('spread_away_odds'), game.get('spread_home_odds'),
                game.get('total'), game.get('over_odds'), game.get('under_odds'),
                game.get('away_ml'), game.get('home_ml'), source
            ))
            
            stored_count += 1
            
        conn.commit()
        logger.info(f"Stored {stored_count} {snapshot_type} odds snapshots for {league} on {game_date}")
        return stored_count
        
    def update_game_result(
        self,
        game_id: str,
        away_score: int,
        home_score: int,
    ) -> bool:
        """Update a game with final score.
        
        Args:
            game_id: Unique game identifier
            away_score: Away team final score
            home_score: Home team final score
            
        Returns:
            True if updated successfully
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        actual_spread = home_score - away_score
        actual_total = away_score + home_score
        
        cursor.execute('''
            UPDATE games SET
                away_score = ?,
                home_score = ?,
                actual_spread = ?,
                actual_total = ?,
                status = 'final',
                updated_at = ?
            WHERE game_id = ?
        ''', (away_score, home_score, actual_spread, actual_total,
              datetime.now().isoformat(), game_id))
        
        conn.commit()
        return cursor.rowcount > 0
        
    def store_prediction(
        self,
        game_id: str,
        model_name: str,
        predicted_spread: float,
        predicted_total: float,
        home_win_prob: float,
        confidence: float = 0.0,
        spread_pick: str = "no_play",
        total_pick: str = "no_play",
        features: dict | None = None,
        model_version: str = "1.0",
    ) -> int:
        """Store a model prediction for a game.
        
        Args:
            game_id: Unique game identifier
            model_name: Name of prediction model
            predicted_spread: Predicted spread (home perspective)
            predicted_total: Predicted total points
            home_win_prob: Home win probability (0-100)
            confidence: Model confidence score
            spread_pick: 'home', 'away', or 'no_play'
            total_pick: 'over', 'under', or 'no_play'
            features: Dictionary of key features used
            model_version: Model version string
            
        Returns:
            Prediction ID
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        # Get closing line for edge calculation
        cursor.execute('''
            SELECT spread, total FROM odds_snapshots
            WHERE game_id = ? AND snapshot_type = 'close'
            ORDER BY snapshot_time DESC LIMIT 1
        ''', (game_id,))
        
        row = cursor.fetchone()
        spread_edge = None
        total_edge = None
        
        if row:
            if row['spread'] is not None:
                spread_edge = predicted_spread - row['spread']
            if row['total'] is not None:
                total_edge = predicted_total - row['total']
        
        cursor.execute('''
            INSERT INTO predictions (
                game_id, prediction_time, model_name, model_version,
                predicted_spread, predicted_total, home_win_prob, confidence,
                spread_edge, total_edge, spread_pick, total_pick, features_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            game_id, datetime.now().isoformat(), model_name, model_version,
            predicted_spread, predicted_total, home_win_prob, confidence,
            spread_edge, total_edge, spread_pick, total_pick,
            json.dumps(features) if features else None
        ))
        
        conn.commit()
        return cursor.lastrowid
        
    def calculate_prediction_results(self, game_id: str) -> dict:
        """Calculate results for all predictions on a completed game.
        
        Args:
            game_id: Unique game identifier
            
        Returns:
            Dictionary with result summary
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        # Get game result
        cursor.execute('''
            SELECT actual_spread, actual_total, away_score, home_score
            FROM games WHERE game_id = ? AND status = 'final'
        ''', (game_id,))
        
        game = cursor.fetchone()
        if not game:
            return {"error": "Game not found or not final"}
            
        actual_spread = game['actual_spread']
        actual_total = game['actual_total']
        
        # Get closing line
        cursor.execute('''
            SELECT spread, total FROM odds_snapshots
            WHERE game_id = ? AND snapshot_type = 'close'
            ORDER BY snapshot_time DESC LIMIT 1
        ''', (game_id,))
        
        closing = cursor.fetchone()
        closing_spread = closing['spread'] if closing else None
        closing_total = closing['total'] if closing else None
        
        # Process each prediction
        cursor.execute('''
            SELECT prediction_id, predicted_spread, predicted_total, spread_pick, total_pick
            FROM predictions WHERE game_id = ?
        ''', (game_id,))
        
        results = []
        for pred in cursor.fetchall():
            # Determine spread result
            spread_result = None
            if pred['spread_pick'] != 'no_play' and closing_spread is not None:
                if pred['spread_pick'] == 'home':
                    # Bet on home to cover (home -spread)
                    if actual_spread < closing_spread:
                        spread_result = 'win'
                    elif actual_spread > closing_spread:
                        spread_result = 'loss'
                    else:
                        spread_result = 'push'
                else:  # away
                    if actual_spread > closing_spread:
                        spread_result = 'win'
                    elif actual_spread < closing_spread:
                        spread_result = 'loss'
                    else:
                        spread_result = 'push'
                        
            # Determine total result
            total_result = None
            if pred['total_pick'] != 'no_play' and closing_total is not None:
                if pred['total_pick'] == 'over':
                    if actual_total > closing_total:
                        total_result = 'win'
                    elif actual_total < closing_total:
                        total_result = 'loss'
                    else:
                        total_result = 'push'
                else:  # under
                    if actual_total < closing_total:
                        total_result = 'win'
                    elif actual_total > closing_total:
                        total_result = 'loss'
                    else:
                        total_result = 'push'
                        
            # Calculate CLV
            clv_spread = None
            clv_total = None
            if closing_spread is not None:
                clv_spread = pred['predicted_spread'] - closing_spread
            if closing_total is not None:
                clv_total = pred['predicted_total'] - closing_total
                
            # Store result
            cursor.execute('''
                INSERT INTO prediction_results (
                    prediction_id, game_id, spread_result, total_result,
                    clv_spread, clv_total, actual_spread, actual_total
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pred['prediction_id'], game_id, spread_result, total_result,
                clv_spread, clv_total, actual_spread, actual_total
            ))
            
            results.append({
                'prediction_id': pred['prediction_id'],
                'spread_result': spread_result,
                'total_result': total_result,
                'clv_spread': clv_spread,
                'clv_total': clv_total,
            })
            
        conn.commit()
        return {'game_id': game_id, 'results': results}
        
    def get_model_performance(
        self,
        model_name: str,
        league: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        """Get performance statistics for a prediction model.
        
        Args:
            model_name: Name of prediction model
            league: Filter by league (optional)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            
        Returns:
            Dictionary with performance statistics
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        # Build query
        query = '''
            SELECT 
                COUNT(*) as total_predictions,
                SUM(CASE WHEN pr.spread_result = 'win' THEN 1 ELSE 0 END) as spread_wins,
                SUM(CASE WHEN pr.spread_result = 'loss' THEN 1 ELSE 0 END) as spread_losses,
                SUM(CASE WHEN pr.spread_result = 'push' THEN 1 ELSE 0 END) as spread_pushes,
                SUM(CASE WHEN pr.total_result = 'win' THEN 1 ELSE 0 END) as total_wins,
                SUM(CASE WHEN pr.total_result = 'loss' THEN 1 ELSE 0 END) as total_losses,
                SUM(CASE WHEN pr.total_result = 'push' THEN 1 ELSE 0 END) as total_pushes,
                AVG(pr.clv_spread) as avg_clv_spread,
                AVG(pr.clv_total) as avg_clv_total
            FROM prediction_results pr
            JOIN predictions p ON pr.prediction_id = p.prediction_id
            JOIN games g ON pr.game_id = g.game_id
            WHERE p.model_name = ?
        '''
        
        params = [model_name]
        
        if league:
            query += ' AND g.league = ?'
            params.append(league)
        if start_date:
            query += ' AND g.game_date >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND g.game_date <= ?'
            params.append(end_date)
            
        cursor.execute(query, params)
        row = cursor.fetchone()
        
        if not row or row['total_predictions'] == 0:
            return {'error': 'No predictions found'}
            
        # Calculate win percentages
        spread_total = (row['spread_wins'] or 0) + (row['spread_losses'] or 0)
        total_total = (row['total_wins'] or 0) + (row['total_losses'] or 0)
        
        spread_pct = (row['spread_wins'] / spread_total * 100) if spread_total > 0 else 0
        total_pct = (row['total_wins'] / total_total * 100) if total_total > 0 else 0
        
        return {
            'model_name': model_name,
            'total_predictions': row['total_predictions'],
            'spread': {
                'wins': row['spread_wins'] or 0,
                'losses': row['spread_losses'] or 0,
                'pushes': row['spread_pushes'] or 0,
                'win_pct': round(spread_pct, 1),
                'record': f"{row['spread_wins'] or 0}-{row['spread_losses'] or 0}-{row['spread_pushes'] or 0}",
            },
            'totals': {
                'wins': row['total_wins'] or 0,
                'losses': row['total_losses'] or 0,
                'pushes': row['total_pushes'] or 0,
                'win_pct': round(total_pct, 1),
                'record': f"{row['total_wins'] or 0}-{row['total_losses'] or 0}-{row['total_pushes'] or 0}",
            },
            'clv': {
                'avg_spread': round(row['avg_clv_spread'] or 0, 2),
                'avg_total': round(row['avg_clv_total'] or 0, 2),
            },
        }
        
    def get_games_by_date(
        self,
        game_date: str,
        league: str | None = None,
    ) -> list[dict]:
        """Get all games for a specific date.
        
        Args:
            game_date: Date string (YYYY-MM-DD)
            league: Filter by league (optional)
            
        Returns:
            List of game dictionaries
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        query = '''
            SELECT g.*, 
                   os.spread as closing_spread,
                   os.total as closing_total,
                   os.away_ml as closing_away_ml,
                   os.home_ml as closing_home_ml
            FROM games g
            LEFT JOIN odds_snapshots os ON g.game_id = os.game_id 
                AND os.snapshot_type = 'close'
            WHERE g.game_date = ?
        '''
        
        params = [game_date]
        if league:
            query += ' AND g.league = ?'
            params.append(league)
            
        query += ' ORDER BY g.game_time'
        
        cursor.execute(query, params)
        
        return [dict(row) for row in cursor.fetchall()]
        
    def get_line_movement(self, game_id: str) -> list[dict]:
        """Get line movement history for a game.
        
        Args:
            game_id: Unique game identifier
            
        Returns:
            List of odds snapshots in chronological order
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM odds_snapshots
            WHERE game_id = ?
            ORDER BY snapshot_time
        ''', (game_id,))
        
        return [dict(row) for row in cursor.fetchall()]


# Convenience functions
def get_db(db_path: Path | str | None = None) -> HistoricalOddsDB:
    """Get a database instance."""
    db = HistoricalOddsDB(db_path)
    db.initialize()
    return db
