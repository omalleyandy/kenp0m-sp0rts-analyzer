"""Line movement tracking for Closing Line Value (CLV) analysis.

CLV (Closing Line Value) measures the difference between the line at which
you bet and the closing line. Positive CLV indicates you got a better number
than the market consensus at close.

CLV = Opening Line - Closing Line (for spread)
CLV = Opening Total - Closing Total (for totals)

Example:
    If you bet Duke -3.5 (opening) and the line closes at Duke -5.5:
    CLV = 3.5 - 5.5 = -2.0 points (negative CLV, line moved away)

    If you bet Duke -5.5 (opening) and line closes at Duke -3.5:
    CLV = 5.5 - 3.5 = +2.0 points (positive CLV, you got a better number)

Sharp money indicators:
- Reverse line movement (RLM): Line moves opposite to public action
- Steam moves: Quick, sharp line moves across multiple books
- Line freeze: Line holds despite heavy one-sided action
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from ..utils.logging import logger


@dataclass
class LineSnapshot:
    """A point-in-time capture of betting lines."""

    game_id: str
    snapshot_type: str  # 'open', 'current', 'close'
    snapshot_time: datetime
    home_team: str
    away_team: str
    spread: float  # Home perspective (negative = home favored)
    total: float
    spread_home_odds: int = -110
    spread_away_odds: int = -110
    over_odds: int = -110
    under_odds: int = -110
    home_ml: int | None = None
    away_ml: int | None = None
    source: str = "overtime.ag"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "game_id": self.game_id,
            "snapshot_type": self.snapshot_type,
            "snapshot_time": self.snapshot_time.isoformat(),
            "home_team": self.home_team,
            "away_team": self.away_team,
            "spread": self.spread,
            "total": self.total,
            "spread_home_odds": self.spread_home_odds,
            "spread_away_odds": self.spread_away_odds,
            "over_odds": self.over_odds,
            "under_odds": self.under_odds,
            "home_ml": self.home_ml,
            "away_ml": self.away_ml,
            "source": self.source,
        }


@dataclass
class LineMovement:
    """Analysis of line movement between two snapshots."""

    game_id: str
    home_team: str
    away_team: str
    # Opening values
    open_spread: float
    open_total: float
    open_time: datetime
    # Closing values
    close_spread: float
    close_total: float
    close_time: datetime
    # Movement calculations
    spread_movement: float  # close - open (negative = moved toward home)
    total_movement: float  # close - open (positive = moved higher)
    # CLV (for bets placed at open)
    spread_clv: float  # open - close (positive = got better number)
    total_clv: float
    # Sharp indicators
    is_reverse_movement: bool = False
    movement_velocity: float = 0.0  # points per hour

    @property
    def has_significant_movement(self) -> bool:
        """Check if movement is significant (>1pt spread, >2 total)."""
        spread_sig = abs(self.spread_movement) >= 1.0
        total_sig = abs(self.total_movement) >= 2.0
        return spread_sig or total_sig

    @property
    def sharp_money_indicator(self) -> str:
        """Classify the type of line movement."""
        if self.is_reverse_movement:
            return "reverse_line_movement"
        if abs(self.movement_velocity) > 0.5:
            return "steam_move"
        if abs(self.spread_movement) < 0.5 and abs(self.total_movement) < 1.0:
            return "line_freeze"
        return "normal"


class LineMovementTracker:
    """Track and analyze line movements for CLV analysis.

    Usage:
        tracker = LineMovementTracker(repository)

        # Capture opening lines
        tracker.capture_opening_lines(games)

        # Capture closing lines (right before game time)
        tracker.capture_closing_lines(games)

        # Calculate CLV for a game
        clv = tracker.calculate_clv(game_id)

        # Get line movement analysis
        movement = tracker.get_line_movement(game_id)
    """

    def __init__(self, repository: Any):
        """Initialize with a KenPom repository.

        Args:
            repository: KenPomRepository instance with database connection.
        """
        self.repository = repository

    def capture_lines(
        self,
        games: list[dict],
        snapshot_type: str = "current",
    ) -> int:
        """Capture betting lines for multiple games.

        Args:
            games: List of game dicts with spread, total, teams.
            snapshot_type: 'open', 'current', or 'close'.

        Returns:
            Number of games captured.
        """
        captured = 0
        now = datetime.now()

        with self.repository.db.connection() as conn:
            for game in games:
                try:
                    # Extract game info
                    game_date = game.get("game_date", str(date.today()))
                    home = game.get("home_team", "")
                    away = game.get("away_team", "")

                    # Skip if already have this snapshot type
                    if snapshot_type in ("open", "close"):
                        existing = conn.execute(
                            """
                            SELECT 1 FROM vegas_odds
                            WHERE home_team = ? AND away_team = ?
                            AND game_date = ? AND snapshot_type = ?
                            """,
                            (home, away, game_date, snapshot_type),
                        ).fetchone()
                        if existing:
                            continue

                    # Insert or update
                    conn.execute(
                        """
                        INSERT INTO vegas_odds (
                            game_date, game_time, away_team, home_team,
                            spread, total, away_ml, home_ml,
                            snapshot_type, snapshot_at, source
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            game_date,
                            game.get("game_time", ""),
                            away,
                            home,
                            game.get("spread"),
                            game.get("total"),
                            game.get("away_ml"),
                            game.get("home_ml"),
                            snapshot_type,
                            now.isoformat(),
                            game.get("source", "overtime.ag"),
                        ),
                    )
                    captured += 1
                except Exception as e:
                    logger.warning(f"Failed to capture {game}: {e}")

            conn.commit()

        logger.info(f"Captured {captured} {snapshot_type} lines")
        return captured

    def get_opening_line(
        self,
        home_team: str,
        away_team: str,
        game_date: date | str,
    ) -> LineSnapshot | None:
        """Get opening line for a game.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            game_date: Game date.

        Returns:
            LineSnapshot or None if not found.
        """
        if isinstance(game_date, date):
            game_date = str(game_date)

        with self.repository.db.connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM vegas_odds
                WHERE home_team = ? AND away_team = ? AND game_date = ?
                AND snapshot_type = 'open'
                ORDER BY snapshot_at ASC
                LIMIT 1
                """,
                (home_team, away_team, game_date),
            ).fetchone()

            if row:
                # Handle snapshot_at as string or datetime
                snapshot_at = row["snapshot_at"]
                if isinstance(snapshot_at, str):
                    snapshot_time = datetime.fromisoformat(snapshot_at)
                else:
                    snapshot_time = snapshot_at

                return LineSnapshot(
                    game_id=f"{game_date}_{away_team}@{home_team}",
                    snapshot_type="open",
                    snapshot_time=snapshot_time,
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    spread=row["spread"] or 0.0,
                    total=row["total"] or 0.0,
                    home_ml=row["home_ml"],
                    away_ml=row["away_ml"],
                )
        return None

    def get_closing_line(
        self,
        home_team: str,
        away_team: str,
        game_date: date | str,
    ) -> LineSnapshot | None:
        """Get closing line for a game.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            game_date: Game date.

        Returns:
            LineSnapshot or None if not found.
        """
        if isinstance(game_date, date):
            game_date = str(game_date)

        with self.repository.db.connection() as conn:
            # First try explicit close snapshot
            row = conn.execute(
                """
                SELECT * FROM vegas_odds
                WHERE home_team = ? AND away_team = ? AND game_date = ?
                AND snapshot_type = 'close'
                ORDER BY snapshot_at DESC
                LIMIT 1
                """,
                (home_team, away_team, game_date),
            ).fetchone()

            # Fall back to most recent snapshot
            if not row:
                row = conn.execute(
                    """
                    SELECT * FROM vegas_odds
                    WHERE home_team = ? AND away_team = ? AND game_date = ?
                    ORDER BY snapshot_at DESC
                    LIMIT 1
                    """,
                    (home_team, away_team, game_date),
                ).fetchone()

            if row:
                # Handle snapshot_at as string or datetime
                snapshot_at = row["snapshot_at"]
                if isinstance(snapshot_at, str):
                    snapshot_time = datetime.fromisoformat(snapshot_at)
                else:
                    snapshot_time = snapshot_at

                return LineSnapshot(
                    game_id=f"{game_date}_{away_team}@{home_team}",
                    snapshot_type=row["snapshot_type"],
                    snapshot_time=snapshot_time,
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    spread=row["spread"] or 0.0,
                    total=row["total"] or 0.0,
                    home_ml=row["home_ml"],
                    away_ml=row["away_ml"],
                )
        return None

    def get_line_movement(
        self,
        home_team: str,
        away_team: str,
        game_date: date | str,
    ) -> LineMovement | None:
        """Calculate line movement for a game.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            game_date: Game date.

        Returns:
            LineMovement analysis or None if insufficient data.
        """
        opening = self.get_opening_line(home_team, away_team, game_date)
        closing = self.get_closing_line(home_team, away_team, game_date)

        if not opening or not closing:
            return None

        spread_movement = closing.spread - opening.spread
        total_movement = closing.total - opening.total

        # Calculate velocity (points per hour)
        delta = closing.snapshot_time - opening.snapshot_time
        time_diff = delta.total_seconds()
        hours = max(time_diff / 3600, 0.1)  # Avoid division by zero
        velocity = abs(spread_movement) / hours

        game_id = f"{game_date}_{away_team}@{home_team}"

        return LineMovement(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            open_spread=opening.spread,
            open_total=opening.total,
            open_time=opening.snapshot_time,
            close_spread=closing.spread,
            close_total=closing.total,
            close_time=closing.snapshot_time,
            spread_movement=spread_movement,
            total_movement=total_movement,
            spread_clv=opening.spread - closing.spread,
            total_clv=opening.total - closing.total,
            is_reverse_movement=False,  # TODO: Add public betting data
            movement_velocity=velocity,
        )

    def calculate_clv(
        self,
        home_team: str,
        away_team: str,
        game_date: date | str,
        bet_spread: float | None = None,
        bet_total: float | None = None,
    ) -> dict[str, float | None]:
        """Calculate Closing Line Value for a bet.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            game_date: Game date.
            bet_spread: Spread at which bet was placed (default: opening).
            bet_total: Total at which bet was placed (default: opening).

        Returns:
            Dict with CLV values (positive = got better number).
        """
        movement = self.get_line_movement(home_team, away_team, game_date)

        if not movement:
            return {"spread_clv": None, "total_clv": None}

        # Use opening if no bet line specified
        if bet_spread is None:
            bet_spread = movement.open_spread
        if bet_total is None:
            bet_total = movement.open_total

        return {
            "spread_clv": bet_spread - movement.close_spread,
            "total_clv": bet_total - movement.close_total,
            "spread_movement": movement.spread_movement,
            "total_movement": movement.total_movement,
            "sharp_indicator": movement.sharp_money_indicator,
        }

    def get_line_history(
        self,
        home_team: str,
        away_team: str,
        game_date: date | str,
    ) -> list[LineSnapshot]:
        """Get all line snapshots for a game.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            game_date: Game date.

        Returns:
            List of LineSnapshot objects in chronological order.
        """
        if isinstance(game_date, date):
            game_date = str(game_date)

        snapshots = []
        with self.repository.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM vegas_odds
                WHERE home_team = ? AND away_team = ? AND game_date = ?
                ORDER BY snapshot_at ASC
                """,
                (home_team, away_team, game_date),
            ).fetchall()

            for row in rows:
                snapshots.append(
                    LineSnapshot(
                        game_id=f"{game_date}_{away_team}@{home_team}",
                        snapshot_type=row["snapshot_type"],
                        snapshot_time=datetime.fromisoformat(row["snapshot_at"]),
                        home_team=row["home_team"],
                        away_team=row["away_team"],
                        spread=row["spread"] or 0.0,
                        total=row["total"] or 0.0,
                        home_ml=row["home_ml"],
                        away_ml=row["away_ml"],
                    )
                )

        return snapshots
