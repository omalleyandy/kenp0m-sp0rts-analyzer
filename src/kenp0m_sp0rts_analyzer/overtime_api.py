"""Overtime.ag Direct API Client.

This module provides direct API access to overtime.ag betting lines
without needing browser automation. Much faster and more reliable.

Discovered API endpoints:
- /Api/Offering.asmx/GetSports - List all sports
- /Api/Offering.asmx/GetSportOffering - Get lines for a sport
- /Api/Login.asmx/CustomerSignIn - Authentication
- /Api/Config.asmx/Get - Site configuration

Requirements:
    pip install httpx python-dotenv

Environment Variables:
    OVERTIME_USER - Overtime.ag username
    OVERTIME_PASSWORD - Overtime.ag password
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

# API Configuration
OVERTIME_BASE_URL = "https://overtime.ag/sports"
OVERTIME_API_BASE = f"{OVERTIME_BASE_URL}/Api"

# Default headers to mimic browser
DEFAULT_HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "Content-Type": "application/json; charset=UTF-8",
    "Origin": "https://overtime.ag",
    "Referer": "https://overtime.ag/sports",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
}


@dataclass
class OvertimeGame:
    """Represents a single game with betting lines."""

    game_num: int
    time: str
    game_date: str
    away_team: str
    home_team: str
    away_rot: int
    home_rot: int
    spread: float | None = None
    spread_away_odds: int | None = None
    spread_home_odds: int | None = None
    total: float | None = None
    over_odds: int | None = None
    under_odds: int | None = None
    away_ml: int | None = None
    home_ml: int | None = None
    sport_type: str = "Basketball"
    sport_subtype: str = "College Basketball"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "time": self.time,
            "game_date": self.game_date,
            "away_team": self.away_team,
            "home_team": self.home_team,
            "away_rot": self.away_rot,
            "home_rot": self.home_rot,
            "spread": self.spread,
            "spread_away_odds": self.spread_away_odds,
            "spread_home_odds": self.spread_home_odds,
            "total": self.total,
            "over_odds": self.over_odds,
            "under_odds": self.under_odds,
            "away_ml": self.away_ml,
            "home_ml": self.home_ml,
        }


@dataclass
class OvertimeSport:
    """Represents a sport/league available for betting."""

    sport_type: str
    sport_subtype: str
    sport_subtype_id: int
    first_rot_num: str
    active: bool
    next_event: datetime | None = None


class OvertimeAPIClient:
    """Direct API client for overtime.ag betting lines.

    This client uses the discovered REST API endpoints to fetch
    betting lines without browser automation.

    Example:
        ```python
        async with OvertimeAPIClient() as client:
            # Get all available sports
            sports = await client.get_sports()

            # Get college basketball lines
            games = await client.get_college_basketball_lines()

            for game in games:
                print(f"{game.away_team} @ {game.home_team}")
                print(f"  Spread: {game.home_team} {game.spread}")
                print(f"  Total: {game.total}")
        ```
    """

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the API client.

        Args:
            username: Overtime.ag username. Reads from OVERTIME_USER env if not provided.
            password: Overtime.ag password. Reads from OVERTIME_PASSWORD env if not provided.
            timeout: Request timeout in seconds.
        """
        self._username = username or os.getenv("OVERTIME_USER")
        self._password = password or os.getenv("OVERTIME_PASSWORD")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._logged_in = False
        self._session_initialized = False

    async def __aenter__(self) -> "OvertimeAPIClient":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def start(self) -> None:
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            headers=DEFAULT_HEADERS,
            follow_redirects=True,
        )
        logger.info("Overtime API client started")

    async def _init_session(self) -> None:
        """Initialize session by visiting main page to get cookies."""
        if self._session_initialized:
            return

        if not self._client:
            raise RuntimeError("Client not started")

        # Visit main page to establish session/cookies
        logger.info("Initializing session...")
        try:
            response = await self._client.get(
                OVERTIME_BASE_URL,
                headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                },
            )
            response.raise_for_status()
            self._session_initialized = True
            logger.info("Session initialized")
        except Exception as e:
            logger.warning(f"Session init warning: {e}")

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
        logger.info("Overtime API client closed")

    async def _post(self, endpoint: str, data: dict | None = None) -> dict:
        """Make a POST request to the API.

        Args:
            endpoint: API endpoint path (e.g., "Offering.asmx/GetSports")
            data: Request body data

        Returns:
            Parsed JSON response
        """
        if not self._client:
            raise RuntimeError(
                "Client not started. Use async context manager."
            )

        # Ensure session is initialized
        await self._init_session()

        url = f"{OVERTIME_API_BASE}/{endpoint}"

        try:
            response = await self._client.post(
                url,
                json=data or {},
            )
            response.raise_for_status()

            result = response.json()

            # Overtime API wraps responses in {"d": {...}}
            if "d" in result:
                return result["d"]
            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {endpoint}")
            raise
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise

    async def login(self) -> bool:
        """Authenticate with overtime.ag.

        Returns:
            True if login successful.
        """
        if not self._username or not self._password:
            logger.warning("No credentials provided - using demo mode")
            return False

        try:
            result = await self._post(
                "Login.asmx/CustomerSignIn",
                {
                    "customer": self._username,
                    "password": self._password,
                },
            )

            if result.get("IsSuccess"):
                self._logged_in = True
                self._session_id = result.get("Data", {}).get("SessionID")
                logger.info("Login successful")
                return True
            else:
                logger.warning(f"Login failed: {result.get('Message')}")
                return False

        except Exception as e:
            logger.error(f"Login error: {e}")
            return False

    async def get_sports(self) -> list[OvertimeSport]:
        """Get list of available sports/leagues.

        Returns:
            List of OvertimeSport objects.
        """
        result = await self._post("Offering.asmx/GetSports", {})

        sports = []
        for item in result.get("Data", []):
            # Parse next event datetime
            next_event = None
            if item.get("NextEventDateTime"):
                try:
                    # Format: /Date(1766106900000)/
                    match = re.search(
                        r"/Date\((\d+)\)/", item["NextEventDateTime"]
                    )
                    if match:
                        timestamp = int(match.group(1)) / 1000
                        next_event = datetime.fromtimestamp(timestamp)
                except Exception:
                    pass

            sports.append(
                OvertimeSport(
                    sport_type=item.get("SportType", ""),
                    sport_subtype=item.get("SportSubType", ""),
                    sport_subtype_id=item.get("SportSubTypeId", 0),
                    first_rot_num=item.get("FirstRotNum", ""),
                    active=bool(item.get("Active", 0)),
                    next_event=next_event,
                )
            )

        return sports

    async def get_sport_offering(
        self,
        sport_type: str,
        sport_subtype: str,
        sport_subtype_id: int,
    ) -> dict:
        """Get betting lines for a specific sport.

        Args:
            sport_type: Sport type (e.g., "Basketball")
            sport_subtype: Sport subtype (e.g., "College Basketball")
            sport_subtype_id: Sport subtype ID

        Returns:
            Raw API response with game lines.
        """
        result = await self._post(
            "Offering.asmx/GetSportOffering",
            {
                "sportType": sport_type,
                "sportSubType": sport_subtype,
                "wagerType": "Straight Bet",
                "hoursAdjustment": 0,
                "periodNumber": None,
                "gameNum": None,
                "parentGameNum": None,
                "teaserName": "",
                "requestMode": None,
            },
        )

        return result.get("Data", {})

    async def get_college_basketball_lines(
        self, include_extra: bool = True
    ) -> list[OvertimeGame]:
        """Get college basketball betting lines.

        Args:
            include_extra: Include "College Extra" games (smaller schools)

        Returns:
            List of OvertimeGame objects with betting lines.
        """
        games = []

        # Get available sports to find CBB
        sports = await self.get_sports()

        # Find college basketball entries
        cbb_sports = [
            s
            for s in sports
            if s.sport_type == "Basketball" and "College" in s.sport_subtype
        ]

        if not cbb_sports:
            logger.warning("College Basketball not found in available sports")
            return games

        for sport in cbb_sports:
            # Skip "College Extra" if not requested
            if not include_extra and "Extra" in sport.sport_subtype:
                continue

            logger.info(f"Fetching lines for: {sport.sport_subtype}")

            try:
                offering = await self.get_sport_offering(
                    sport.sport_type,
                    sport.sport_subtype,
                    sport.sport_subtype_id,
                )

                game_lines = offering.get("GameLines", [])

                for gl in game_lines:
                    game = self._parse_game_line(gl, sport)
                    if game:
                        games.append(game)

            except Exception as e:
                logger.error(f"Error fetching {sport.sport_subtype}: {e}")
                continue

        logger.info(f"Found {len(games)} college basketball games")
        return games

    def _parse_game_line(
        self, gl: dict, sport: OvertimeSport
    ) -> OvertimeGame | None:
        """Parse a game line from the API response.

        Args:
            gl: Raw game line data from API
            sport: Sport information

        Returns:
            OvertimeGame object or None if parsing fails.
        """
        try:
            # Parse game datetime
            game_date = ""
            game_time = ""
            if gl.get("GameDateTime"):
                try:
                    match = re.search(r"/Date\((\d+)\)/", gl["GameDateTime"])
                    if match:
                        timestamp = int(match.group(1)) / 1000
                        dt = datetime.fromtimestamp(timestamp)
                        game_date = dt.strftime("%Y-%m-%d")
                        game_time = dt.strftime("%I:%M %p").lstrip("0")
                except Exception:
                    pass

            # Skip half/quarter lines - only process full game lines
            period_desc = gl.get("PeriodDescription", "")
            if period_desc and period_desc != "Game":
                return None

            # Extract team names (API uses Team1ID/Team2ID, not Team1/Team2)
            away_team = gl.get("Team1ID", "").strip()
            home_team = gl.get("Team2ID", "").strip()

            if not away_team or not home_team:
                return None

            # Extract spread (negative = home team favored)
            # Spread1 is for Team1 (away), Spread2 is for Team2 (home)
            spread = None
            spread_away_odds = None
            spread_home_odds = None

            if gl.get("Spread2") is not None:
                spread = float(gl["Spread2"])  # Home team spread
                spread_away_odds = gl.get("SpreadAdj1") or gl.get(
                    "OrigSpreadAdj1"
                )
                spread_home_odds = gl.get("SpreadAdj2") or gl.get(
                    "OrigSpreadAdj2"
                )

            # Extract total
            total = None
            over_odds = None
            under_odds = None

            if gl.get("TotalPoints1") is not None:
                total = float(gl["TotalPoints1"])
                over_odds = gl.get("TtlPtsAdj1") or gl.get("OrigTtlPtsAdj1")
                under_odds = gl.get("TtlPtsAdj2") or gl.get("OrigTtlPtsAdj2")

            # Extract moneylines
            away_ml = gl.get("MoneyLine1")
            home_ml = gl.get("MoneyLine2")

            return OvertimeGame(
                game_num=gl.get("GameNum", 0),
                time=game_time,
                game_date=game_date,
                away_team=away_team,
                home_team=home_team,
                away_rot=gl.get("Team1RotNum", 0),
                home_rot=gl.get("Team2RotNum", 0),
                spread=spread,
                spread_away_odds=spread_away_odds,
                spread_home_odds=spread_home_odds,
                total=total,
                over_odds=over_odds,
                under_odds=under_odds,
                away_ml=away_ml,
                home_ml=home_ml,
                sport_type=sport.sport_type,
                sport_subtype=sport.sport_subtype,
            )

        except Exception as e:
            logger.debug(f"Error parsing game line: {e}")
            return None


async def fetch_overtime_lines(
    target_date: date | None = None,
    include_extra: bool = True,
) -> list[dict]:
    """Fetch college basketball lines from overtime.ag API.

    Args:
        target_date: Date to filter games (default: all available)
        include_extra: Include "College Extra" games

    Returns:
        List of game dictionaries with betting lines.
    """
    async with OvertimeAPIClient() as client:
        games = await client.get_college_basketball_lines(
            include_extra=include_extra
        )

        # Filter by date if specified
        if target_date:
            date_str = target_date.strftime("%Y-%m-%d")
            games = [g for g in games if g.game_date == date_str]

        return [g.to_dict() for g in games]


def save_vegas_lines(
    games: list[dict],
    date_str: str,
    output_path: Path | None = None,
) -> str:
    """Save games to vegas_lines.json format.

    Args:
        games: List of game dictionaries
        date_str: Date string (YYYY-MM-DD)
        output_path: Output file path (default: data/vegas_lines.json)

    Returns:
        Path to saved file.
    """
    if output_path is None:
        output_path = (
            Path(__file__).parent.parent.parent.parent
            / "data"
            / "vegas_lines.json"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "source": "overtime.ag",
        "date": date_str,
        "scraped_at": datetime.now().isoformat(),
        "games": games,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return str(output_path)


# CLI entry point
if __name__ == "__main__":
    import argparse

    async def main():
        parser = argparse.ArgumentParser(description="Overtime.ag API Client")
        parser.add_argument(
            "--date", "-d", type=str, help="Target date (YYYY-MM-DD)"
        )
        parser.add_argument(
            "--sports", action="store_true", help="List available sports"
        )
        parser.add_argument(
            "--no-extra",
            action="store_true",
            help="Exclude College Extra games",
        )
        parser.add_argument(
            "--output", "-o", type=str, help="Output file path"
        )

        args = parser.parse_args()

        print("=" * 70)
        print("OVERTIME.AG API CLIENT")
        print("=" * 70)

        async with OvertimeAPIClient() as client:
            if args.sports:
                # List available sports
                sports = await client.get_sports()
                print(f"\nFound {len(sports)} available sports:\n")
                for s in sports:
                    status = "✓" if s.active else "✗"
                    print(
                        f"  {status} {s.sport_type} - {s.sport_subtype} (ID: {s.sport_subtype_id})"
                    )
                return

            # Get college basketball lines
            print("\nFetching college basketball lines...")

            games = await client.get_college_basketball_lines(
                include_extra=not args.no_extra
            )

            if not games:
                print("\n⚠️  No games found")
                return

            # Filter by date if specified
            if args.date:
                games = [g for g in games if g.game_date == args.date]
                if not games:
                    print(f"\n⚠️  No games found for {args.date}")
                    return

            print(f"\n✅ Found {len(games)} games\n")

            # Print games
            print(f"{'MATCHUP':<45} {'SPREAD':>10} {'TOTAL':>8} {'ML':>12}")
            print("-" * 80)

            for g in games[:20]:  # Show first 20
                matchup = f"{g.away_team[:20]} @ {g.home_team[:20]}"
                spread = f"{g.spread:+.1f}" if g.spread else "N/A"
                total = f"{g.total}" if g.total else "N/A"
                ml = f"{g.away_ml}/{g.home_ml}" if g.away_ml else "N/A"
                print(f"{matchup:<45} {spread:>10} {total:>8} {ml:>12}")

            if len(games) > 20:
                print(f"\n... and {len(games) - 20} more games")

            # Save to file
            if args.output or True:  # Always save
                target_date = args.date or date.today().strftime("%Y-%m-%d")
                output_path = Path(args.output) if args.output else None

                game_dicts = [g.to_dict() for g in games]
                saved_path = save_vegas_lines(
                    game_dicts, target_date, output_path
                )
                print(f"\n📁 Saved to: {saved_path}")

    asyncio.run(main())
