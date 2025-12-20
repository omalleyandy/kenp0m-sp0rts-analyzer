"""KenPom Player Stats Scraper.

Scrapes player statistics from kenpom.com/playerstats.php to populate
the player impact tracking system.

Since KenPom API doesn't have a player stats endpoint, we scrape the
HTML table directly while respecting rate limits and using our API credentials.
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime

import httpx
import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class PlayerStats:
    """Player statistics from KenPom."""

    # Identity
    player_name: str
    team_name: str
    team_id: int | None
    position: str
    class_year: str
    height: str

    # Season Context
    season: int
    games_played: int

    # Tempo-Free Stats (per 100 possessions)
    ortg: float  # Offensive Rating
    pct_poss: float  # Percent of possessions used
    ppp: float  # Points per possession (shooting)

    # Shooting Percentages
    efg_pct: float  # Effective FG%
    ts_pct: float  # True Shooting%
    three_pt_rate: float  # % of FGA that are 3s
    ft_rate: float  # FT Rate (FTA/FGA)

    # Per-Game Stats
    ppg: float
    rpg: float
    apg: float

    # Advanced
    ast_rate: float  # Assist Rate
    to_rate: float  # Turnover Rate
    or_pct: float  # Offensive Rebound %
    dr_pct: float  # Defensive Rebound %
    fc_per_40: float  # Fouls committed per 40 min

    # Minutes
    minutes_pct: float  # % of team minutes played


class KenPomPlayerScraper:
    """Scraper for KenPom player statistics."""

    BASE_URL = "https://kenpom.com"
    LOGIN_URL = "https://kenpom.com/handlers/login_handler.php"
    PLAYER_STATS_URL = "https://kenpom.com/playerstats.php"

    def __init__(
        self, email: str | None = None, password: str | None = None
    ):
        """Initialize scraper with KenPom credentials.

        Args:
            email: KenPom account email (uses KENPOM_EMAIL env var if not provided)
            password: KenPom account password (uses KENPOM_PASSWORD env var)
        """
        self.email = email or os.getenv("KENPOM_EMAIL")
        self.password = password or os.getenv("KENPOM_PASSWORD")

        if not self.email or not self.password:
            raise ValueError(
                "KenPom credentials required. Set KENPOM_EMAIL and KENPOM_PASSWORD "
                "environment variables or pass email/password parameters."
            )

        self.client = httpx.Client(
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36",
            },
            timeout=30.0,
            follow_redirects=True,
        )
        self._logged_in = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close HTTP client."""
        self.client.close()

    def login(self) -> bool:
        """Login to KenPom to establish authenticated session.

        Returns:
            True if login successful, False otherwise

        Raises:
            httpx.HTTPError: If login request fails
        """
        if self._logged_in:
            return True

        logger.info("Logging in to KenPom...")

        # POST login credentials
        login_data = {
            "email": self.email,
            "password": self.password,
            "submit": "Login",
        }

        response = self.client.post(self.LOGIN_URL, data=login_data)
        response.raise_for_status()

        # Check if login was successful by looking for session cookie
        if "PHPSESSID" in self.client.cookies:
            self._logged_in = True
            logger.info("Login successful")
            return True
        else:
            logger.error("Login failed - no session cookie received")
            return False

    def scrape_player_stats(
        self,
        year: int = 2025,
        page: int = 1,
        max_players: int = 200,
    ) -> list[PlayerStats]:
        """Scrape player statistics from KenPom.

        Args:
            year: Season year (e.g., 2025 for 2024-25 season)
            page: Page number to scrape (1 = top 100, 2 = 101-200, etc.)
            max_players: Maximum number of players to scrape

        Returns:
            List of PlayerStats objects

        Raises:
            httpx.HTTPError: If request fails
            ValueError: If unable to parse table
        """
        # Ensure we're logged in
        if not self._logged_in:
            if not self.login():
                raise RuntimeError("Failed to login to KenPom")

        players = []
        current_page = page
        players_scraped = 0

        while players_scraped < max_players:
            logger.info(f"Scraping page {current_page} (year={year})...")

            # Request page
            params = {"y": year, "p": current_page}
            response = self.client.get(self.PLAYER_STATS_URL, params=params)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Find the player stats table
            table = soup.find("table", {"id": "playerstats"})
            if not table:
                logger.warning(f"No player stats table found on page {current_page}")
                break

            # Parse table rows
            tbody = table.find("tbody")
            if not tbody:
                logger.warning("No tbody found in player stats table")
                break

            rows = tbody.find_all("tr")
            if not rows:
                logger.warning(f"No data rows found on page {current_page}")
                break

            page_players = 0
            for row in rows:
                if players_scraped >= max_players:
                    break

                try:
                    player = self._parse_player_row(row, year)
                    if player:
                        players.append(player)
                        players_scraped += 1
                        page_players += 1
                except Exception as e:
                    logger.error(f"Error parsing player row: {e}")
                    continue

            logger.info(
                f"Page {current_page}: Scraped {page_players} players "
                f"({players_scraped}/{max_players} total)"
            )

            # If we got fewer players than expected, we've reached the end
            if page_players == 0:
                break

            # Rate limiting (be respectful)
            if players_scraped < max_players:
                time.sleep(2)  # 2 second delay between pages
                current_page += 1

        logger.info(f"Scraping complete: {len(players)} players retrieved")
        return players

    def _parse_player_row(self, row, year: int) -> PlayerStats | None:
        """Parse a single player table row.

        Args:
            row: BeautifulSoup TR element
            year: Season year

        Returns:
            PlayerStats object or None if parsing fails
        """
        cols = row.find_all("td")
        if len(cols) < 20:  # Ensure we have all columns
            return None

        try:
            # Column 0: Rank (ignore)
            # Column 1: Player name + team
            player_cell = cols[1].get_text(strip=True)
            player_parts = player_cell.split(",")
            if len(player_parts) < 2:
                return None

            player_name = player_parts[0].strip()
            team_name = player_parts[1].strip()

            # Column 2: Conf (ignore)
            # Column 3: Height-Class-Pos (e.g., "6-5 Jr. G")
            hcp = cols[3].get_text(strip=True).split()
            height = hcp[0] if len(hcp) > 0 else ""
            class_year = hcp[1].rstrip(".") if len(hcp) > 1 else ""
            position = hcp[2] if len(hcp) > 2 else ""

            # Extract numeric stats
            def safe_float(col_idx):
                """Extract float from column, return 0 if parsing fails."""
                try:
                    text = cols[col_idx].get_text(strip=True)
                    return float(text) if text else 0.0
                except (ValueError, AttributeError):
                    return 0.0

            def safe_int(col_idx):
                """Extract int from column, return 0 if parsing fails."""
                try:
                    text = cols[col_idx].get_text(strip=True)
                    return int(text) if text else 0
                except (ValueError, AttributeError):
                    return 0

            # Column indices based on KenPom player stats table structure
            # May need adjustment based on actual table layout
            return PlayerStats(
                player_name=player_name,
                team_name=team_name,
                team_id=None,  # Will resolve later from teams table
                position=position,
                class_year=class_year,
                height=height,
                season=year,
                games_played=safe_int(4),  # GP
                ortg=safe_float(5),  # ORtg
                pct_poss=safe_float(6),  # %Poss
                ppp=safe_float(7),  # PPP
                efg_pct=safe_float(8),  # eFG%
                ts_pct=safe_float(9),  # TS%
                three_pt_rate=safe_float(10),  # 3PR
                ft_rate=safe_float(11),  # FTR
                ppg=safe_float(12),  # PPG
                rpg=safe_float(13),  # RPG
                apg=safe_float(14),  # APG
                ast_rate=safe_float(15),  # ARate
                to_rate=safe_float(16),  # TORate
                or_pct=safe_float(17),  # OR%
                dr_pct=safe_float(18),  # DR%
                fc_per_40=safe_float(19),  # FC/40
                minutes_pct=safe_float(20),  # Min%
            )

        except Exception as e:
            logger.error(f"Error parsing player row: {e}")
            return None

    def to_dataframe(self, players: list[PlayerStats]) -> pd.DataFrame:
        """Convert player stats to pandas DataFrame.

        Args:
            players: List of PlayerStats objects

        Returns:
            DataFrame with player statistics
        """
        data = [
            {
                "name": p.player_name,
                "team": p.team_name,
                "position": p.position,
                "class_year": p.class_year,
                "height": p.height,
                "season": p.season,
                "games_played": p.games_played,
                "ortg": p.ortg,
                "pct_poss": p.pct_poss,
                "ppp": p.ppp,
                "efg_pct": p.efg_pct,
                "ts_pct": p.ts_pct,
                "three_pt_rate": p.three_pt_rate,
                "ft_rate": p.ft_rate,
                "ppg": p.ppg,
                "rpg": p.rpg,
                "apg": p.apg,
                "ast_rate": p.ast_rate,
                "to_rate": p.to_rate,
                "or_pct": p.or_pct,
                "dr_pct": p.dr_pct,
                "fc_per_40": p.fc_per_40,
                "minutes_pct": p.minutes_pct,
            }
            for p in players
        ]

        return pd.DataFrame(data)

    def save_to_csv(self, players: list[PlayerStats], filepath: str):
        """Save player stats to CSV file.

        Args:
            players: List of PlayerStats objects
            filepath: Path to save CSV
        """
        df = self.to_dataframe(players)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(players)} players to {filepath}")


def main():
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Scrape KenPom player statistics")
    parser.add_argument("--year", type=int, default=2025, help="Season year")
    parser.add_argument(
        "--max-players", type=int, default=200, help="Maximum players to scrape"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/kenpom_players.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Scrape players
    with KenPomPlayerScraper() as scraper:
        players = scraper.scrape_player_stats(
            year=args.year, max_players=args.max_players
        )

        # Save to CSV
        scraper.save_to_csv(players, args.output)

        # Print summary
        print(f"\n{'='*80}")
        print(f"SCRAPING COMPLETE")
        print(f"{'='*80}")
        print(f"Total Players: {len(players)}")
        print(f"Season: {args.year}")
        print(f"Output: {args.output}")

        if players:
            df = scraper.to_dataframe(players)
            print(f"\nTop 10 Players by ORtg:")
            print(df.nlargest(10, "ortg")[["name", "team", "ppg", "ortg"]])


if __name__ == "__main__":
    main()
