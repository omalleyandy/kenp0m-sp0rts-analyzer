"""KenPom Player Stats Scraper using Playwright.

Scrapes player statistics from kenpom.com/playerstats.php using browser automation
to handle authentication and JavaScript rendering.
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, Page

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class PlayerStats:
    """Player statistics from KenPom."""

    # Identity
    rank: int
    player_name: str
    team_name: str
    conference: str
    position: str
    class_year: str
    height: str

    # Season Context
    season: int
    games_played: int

    # Tempo-Free Stats (per 100 possessions)
    ortg: float  # Offensive Rating
    pct_poss: float  # Percent of possessions used

    # Shooting
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
    minutes_pct: float  # % of team minutes played


class KenPomPlayerScraper:
    """Scraper for KenPom player statistics using Playwright."""

    BASE_URL = "https://kenpom.com"
    LOGIN_URL = "https://kenpom.com/index.php"
    PLAYER_STATS_URL = "https://kenpom.com/playerstats.php"

    def __init__(self, email: str | None = None, password: str | None = None):
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
                "environment variables."
            )

    def scrape_player_stats(
        self, year: int = 2025, max_players: int = 200
    ) -> list[PlayerStats]:
        """Scrape player statistics using Playwright browser automation.

        Args:
            year: Season year (e.g., 2025 for 2024-25 season)
            max_players: Maximum number of players to scrape

        Returns:
            List of PlayerStats objects
        """
        players = []

        with sync_playwright() as p:
            # Launch browser (headless mode)
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()

            try:
                # Login
                logger.info("Logging in to KenPom...")
                if not self._login(page):
                    raise RuntimeError("Failed to login to KenPom")

                logger.info("Login successful")

                # Navigate to player stats
                logger.info(f"Navigating to player stats (year={year})...")
                page.goto(f"{self.PLAYER_STATS_URL}?y={year}")
                page.wait_for_load_state("networkidle")

                # Parse player table
                players = self._parse_player_table(page, year, max_players)

            finally:
                browser.close()

        logger.info(f"Scraped {len(players)} players")
        return players

    def _login(self, page: Page) -> bool:
        """Login to KenPom.

        Args:
            page: Playwright page object

        Returns:
            True if login successful
        """
        try:
            # Navigate to login page
            page.goto(self.LOGIN_URL)
            page.wait_for_load_state("networkidle")

            # Fill login form
            page.fill('input[name="email"]', self.email)
            page.fill('input[name="password"]', self.password)

            # Click login button
            page.click('input[type="submit"]')
            page.wait_for_load_state("networkidle")

            # Check if login was successful by looking for logout link or user menu
            # KenPom shows user's name/email in top right when logged in
            if page.query_selector('a[href*="logout"]') or "@" in page.content():
                return True

            logger.error("Login failed - could not verify successful login")
            return False

        except Exception as e:
            logger.error(f"Login error: {e}")
            return False

    def _parse_player_table(
        self, page: Page, year: int, max_players: int
    ) -> list[PlayerStats]:
        """Parse the player stats table from the page.

        Args:
            page: Playwright page with player stats loaded
            year: Season year
            max_players: Maximum number of players to parse

        Returns:
            List of PlayerStats objects
        """
        players = []

        # Wait for table to load
        page.wait_for_selector("#playerstats", timeout=10000)

        # Get all table rows
        rows = page.query_selector_all("#playerstats tbody tr")
        logger.info(f"Found {len(rows)} player rows in table")

        for i, row in enumerate(rows[:max_players]):
            try:
                player = self._parse_player_row(row, year, i + 1)
                if player:
                    players.append(player)

                if (i + 1) % 50 == 0:
                    logger.info(f"Parsed {i + 1} players...")

            except Exception as e:
                logger.error(f"Error parsing row {i + 1}: {e}")
                continue

        return players

    def _parse_player_row(self, row, year: int, rank: int) -> PlayerStats | None:
        """Parse a single player table row.

        Args:
            row: Playwright element handle for table row
            year: Season year
            rank: Player's rank (1-based)

        Returns:
            PlayerStats object or None if parsing fails
        """
        try:
            cols = row.query_selector_all("td")
            if len(cols) < 20:
                return None

            def text(idx):
                """Get text from column index."""
                return cols[idx].inner_text().strip()

            def float_val(idx):
                """Get float value from column."""
                try:
                    val = text(idx)
                    return float(val) if val and val != "-" else 0.0
                except ValueError:
                    return 0.0

            def int_val(idx):
                """Get int value from column."""
                try:
                    val = text(idx)
                    return int(val) if val and val != "-" else 0
                except ValueError:
                    return 0

            # Column 1: Player name, Team
            player_text = text(1)
            if "," not in player_text:
                return None

            parts = player_text.split(",", 1)
            player_name = parts[0].strip()
            team_name = parts[1].strip() if len(parts) > 1 else ""

            # Column 2: Conference
            conference = text(2)

            # Column 3: Height-Class-Pos (e.g., "6-5 Jr. G")
            hcp_text = text(3)
            hcp_parts = hcp_text.split()
            height = hcp_parts[0] if len(hcp_parts) > 0 else ""
            class_year = hcp_parts[1].rstrip(".") if len(hcp_parts) > 1 else ""
            position = hcp_parts[2] if len(hcp_parts) > 2 else ""

            return PlayerStats(
                rank=rank,
                player_name=player_name,
                team_name=team_name,
                conference=conference,
                position=position,
                class_year=class_year,
                height=height,
                season=year,
                games_played=int_val(4),  # GP
                ortg=float_val(5),  # ORtg
                pct_poss=float_val(6),  # %Poss
                efg_pct=float_val(7),  # eFG%
                ts_pct=float_val(8),  # TS%
                three_pt_rate=float_val(9),  # 3PR
                ft_rate=float_val(10),  # FTR
                ppg=float_val(11),  # PPG
                rpg=float_val(12),  # RPG
                apg=float_val(13),  # APG
                ast_rate=float_val(14),  # ARate
                to_rate=float_val(15),  # TORate
                or_pct=float_val(16),  # OR%
                dr_pct=float_val(17),  # DR%
                fc_per_40=float_val(18),  # FC/40
                minutes_pct=float_val(19),  # Min%
            )

        except Exception as e:
            logger.error(f"Error parsing player row (rank {rank}): {e}")
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
                "rank": p.rank,
                "name": p.player_name,
                "team": p.team_name,
                "conference": p.conference,
                "position": p.position,
                "class_year": p.class_year,
                "height": p.height,
                "season": p.season,
                "games_played": p.games_played,
                "ortg": p.ortg,
                "pct_poss": p.pct_poss,
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
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Scrape KenPom player statistics")
    parser.add_argument("--year", type=int, default=2025, help="Season year")
    parser.add_argument(
        "--max-players", type=int, default=200, help="Maximum players to scrape"
    )
    parser.add_argument(
        "--output", type=str, default="data/kenpom_players.csv", help="Output CSV"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Scrape players
    scraper = KenPomPlayerScraper()
    players = scraper.scrape_player_stats(year=args.year, max_players=args.max_players)

    # Save to CSV
    scraper.save_to_csv(players, args.output)

    # Print summary
    if players:
        df = scraper.to_dataframe(players)
        print(f"\n{'='*80}")
        print(f"SCRAPING COMPLETE")
        print(f"{'='*80}")
        print(f"Total Players: {len(players)}")
        print(f"Season: {args.year}")
        print(f"Output: {args.output}\n")
        print("Top 10 Players by ORtg:")
        print(df.nlargest(10, "ortg")[["rank", "name", "team", "ppg", "ortg"]])


if __name__ == "__main__":
    main()
