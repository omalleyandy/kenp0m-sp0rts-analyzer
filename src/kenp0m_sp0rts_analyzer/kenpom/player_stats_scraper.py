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
            # Launch browser (non-headless to avoid Cloudflare detection)
            browser = p.chromium.launch(
                headless=False,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                ],
            )

            # More realistic browser context
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
                timezone_id="America/Los_Angeles",
            )

            # Remove automation indicators
            context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)

            page = context.new_page()

            try:
                # Login
                logger.info("Logging in to KenPom...")
                if not self._login(page):
                    raise RuntimeError("Failed to login to KenPom")

                logger.info("Login successful")

                # Navigate to player stats
                logger.info(f"Navigating to player stats (year={year})...")
                page.goto(f"{self.PLAYER_STATS_URL}?y={year}", wait_until="domcontentloaded")

                # Wait longer for Cloudflare challenge to potentially resolve
                logger.info("Waiting for page to fully load (Cloudflare check)...")
                page.wait_for_timeout(5000)

                # Check for Cloudflare challenge
                cloudflare_challenge = page.query_selector("text=Verify you are human")
                if cloudflare_challenge:
                    logger.warning("Cloudflare challenge detected - waiting for manual completion...")
                    logger.warning("Please complete the CAPTCHA in the browser window...")
                    # Wait up to 60 seconds for user to complete challenge
                    try:
                        page.wait_for_selector("#playerstats", timeout=60000)
                        logger.info("Cloudflare challenge completed - table loaded")
                    except Exception:
                        logger.error("Timeout waiting for Cloudflare challenge completion")
                        raise RuntimeError("Cloudflare challenge not completed in time")

                # Debug: Save screenshot
                page.screenshot(path="data/kenpom_playerstats_debug.png")
                logger.info("Saved player stats page screenshot")

                # Log current URL
                logger.info(f"Current URL: {page.url}")

                # Scrape all statistical categories
                players = self._scrape_all_categories(page, year, max_players)

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
            # Navigate to home page (has login form)
            logger.info("Navigating to KenPom home page...")
            page.goto("https://kenpom.com/", wait_until="domcontentloaded")

            # Wait for potential Cloudflare challenge
            page.wait_for_timeout(2000)

            # Check for Cloudflare on homepage
            cloudflare_challenge = page.query_selector("text=Verify you are human")
            if cloudflare_challenge:
                logger.warning("Cloudflare challenge on homepage - waiting...")
                page.wait_for_timeout(10000)  # Give it time to auto-resolve

            # Wait for login form to be visible
            page.wait_for_selector('input[placeholder="E-mail"]', timeout=15000)

            # Fill login form with correct selectors (human-like delays)
            logger.info("Filling login form...")
            page.fill('input[placeholder="E-mail"]', self.email)
            page.wait_for_timeout(500)  # Human-like delay
            page.fill('input[placeholder="Password"]', self.password)
            page.wait_for_timeout(500)

            # Click login button
            logger.info("Clicking login button...")
            page.click('input[value="Login!"]')

            # Wait for page to reload after login
            page.wait_for_timeout(4000)

            # Debug: Save screenshot
            page.screenshot(path="data/kenpom_login_debug.png")
            logger.info("Saved debug screenshot to data/kenpom_login_debug.png")

            # Check for login error message
            error_div = page.query_selector(".alert-danger")
            if error_div:
                error_text = error_div.inner_text()
                logger.error(f"Login error message: {error_text}")
                return False

            # Check if we're still on the same page (login failed)
            current_url = page.url
            logger.info(f"Current URL after login: {current_url}")

            # Simpler check: look for any link that contains "logout" (case insensitive)
            logout_link = page.query_selector('a[href*="logout"], a[href*="Logout"]')
            if logout_link:
                logger.info("Login successful - logout link found")
                return True

            # Alternative: check if login form is gone (successful login usually hides it)
            login_form = page.query_selector('input[value="Login!"]')
            if not login_form:
                logger.info("Login successful - login form disappeared")
                return True

            # Check for player stats link
            player_stats_link = page.query_selector('a[href="playerstats.php"]')
            if player_stats_link:
                logger.info("Login successful - player stats link found")
                return True

            logger.error("Login verification failed - no indicators of successful login found")
            logger.error(f"Current page title: {page.title()}")
            return False

        except Exception as e:
            logger.error(f"Login error: {e}")
            return False

    def _scrape_all_categories(
        self, page: Page, year: int, max_players: int
    ) -> list[PlayerStats]:
        """Scrape player stats across all statistical categories.

        Args:
            page: Playwright page with player stats loaded
            year: Season year
            max_players: Maximum number of players to scrape

        Returns:
            List of PlayerStats objects with data from all categories
        """
        # Statistical categories to scrape in order
        stat_categories = [
            "ORtg",    # Offensive Rating
            "%Min",    # Percentage of Minutes
            "eFG%",    # Effective Field Goal %
            "%Poss",   # Percentage of Possessions
            "%Shots",  # Percentage of Shots
            "OR%",     # Offensive Rebounding %
            "DR%",     # Defensive Rebounding %
            "TO%",     # Turnover %
            "ARate",   # Assist Rate
            "Blk%",    # Block %
            "FTRate",  # Free Throw Rate
            "Stl%",    # Steal %
            "TS%",     # True Shooting %
            "FC/40",   # Fouls Committed per 40
            "FD/40",   # Fouls Drawn per 40
            "2P%",     # 2-Point %
            "3P%",     # 3-Point %
            "FT%",     # Free Throw %
        ]

        # Dictionary to store all player data by (name, team) key
        players_dict = {}

        logger.info(f"Scraping {len(stat_categories)} statistical categories...")

        for i, stat in enumerate(stat_categories, 1):
            try:
                logger.info(f"[{i}/{len(stat_categories)}] Scraping {stat} leaderboard...")

                # Click on the stat category link
                stat_link = page.query_selector(f'a:has-text("{stat}")')
                if stat_link:
                    stat_link.click()
                    page.wait_for_timeout(2000)  # Wait for table to reload

                    # Get stat values from column 3
                    stat_values = self._extract_stat_column(page, max_players)

                    # Parse the table for player identities
                    category_players = self._parse_player_table(page, year, max_players)

                    # Merge players with their stats
                    for idx, player in enumerate(category_players):
                        key = (player.player_name, player.team_name)

                        # Get stat value for this player
                        stat_val = stat_values[idx] if idx < len(stat_values) else 0.0

                        if key not in players_dict:
                            # New player - add to dict
                            players_dict[key] = player

                        # Map stat to appropriate field
                        existing = players_dict[key]
                        self._set_stat_value(existing, stat, stat_val)

                    logger.info(f"  -> Found {len(category_players)} players in {stat}")
                else:
                    logger.warning(f"  -> Could not find link for {stat}")

            except Exception as e:
                logger.error(f"Error scraping {stat} category: {e}")
                continue

        players = list(players_dict.values())
        logger.info(f"Total unique players collected: {len(players)}")

        return players[:max_players]

    def _extract_stat_column(self, page: Page, max_players: int) -> list[float]:
        """Extract stat values from column 3 of the current table.

        Args:
            page: Playwright page with stats table
            max_players: Maximum number of rows to extract

        Returns:
            List of stat values
        """
        stat_values = []

        try:
            rows = page.query_selector_all("table tbody tr")

            for row in rows[:max_players]:
                cols = row.query_selector_all("td")
                if len(cols) >= 4:
                    # Column 3 has the stat value
                    stat_text = cols[3].inner_text().strip()

                    # Parse float (handle values like "129.7 (29.0)")
                    try:
                        stat_text = stat_text.split("(")[0].strip()
                        stat_val = float(stat_text) if stat_text and stat_text != "-" else 0.0
                        stat_values.append(stat_val)
                    except ValueError:
                        stat_values.append(0.0)

        except Exception as e:
            logger.warning(f"Error extracting stat column: {e}")

        return stat_values

    def _set_stat_value(self, player: PlayerStats, stat_name: str, value: float):
        """Map a stat value to the appropriate PlayerStats field.

        Args:
            player: PlayerStats object to update
            stat_name: Name of the stat (e.g., "ORtg", "%Min")
            value: Stat value to set
        """
        # Map KenPom stat names to PlayerStats fields
        stat_mapping = {
            "ORtg": "ortg",
            "%Min": "minutes_pct",
            "eFG%": "efg_pct",
            "%Poss": "pct_poss",
            "%Shots": "three_pt_rate",  # Approximate mapping
            "OR%": "or_pct",
            "DR%": "dr_pct",
            "TO%": "to_rate",
            "ARate": "ast_rate",
            "Blk%": None,  # Not in PlayerStats
            "FTRate": "ft_rate",
            "Stl%": None,  # Not in PlayerStats
            "TS%": "ts_pct",
            "FC/40": "fc_per_40",
            "FD/40": None,  # Not in PlayerStats
            "2P%": None,  # Not in PlayerStats (can derive from eFG)
            "3P%": None,  # Not in PlayerStats
            "FT%": None,  # Not in PlayerStats
        }

        field_name = stat_mapping.get(stat_name)
        if field_name:
            setattr(player, field_name, value)

    def _parse_player_table(
        self, page: Page, year: int, max_players: int
    ) -> list[PlayerStats]:
        """Parse the player stats table from the current page.

        Args:
            page: Playwright page with player stats loaded
            year: Season year
            max_players: Maximum number of players to parse

        Returns:
            List of PlayerStats objects
        """
        players = []

        # Wait for any table to load (more flexible selector)
        try:
            page.wait_for_selector("table tbody tr", timeout=5000)
        except Exception:
            logger.warning("No table rows found on page")
            return players

        # Get all table rows (use flexible selector)
        rows = page.query_selector_all("table tbody tr")
        logger.info(f"Found {len(rows)} rows in table")

        for i, row in enumerate(rows[:max_players]):
            try:
                player = self._parse_player_row(row, year, i + 1)
                if player:
                    players.append(player)

            except Exception as e:
                logger.debug(f"Error parsing row {i + 1}: {e}")
                continue

        logger.info(f"Successfully parsed {len(players)} players from table")
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

            # Need at least 4 columns: rank, player, team, stat
            if len(cols) < 4:
                return None

            def text(idx):
                """Get text from column index."""
                if idx < len(cols):
                    return cols[idx].inner_text().strip()
                return ""

            def float_val(val_str):
                """Parse float from string."""
                try:
                    # Handle values like "129.7 (29.0)" - extract first number
                    val_str = val_str.split("(")[0].strip()
                    return float(val_str) if val_str and val_str != "-" else 0.0
                except ValueError:
                    return 0.0

            # Extract player info from table
            # Col 0: Rank
            # Col 1: Player name
            # Col 2: Team
            # Col 3: Stat value (or made for shooting %)
            # Col 4: Height (or attempted for shooting %)
            # Col 5: Weight (or percentage for shooting %)
            # Col 6: Class (or height for shooting %)

            player_name = text(1)
            team_name = text(2)

            if not player_name or not team_name:
                return None

            # Determine table type by column count
            if len(cols) >= 9:  # Shooting percentage table (2P%, 3P%, FT%)
                height = text(6)
                weight_str = text(7)
                class_year = text(8)
            else:  # Regular stat table (7 columns)
                height = text(4)
                weight_str = text(5)
                class_year = text(6)

            # Parse weight
            try:
                weight = int(weight_str) if weight_str and weight_str != "-" else 0
            except ValueError:
                weight = 0

            # Parse height into inches
            height_inches = 0
            if height and "-" in height:
                try:
                    feet, inches = height.split("-")
                    height_inches = int(feet) * 12 + int(inches)
                except ValueError:
                    pass

            # Clean up class year
            class_year = class_year.rstrip(".")

            # Create basic PlayerStats with identity fields
            # Stats will be filled in by category-specific parsing
            return PlayerStats(
                rank=rank,
                player_name=player_name,
                team_name=team_name,
                conference="",  # Not available in player stats tables
                position="",     # Not available in player stats tables
                class_year=class_year,
                height=height,
                season=year,
                games_played=0,
                ortg=0.0,
                pct_poss=0.0,
                efg_pct=0.0,
                ts_pct=0.0,
                three_pt_rate=0.0,
                ft_rate=0.0,
                ppg=0.0,
                rpg=0.0,
                apg=0.0,
                ast_rate=0.0,
                to_rate=0.0,
                or_pct=0.0,
                dr_pct=0.0,
                fc_per_40=0.0,
                minutes_pct=0.0,
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
