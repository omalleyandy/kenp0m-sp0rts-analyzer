#!/usr/bin/env python3
"""Backfill KenPom efficiency & tempo data from 2015-2024.

Scrapes kenpom.com/summary.php for each season from 2015 through 2024
and populates the efficiency_tempo table in the database.
"""

import argparse
import logging
import os
import sqlite3
import time
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, Page

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class KenPomSummaryScraper:
    """Scraper for KenPom summary page (efficiency & tempo data)."""

    BASE_URL = "https://kenpom.com"
    SUMMARY_URL = "https://kenpom.com/summary.php"

    def __init__(self, email: str | None = None, password: str | None = None):
        """Initialize scraper with KenPom credentials.

        Args:
            email: KenPom account email
            password: KenPom account password
        """
        self.email = email or os.getenv("KENPOM_EMAIL")
        self.password = password or os.getenv("KENPOM_PASSWORD")

        if not self.email or not self.password:
            raise ValueError(
                "KenPom credentials required. Set KENPOM_EMAIL and KENPOM_PASSWORD"
            )

    def scrape_season(self, year: int) -> list[dict]:
        """Scrape summary data for a specific season.

        Args:
            year: Season year (e.g., 2024 for 2023-24 season)

        Returns:
            List of team data dictionaries
        """
        teams_data = []

        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(
                headless=False,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                ],
            )

            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080},
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

                logger.info(f"Scraping summary data for {year} season...")
                page.goto(f"{self.SUMMARY_URL}?y={year}")
                page.wait_for_timeout(3000)

                # Check for Cloudflare
                if page.query_selector("text=Verify you are human"):
                    logger.warning("Cloudflare challenge - waiting...")
                    page.wait_for_timeout(10000)

                # Parse summary table
                teams_data = self._parse_summary_table(page, year)

                logger.info(f"Scraped {len(teams_data)} teams for {year}")

            finally:
                browser.close()

        return teams_data

    def _login(self, page: Page) -> bool:
        """Login to KenPom."""
        try:
            page.goto("https://kenpom.com/")
            page.wait_for_timeout(2000)

            page.wait_for_selector('input[placeholder="E-mail"]', timeout=15000)

            page.fill('input[placeholder="E-mail"]', self.email)
            page.wait_for_timeout(500)
            page.fill('input[placeholder="Password"]', self.password)
            page.wait_for_timeout(500)

            page.click('input[value="Login!"]')
            page.wait_for_timeout(4000)

            # Check if login succeeded
            login_form = page.query_selector('input[value="Login!"]')
            if not login_form:
                logger.info("Login successful")
                return True

            return False

        except Exception as e:
            logger.error(f"Login error: {e}")
            return False

    def _parse_summary_table(self, page: Page, year: int) -> list[dict]:
        """Parse the summary statistics table.

        Args:
            page: Playwright page with summary data
            year: Season year

        Returns:
            List of team data dictionaries
        """
        teams = []

        try:
            # Wait for table
            page.wait_for_selector("table", timeout=10000)

            # Get all table rows
            rows = page.query_selector_all("table tbody tr")
            logger.info(f"Found {len(rows)} team rows")

            for i, row in enumerate(rows, 1):
                try:
                    team_data = self._parse_team_row(row, year)
                    if team_data:
                        teams.append(team_data)

                    if i % 50 == 0:
                        logger.info(f"  Parsed {i} teams...")

                except Exception as e:
                    logger.debug(f"Error parsing row {i}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error parsing summary table: {e}")

        return teams

    def _parse_team_row(self, row, year: int) -> dict | None:
        """Parse a single team row from summary table.

        Args:
            row: Playwright row element
            year: Season year

        Returns:
            Dictionary with team data
        """
        try:
            cols = row.query_selector_all("td")

            if len(cols) < 5:
                return None

            def text(idx):
                """Get text from column."""
                if idx < len(cols):
                    return cols[idx].inner_text().strip()
                return ""

            def float_val(idx):
                """Parse float from column."""
                try:
                    val = text(idx)
                    return float(val) if val and val != "-" else None
                except ValueError:
                    return None

            def int_val(idx):
                """Parse int from column."""
                try:
                    val = text(idx)
                    return int(val) if val and val != "-" else None
                except ValueError:
                    return None

            # Column 0: Rank
            rank = int_val(0)

            # Column 1: Team name
            team_name = text(1)

            # Column 2: Conference
            conference = text(2)

            # Column 3: Record (W-L)
            record = text(3)
            wins, losses = 0, 0
            if record and "-" in record:
                try:
                    w, l = record.split("-")
                    wins = int(w)
                    losses = int(l)
                except ValueError:
                    pass

            # Column 4: AdjEM
            adj_em = float_val(4)

            # Column 5: AdjO
            adj_oe = float_val(5)

            # Column 6: AdjD
            adj_de = float_val(6)

            # Column 7: AdjT
            adj_tempo = float_val(7)

            # Column 8: Luck
            luck = float_val(8)

            # Column 9: SOS AdjEM
            sos_adj_em = float_val(9)

            # Column 10: OppO (SOS Offense)
            sos_adj_oe = float_val(10)

            # Column 11: OppD (SOS Defense)
            sos_adj_de = float_val(11)

            # Column 12: NCSOS
            nc_sos = float_val(12)

            if not team_name or not adj_em:
                return None

            return {
                "team_name": team_name,
                "conference": conference,
                "season": year,
                "snapshot_date": date.today(),
                "rank_overall": rank,
                "adj_em": adj_em,
                "adj_oe": adj_oe,
                "adj_de": adj_de,
                "adj_tempo": adj_tempo,
                "luck": luck,
                "sos_adj_em": sos_adj_em,
                "sos_adj_oe": sos_adj_oe,
                "sos_adj_de": sos_adj_de,
                "nc_sos_adj_em": nc_sos,
                "wins": wins,
                "losses": losses,
                "win_pct": wins / (wins + losses) if (wins + losses) > 0 else None,
            }

        except Exception as e:
            logger.debug(f"Error parsing team row: {e}")
            return None


def populate_database(teams_data: list[dict], db_path: str):
    """Populate efficiency_tempo table with scraped data.

    Args:
        teams_data: List of team data dictionaries
        db_path: Path to SQLite database
    """
    logger.info(f"Populating database at {db_path}...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create schema if needed
    schema_file = (
        Path(__file__).parent.parent
        / "src/kenp0m_sp0rts_analyzer/kenpom/efficiency_tempo_schema.sql"
    )
    if schema_file.exists():
        with open(schema_file) as f:
            conn.executescript(f.read())
        logger.info("Schema created/updated")

    inserted = 0
    updated = 0

    for team in teams_data:
        # Check if record exists
        cursor.execute(
            """
            SELECT efficiency_tempo_id FROM efficiency_tempo
            WHERE team_name = ? AND season = ? AND snapshot_date = ?
            """,
            (team["team_name"], team["season"], team["snapshot_date"]),
        )

        existing = cursor.fetchone()

        if existing:
            # Update existing
            cursor.execute(
                """
                UPDATE efficiency_tempo SET
                    conference = ?,
                    rank_overall = ?,
                    adj_em = ?,
                    adj_oe = ?,
                    adj_de = ?,
                    adj_tempo = ?,
                    luck = ?,
                    sos_adj_em = ?,
                    sos_adj_oe = ?,
                    sos_adj_de = ?,
                    nc_sos_adj_em = ?,
                    wins = ?,
                    losses = ?,
                    win_pct = ?,
                    scraped_at = CURRENT_TIMESTAMP
                WHERE efficiency_tempo_id = ?
                """,
                (
                    team.get("conference"),
                    team.get("rank_overall"),
                    team.get("adj_em"),
                    team.get("adj_oe"),
                    team.get("adj_de"),
                    team.get("adj_tempo"),
                    team.get("luck"),
                    team.get("sos_adj_em"),
                    team.get("sos_adj_oe"),
                    team.get("sos_adj_de"),
                    team.get("nc_sos_adj_em"),
                    team.get("wins"),
                    team.get("losses"),
                    team.get("win_pct"),
                    existing[0],
                ),
            )
            updated += 1
        else:
            # Insert new
            cursor.execute(
                """
                INSERT INTO efficiency_tempo (
                    team_name, conference, season, snapshot_date,
                    rank_overall, adj_em, adj_oe, adj_de, adj_tempo,
                    luck, sos_adj_em, sos_adj_oe, sos_adj_de, nc_sos_adj_em,
                    wins, losses, win_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    team["team_name"],
                    team.get("conference"),
                    team["season"],
                    team["snapshot_date"],
                    team.get("rank_overall"),
                    team.get("adj_em"),
                    team.get("adj_oe"),
                    team.get("adj_de"),
                    team.get("adj_tempo"),
                    team.get("luck"),
                    team.get("sos_adj_em"),
                    team.get("sos_adj_oe"),
                    team.get("sos_adj_de"),
                    team.get("nc_sos_adj_em"),
                    team.get("wins"),
                    team.get("losses"),
                    team.get("win_pct"),
                ),
            )
            inserted += 1

    conn.commit()
    conn.close()

    logger.info(f"Database updated: {inserted} inserted, {updated} updated")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Backfill KenPom efficiency & tempo data (2015-2024)"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
        help="Start year for backfill (default: 2015)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="End year for backfill (default: 2024)",
    )
    parser.add_argument(
        "--db", type=str, default="data/kenpom.db", help="Database path"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="Delay between seasons in seconds (default: 5)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 80)
    logger.info("KENPOM EFFICIENCY & TEMPO BACKFILL (2015-2024)")
    logger.info("=" * 80)
    logger.info(f"Seasons: {args.start_year} through {args.end_year}")
    logger.info(f"Database: {args.db}")
    logger.info(f"Delay between seasons: {args.delay}s")
    logger.info("=" * 80)

    scraper = KenPomSummaryScraper()

    total_teams = 0
    seasons_scraped = 0

    for year in range(args.start_year, args.end_year + 1):
        try:
            logger.info(f"\n[{year}] Starting season scrape...")

            # Scrape season data
            teams_data = scraper.scrape_season(year)

            if teams_data:
                # Populate database
                populate_database(teams_data, args.db)

                total_teams += len(teams_data)
                seasons_scraped += 1

                logger.info(f"[{year}] Complete - {len(teams_data)} teams scraped")

                # Delay between seasons (respectful scraping)
                if year < args.end_year:
                    logger.info(f"Waiting {args.delay}s before next season...")
                    time.sleep(args.delay)
            else:
                logger.warning(f"[{year}] No data scraped")

        except Exception as e:
            logger.error(f"[{year}] Error during scrape: {e}", exc_info=True)
            continue

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Seasons scraped: {seasons_scraped}")
    logger.info(f"Total teams: {total_teams}")
    logger.info(f"Database: {args.db}")
    logger.info("=" * 80)


if __name__ == "__main__":
    exit(main())
