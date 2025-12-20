"""ESPN Historical Game Results Scraper.

Scrapes historical college basketball game results from ESPN for training
the XGBoost prediction model with real outcomes instead of simulated data.

Workflow:
1. Navigate to ESPN Men's College Basketball scoreboard
2. Use date parameter to access historical dates
3. Extract all game results (teams, scores)
4. Save to CSV format compatible with HistoricalDataLoader

Requirements:
    uv add playwright
    playwright install chromium

Usage:
    from kenp0m_sp0rts_analyzer.espn_results_scraper import ESPNResultsScraper

    scraper = ESPNResultsScraper()
    results = await scraper.scrape_date(date(2024, 12, 15))
    scraper.save_to_csv(results, "game_results.csv")
"""

import asyncio
import csv
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from .helpers import normalize_team_name
from .utils.logging import logger

# ESPN URLs
ESPN_SCOREBOARD_URL = "https://www.espn.com/mens-college-basketball/scoreboard"


@dataclass
class GameResult:
    """A single game result with scores."""

    game_date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    game_id: str = ""
    venue: str = ""
    is_neutral: bool = False
    is_conference: bool = False
    overtime: bool = False

    @property
    def margin(self) -> int:
        """Home margin (positive = home won)."""
        return self.home_score - self.away_score

    @property
    def total(self) -> int:
        """Combined score."""
        return self.home_score + self.away_score

    @property
    def home_team_normalized(self) -> str:
        """KenPom-normalized home team name."""
        return normalize_team_name(self.home_team)

    @property
    def away_team_normalized(self) -> str:
        """KenPom-normalized away team name."""
        return normalize_team_name(self.away_team)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "game_date": str(self.game_date),
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "margin": self.margin,
            "total": self.total,
            "game_id": self.game_id,
            "venue": self.venue,
            "is_neutral": self.is_neutral,
            "overtime": self.overtime,
        }


class ESPNResultsScraper:
    """Scrape historical game results from ESPN.

    Uses Playwright for browser automation to handle JavaScript-rendered
    content on ESPN's scoreboard pages.

    Example:
        >>> scraper = ESPNResultsScraper()
        >>> results = await scraper.scrape_date(date(2024, 12, 15))
        >>> print(f"Found {len(results)} games")
        >>> scraper.save_to_csv(results, "results.csv")
    """

    def __init__(
        self,
        headless: bool = True,
        output_dir: str = "data/espn_results",
    ):
        """Initialize scraper.

        Args:
            headless: Run browser in headless mode.
            output_dir: Directory to save scraped results.
        """
        self.headless = headless
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_url(self, target_date: date) -> str:
        """Build ESPN scoreboard URL for a specific date."""
        date_str = target_date.strftime("%Y%m%d")
        return f"{ESPN_SCOREBOARD_URL}/_/date/{date_str}/group/50"

    async def scrape_date(self, target_date: date) -> list[GameResult]:
        """Scrape all game results for a specific date.

        Args:
            target_date: The date to scrape results for.

        Returns:
            List of GameResult objects for all games on that date.
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError as e:
            raise ImportError(
                "Playwright is required. Install with: "
                "uv add playwright && playwright install chromium"
            ) from e

        results = []
        url = self._build_url(target_date)
        logger.info(f"Scraping ESPN results for {target_date}: {url}")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            page = await context.new_page()

            try:
                # Navigate to scoreboard (domcontentloaded, not networkidle)
                await page.goto(
                    url, wait_until="domcontentloaded", timeout=60000
                )
                await asyncio.sleep(4)  # Let JS render

                # Wait for scoreboard to render - try multiple selectors
                try:
                    await page.wait_for_selector(
                        ".Scoreboard, .ScoreboardScoreCell, .ScoreCell",
                        timeout=15000,
                    )
                except Exception:
                    # Page might already have content, continue anyway
                    logger.info("Selector wait timed out, proceeding...")

                # Extra wait for JS rendering
                await asyncio.sleep(2)

                # Extract game data
                results = await self._extract_games(page, target_date)

                logger.info(f"Scraped {len(results)} games for {target_date}")

            except Exception as e:
                logger.error(f"Failed to scrape {target_date}: {e}")
                # Save screenshot for debugging
                screenshot_path = self.output_dir / f"error_{target_date}.png"
                await page.screenshot(path=str(screenshot_path))
                logger.info(f"Saved error screenshot to {screenshot_path}")

            finally:
                await browser.close()

        return results

    async def _extract_games(
        self,
        page: Any,
        target_date: date,
    ) -> list[GameResult]:
        """Extract game results from the scoreboard page.

        Args:
            page: Playwright page object.
            target_date: Date being scraped.

        Returns:
            List of GameResult objects.
        """
        # Try JavaScript extraction FIRST - it's more reliable
        logger.info("Extracting via JavaScript API...")
        results = await self._extract_via_js(page, target_date)

        if results:
            logger.info(f"JS extraction found {len(results)} games")
            return results

        # Fallback to DOM parsing if JS fails
        logger.info("JS extraction failed, trying DOM parsing...")
        results = []

        # ESPN structure: look for game cells directly
        game_cells = await page.query_selector_all(
            ".ScoreboardScoreCell--post"
        )

        logger.info(f"Found {len(game_cells)} completed game cells")

        for cell in game_cells:
            try:
                result = await self._parse_game_cell(cell, target_date)
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Failed to parse game cell: {e}")
                continue

        return results

    async def _parse_game_cell(
        self,
        cell: Any,
        target_date: date,
    ) -> GameResult | None:
        """Parse a ScoreboardScoreCell element.

        Args:
            cell: The ScoreboardScoreCell element.
            target_date: Date being scraped.

        Returns:
            GameResult or None if parsing fails.
        """
        # Find away and home team rows
        away_row = await cell.query_selector(
            "li.ScoreboardScoreCell__Item--away"
        )
        home_row = await cell.query_selector(
            "li.ScoreboardScoreCell__Item--home"
        )

        if not away_row or not home_row:
            return None

        # Parse away team
        away_name_el = await away_row.query_selector(".ScoreCell__TeamName")
        away_score_el = await away_row.query_selector(".ScoreCell__Score")
        if not away_name_el or not away_score_el:
            return None

        away_team = (await away_name_el.inner_text()).strip()
        away_score_text = (await away_score_el.inner_text()).strip()

        # Parse home team
        home_name_el = await home_row.query_selector(".ScoreCell__TeamName")
        home_score_el = await home_row.query_selector(".ScoreCell__Score")
        if not home_name_el or not home_score_el:
            return None

        home_team = (await home_name_el.inner_text()).strip()
        home_score_text = (await home_score_el.inner_text()).strip()

        # Parse scores
        try:
            away_score = int(away_score_text)
            home_score = int(home_score_text)
        except ValueError:
            return None

        # Validate basketball scores
        if not (20 <= away_score <= 200 and 20 <= home_score <= 200):
            return None

        # Check for overtime
        cell_text = await cell.inner_text()
        overtime = "OT" in cell_text

        game_id = f"{target_date}_{away_team}@{home_team}"

        logger.info(
            f"Parsed: {away_team} {away_score} @ {home_team} {home_score}"
        )

        return GameResult(
            game_date=target_date,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            game_id=game_id,
            overtime=overtime,
        )

    async def _parse_game_container(
        self,
        container: Any,
        target_date: date,
    ) -> GameResult | None:
        """Parse a single game container element.

        ESPN Scoreboard structure (Dec 2024):
        section.Scoreboard
          > div (teams wrapper)
            > ul.ScoreboardScoreCell
              > li.ScoreboardScoreCell__Item (away team row)
                > div.ScoreCell__TeamName
                > div.ScoreCell__Score
              > li.ScoreboardScoreCell__Item (home team row)
                > div.ScoreCell__TeamName
                > div.ScoreCell__Score

        Args:
            container: Playwright element handle for game container.
            target_date: Date of the game.

        Returns:
            GameResult or None if parsing fails.
        """
        # Check if game is completed (Final)
        status_el = await container.query_selector(".ScoreCell__Time")
        if not status_el:
            return None
        status_text = await status_el.inner_text()
        if "Final" not in status_text:
            return None

        # Find away and home team rows specifically by class
        away_row = await container.query_selector(
            "li.ScoreboardScoreCell__Item--away"
        )
        home_row = await container.query_selector(
            "li.ScoreboardScoreCell__Item--home"
        )

        if not away_row or not home_row:
            # Debug: log what we found
            all_li = await container.query_selector_all("li")
            logger.debug(f"Found {len(all_li)} li elements, no away/home rows")
            return None

        # Parse away team
        away_name_el = await away_row.query_selector(".ScoreCell__TeamName")
        away_score_el = await away_row.query_selector(".ScoreCell__Score")
        if not away_name_el or not away_score_el:
            return None

        away_team = (await away_name_el.inner_text()).strip()
        away_score_text = (await away_score_el.inner_text()).strip()

        # Parse home team
        home_name_el = await home_row.query_selector(".ScoreCell__TeamName")
        home_score_el = await home_row.query_selector(".ScoreCell__Score")
        if not home_name_el or not home_score_el:
            return None

        home_team = (await home_name_el.inner_text()).strip()
        home_score_text = (await home_score_el.inner_text()).strip()

        # Parse scores
        try:
            away_score = int(away_score_text)
            home_score = int(home_score_text)
        except ValueError:
            return None

        # Validate basketball scores (reasonable range)
        if not (20 <= away_score <= 200 and 20 <= home_score <= 200):
            logger.debug(
                f"Invalid scores: {away_team} {away_score} @ "
                f"{home_team} {home_score}"
            )
            return None

        # Check for overtime
        overtime = False
        game_text = await container.inner_text()
        if "OT" in game_text:
            overtime = True

        # Generate game ID
        game_id = f"{target_date}_{away_team}@{home_team}"

        logger.info(
            f"Parsed: {away_team} {away_score} @ {home_team} {home_score}"
        )

        return GameResult(
            game_date=target_date,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            game_id=game_id,
            overtime=overtime,
        )

    async def _extract_via_js(
        self,
        page: Any,
        target_date: date,
    ) -> list[GameResult]:
        """Extract games using JavaScript evaluation.

        ESPN stores page data in window.__ESPN_INITIAL_STATE__ which is
        more reliable than DOM parsing.
        """
        results = []

        # ESPN data structure extraction
        js_code = """
        () => {
            const games = [];

            // Try ESPN state object - multiple possible paths
            if (window.__ESPN_INITIAL_STATE__) {
                const state = window.__ESPN_INITIAL_STATE__;

                // Path 1: page.content.events
                let events = state?.page?.content?.events || [];

                // Path 2: scoreboard.events
                if (!events.length && state?.scoreboard?.events) {
                    events = state.scoreboard.events;
                }

                // Path 3: Check various content paths
                if (!events.length) {
                    const content = state?.page?.content;
                    if (content) {
                        for (const key of Object.keys(content)) {
                            if (Array.isArray(content[key])) {
                                const arr = content[key];
                                if (arr.length && arr[0]?.competitions) {
                                    events = arr;
                                    break;
                                }
                            }
                        }
                    }
                }

                for (const event of events) {
                    const comp = event?.competitions?.[0];
                    const teams = comp?.competitors;
                    if (teams && teams.length === 2) {
                        const away = teams.find(c => c.homeAway === 'away');
                        const home = teams.find(c => c.homeAway === 'home');
                        if (away && home && away.score && home.score) {
                            const awayScore = parseInt(away.score);
                            const homeScore = parseInt(home.score);
                            // Valid basketball scores: 20-200 typically
                            if (awayScore >= 20 && awayScore <= 200 &&
                                homeScore >= 20 && homeScore <= 200) {
                                games.push({
                                    away_team: away.team?.displayName ||
                                               away.team?.name || '',
                                    home_team: home.team?.displayName ||
                                               home.team?.name || '',
                                    away_score: awayScore,
                                    home_score: homeScore,
                                    status: event?.status?.type?.name || '',
                                });
                            }
                        }
                    }
                }
            }

            // Fallback: careful DOM parsing
            if (games.length === 0) {
                const scoreboards = document.querySelectorAll(
                    'section.Scoreboard'
                );
                for (const sb of scoreboards) {
                    // Look for completed games (Final status)
                    const status = sb.querySelector('.ScoreCell__Time');
                    if (!status || !status.innerText.includes('Final')) {
                        continue;
                    }

                    const teams = sb.querySelectorAll('.ScoreCell__TeamName');
                    const scores = sb.querySelectorAll(
                        '.ScoreCell__Score--scoreboard'
                    );

                    if (teams.length >= 2 && scores.length >= 2) {
                        const awayScore = parseInt(scores[0].innerText);
                        const homeScore = parseInt(scores[1].innerText);
                        if (awayScore >= 20 && awayScore <= 200 &&
                            homeScore >= 20 && homeScore <= 200) {
                            games.push({
                                away_team: teams[0].innerText.trim(),
                                home_team: teams[1].innerText.trim(),
                                away_score: awayScore,
                                home_score: homeScore,
                                status: 'Final',
                            });
                        }
                    }
                }
            }

            return games;
        }
        """

        try:
            games_data = await page.evaluate(js_code)
            logger.info(f"JS found {len(games_data)} raw games")

            for game in games_data:
                # Skip games without valid team names
                if not game.get("away_team") or not game.get("home_team"):
                    continue

                # Only include completed games
                status = game.get("status", "")
                if (
                    status
                    and "Final" not in status
                    and "STATUS_FINAL" not in status
                ):
                    continue

                results.append(
                    GameResult(
                        game_date=target_date,
                        home_team=game["home_team"],
                        away_team=game["away_team"],
                        home_score=game["home_score"],
                        away_score=game["away_score"],
                        game_id=(
                            f"{target_date}_"
                            f"{game['away_team']}@{game['home_team']}"
                        ),
                    )
                )
        except Exception as e:
            logger.warning(f"JavaScript extraction failed: {e}")

        return results

    async def scrape_date_range(
        self,
        start_date: date,
        end_date: date,
        delay_seconds: float = 2.0,
    ) -> list[GameResult]:
        """Scrape results for a range of dates.

        Args:
            start_date: First date to scrape.
            end_date: Last date to scrape (inclusive).
            delay_seconds: Delay between requests to avoid rate limiting.

        Returns:
            List of all GameResult objects.
        """
        all_results = []
        current = start_date

        while current <= end_date:
            results = await self.scrape_date(current)
            all_results.extend(results)

            # Progress logging
            logger.info(
                f"Progress: {current} - {len(results)} games "
                f"({len(all_results)} total)"
            )

            # Delay between requests
            await asyncio.sleep(delay_seconds)
            current += timedelta(days=1)

        return all_results

    def save_to_csv(
        self,
        results: list[GameResult],
        filename: str = "game_results.csv",
    ) -> Path:
        """Save results to CSV file.

        The CSV format is compatible with HistoricalDataLoader.load_from_csv().

        Args:
            results: List of GameResult objects.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        output_path = self.output_dir / filename

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "game_date",
                    "home_team",
                    "away_team",
                    "margin",
                    "total",
                    "home_score",
                    "away_score",
                    "overtime",
                ],
            )
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "game_date": str(result.game_date),
                        "home_team": result.home_team_normalized,
                        "away_team": result.away_team_normalized,
                        "margin": result.margin,
                        "total": result.total,
                        "home_score": result.home_score,
                        "away_score": result.away_score,
                        "overtime": result.overtime,
                    }
                )

        logger.info(f"Saved {len(results)} results to {output_path}")
        return output_path

    def save_to_json(
        self,
        results: list[GameResult],
        filename: str = "game_results.json",
    ) -> Path:
        """Save results to JSON file.

        Args:
            results: List of GameResult objects.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        output_path = self.output_dir / filename

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                [r.to_dict() for r in results],
                f,
                indent=2,
            )

        logger.info(f"Saved {len(results)} results to {output_path}")
        return output_path


async def main():
    """CLI entry point for scraping ESPN results."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape ESPN college basketball results"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Single date to scrape (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for range (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for range (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="game_results.csv",
        help="Output filename",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run in headless mode",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Show browser window",
    )

    args = parser.parse_args()

    # Determine dates
    if args.date:
        target = datetime.strptime(args.date, "%Y-%m-%d").date()
        start = end = target
    elif args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    else:
        # Default: yesterday
        target = date.today() - timedelta(days=1)
        start = end = target

    # Run scraper
    headless = not args.visible
    scraper = ESPNResultsScraper(headless=headless)

    if start == end:
        results = await scraper.scrape_date(start)
    else:
        results = await scraper.scrape_date_range(start, end)

    # Save results
    if results:
        csv_path = scraper.save_to_csv(results, args.output)
        print(f"Saved {len(results)} games to {csv_path}")
    else:
        print("No games found")


if __name__ == "__main__":
    asyncio.run(main())
