"""KenPom web scraper using stealth browser automation.

This module provides direct scraping of KenPom.com using Playwright
with stealth techniques for reliable data extraction.

Requirements:
    pip install kenp0m-sp0rts-analyzer[browser]
    playwright install chromium
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .browser import BrowserConfig, StealthBrowser, run_sync
from .utils import get_credentials

logger = logging.getLogger(__name__)

# KenPom URLs
KENPOM_BASE_URL = "https://kenpom.com"
KENPOM_LOGIN_URL = f"{KENPOM_BASE_URL}/login.php"
KENPOM_API_DOCS_URL = f"{KENPOM_BASE_URL}/api-documentation.php"


@dataclass
class KenPomPage:
    """Represents a KenPom page with its data."""

    url: str
    title: str
    html: str
    timestamp: datetime


class KenPomScraper:
    """Stealth scraper for KenPom.com data.

    This scraper uses Playwright with stealth configuration to:
    - Authenticate with KenPom subscription
    - Navigate pages like a real user
    - Extract data from tables and page content
    - Access the API documentation

    Example:
        ```python
        # Async usage
        async with KenPomScraper(headless=False) as scraper:
            await scraper.login()
            ratings = await scraper.get_ratings()
            print(ratings.head())

        # Sync usage
        scraper = KenPomScraper(headless=False)
        scraper.login_sync()
        ratings = scraper.get_ratings_sync()
        ```
    """

    def __init__(
        self,
        email: str | None = None,
        password: str | None = None,
        headless: bool = False,
        user_data_dir: Path | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the KenPom scraper.

        Args:
            email: KenPom subscription email. If not provided, reads from env.
            password: KenPom subscription password. If not provided, reads from env.
            headless: Run browser in headless mode (default: False for visible).
            user_data_dir: Directory for persistent browser data/sessions.
            **kwargs: Additional arguments for BrowserConfig.
        """
        self._email = email
        self._password = password

        # Configure browser
        self._config = BrowserConfig(
            headless=headless,
            user_data_dir=user_data_dir,
            **kwargs,
        )

        self._browser: StealthBrowser | None = None
        self._page: Any = None
        self._logged_in = False

    async def __aenter__(self) -> "KenPomScraper":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def start(self) -> None:
        """Start the browser."""
        self._browser = StealthBrowser(config=self._config)
        await self._browser.start()
        self._page = await self._browser.new_page()

    async def close(self) -> None:
        """Close the browser."""
        if self._browser:
            await self._browser.close()

    def _get_credentials(self) -> tuple[str, str]:
        """Get credentials from instance or environment."""
        if self._email and self._password:
            return self._email, self._password
        return get_credentials()

    async def login(self) -> bool:
        """Login to KenPom with subscription credentials.

        Returns:
            True if login successful.
        """
        if not self._page:
            raise RuntimeError("Browser not started. Use async context manager.")

        email, password = self._get_credentials()

        logger.info("Navigating to KenPom login page...")
        await self._page.goto(KENPOM_LOGIN_URL)

        # Wait for page to load
        await self._page.wait_for_load_state("networkidle")

        # Check if already logged in (redirect to main page)
        if "login" not in self._page.url.lower():
            logger.info("Already logged in from previous session")
            self._logged_in = True
            return True

        # Fill login form
        logger.info("Filling login form...")
        await self._page.fill('input[name="email"]', email)
        await self._page.fill('input[name="password"]', password)

        # Human-like delay before clicking
        await asyncio.sleep(0.5 + 0.5 * (await self._random_delay()))

        # Click login button
        await self._page.click('input[type="submit"]')

        # Wait for navigation
        await self._page.wait_for_load_state("networkidle")

        # Check for successful login
        if "login" in self._page.url.lower():
            # Check for error message
            error = await self._page.query_selector(".error")
            if error:
                error_text = await error.inner_text()
                raise ValueError(f"Login failed: {error_text}")
            raise ValueError("Login failed: Unknown error")

        logger.info("Successfully logged in to KenPom")
        self._logged_in = True
        return True

    async def _random_delay(self) -> float:
        """Generate random delay for human-like behavior."""
        import random

        return random.random()

    async def _ensure_logged_in(self) -> None:
        """Ensure user is logged in before accessing data."""
        if not self._logged_in:
            await self.login()

    async def get_page(self, url: str) -> KenPomPage:
        """Fetch a KenPom page.

        Args:
            url: Full URL or path relative to kenpom.com.

        Returns:
            KenPomPage with page data.
        """
        await self._ensure_logged_in()

        # Make URL absolute if needed
        if not url.startswith("http"):
            url = f"{KENPOM_BASE_URL}/{url.lstrip('/')}"

        logger.info(f"Fetching: {url}")
        await self._page.goto(url)
        await self._page.wait_for_load_state("networkidle")

        return KenPomPage(
            url=self._page.url,
            title=await self._page.title(),
            html=await self._page.content(),
            timestamp=datetime.now(),
        )

    async def get_ratings(self, season: int | None = None) -> pd.DataFrame:
        """Get the main KenPom ratings table.

        Args:
            season: Season year. Defaults to current season.

        Returns:
            DataFrame with team ratings.
        """
        url = KENPOM_BASE_URL
        if season:
            url = f"{url}/index.php?y={season}"

        page = await self.get_page(url)
        return self._parse_ratings_table(page.html)

    async def get_efficiency(self, season: int | None = None) -> pd.DataFrame:
        """Get efficiency and tempo statistics.

        Args:
            season: Season year. Defaults to current season.

        Returns:
            DataFrame with efficiency data.
        """
        url = f"{KENPOM_BASE_URL}/summary.php"
        if season:
            url = f"{url}?y={season}"

        page = await self.get_page(url)
        return self._parse_summary_table(page.html)

    async def get_four_factors(self, season: int | None = None) -> pd.DataFrame:
        """Get Four Factors analysis.

        Args:
            season: Season year. Defaults to current season.

        Returns:
            DataFrame with Four Factors data.
        """
        url = f"{KENPOM_BASE_URL}/stats.php?s=RankAdjOE"
        if season:
            url = f"{url}&y={season}"

        page = await self.get_page(url)
        return self._parse_stats_table(page.html)

    async def get_team_schedule(
        self, team: str, season: int | None = None
    ) -> pd.DataFrame:
        """Get a team's schedule with game-by-game data.

        Args:
            team: Team name.
            season: Season year. Defaults to current season.

        Returns:
            DataFrame with schedule data.
        """
        # Get team URL (need to find the team page first)
        await self._ensure_logged_in()

        # Navigate to team page
        url = f"{KENPOM_BASE_URL}/team.php?team={team.replace(' ', '+')}"
        if season:
            url = f"{url}&y={season}"

        page = await self.get_page(url)
        return self._parse_schedule_table(page.html)

    async def get_api_documentation(self) -> dict[str, Any]:
        """Scrape the KenPom API documentation page.

        Returns:
            Dictionary with API documentation details.
        """
        await self._ensure_logged_in()

        logger.info("Fetching API documentation...")
        page = await self.get_page(KENPOM_API_DOCS_URL)

        return self._parse_api_docs(page.html)

    async def screenshot(self, path: str = "screenshot.png") -> str:
        """Take a screenshot of the current page.

        Args:
            path: Path to save screenshot.

        Returns:
            Path to saved screenshot.
        """
        await self._page.screenshot(path=path, full_page=True)
        logger.info(f"Screenshot saved to: {path}")
        return path

    async def get_cdp_session(self) -> Any:
        """Get Chrome DevTools Protocol session.

        Returns:
            CDP session for advanced browser control.
        """
        if not self._browser:
            raise RuntimeError("Browser not started")
        return await self._browser.get_cdp_session(self._page)

    # ==================== Parsing Methods ====================

    def _parse_ratings_table(self, html: str) -> pd.DataFrame:
        """Parse the main ratings table from HTML."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", {"id": "ratings-table"})

        if not table:
            # Try alternative table selector
            table = soup.find("table")

        if not table:
            raise ValueError("Could not find ratings table")

        # Extract headers
        headers = []
        header_row = table.find("tr")
        if header_row:
            headers = [
                th.get_text(strip=True) for th in header_row.find_all(["th", "td"])
            ]

        # Extract data rows
        rows = []
        for tr in table.find_all("tr")[1:]:  # Skip header
            cells = tr.find_all(["td", "th"])
            if cells:
                row = [cell.get_text(strip=True) for cell in cells]
                rows.append(row)

        df = pd.DataFrame(rows, columns=headers if headers else None)
        return df

    def _parse_summary_table(self, html: str) -> pd.DataFrame:
        """Parse summary/efficiency table from HTML."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")

        # Find the main data table
        tables = soup.find_all("table")
        for table in tables:
            if table.find("th") and "Team" in table.get_text():
                return self._extract_table_data(table)

        raise ValueError("Could not find efficiency table")

    def _parse_stats_table(self, html: str) -> pd.DataFrame:
        """Parse stats table from HTML."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        tables = soup.find_all("table")

        for table in tables:
            if table.find("th"):
                return self._extract_table_data(table)

        raise ValueError("Could not find stats table")

    def _parse_schedule_table(self, html: str) -> pd.DataFrame:
        """Parse team schedule table from HTML."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")

        # Find schedule table (usually has "Schedule" nearby)
        tables = soup.find_all("table")
        for table in tables:
            if "Date" in table.get_text() and "Opponent" in table.get_text():
                return self._extract_table_data(table)

        raise ValueError("Could not find schedule table")

    def _parse_api_docs(self, html: str) -> dict[str, Any]:
        """Parse API documentation page."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")

        docs = {
            "title": soup.title.string if soup.title else "KenPom API Documentation",
            "sections": [],
            "endpoints": [],
            "raw_text": "",
        }

        # Extract main content
        content = soup.find("div", {"id": "content"}) or soup.find("main") or soup.body
        if content:
            docs["raw_text"] = content.get_text(separator="\n", strip=True)

            # Find headers and sections
            for header in content.find_all(["h1", "h2", "h3"]):
                docs["sections"].append(
                    {
                        "level": header.name,
                        "title": header.get_text(strip=True),
                    }
                )

            # Look for code blocks or API endpoint patterns
            for code in content.find_all(["code", "pre"]):
                text = code.get_text(strip=True)
                if text:
                    docs["endpoints"].append(text)

            # Look for URL patterns
            url_pattern = re.compile(r'(/api/[^\s<>"]+|https?://kenpom\.com/[^\s<>"]+)')
            urls = url_pattern.findall(str(content))
            docs["endpoints"].extend(urls)

        return docs

    def _extract_table_data(self, table: Any) -> pd.DataFrame:
        """Extract data from an HTML table element."""
        headers = []
        rows = []

        # Get headers
        header_row = table.find("thead")
        if header_row:
            headers = [
                th.get_text(strip=True) for th in header_row.find_all(["th", "td"])
            ]
        else:
            first_row = table.find("tr")
            if first_row:
                headers = [
                    th.get_text(strip=True) for th in first_row.find_all(["th", "td"])
                ]

        # Get data rows
        body = table.find("tbody") or table
        for tr in body.find_all("tr"):
            cells = tr.find_all(["td"])
            if cells:
                row = [cell.get_text(strip=True) for cell in cells]
                if len(row) == len(headers) or not headers:
                    rows.append(row)

        df = pd.DataFrame(rows, columns=headers) if headers else pd.DataFrame(rows)

        return df

    # ==================== Sync Wrappers ====================

    def login_sync(self) -> bool:
        """Synchronous wrapper for login."""
        return run_sync(self._run_with_browser(self.login))

    def get_ratings_sync(self, season: int | None = None) -> pd.DataFrame:
        """Synchronous wrapper for get_ratings."""
        return run_sync(self._run_with_browser(lambda: self.get_ratings(season)))

    def get_efficiency_sync(self, season: int | None = None) -> pd.DataFrame:
        """Synchronous wrapper for get_efficiency."""
        return run_sync(self._run_with_browser(lambda: self.get_efficiency(season)))

    def get_api_documentation_sync(self) -> dict[str, Any]:
        """Synchronous wrapper for get_api_documentation."""
        return run_sync(self._run_with_browser(self.get_api_documentation))

    async def _run_with_browser(self, coro_func: Any) -> Any:
        """Run a coroutine with browser lifecycle management."""
        await self.start()
        try:
            return await coro_func()
        finally:
            await self.close()


async def scrape_kenpom(
    headless: bool = False,
    season: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Quick function to scrape main KenPom data.

    Args:
        headless: Run browser in headless mode.
        season: Season year.

    Returns:
        Dictionary with DataFrames for different data types.
    """
    async with KenPomScraper(headless=headless) as scraper:
        await scraper.login()

        data = {
            "ratings": await scraper.get_ratings(season),
            "efficiency": await scraper.get_efficiency(season),
        }

        return data
