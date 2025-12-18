"""Overtime.ag API Discovery and Scraper.

This module provides scraping capabilities for overtime.ag betting lines
using Playwright with stealth techniques (same approach as KenPom scraper).

Requirements:
    pip install kenp0m-sp0rts-analyzer[browser]
    playwright install chromium

Environment Variables:
    OVERTIME_USER - Overtime.ag username
    OVERTIME_PASSWORD - Overtime.ag password
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .overtime_timing import (
    TimingDatabase,
    find_timestamp_fields,
    generate_game_id,
    get_monitoring_dir,
)

logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

# Overtime.ag URLs
OVERTIME_BASE_URL = "https://overtime.ag"
OVERTIME_SPORTS_URL = f"{OVERTIME_BASE_URL}/sports#/"
# Multiple possible URLs for College Basketball
OVERTIME_CBB_URLS = [
    f"{OVERTIME_BASE_URL}/sports#/basketball-college-basketball",
    f"{OVERTIME_BASE_URL}/sports#/basketball/college-basketball",
    f"{OVERTIME_BASE_URL}/sports#/college-basketball",
]


# Known NFL teams to filter out
NFL_TEAMS = {
    'arizona cardinals', 'atlanta falcons', 'baltimore ravens', 'buffalo bills',
    'carolina panthers', 'chicago bears', 'cincinnati bengals', 'cleveland browns',
    'dallas cowboys', 'denver broncos', 'detroit lions', 'green bay packers',
    'houston texans', 'indianapolis colts', 'jacksonville jaguars', 'kansas city chiefs',
    'las vegas raiders', 'los angeles chargers', 'los angeles rams', 'miami dolphins',
    'minnesota vikings', 'new england patriots', 'new orleans saints', 'new york giants',
    'new york jets', 'philadelphia eagles', 'pittsburgh steelers', 'san francisco 49ers',
    'seattle seahawks', 'tampa bay buccaneers', 'tennessee titans', 'washington commanders',
    # Shortened versions
    'cardinals', 'falcons', 'ravens', 'bills', 'panthers', 'bears', 'bengals', 'browns',
    'cowboys', 'broncos', 'lions', 'packers', 'texans', 'colts', 'jaguars', 'chiefs',
    'raiders', 'chargers', 'rams', 'dolphins', 'vikings', 'patriots', 'saints', 'giants',
    'jets', 'eagles', 'steelers', '49ers', 'seahawks', 'buccaneers', 'titans', 'commanders',
}


def is_nfl_team(team_name: str) -> bool:
    """Check if a team name is an NFL team."""
    normalized = team_name.lower().strip()
    # Check exact match
    if normalized in NFL_TEAMS:
        return True
    # Check if any NFL team name is in the string
    for nfl_team in NFL_TEAMS:
        if nfl_team in normalized or normalized in nfl_team:
            return True
    return False


@dataclass
class OvertimeGame:
    """Represents a single game with betting lines."""
    
    time: str
    away_team: str
    home_team: str
    spread: float | None = None
    spread_away_odds: int | None = None
    spread_home_odds: int | None = None
    total: float | None = None
    over_odds: int | None = None
    under_odds: int | None = None
    away_ml: int | None = None
    home_ml: int | None = None
    category: str = "college_basketball"
    
    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "away_team": self.away_team,
            "home_team": self.home_team,
            "spread": self.spread,
            "total": self.total,
            "away_ml": self.away_ml,
            "home_ml": self.home_ml,
            "category": self.category,
        }


@dataclass 
class NetworkCapture:
    """Captured network request/response."""
    
    url: str
    method: str
    request_headers: dict = field(default_factory=dict)
    response_headers: dict = field(default_factory=dict)
    response_body: str = ""
    status: int = 0
    resource_type: str = ""


class OvertimeScraper:
    """Scraper for overtime.ag betting lines.
    
    Uses Playwright with stealth configuration to:
    - Authenticate with overtime.ag account
    - Navigate to college basketball section
    - Extract betting lines (spreads, totals, moneylines)
    - Optionally capture API requests for direct access
    
    Example:
        ```python
        async with OvertimeScraper(headless=False) as scraper:
            await scraper.login()
            games = await scraper.get_college_basketball_lines()
            
            for game in games:
                print(f"{game.away_team} @ {game.home_team}: {game.spread}")
        ```
    """
    
    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        headless: bool = False,
        capture_network: bool = True,
        timing_db: TimingDatabase | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Overtime scraper.

        Args:
            username: Overtime.ag username. From OVERTIME_USER env.
            password: Overtime.ag password. From OVERTIME_PASSWORD env.
            headless: Run browser in headless mode (default: False).
            capture_network: Capture network requests for API discovery.
            timing_db: TimingDatabase for odds release timing analysis.
            **kwargs: Additional arguments for browser configuration.
        """
        self._username = username or os.getenv("OVERTIME_USER")
        self._password = password or os.getenv("OVERTIME_PASSWORD")
        self._headless = headless
        self._capture_network = capture_network
        self._timing_db = timing_db

        self._browser = None
        self._context = None
        self._page = None
        self._logged_in = False

        # Network capture storage
        self._captured_requests: list[NetworkCapture] = []
        self._api_endpoints: list[str] = []

        # Timing snapshot storage (for current capture)
        self._captured_games: dict[str, dict] = {}
        
    async def __aenter__(self) -> "OvertimeScraper":
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
        
    async def start(self) -> None:
        """Start the browser with stealth configuration."""
        try:
            from playwright.async_api import async_playwright
        except ImportError as e:
            raise ImportError(
                "Browser automation requires Playwright. "
                "Install with: pip install playwright && playwright install chromium"
            ) from e
            
        self._playwright = await async_playwright().start()
        
        # Stealth launch arguments
        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--disable-infobars",
            "--no-first-run",
        ]
        
        self._browser = await self._playwright.chromium.launch(
            headless=self._headless,
            slow_mo=50,
            args=launch_args,
        )
        
        self._context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="en-US",
            timezone_id="America/New_York",
        )
        
        self._page = await self._context.new_page()
        
        # Set up network capture
        if self._capture_network:
            await self._setup_network_capture()
            
        # Apply stealth scripts
        await self._apply_stealth()
        
        logger.info("Browser started successfully")
        
    async def _apply_stealth(self) -> None:
        """Apply stealth techniques to avoid detection."""
        await self._page.add_init_script("""
            // Remove webdriver flag
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            
            // Mock plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            
            // Mock chrome object
            window.chrome = { runtime: {} };
        """)
        
    async def _setup_network_capture(self) -> None:
        """Set up network request/response capture."""
        
        async def handle_response(response):
            url = response.url
            
            # Capture API-like requests (JSON, XHR, fetch)
            content_type = response.headers.get("content-type", "")
            
            if any(x in url.lower() for x in ["api", "odds", "lines", "events", "games", "sports"]):
                capture = NetworkCapture(
                    url=url,
                    method=response.request.method,
                    status=response.status,
                    resource_type=response.request.resource_type,
                    response_headers=dict(response.headers),
                )
                
                # Try to get response body for JSON responses
                if "json" in content_type or response.request.resource_type in ["xhr", "fetch"]:
                    try:
                        body = await response.text()
                        capture.response_body = body[:5000]  # Limit size
                        
                        # Check if it's valid JSON with odds data
                        if body.strip().startswith(("{", "[")):
                            self._api_endpoints.append(url)
                            logger.info(f"📡 Captured API endpoint: {url}")
                    except Exception:
                        pass
                        
                self._captured_requests.append(capture)
                
        self._page.on("response", handle_response)
        
    async def close(self) -> None:
        """Close the browser."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser closed")
        
    async def login(self) -> bool:
        """Login to overtime.ag (optional - data is publicly accessible).
        
        Returns:
            True if login successful or skipped.
        """
        # Data is publicly accessible - skip login to avoid dialogs
        logger.info("Skipping login - data is publicly accessible")
        
        # Just navigate to main page
        await self._page.goto(OVERTIME_SPORTS_URL)
        await self._page.wait_for_load_state("networkidle", timeout=30000)
        
        # Dismiss any login dialogs that might appear
        try:
            ok_button = await self._page.query_selector('button:has-text("Ok"), button:has-text("OK"), .modal button')
            if ok_button:
                await ok_button.click()
                await asyncio.sleep(0.5)
        except Exception:
            pass
            
        return True
        
    async def navigate_to_college_basketball(self) -> bool:
        """Navigate to college basketball section.
        
        Returns:
            True if navigation succeeded, False otherwise.
        """
        logger.info("Navigating to college basketball...")
        
        # Try direct URL navigation first (most reliable)
        logger.info("Trying direct URL navigation...")
        for cbb_url in OVERTIME_CBB_URLS:
            try:
                logger.debug(f"Trying URL: {cbb_url}")
                await self._page.goto(cbb_url)
                await self._page.wait_for_load_state("networkidle", timeout=30000)
                await asyncio.sleep(3)
                await self._dismiss_dialogs()
                
                # Verify we're on the right page
                if await self._verify_college_basketball_page():
                    logger.info(f"Direct URL navigation succeeded: {cbb_url}")
                    return True
                else:
                    logger.debug(f"URL {cbb_url} didn't land on CBB page")
            except Exception as e:
                logger.debug(f"URL {cbb_url} failed: {e}")
        
        logger.info("Direct URL navigation didn't work, trying sidebar...")
        
        # Fallback: Try sidebar navigation
        await self._page.goto(OVERTIME_SPORTS_URL)
        await self._page.wait_for_load_state("networkidle", timeout=30000)
        await self._dismiss_dialogs()
        await asyncio.sleep(1)
        
        # Step 1: Click "Basketball" in sidebar to expand submenu
        logger.info("Clicking Basketball in sidebar...")
        basketball_clicked = False
        
        selectors_to_try = [
            ('get_by_text', "Basketball"),
            ('query_selector', 'text=Basketball'),
            ('query_selector', '[class*="Basketball"]'),
            ('query_selector', 'a:has-text("Basketball")'),
            ('query_selector', 'span:has-text("Basketball")'),
        ]
        
        for method, selector in selectors_to_try:
            try:
                if method == 'get_by_text':
                    btn = self._page.get_by_text(selector, exact=True)
                    await btn.click(timeout=5000)
                else:
                    btn = await self._page.query_selector(selector)
                    if btn:
                        await btn.click()
                basketball_clicked = True
                logger.info(f"Clicked Basketball using {method}: {selector}")
                await asyncio.sleep(1)
                await self._dismiss_dialogs()
                break
            except Exception as e:
                logger.debug(f"Failed with {method} {selector}: {e}")
                continue
        
        if not basketball_clicked:
            logger.warning("Could not click Basketball in sidebar")
            return False
        
        # Step 2: Click "College Basketball" in submenu
        logger.info("Clicking College Basketball...")
        cbb_clicked = False

        cbb_selectors = [
            # Try exact matches first
            ('get_by_text', "COLLEGE BASKET"),
            ('get_by_text', "College Basket"),
            ('get_by_text', "COLLEGE BASKETBALL"),
            ('get_by_text', "College Basketball"),
            # Try contains/partial matches
            ('query_selector', 'text=/COLLEGE BASKET/i'),
            ('query_selector', 'text=/College Basketball/i'),
            ('query_selector', 'a:has-text("COLLEGE BASKET")'),
            ('query_selector', 'a:has-text("College Basketball")'),
            ('query_selector', 'span:has-text("COLLEGE BASKET")'),
            ('query_selector', 'span:has-text("College Basketball")'),
            ('query_selector', '[aria-label*="College Basketball"]'),
            ('query_selector', '[aria-label*="COLLEGE BASKET"]'),
        ]
        
        for method, selector in cbb_selectors:
            try:
                if method == 'get_by_text':
                    btn = self._page.get_by_text(selector, exact=True)
                    await btn.click(timeout=5000)
                else:
                    btn = await self._page.query_selector(selector)
                    if btn:
                        await btn.click()
                cbb_clicked = True
                logger.info(f"Clicked College Basketball using {method}: {selector}")
                await self._page.wait_for_load_state("networkidle", timeout=30000)
                await asyncio.sleep(2)
                await self._dismiss_dialogs()
                break
            except Exception as e:
                logger.debug(f"Failed with {method} {selector}: {e}")
                continue
        
        if not cbb_clicked:
            logger.warning("Could not click College Basketball - navigation failed")
            return False
        
        # Verify we're on the right page
        if await self._verify_college_basketball_page():
            logger.info("Navigation complete - verified on College Basketball page")
            return True
        else:
            logger.warning("Navigation may have failed - page doesn't look like College Basketball")
            return False
    
    async def _verify_college_basketball_page(self) -> bool:
        """Verify we're actually on the College Basketball page, not NFL.
        
        Returns:
            True if we appear to be on CBB page.
        """
        try:
            # Check the URL
            current_url = self._page.url
            if 'basketball' in current_url.lower() and 'college' in current_url.lower():
                return True
            
            # Check page content for college basketball indicators
            page_text = await self._page.evaluate('() => document.body.innerText')
            page_text_lower = page_text.lower()
            
            # Look for college basketball indicators
            cbb_indicators = ['college basketball', 'ncaa', 'duke', 'kentucky', 'kansas', 'north carolina', 'gonzaga']
            nfl_indicators = ['nfl', 'touchdown', 'quarterback', 'rams', 'seahawks', 'eagles', 'cowboys', 'patriots']
            
            cbb_score = sum(1 for ind in cbb_indicators if ind in page_text_lower)
            nfl_score = sum(1 for ind in nfl_indicators if ind in page_text_lower)
            
            logger.debug(f"Page verification - CBB indicators: {cbb_score}, NFL indicators: {nfl_score}")
            
            # If we see more NFL than CBB, we're on the wrong page
            if nfl_score > cbb_score and nfl_score >= 3:
                return False
                
            return cbb_score >= 1 or nfl_score == 0
            
        except Exception as e:
            logger.debug(f"Page verification error: {e}")
            return False
    
    async def navigate_to_college_extra(self) -> bool:
        """Navigate to College Extra section for additional games.
        
        Returns:
            True if navigation succeeded.
        """
        logger.info("Navigating to College Extra...")

        extra_selectors = [
            # Try exact matches first
            ('get_by_text', "COLLEGE EXTRA"),
            ('get_by_text', "College Extra"),
            # Try contains/partial matches
            ('query_selector', 'text=/COLLEGE EXTRA/i'),
            ('query_selector', 'text=/College Extra/i'),
            ('query_selector', 'a:has-text("COLLEGE EXTRA")'),
            ('query_selector', 'a:has-text("College Extra")'),
            ('query_selector', 'span:has-text("COLLEGE EXTRA")'),
            ('query_selector', 'span:has-text("College Extra")'),
            ('query_selector', '[aria-label*="College Extra"]'),
            ('query_selector', '[aria-label*="COLLEGE EXTRA"]'),
        ]
        
        for method, selector in extra_selectors:
            try:
                if method == 'get_by_text':
                    btn = self._page.get_by_text(selector, exact=True)
                    await btn.click(timeout=5000)
                else:
                    btn = await self._page.query_selector(selector)
                    if btn:
                        await btn.click()
                logger.info(f"Clicked College Extra using {method}: {selector}")
                await self._page.wait_for_load_state("networkidle", timeout=30000)
                await asyncio.sleep(2)
                await self._dismiss_dialogs()
                return True
            except Exception as e:
                logger.debug(f"Failed with {method} {selector}: {e}")
                continue
        
        logger.warning("Could not click College Extra")
        return False
    
    async def _dismiss_dialogs(self) -> None:
        """Dismiss any modal dialogs that might be blocking the page."""
        try:
            # Common dialog dismiss buttons
            selectors = [
                'button:has-text("Ok")',
                'button:has-text("OK")', 
                'button:has-text("Close")',
                'button:has-text("×")',
                '.modal button',
                '.dialog button',
                '[class*="close"]',
                'button.close',
            ]
            
            for selector in selectors:
                try:
                    button = await self._page.query_selector(selector)
                    if button and await button.is_visible():
                        await button.click()
                        await asyncio.sleep(0.3)
                        logger.info(f"Dismissed dialog with: {selector}")
                except Exception:
                    pass
                    
        except Exception as e:
            logger.debug(f"Dialog dismiss: {e}")
                
    async def get_college_basketball_lines(self, include_extra: bool = True) -> list[OvertimeGame]:
        """Scrape college basketball betting lines.
        
        Args:
            include_extra: Also scrape "College Extra" section
        
        Returns:
            List of OvertimeGame objects with betting lines.
        """
        all_games = []
        seen_matchups = set()  # Track seen games to avoid duplicates
        
        # Navigate to College Basketball
        nav_success = await self.navigate_to_college_basketball()
        
        if not nav_success:
            logger.error("Failed to navigate to College Basketball page")
            logger.error("This may be due to site changes or no CBB games being available.")
            # Return empty list - do NOT scrape NFL games!
            return []
        
        # Check if section is available (may show "No games available")
        no_games_msg = await self._check_no_games_message()
        if no_games_msg:
            logger.info(f"College Basketball: {no_games_msg}")
        else:
            # Scrape main College Basketball games
            logger.info("Scraping College Basketball games...")
            games = await self._scrape_current_page_games()
            
            # Filter and deduplicate
            filtered_count = 0
            for g in games:
                g.category = "college_basketball"
                matchup_key = f"{g.away_team.lower()}_{g.home_team.lower()}"
                
                # Skip NFL games (safety check)
                if is_nfl_team(g.away_team) or is_nfl_team(g.home_team):
                    logger.warning(f"Filtered NFL game (should not happen): {g.away_team} @ {g.home_team}")
                    filtered_count += 1
                    continue
                
                # Skip duplicates
                if matchup_key in seen_matchups:
                    logger.debug(f"Skipped duplicate: {g.away_team} @ {g.home_team}")
                    continue
                
                seen_matchups.add(matchup_key)
                all_games.append(g)
            
            if filtered_count > 0:
                logger.warning(f"Filtered {filtered_count} NFL games - navigation may have failed!")
                
            logger.info(f"Found {len(all_games)} College Basketball games (after filtering)")
        
        # Optionally scrape College Extra
        if include_extra:
            extra_nav_success = await self.navigate_to_college_extra()
            
            if not extra_nav_success:
                logger.info("College Extra section not available or navigation failed")
            else:
                no_games_msg = await self._check_no_games_message()
                if no_games_msg:
                    logger.info(f"College Extra: {no_games_msg}")
                else:
                    logger.info("Scraping College Extra games...")
                    extra_games = await self._scrape_current_page_games()
                    
                    extra_count = 0
                    for g in extra_games:
                        g.category = "college_extra"
                        matchup_key = f"{g.away_team.lower()}_{g.home_team.lower()}"
                        
                        # Skip NFL games
                        if is_nfl_team(g.away_team) or is_nfl_team(g.home_team):
                            continue
                        
                        # Skip duplicates
                        if matchup_key in seen_matchups:
                            continue
                        
                        seen_matchups.add(matchup_key)
                        all_games.append(g)
                        extra_count += 1
                        
                    logger.info(f"Found {extra_count} College Extra games (after filtering)")
        
        # Final validation - if we only got NFL games, something went wrong
        if all_games:
            nfl_check = [g for g in all_games if is_nfl_team(g.away_team) or is_nfl_team(g.home_team)]
            if nfl_check:
                logger.error(f"ERROR: {len(nfl_check)} NFL games in results - clearing results!")
                all_games = [g for g in all_games if not (is_nfl_team(g.away_team) or is_nfl_team(g.home_team))]
        
        logger.info(f"Total: {len(all_games)} college basketball games")
        return all_games
    
    async def _check_no_games_message(self) -> str | None:
        """Check if page shows 'no games available' message.
        
        Returns:
            Message string if no games, None if games are available.
        """
        try:
            # Common "no games" indicators
            no_game_selectors = [
                'text=No games available',
                'text=No events available', 
                'text=No lines available',
                'text=Check back later',
                '.no-games',
                '.no-events',
            ]
            
            for selector in no_game_selectors:
                try:
                    element = await self._page.query_selector(selector)
                    if element and await element.is_visible():
                        text = await element.text_content()
                        return text.strip() if text else "No games available"
                except Exception:
                    pass
            
            # Also check if the games container is empty
            page_text = await self._page.evaluate('() => document.body.innerText')
            if 'no games' in page_text.lower() or 'no events' in page_text.lower():
                return "No games currently available"
                
            # Check if there are any game elements at all
            game_elements = await self._page.query_selector_all('.gameLineInfo, [class*="gameLine"], .game-row')
            if not game_elements:
                # Could be loading or truly no games
                await asyncio.sleep(2)  # Wait a bit more
                game_elements = await self._page.query_selector_all('.gameLineInfo, [class*="gameLine"], .game-row')
                if not game_elements:
                    return "No games currently listed"
                    
        except Exception as e:
            logger.debug(f"Error checking for no-games message: {e}")
            
        return None
    
    async def _scrape_current_page_games(self) -> list[OvertimeGame]:
        """Scrape games from the currently displayed page."""
        games = []
        
        # Wait for Angular to render content
        await asyncio.sleep(3)
        
        # Try to extract games using JavaScript evaluation
        # This is more reliable for Angular sites
        try:
            games_data = await self._page.evaluate('''
                () => {
                    const games = [];
                    
                    // Find all game container divs - look for the pattern with two team rows
                    // Overtime.ag uses a structure where each game has a container with team info
                    const allText = document.body.innerText;
                    
                    // Find all buttons with betting data
                    const allButtons = document.querySelectorAll('button');
                    
                    // Build a map of game data from buttons
                    const gameMap = new Map();
                    
                    allButtons.forEach(btn => {
                        const text = btn.textContent.trim();
                        
                        // Skip empty or non-betting buttons
                        if (!text || text.length > 20) return;
                        
                        // Find parent row to get game context
                        let parent = btn.closest('.gameLineInfo, .game-line, [class*="game"]');
                        if (!parent) return;
                        
                        // Get any team name nearby
                        const teamEl = parent.querySelector('[class*="team"], span:not(:empty)');
                        if (teamEl) {
                            // Store this data
                        }
                    });
                    
                    // Alternative: Look for structured game containers
                    const gameContainers = document.querySelectorAll('.gameLineInfo, [class*="childGameLine"]');
                    
                    gameContainers.forEach(container => {
                        try {
                            // Get all text spans that might be team names
                            const spans = container.querySelectorAll('span');
                            const buttons = container.querySelectorAll('button');
                            
                            // Extract team names (usually in spans with class containing "team")
                            let teams = [];
                            spans.forEach(s => {
                                const text = s.textContent.trim();
                                // Team names are usually 2+ words, not numbers
                                if (text && text.length > 3 && !/^[\\d\\s+\\-\\.]+$/.test(text) &&  
                                    !text.includes('O ') && !text.includes('U ')) {
                                    // Check if it looks like a team name
                                    if (text.match(/^[A-Za-z\\s\\.]+$/)) {
                                        teams.push(text);
                                    }
                                }
                            });
                            
                            // Extract betting lines from buttons
                            let spread = null, total = null, awayMl = null, homeMl = null;
                            
                            buttons.forEach(b => {
                                const text = b.textContent.trim();
                                
                                // Spread pattern: +4 -110 or -4 -110
                                const spreadMatch = text.match(/^([+-]?\\d+\\.?\\d*)\\s*[+-]\\d+$/);
                                if (spreadMatch && spread === null) {
                                    spread = parseFloat(spreadMatch[1]);
                                }
                                
                                // Total pattern: O 160.5 -110 or U 160.5 -110
                                const totalMatch = text.match(/^[OU]\\s*(\\d+\\.?\\d*)/i);
                                if (totalMatch && total === null) {
                                    total = parseFloat(totalMatch[1]);
                                }
                                
                                // Moneyline pattern: +150 or -200 (standalone)
                                const mlMatch = text.match(/^([+-]\\d+)$/);
                                if (mlMatch) {
                                    const ml = parseInt(mlMatch[1]);
                                    if (awayMl === null) awayMl = ml;
                                    else if (homeMl === null) homeMl = ml;
                                }
                            });
                            
                            if (teams.length >= 2) {
                                games.push({
                                    away_team: teams[0],
                                    home_team: teams[1],
                                    spread: spread,
                                    total: total,
                                    away_ml: awayMl,
                                    home_ml: homeMl
                                });
                            }
                        } catch (e) {
                            console.log('Parse error:', e);
                        }
                    });
                    
                    return games;
                }
            ''')
            
            for gd in games_data:
                game = OvertimeGame(
                    time="",
                    away_team=gd.get('away_team', ''),
                    home_team=gd.get('home_team', ''),
                    spread=gd.get('spread'),
                    total=gd.get('total'),
                    away_ml=gd.get('away_ml'),
                    home_ml=gd.get('home_ml'),
                )
                if game.away_team and game.home_team:
                    games.append(game)
                    
        except Exception as e:
            logger.warning(f"JavaScript extraction failed: {e}")
        
        # Fallback: Try standard DOM parsing
        if not games:
            logger.info("Trying fallback DOM parsing...")
            games = await self._parse_games_from_dom()
        
        # Last resort: Parse from page text
        if not games:
            logger.info("Trying text-based extraction...")
            games = await self._extract_games_from_text()
                    
        logger.info(f"Found {len(games)} games on current page")
        return games
    
    async def _extract_games_from_text(self) -> list[OvertimeGame]:
        """Extract games by parsing page text content."""
        games = []
        
        try:
            # Get all text from the page
            page_text = await self._page.evaluate('() => document.body.innerText')
            lines = page_text.split('\n')
            
            current_game = {}
            team_count = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for rotation numbers (3-6 digit numbers at start of line)
                rot_match = re.match(r'^(\d{3,6})\s+(.+)', line)
                if rot_match:
                    rot_num = rot_match.group(1)
                    rest = rot_match.group(2).strip()
                    
                    # This is likely a team line
                    # Extract team name (text before any numbers/odds)
                    team_match = re.match(r'^([A-Za-z][A-Za-z\.\s\'\-]+)', rest)
                    if team_match:
                        team_name = team_match.group(1).strip()
                        
                        if team_count == 0:
                            # Away team
                            current_game = {'away_team': team_name, 'away_rot': rot_num}
                            team_count = 1
                        else:
                            # Home team - complete the game
                            current_game['home_team'] = team_name
                            current_game['home_rot'] = rot_num
                            
                            # Try to extract spread from the line
                            spread_match = re.search(r'([+-]?\d+\.?\d*)\s*-\d+', rest)
                            if spread_match:
                                current_game['spread'] = float(spread_match.group(1))
                            
                            # Try to extract total
                            total_match = re.search(r'[OU]\s*(\d+\.?\d*)', rest)
                            if total_match:
                                current_game['total'] = float(total_match.group(1))
                            
                            # Try to extract moneyline
                            ml_match = re.search(r'\s([+-]\d{3,4})\s', rest)
                            if ml_match:
                                current_game['home_ml'] = int(ml_match.group(1))
                            
                            # Create game object
                            if current_game.get('away_team') and current_game.get('home_team'):
                                game = OvertimeGame(
                                    time="",
                                    away_team=current_game['away_team'],
                                    home_team=current_game['home_team'],
                                    spread=current_game.get('spread'),
                                    total=current_game.get('total'),
                                    away_ml=current_game.get('away_ml'),
                                    home_ml=current_game.get('home_ml'),
                                )
                                games.append(game)
                            
                            # Reset for next game
                            current_game = {}
                            team_count = 0
                            
        except Exception as e:
            logger.warning(f"Text extraction failed: {e}")
            
        return games
    
    async def _parse_games_from_dom(self) -> list[OvertimeGame]:
        """Fallback DOM parsing method."""
        games = []
        
        # Try multiple selectors for game rows
        game_rows = await self._page.query_selector_all(
            '.gameLineInfo, .game-row, .event-row, [class*="gameLine"]'
        )
        
        for row in game_rows:
            try:
                game = await self._parse_game_row(row)
                if game:
                    games.append(game)
            except Exception as e:
                logger.debug(f"Failed to parse game row: {e}")
                
        return games
        
    async def _parse_game_row(self, row) -> OvertimeGame | None:
        """Parse a single game row element."""
        try:
            # Extract team names
            team_elements = await row.query_selector_all('[class*="team"], [class*="participant"]')
            if len(team_elements) < 2:
                return None
                
            away_team = await team_elements[0].inner_text()
            home_team = await team_elements[1].inner_text()
            
            # Clean team names
            away_team = away_team.strip().split('\n')[0]
            home_team = home_team.strip().split('\n')[0]
            
            if not away_team or not home_team:
                return None
                
            game = OvertimeGame(
                time="",
                away_team=away_team,
                home_team=home_team,
            )
            
            # Try to extract spread
            spread_elements = await row.query_selector_all('[class*="spread"]')
            if spread_elements:
                for elem in spread_elements:
                    text = await elem.inner_text()
                    spread_match = re.search(r'([+-]?\d+\.?\d*)', text)
                    if spread_match:
                        game.spread = float(spread_match.group(1))
                        break
                        
            # Try to extract total
            total_elements = await row.query_selector_all('[class*="total"], [class*="over"]')
            if total_elements:
                for elem in total_elements:
                    text = await elem.inner_text()
                    total_match = re.search(r'[OU]\s*(\d+\.?\d*)', text)
                    if total_match:
                        game.total = float(total_match.group(1))
                        break
                        
            # Try to extract moneylines
            ml_elements = await row.query_selector_all('[class*="money"], [class*="ml"]')
            if len(ml_elements) >= 2:
                away_ml_text = await ml_elements[0].inner_text()
                home_ml_text = await ml_elements[1].inner_text()
                
                away_match = re.search(r'([+-]?\d+)', away_ml_text)
                home_match = re.search(r'([+-]?\d+)', home_ml_text)
                
                if away_match:
                    game.away_ml = int(away_match.group(1))
                if home_match:
                    game.home_ml = int(home_match.group(1))
                    
            return game
            
        except Exception as e:
            logger.debug(f"Error parsing game row: {e}")
            return None
            
    async def _parse_games_from_page(self) -> list[OvertimeGame]:
        """Parse games from page content when standard selectors fail."""
        games = []
        
        # Get all text content
        content = await self._page.content()
        
        # Look for game patterns in the HTML
        # This is a fallback - the actual parsing will depend on the site structure
        
        # Try to find game containers
        containers = await self._page.query_selector_all('div, tr, li')
        
        for container in containers:
            text = await container.inner_text()
            
            # Look for patterns like "Team A @ Team B" or team name pairs
            if '@' in text or 'vs' in text.lower():
                # Try to parse this as a game
                lines = text.strip().split('\n')
                if len(lines) >= 2:
                    try:
                        game = self._parse_game_text(lines)
                        if game and game.away_team and game.home_team:
                            games.append(game)
                    except Exception:
                        pass
                        
        return games
        
    def _parse_game_text(self, lines: list[str]) -> OvertimeGame | None:
        """Parse game information from text lines."""
        # This is a heuristic parser - adjust based on actual site structure
        game = OvertimeGame(time="", away_team="", home_team="")
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Look for spread pattern (e.g., "-4 -110" or "+4 -110")
            spread_match = re.match(r'^([+-]?\d+\.?\d*)\s*[-+]?\d+$', line)
            if spread_match and game.spread is None:
                game.spread = float(spread_match.group(1))
                continue
                
            # Look for total pattern (e.g., "O 160.5 -110" or "U 160.5 -110")
            total_match = re.match(r'^[OU]\s*(\d+\.?\d*)', line)
            if total_match and game.total is None:
                game.total = float(total_match.group(1))
                continue
                
            # Look for moneyline pattern (e.g., "+150" or "-200")
            ml_match = re.match(r'^([+-]\d+)$', line)
            if ml_match:
                ml = int(ml_match.group(1))
                if game.away_ml is None:
                    game.away_ml = ml
                elif game.home_ml is None:
                    game.home_ml = ml
                continue
                
            # Look for time pattern (e.g., "7:30 PM")
            time_match = re.match(r'^\d{1,2}:\d{2}\s*(AM|PM|EST)?', line, re.IGNORECASE)
            if time_match:
                game.time = line
                continue
                
            # Otherwise, assume it's a team name
            if len(line) > 2 and not line.isdigit():
                if not game.away_team:
                    game.away_team = line
                elif not game.home_team:
                    game.home_team = line
                    
        return game if game.away_team and game.home_team else None
        
    # =========================================================================
    # TIMING ANALYSIS METHODS
    # =========================================================================

    async def capture_timing_snapshot(self) -> dict:
        """Capture a snapshot for odds release timing analysis.

        Navigates to college basketball, captures API responses,
        and records timing data for later analysis.

        Returns:
            Snapshot summary with response/game counts
        """
        logger.info("Capturing timing snapshot...")

        # Reset capture storage
        self._captured_responses = []
        self._captured_games = {}

        # Navigate and capture
        await self.login()
        await self.navigate_to_college_basketball()
        await asyncio.sleep(3)

        # Also try College Extra
        await self.navigate_to_college_extra()
        await asyncio.sleep(2)

        # Save snapshot to database if available
        snapshot = self._save_timing_snapshot()

        # Take screenshot for debugging
        monitoring_dir = get_monitoring_dir()
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        screenshot_path = monitoring_dir / f"screenshot_{ts}.png"
        await self.screenshot(str(screenshot_path))

        return snapshot

    def _save_timing_snapshot(self) -> dict:
        """Save captured data to timing database."""
        timestamp = datetime.now()

        if self._timing_db:
            # Save API snapshots
            for capture in self._captured_requests:
                if capture.response_body:
                    try:
                        data = json.loads(capture.response_body)
                        self._timing_db.save_api_snapshot(
                            endpoint=capture.url,
                            response_data=data,
                            game_count=len(self._captured_games),
                            captured_at=timestamp,
                        )
                    except json.JSONDecodeError:
                        pass

            # Track games
            for game_id, game_data in self._captured_games.items():
                data = game_data.get('data', {})
                timestamps = find_timestamp_fields(data, db=self._timing_db)

                self._timing_db.track_game(
                    game_id=game_id,
                    home_team=self._extract_team(data, 'home'),
                    away_team=self._extract_team(data, 'away'),
                    game_time=data.get('time', data.get('startTime', '')),
                    timestamps=timestamps,
                    raw_data=data,
                    first_seen_at=timestamp,
                )

        snapshot = {
            'timestamp': timestamp.isoformat(),
            'responses_captured': len(self._captured_requests),
            'games_found': len(self._captured_games),
            'new_games': list(self._captured_games.keys()),
        }

        logger.info(
            f"Snapshot: {len(self._captured_requests)} responses, "
            f"{len(self._captured_games)} games"
        )
        return snapshot

    def _extract_team(self, data: dict, team_type: str) -> str:
        """Extract team name from game data."""
        key_variants = [
            team_type,
            f"{team_type}Team",
            f"{team_type}_team",
        ]
        for key in key_variants:
            if key in data:
                val = data[key]
                if isinstance(val, dict):
                    return val.get('name', val.get('team', str(val)))
                return str(val)
        return ''

    async def run_continuous_monitoring(
        self,
        interval_minutes: int = 30,
        duration_hours: int = 24,
    ) -> list[dict]:
        """Run continuous monitoring at specified intervals.

        Args:
            interval_minutes: Minutes between captures (default: 30)
            duration_hours: Total duration in hours (default: 24)

        Returns:
            List of snapshot summaries
        """
        logger.info(
            f"Starting continuous monitoring: "
            f"{interval_minutes}min intervals for {duration_hours}h"
        )

        snapshots = []
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        iteration = 0

        while datetime.now() < end_time:
            iteration += 1
            logger.info(f"[Iteration {iteration}] {datetime.now():%H:%M:%S}")

            try:
                snapshot = await self.capture_timing_snapshot()
                snapshots.append(snapshot)
            except Exception as e:
                logger.error(f"Capture failed: {e}")
                snapshots.append({'error': str(e)})

            # Sleep until next interval
            next_run = datetime.now() + timedelta(minutes=interval_minutes)
            if next_run < end_time:
                sleep_secs = interval_minutes * 60
                logger.info(f"Sleeping {interval_minutes} minutes...")
                await asyncio.sleep(sleep_secs)

        logger.info(f"Monitoring complete: {len(snapshots)} snapshots")
        return snapshots

    async def screenshot(self, path: str = "overtime_screenshot.png") -> str:
        """Take a screenshot of the current page."""
        await self._page.screenshot(path=path, full_page=True)
        logger.info(f"Screenshot saved to: {path}")
        return path
        
    def get_captured_api_endpoints(self) -> list[str]:
        """Get list of discovered API endpoints."""
        return list(set(self._api_endpoints))
        
    def get_network_captures(self) -> list[NetworkCapture]:
        """Get all captured network requests."""
        return self._captured_requests
        
    def save_network_captures(self, path: str = "overtime_network_capture.json") -> str:
        """Save network captures to JSON file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "api_endpoints": self.get_captured_api_endpoints(),
            "requests": [
                {
                    "url": r.url,
                    "method": r.method,
                    "status": r.status,
                    "resource_type": r.resource_type,
                    "response_preview": r.response_body[:500] if r.response_body else "",
                }
                for r in self._captured_requests
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Network captures saved to: {path}")
        return path


async def discover_overtime_api(headless: bool = False) -> dict:
    """Discover overtime.ag API endpoints by capturing network traffic.
    
    Args:
        headless: Run browser in headless mode.
        
    Returns:
        Dictionary with discovered API information.
    """
    print("=" * 60)
    print("OVERTIME.AG API DISCOVERY")
    print("=" * 60)
    print("\nStarting browser and capturing network traffic...")
    print("This will help us understand how the site fetches odds data.\n")
    
    async with OvertimeScraper(headless=headless, capture_network=True) as scraper:
        # Login if credentials available
        await scraper.login()
        
        # Navigate to college basketball
        await scraper.navigate_to_college_basketball()
        
        # Wait for all data to load
        print("Waiting for odds to load...")
        await asyncio.sleep(5)
        
        # Scroll to trigger lazy loading
        print("Scrolling page to load more data...")
        await scraper._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(2)
        
        # Take screenshot for reference
        screenshot_path = await scraper.screenshot("overtime_cbb_screenshot.png")
        print(f"Screenshot saved: {screenshot_path}")
        
        # Get captured API endpoints
        endpoints = scraper.get_captured_api_endpoints()
        
        # Save full network capture
        capture_path = scraper.save_network_captures("overtime_api_discovery.json")
        
        print("\n" + "=" * 60)
        print("DISCOVERY RESULTS")
        print("=" * 60)
        
        if endpoints:
            print(f"\n✅ Found {len(endpoints)} potential API endpoints:\n")
            for i, url in enumerate(endpoints, 1):
                print(f"  {i}. {url[:100]}...")
        else:
            print("\n⚠️  No obvious API endpoints found")
            print("   The site may use WebSocket or embedded data")
            
        print(f"\n📁 Full network capture saved to: {capture_path}")
        print(f"📸 Screenshot saved to: {screenshot_path}")
        
        return {
            "endpoints": endpoints,
            "captures_file": capture_path,
            "screenshot": screenshot_path,
        }


async def scrape_overtime_lines(
    target_date: date | None = None,
    headless: bool = False,
) -> list[dict]:
    """Scrape college basketball lines from overtime.ag.
    
    Args:
        target_date: Date to scrape (default: today).
        headless: Run browser in headless mode.
        
    Returns:
        List of game dictionaries with betting lines.
    """
    target_date = target_date or date.today()
    
    print("=" * 60)
    print(f"SCRAPING OVERTIME.AG - {target_date}")
    print("=" * 60)
    
    async with OvertimeScraper(headless=headless) as scraper:
        await scraper.login()
        games = await scraper.get_college_basketball_lines()
        
        # Convert to dictionaries
        game_dicts = [g.to_dict() for g in games]
        
        print(f"\n✅ Scraped {len(games)} games")
        
        return game_dicts


def save_vegas_lines(games: list[dict], date_str: str, output_dir: Path | None = None) -> str:
    """Save scraped games to vegas_lines.json format.
    
    Args:
        games: List of game dictionaries.
        date_str: Date string (YYYY-MM-DD).
        output_dir: Output directory (default: data/).
        
    Returns:
        Path to saved file.
    """
    output_dir = output_dir or Path(__file__).parent.parent.parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "vegas_lines.json"
    
    data = {
        "source": "overtime.ag",
        "date": date_str,
        "scraped_at": datetime.now().isoformat(),
        "games": games,
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"✅ Saved to: {output_file}")
    return str(output_file)


async def run_timing_monitor(
    interval: int = 30,
    duration: int = 24,
    headless: bool = True,
) -> None:
    """Run timing monitor to discover odds release patterns.

    Args:
        interval: Minutes between captures
        duration: Total hours to run
        headless: Run in headless mode
    """
    print("=" * 60)
    print("OVERTIME.AG TIMING MONITOR")
    print("=" * 60)
    print(f"\nInterval: {interval} minutes")
    print(f"Duration: {duration} hours")
    print(f"Headless: {headless}\n")

    # Initialize database
    db = TimingDatabase()
    print(f"Database: {db.db_path}")

    async with OvertimeScraper(
        headless=headless,
        capture_network=True,
        timing_db=db,
    ) as scraper:
        snapshots = await scraper.run_continuous_monitoring(
            interval_minutes=interval,
            duration_hours=duration,
        )

    print(f"\n✅ Completed {len(snapshots)} snapshots")
    print(f"\nAnalyze results with:")
    print(f"  python -m kenp0m_sp0rts_analyzer.overtime_timing")


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Overtime.ag Scraper")
    parser.add_argument(
        "--discover", action="store_true",
        help="Run API discovery mode"
    )
    parser.add_argument(
        "--scrape", action="store_true",
        help="Scrape college basketball lines"
    )
    parser.add_argument(
        "--monitor", action="store_true",
        help="Run timing monitor to discover odds release patterns"
    )
    parser.add_argument(
        "--date", type=str,
        help="Target date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=30,
        help="Monitoring interval in minutes (default: 30)"
    )
    parser.add_argument(
        "--duration", "-d", type=int, default=24,
        help="Monitoring duration in hours (default: 24)"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run in headless mode"
    )

    args = parser.parse_args()

    if args.discover:
        asyncio.run(discover_overtime_api(headless=args.headless))
    elif args.scrape:
        target = (
            datetime.strptime(args.date, "%Y-%m-%d").date()
            if args.date else date.today()
        )
        games = asyncio.run(
            scrape_overtime_lines(target_date=target, headless=args.headless)
        )
        if games:
            save_vegas_lines(games, target.strftime("%Y-%m-%d"))
    elif args.monitor:
        asyncio.run(run_timing_monitor(
            interval=args.interval,
            duration=args.duration,
            headless=args.headless,
        ))
    else:
        print("Usage:")
        print("  python overtime_scraper.py --discover")
        print("  python overtime_scraper.py --scrape")
        print("  python overtime_scraper.py --scrape --date 2025-12-17")
        print("  python overtime_scraper.py --monitor --interval 30 --duration 24")
