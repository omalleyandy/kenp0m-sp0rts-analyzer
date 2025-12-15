"""Browser automation with stealth configuration for KenPom scraping.

This module provides a Playwright-based browser with stealth techniques
to access KenPom.com in a way that mimics regular browser behavior.

Requirements:
    pip install kenp0m-sp0rts-analyzer[browser]
    playwright install chromium
"""

import asyncio
import logging
import random
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default viewport sizes that mimic real devices
VIEWPORT_SIZES = [
    {"width": 1920, "height": 1080},
    {"width": 1366, "height": 768},
    {"width": 1536, "height": 864},
    {"width": 1440, "height": 900},
    {"width": 1280, "height": 720},
]

# Common user agents for Chrome on different platforms
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]


@dataclass
class BrowserConfig:
    """Configuration for stealth browser."""

    headless: bool = False
    slow_mo: int = 50  # Milliseconds between actions
    timeout: int = 30000  # Default timeout in ms
    viewport: dict[str, int] = field(
        default_factory=lambda: {"width": 1920, "height": 1080}
    )
    user_agent: str | None = None
    locale: str = "en-US"
    timezone_id: str = "America/New_York"

    # Stealth options
    disable_webdriver: bool = True
    randomize_viewport: bool = True
    randomize_user_agent: bool = True

    # Chrome DevTools Protocol options
    enable_cdp: bool = True

    # Persistence
    user_data_dir: Path | None = None

    def __post_init__(self) -> None:
        """Apply randomization if enabled."""
        if self.randomize_viewport:
            self.viewport = random.choice(VIEWPORT_SIZES)
        if self.randomize_user_agent and not self.user_agent:
            self.user_agent = random.choice(USER_AGENTS)


class StealthBrowser:
    """Playwright browser with stealth techniques and CDP access.

    This browser is configured to:
    - Mimic real Chrome browser fingerprints
    - Disable webdriver detection flags
    - Use realistic viewport sizes and user agents
    - Support Chrome DevTools Protocol for advanced control
    - Persist sessions across runs (optional)

    Example:
        ```python
        async with StealthBrowser(headless=False) as browser:
            page = await browser.new_page()
            await page.goto("https://kenpom.com")

            # Access CDP session for advanced control
            cdp = await page.context.new_cdp_session(page)
            await cdp.send("Network.enable")
        ```
    """

    def __init__(self, config: BrowserConfig | None = None, **kwargs: Any) -> None:
        """Initialize the stealth browser.

        Args:
            config: BrowserConfig instance. If not provided, creates one from kwargs.
            **kwargs: Arguments passed to BrowserConfig if config not provided.
        """
        self.config = config or BrowserConfig(**kwargs)
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None

    async def __aenter__(self) -> "StealthBrowser":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def start(self) -> None:
        """Start the browser with stealth configuration."""
        try:
            from playwright.async_api import async_playwright
            from playwright_stealth import stealth_async
        except ImportError as e:
            raise ImportError(
                "Browser automation requires extra dependencies. "
                "Install with: pip install kenp0m-sp0rts-analyzer[browser] && playwright install chromium"
            ) from e

        self._playwright = await async_playwright().start()
        self._stealth_async = stealth_async

        # Browser launch arguments for stealth
        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--disable-infobars",
            "--disable-background-networking",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-breakpad",
            "--disable-component-extensions-with-background-pages",
            "--disable-component-update",
            "--disable-default-apps",
            "--disable-extensions",
            "--disable-features=TranslateUI",
            "--disable-hang-monitor",
            "--disable-ipc-flooding-protection",
            "--disable-popup-blocking",
            "--disable-prompt-on-repost",
            "--disable-renderer-backgrounding",
            "--disable-sync",
            "--enable-features=NetworkService,NetworkServiceInProcess",
            "--force-color-profile=srgb",
            "--metrics-recording-only",
            "--no-first-run",
            "--password-store=basic",
            "--use-mock-keychain",
        ]

        # Launch browser
        launch_options: dict[str, Any] = {
            "headless": self.config.headless,
            "slow_mo": self.config.slow_mo,
            "args": launch_args,
        }

        # Use persistent context if user_data_dir specified
        if self.config.user_data_dir:
            self.config.user_data_dir.mkdir(parents=True, exist_ok=True)
            self._context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.config.user_data_dir),
                **launch_options,
                viewport=self.config.viewport,
                user_agent=self.config.user_agent,
                locale=self.config.locale,
                timezone_id=self.config.timezone_id,
                permissions=["geolocation"],
                geolocation={"latitude": 40.7128, "longitude": -74.0060},  # NYC
                color_scheme="light",
                java_script_enabled=True,
            )
        else:
            self._browser = await self._playwright.chromium.launch(**launch_options)
            self._context = await self._browser.new_context(
                viewport=self.config.viewport,
                user_agent=self.config.user_agent,
                locale=self.config.locale,
                timezone_id=self.config.timezone_id,
                permissions=["geolocation"],
                geolocation={"latitude": 40.7128, "longitude": -74.0060},
                color_scheme="light",
                java_script_enabled=True,
            )

        # Set default timeout
        self._context.set_default_timeout(self.config.timeout)

        logger.info(
            f"Browser started (headless={self.config.headless}, "
            f"viewport={self.config.viewport['width']}x{self.config.viewport['height']})"
        )

    async def close(self) -> None:
        """Close the browser and cleanup."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser closed")

    async def new_page(self) -> Any:
        """Create a new page with stealth applied.

        Returns:
            Playwright Page object with stealth configuration.
        """
        if not self._context:
            raise RuntimeError(
                "Browser not started. Call start() or use async context manager."
            )

        page = await self._context.new_page()

        # Apply stealth techniques
        await self._stealth_async(page)

        # Additional stealth via CDP
        if self.config.enable_cdp:
            cdp = await self._context.new_cdp_session(page)

            # Remove webdriver flag
            if self.config.disable_webdriver:
                await cdp.send(
                    "Page.addScriptToEvaluateOnNewDocument",
                    {
                        "source": """
                            Object.defineProperty(navigator, 'webdriver', {
                                get: () => undefined
                            });

                            // Overwrite plugins
                            Object.defineProperty(navigator, 'plugins', {
                                get: () => [1, 2, 3, 4, 5]
                            });

                            // Overwrite languages
                            Object.defineProperty(navigator, 'languages', {
                                get: () => ['en-US', 'en']
                            });

                            // Mock chrome object
                            window.chrome = {
                                runtime: {}
                            };

                            // Mock permissions
                            const originalQuery = window.navigator.permissions.query;
                            window.navigator.permissions.query = (parameters) => (
                                parameters.name === 'notifications' ?
                                    Promise.resolve({ state: Notification.permission }) :
                                    originalQuery(parameters)
                            );
                        """
                    },
                )

        return page

    async def get_cdp_session(self, page: Any) -> Any:
        """Get Chrome DevTools Protocol session for a page.

        Args:
            page: Playwright Page object.

        Returns:
            CDP session for advanced browser control.
        """
        return await self._context.new_cdp_session(page)

    @property
    def context(self) -> Any:
        """Get the browser context."""
        return self._context


@asynccontextmanager
async def create_stealth_browser(
    headless: bool = False,
    **kwargs: Any,
) -> AsyncGenerator[StealthBrowser, None]:
    """Create a stealth browser as an async context manager.

    Args:
        headless: Run browser in headless mode (default: False for visible browser).
        **kwargs: Additional arguments for BrowserConfig.

    Yields:
        StealthBrowser instance.

    Example:
        ```python
        async with create_stealth_browser(headless=False) as browser:
            page = await browser.new_page()
            await page.goto("https://kenpom.com")
        ```
    """
    browser = StealthBrowser(headless=headless, **kwargs)
    try:
        await browser.start()
        yield browser
    finally:
        await browser.close()


def run_sync(coro: Any) -> Any:
    """Run an async coroutine synchronously.

    Useful for running async browser code in synchronous contexts.

    Args:
        coro: Async coroutine to run.

    Returns:
        Result of the coroutine.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)
