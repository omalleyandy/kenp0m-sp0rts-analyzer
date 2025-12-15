#!/usr/bin/env python3
"""Stealth browser scraper example for KenPom.

This script demonstrates how to use the stealth browser to scrape
KenPom.com with a visible Chrome browser that mimics real user behavior.

Requirements:
    pip install kenp0m-sp0rts-analyzer[browser]
    playwright install chromium

Environment Variables:
    KENPOM_EMAIL - Your kenpom.com subscription email
    KENPOM_PASSWORD - Your kenpom.com subscription password
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Run the stealth scraper example."""
    load_dotenv()

    # Verify credentials
    if not os.getenv("KENPOM_EMAIL") or not os.getenv("KENPOM_PASSWORD"):
        print("Error: KENPOM_EMAIL and KENPOM_PASSWORD environment variables required")
        print("\nCreate a .env file with:")
        print("  KENPOM_EMAIL=your-email@example.com")
        print("  KENPOM_PASSWORD=your-password")
        sys.exit(1)

    try:
        from kenp0m_sp0rts_analyzer import KenPomScraper, create_stealth_browser
    except ImportError as e:
        print("Error: Browser dependencies not installed")
        print("\nInstall with:")
        print("  pip install kenp0m-sp0rts-analyzer[browser]")
        print("  playwright install chromium")
        sys.exit(1)

    print("=" * 60)
    print("KenPom Stealth Scraper Example")
    print("=" * 60)

    # Create a persistent data directory for browser session
    user_data_dir = Path.home() / ".kenpom_browser_data"

    # Use the KenPomScraper with headless=False for visible browser
    async with KenPomScraper(
        headless=False,  # Set to True for invisible browser
        user_data_dir=user_data_dir,  # Persist login sessions
        slow_mo=100,  # Slow down actions for visibility
    ) as scraper:
        # Login to KenPom
        print("\n[1] Logging in to KenPom...")
        await scraper.login()
        print("    Successfully logged in!")

        # Take a screenshot of the main page
        print("\n[2] Taking screenshot of main page...")
        await scraper.screenshot("kenpom_main.png")
        print("    Saved: kenpom_main.png")

        # Get the main ratings
        print("\n[3] Fetching ratings table...")
        ratings = await scraper.get_ratings()
        print(f"    Retrieved {len(ratings)} teams")
        print("\n    Top 10 Teams:")
        print(ratings.head(10).to_string())

        # Get efficiency data
        print("\n[4] Fetching efficiency data...")
        try:
            efficiency = await scraper.get_efficiency()
            print(f"    Retrieved efficiency data for {len(efficiency)} teams")
        except Exception as e:
            logger.warning(f"Could not fetch efficiency: {e}")

        # Try to get API documentation
        print("\n[5] Fetching API documentation...")
        try:
            api_docs = await scraper.get_api_documentation()
            print(f"    Title: {api_docs.get('title', 'N/A')}")
            print(f"    Sections found: {len(api_docs.get('sections', []))}")
            print(f"    Endpoints found: {len(api_docs.get('endpoints', []))}")

            if api_docs.get("sections"):
                print("\n    Documentation Sections:")
                for section in api_docs["sections"][:10]:
                    print(f"      - [{section['level']}] {section['title']}")

            if api_docs.get("endpoints"):
                print("\n    API Endpoints/URLs Found:")
                for endpoint in api_docs["endpoints"][:10]:
                    print(f"      - {endpoint}")
        except Exception as e:
            logger.warning(f"Could not fetch API docs: {e}")

        # Access CDP for advanced operations
        print("\n[6] Chrome DevTools Protocol access...")
        cdp = await scraper.get_cdp_session()
        print("    CDP session established")

        # Example: Get performance metrics via CDP
        await cdp.send("Performance.enable")
        metrics = await cdp.send("Performance.getMetrics")
        print("    Performance metrics available:")
        for metric in metrics.get("metrics", [])[:5]:
            print(f"      - {metric['name']}: {metric['value']:.2f}")

        print("\n" + "=" * 60)
        print("Scraping complete!")
        print("=" * 60)

        # Keep browser open for inspection (optional)
        print("\nBrowser will close in 5 seconds...")
        print("(Increase this delay to inspect the browser)")
        await asyncio.sleep(5)


async def demo_stealth_browser():
    """Demonstrate the low-level stealth browser API."""
    from kenp0m_sp0rts_analyzer.browser import create_stealth_browser

    print("\n" + "=" * 60)
    print("Low-Level Stealth Browser Demo")
    print("=" * 60)

    async with create_stealth_browser(
        headless=False,
        slow_mo=50,
        randomize_viewport=True,
        randomize_user_agent=True,
    ) as browser:
        page = await browser.new_page()

        # Navigate to KenPom
        print("\nNavigating to kenpom.com...")
        await page.goto("https://kenpom.com")

        # Check webdriver detection
        is_webdriver = await page.evaluate("navigator.webdriver")
        print(f"navigator.webdriver = {is_webdriver}")

        # Check user agent
        user_agent = await page.evaluate("navigator.userAgent")
        print(f"User Agent: {user_agent[:60]}...")

        # Check viewport
        viewport = await page.evaluate(
            "({ width: window.innerWidth, height: window.innerHeight })"
        )
        print(f"Viewport: {viewport['width']}x{viewport['height']}")

        # Take screenshot
        await page.screenshot(path="stealth_demo.png")
        print("\nScreenshot saved: stealth_demo.png")

        await asyncio.sleep(3)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KenPom stealth scraper example")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run low-level stealth browser demo instead of full scraper",
    )

    args = parser.parse_args()

    if args.demo:
        asyncio.run(demo_stealth_browser())
    else:
        asyncio.run(main())
