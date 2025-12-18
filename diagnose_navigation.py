"""Diagnose overtime.ag navigation issues."""
import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from playwright.async_api import async_playwright

# URLs to test
OVERTIME_SPORTS_URL = "https://overtime.ag/sports#/"
OVERTIME_CBB_URLS = [
    "https://overtime.ag/sports#/basketball-college-basketball",
    "https://overtime.ag/sports#/basketball/college-basketball",
    "https://overtime.ag/sports#/college-basketball",
]


async def diagnose():
    """Diagnose navigation issues."""
    print("=" * 70)
    print("OVERTIME.AG NAVIGATION DIAGNOSTIC")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    screenshots_dir = Path(__file__).parent / "data" / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=100)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        )
        page = await context.new_page()

        # Test 1: Main sports page
        print("[TEST 1] Loading main sports page...")
        await page.goto(OVERTIME_SPORTS_URL)
        await page.wait_for_load_state("networkidle", timeout=30000)
        await asyncio.sleep(2)

        screenshot1 = screenshots_dir / "01_main_page.png"
        await page.screenshot(path=screenshot1, full_page=True)
        print(f"  URL: {page.url}")
        print(f"  Screenshot: {screenshot1}")

        # Get available sport options
        print("\n[TEST 2] Finding Basketball/College Basketball options...")
        try:
            # Try to find Basketball menu item
            basketball_elements = await page.query_selector_all('text=/Basketball/i')
            print(f"  Found {len(basketball_elements)} 'Basketball' elements")

            # Click first Basketball element if found
            if basketball_elements:
                print("  Clicking first Basketball element...")
                await basketball_elements[0].click()
                await asyncio.sleep(2)

                screenshot2 = screenshots_dir / "02_after_basketball_click.png"
                await page.screenshot(path=screenshot2, full_page=True)
                print(f"  Screenshot: {screenshot2}")

                # Look for College Basketball
                cbb_elements = await page.query_selector_all('text=/College Basketball/i')
                print(f"  Found {len(cbb_elements)} 'College Basketball' elements")

                if cbb_elements:
                    print("  Clicking first College Basketball element...")
                    await cbb_elements[0].click()
                    await page.wait_for_load_state("networkidle", timeout=30000)
                    await asyncio.sleep(2)

                    screenshot3 = screenshots_dir / "03_college_basketball_page.png"
                    await page.screenshot(path=screenshot3, full_page=True)
                    print(f"  URL: {page.url}")
                    print(f"  Screenshot: {screenshot3}")
                else:
                    print("  [WARNING] No 'College Basketball' elements found")

        except Exception as e:
            print(f"  [ERROR] Sidebar navigation failed: {e}")

        # Test 3: Direct URL navigation
        print("\n[TEST 3] Testing direct URLs...")
        for i, url in enumerate(OVERTIME_CBB_URLS, 1):
            try:
                print(f"\n  URL {i}: {url}")
                await page.goto(url)
                await page.wait_for_load_state("networkidle", timeout=30000)
                await asyncio.sleep(2)

                screenshot = screenshots_dir / f"04_direct_url_{i}.png"
                await page.screenshot(path=screenshot, full_page=True)
                print(f"    Final URL: {page.url}")
                print(f"    Screenshot: {screenshot}")

                # Check for game-related content
                page_text = await page.evaluate('() => document.body.innerText')
                has_games = any(word in page_text.lower() for word in ['spread', 'total', 'moneyline', 'odds'])
                print(f"    Game content detected: {has_games}")

            except Exception as e:
                print(f"    [ERROR] {e}")

        # Test 4: Check what sports are currently available
        print("\n[TEST 4] Checking available sports...")
        await page.goto(OVERTIME_SPORTS_URL)
        await page.wait_for_load_state("networkidle", timeout=30000)
        await asyncio.sleep(2)

        # Get all visible text
        all_text = await page.evaluate('() => document.body.innerText')

        # Look for sport names
        sports = ['basketball', 'football', 'baseball', 'hockey', 'soccer', 'nfl', 'nba', 'mlb', 'nhl']
        found_sports = [sport for sport in sports if sport in all_text.lower()]

        print(f"  Sports mentioned on page: {', '.join(found_sports)}")

        # Check if college basketball is mentioned
        if 'college basketball' in all_text.lower():
            print("  [OK] 'College Basketball' found in page text")
        else:
            print("  [WARNING] 'College Basketball' NOT found in page text")

        print("\n" + "=" * 70)
        print("DIAGNOSTIC COMPLETE")
        print("=" * 70)
        print(f"\nScreenshots saved to: {screenshots_dir}")
        print("\nReview screenshots to see:")
        print("  1. What the main page looks like")
        print("  2. What happens when clicking Basketball")
        print("  3. What each direct URL shows")
        print("\nKeeping browser open for 30 seconds for manual inspection...")
        print("Press Ctrl+C to close early.\n")

        await asyncio.sleep(30)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(diagnose())
