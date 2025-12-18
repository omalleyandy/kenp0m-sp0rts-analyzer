"""Check if NBA lines are available since college basketball isn't."""
import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from playwright.async_api import async_playwright


async def check_nba():
    """Check NBA availability."""
    print("=" * 70)
    print("CHECKING NBA AVAILABILITY ON OVERTIME.AG")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nSince College Basketball isn't available, checking NBA instead...\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=100)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
        )
        page = await context.new_page()

        # Navigate to main page
        print("[1] Loading overtime.ag sports page...")
        await page.goto("https://overtime.ag/sports#/")
        await page.wait_for_load_state("networkidle", timeout=30000)
        await asyncio.sleep(2)

        # Click Basketball
        print("[2] Clicking Basketball in sidebar...")
        basketball_elements = await page.query_selector_all('text=/^Basketball$/i')
        if basketball_elements:
            await basketball_elements[0].click()
            await asyncio.sleep(2)
            print("    Basketball menu expanded")

            # Click NBA
            print("[3] Clicking NBA...")
            nba_elements = await page.query_selector_all('text=/^NBA$/i')
            if nba_elements:
                await nba_elements[0].click()
                await page.wait_for_load_state("networkidle", timeout=30000)
                await asyncio.sleep(3)

                print(f"\n    Final URL: {page.url}")

                # Check for games
                page_text = await page.evaluate('() => document.body.innerText')

                # Look for game indicators
                has_spread = 'spread' in page_text.lower()
                has_total = 'total' in page_text.lower()
                has_money_line = 'money line' in page_text.lower()

                print(f"\n[RESULTS]")
                print(f"  Spread column: {has_spread}")
                print(f"  Total column: {has_total}")
                print(f"  Money line column: {has_money_line}")

                # Try to count games
                game_rows = await page.query_selector_all('tr:has(td)')
                print(f"  Potential game rows: {len(game_rows)}")

                # Take screenshot
                screenshots_dir = Path(__file__).parent / "data" / "screenshots"
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                screenshot = screenshots_dir / "nba_page.png"
                await page.screenshot(path=screenshot, full_page=True)
                print(f"\n  Screenshot: {screenshot}")

            else:
                print("    [ERROR] NBA element not found")
        else:
            print("    [ERROR] Basketball element not found")

        # Check what sports ARE available
        print("\n" + "=" * 70)
        print("AVAILABLE SPORTS ON OVERTIME.AG")
        print("=" * 70)

        await page.goto("https://overtime.ag/sports#/")
        await page.wait_for_load_state("networkidle", timeout=30000)

        # Get sidebar text
        sidebar = await page.query_selector('.sidebar, [class*="sidebar"], nav')
        if sidebar:
            sidebar_text = await sidebar.inner_text()
            sports_found = []
            for line in sidebar_text.split('\n'):
                line = line.strip()
                if line and len(line) < 30 and line.upper() == line:
                    continue  # Skip all-caps labels
                if line and len(line) < 30:
                    sports_found.append(line)

            print("\nSports/Leagues visible in sidebar:")
            for sport in sports_found[:30]:  # First 30 items
                print(f"  - {sport}")

        print("\n" + "=" * 70)
        print("\nKEY FINDING: College Basketball is NOT available on overtime.ag")
        print("             Only NBA, EUROLEAGUE, BRAZIL NBB, CHINA CBA are shown")
        print("\nPOSSIBLE REASONS:")
        print("  1. Off-season (but December should be active)")
        print("  2. overtime.ag may have removed college basketball")
        print("  3. May need special access or different account")
        print("\nRECOMMENDATION:")
        print("  - Check overtime.ag manually to confirm availability")
        print("  - May need to find alternative source for college basketball odds")
        print("  - NBA monitoring can still work with this infrastructure")
        print("=" * 70)

        print("\nKeeping browser open for 30 seconds...")
        await asyncio.sleep(30)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(check_nba())
