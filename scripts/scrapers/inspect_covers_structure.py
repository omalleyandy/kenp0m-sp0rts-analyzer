"""Inspect Covers.com injury page structure to find correct selectors.

This diagnostic tool loads the page and helps identify the correct CSS selectors
for extracting injury data.
"""

from pathlib import Path

from playwright.sync_api import sync_playwright


def inspect_page_structure():
    """Load Covers.com injury page and analyze structure."""
    print("\n" + "="*80)
    print("COVERS.COM PAGE STRUCTURE INSPECTOR")
    print("="*80 + "\n")

    with sync_playwright() as p:
        # Launch browser with visible window
        print("Launching browser...")
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # Navigate to injury page
        url = 'https://www.covers.com/sport/basketball/ncaab/injuries'
        print(f"Loading: {url}")
        page.goto(url, wait_until='networkidle')

        # Wait for content
        print("Waiting for page to fully load...")
        page.wait_for_timeout(5000)

        # Save full page source
        html_content = page.content()
        output_file = Path('data/covers_full_page.html')
        output_file.parent.mkdir(exist_ok=True)
        output_file.write_text(html_content, encoding='utf-8')
        print(f"[OK] Saved full page HTML to {output_file}")

        # Try to find injury-related elements
        print("\n" + "-"*80)
        print("SEARCHING FOR INJURY DATA ELEMENTS")
        print("-"*80 + "\n")

        # Look for various table patterns
        selectors_to_try = [
            "table",
            "div.injury-report",
            "div.injuries",
            "div[class*='injury']",
            "div[class*='player']",
            "tbody tr",
            ".table-container",
            "[data-injury]",
            "article",
            "main",
        ]

        for selector in selectors_to_try:
            elements = page.query_selector_all(selector)
            if elements:
                print(f"[{len(elements):3d} found] {selector}")

                # For tables, show first row structure
                if selector in ["table", "tbody tr"]:
                    for i, elem in enumerate(elements[:3]):
                        print(f"        Element {i+1} classes: {elem.get_attribute('class')}")
                        # Get text preview
                        text = elem.text_content()[:100] if elem.text_content() else ""
                        if text and "cookie" not in text.lower():
                            print(f"        Preview: {text}...")

        # Look for specific team names to locate injury data
        print("\n" + "-"*80)
        print("SEARCHING FOR KNOWN TEAM NAMES")
        print("-"*80 + "\n")

        test_teams = ["Duke", "Kentucky", "Kansas", "North Carolina", "Louisville"]
        for team in test_teams:
            team_elems = page.get_by_text(team, exact=False)
            count = team_elems.count()
            if count > 0:
                print(f"[{count:2d} found] '{team}'")
                # Get parent element info
                if count > 0:
                    first = team_elems.first
                    parent = first.evaluate("el => el.parentElement.className")
                    grandparent = first.evaluate("el => el.parentElement.parentElement.className")
                    print(f"        Parent class: {parent}")
                    print(f"        Grandparent class: {grandparent}")

        # Get main content area
        print("\n" + "-"*80)
        print("MAIN CONTENT STRUCTURE")
        print("-"*80 + "\n")

        main = page.query_selector("main, #main-content, .main, article")
        if main:
            print("[OK] Found main content area")
            main_html = main.inner_html()[:500]
            print(f"Preview:\n{main_html}...")
        else:
            print("[WARN] Could not find main content area")

        # Wait for user to inspect
        print("\n" + "="*80)
        print("Browser window is open. Inspect the page structure.")
        print("Press Enter when done inspecting...")
        print("="*80)
        input()

        browser.close()

    print("\n[OK] Inspection complete")
    print(f"[OK] Page source saved to {output_file}")
    print("\nNext: Analyze the HTML file to find injury table selectors")


if __name__ == '__main__':
    inspect_page_structure()
