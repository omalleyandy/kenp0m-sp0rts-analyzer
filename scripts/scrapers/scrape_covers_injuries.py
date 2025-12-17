"""Scrape college basketball injury reports from Covers.com.

This scraper uses Playwright to load the JavaScript-heavy Covers.com injury page
and extracts current injury data for all NCAA Division I teams.

Usage:
    python scripts/scrapers/scrape_covers_injuries.py
    python scripts/scrapers/scrape_covers_injuries.py --output injuries.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright


def scrape_covers_injuries(headless: bool = True) -> list[dict]:
    """Scrape NCAAB injury report from Covers.com.

    Args:
        headless: Run browser in headless mode (default: True)

    Returns:
        List of injury dictionaries with team, player, status, etc.
    """
    print("\n" + "="*80)
    print("COVERS.COM INJURY SCRAPER")
    print("="*80 + "\n")

    injuries = []

    with sync_playwright() as p:
        # Launch browser
        print(f"Launching browser (headless={headless})...")
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()

        # Navigate to injury page
        url = 'https://www.covers.com/sport/basketball/ncaab/injuries'
        print(f"Loading: {url}")
        page.goto(url, wait_until='networkidle')

        # Wait for page to fully load
        print("Waiting for content to load...")
        page.wait_for_timeout(3000)  # Give JavaScript time to render

        # Try to find injury data
        # Covers.com uses various class names, we'll try multiple selectors
        print("Extracting injury data...")

        # Method 1: Try to find injury table rows
        injury_rows = page.query_selector_all('tr.injury-row, tr[class*="injury"], tbody tr')

        if injury_rows:
            print(f"Found {len(injury_rows)} potential injury rows")

            for row in injury_rows:
                try:
                    # Extract text from all cells
                    cells = row.query_selector_all('td')
                    if len(cells) >= 4:  # Need at least team, player, status, injury
                        injury = {
                            'team': cells[0].text_content().strip() if len(cells) > 0 else '',
                            'player': cells[1].text_content().strip() if len(cells) > 1 else '',
                            'position': cells[2].text_content().strip() if len(cells) > 2 else '',
                            'status': cells[3].text_content().strip() if len(cells) > 3 else '',
                            'injury_type': cells[4].text_content().strip() if len(cells) > 4 else '',
                            'date_updated': cells[5].text_content().strip() if len(cells) > 5 else '',
                            'scraped_at': datetime.now().isoformat()
                        }

                        # Only add if we have team and player
                        if injury['team'] and injury['player']:
                            injuries.append(injury)

                except Exception as e:
                    print(f"Warning: Error parsing row: {e}")
                    continue

        # Method 2: If no structured table, try to extract from text
        if not injuries:
            print("No structured table found, trying alternative extraction...")

            # Look for team sections
            team_sections = page.query_selector_all('div[class*="team"], div[class*="injury"]')
            print(f"Found {len(team_sections)} potential sections")

            # Save page content for debugging
            content = page.content()
            debug_file = Path('data/covers_debug.html')
            debug_file.parent.mkdir(exist_ok=True)
            debug_file.write_text(content, encoding='utf-8')
            print(f"Saved page HTML to {debug_file} for debugging")

        browser.close()

    print(f"\n[OK] Scraped {len(injuries)} injuries")
    return injuries


def save_injuries(injuries: list[dict], output_file: str | Path):
    """Save injuries to JSON file.

    Args:
        injuries: List of injury dictionaries
        output_file: Path to output JSON file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'scraped_at': datetime.now().isoformat(),
            'source': 'covers.com',
            'count': len(injuries),
            'injuries': injuries
        }, f, indent=2)

    print(f"[OK] Saved {len(injuries)} injuries to {output_path}")


def print_injury_summary(injuries: list[dict]):
    """Print summary of scraped injuries.

    Args:
        injuries: List of injury dictionaries
    """
    print("\n" + "="*80)
    print("INJURY SUMMARY")
    print("="*80 + "\n")

    if not injuries:
        print("No injuries found!")
        return

    # Group by status
    by_status = {}
    for injury in injuries:
        status = injury.get('status', 'Unknown')
        by_status.setdefault(status, []).append(injury)

    print(f"Total Injuries: {len(injuries)}\n")

    for status, inj_list in sorted(by_status.items()):
        print(f"{status}: {len(inj_list)}")
        for inj in inj_list[:5]:  # Show first 5
            print(f"  - {inj.get('player', 'Unknown')} ({inj.get('team', 'Unknown')})")
        if len(inj_list) > 5:
            print(f"  ... and {len(inj_list) - 5} more")
        print()

    # Show teams with multiple injuries
    by_team = {}
    for injury in injuries:
        team = injury.get('team', 'Unknown')
        by_team.setdefault(team, []).append(injury)

    teams_multiple = {t: inj for t, inj in by_team.items() if len(inj) > 1}
    if teams_multiple:
        print("Teams with Multiple Injuries:")
        for team, inj_list in sorted(teams_multiple.items(), key=lambda x: -len(x[1])):
            print(f"  {team}: {len(inj_list)} injured")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Scrape college basketball injuries from Covers.com'
    )
    parser.add_argument(
        '--output',
        '-o',
        default='data/injuries_covers.json',
        help='Output JSON file (default: data/injuries_covers.json)'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        default=True,
        help='Run browser in headless mode (default: True)'
    )
    parser.add_argument(
        '--show-browser',
        action='store_true',
        help='Show browser window (overrides --headless)'
    )

    args = parser.parse_args()

    # Scrape injuries
    headless = args.headless and not args.show_browser
    injuries = scrape_covers_injuries(headless=headless)

    # Save to file
    save_injuries(injuries, args.output)

    # Print summary
    print_injury_summary(injuries)

    print("\n" + "="*80)
    print("SCRAPING COMPLETE")
    print("="*80)

    # Return non-zero exit code if no injuries found (might indicate scraping issue)
    if not injuries:
        print("\n[WARN] No injuries found - scraper may need updating")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
