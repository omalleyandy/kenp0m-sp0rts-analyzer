"""Scrape college basketball injury reports from Covers.com (Fixed Version).

This scraper uses the correct selectors for Covers.com's injury page structure.

Usage:
    python scripts/scrapers/scrape_covers_injuries_v2.py
    python scripts/scrapers/scrape_covers_injuries_v2.py --output injuries.json
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright


def extract_team_name(team_container):
    """Extract team name from container.

    Args:
        team_container: Playwright element for team injury container

    Returns:
        Team name string or None
    """
    try:
        # Team name is in the header within the injury container
        team_elem = team_container.evaluate(
            """el => {
                let teamName = el.querySelector('.covers-CoversMatchups-teamName');
                return teamName ? teamName.textContent.trim() : null;
            }"""
        )
        return team_elem
    except Exception as e:
        return None


def scrape_covers_injuries(headless: bool = True) -> list[dict]:
    """Scrape NCAAB injury report from Covers.com.

    Args:
        headless: Run browser in headless mode (default: True)

    Returns:
        List of injury dictionaries with team, player, status, etc.
    """
    print("\n" + "="*80)
    print("COVERS.COM INJURY SCRAPER V2")
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
        page.wait_for_timeout(5000)

        # Find all team injury containers
        print("Extracting injury data...")
        team_containers = page.query_selector_all('.covers-CoversSeasonInjuries-blockContainer')

        print(f"Found {len(team_containers)} team sections")

        for team_container in team_containers:
            try:
                # Get team name
                team_elem = team_container.query_selector('.covers-CoversMatchups-teamName')
                if not team_elem:
                    continue

                # Use inner_text to preserve line breaks, then normalize whitespace
                team_name = team_elem.inner_text().strip()
                team_name = ' '.join(team_name.split())  # Normalize whitespace

                # Find injury table within this team's section
                injury_table = team_container.query_selector('table')
                if not injury_table:
                    continue

                # Get all injury rows (skip header row)
                injury_rows = injury_table.query_selector_all('tbody > tr')

                for row in injury_rows:
                    try:
                        # Skip collapsible detail rows
                        if 'collapse' in (row.get_attribute('class') or ''):
                            continue

                        # Extract cells
                        cells = row.query_selector_all('td')
                        if len(cells) < 3:
                            continue

                        # Parse injury data
                        # Structure: [Name] [Position] [Status - Injury] [Date] [Expand Icon]
                        player_name = ' '.join(cells[0].inner_text().split())
                        position = ' '.join(cells[1].inner_text().split())

                        # Status cell contains: "<b>Status - InjuryType</b><br>(Date)"
                        status_cell = cells[2]
                        status_text = status_cell.inner_text().strip()

                        # Parse status and injury type
                        status = "Unknown"
                        injury_type = ""
                        date_updated = ""

                        # Extract from bold tag
                        bold_elem = status_cell.query_selector('b')
                        if bold_elem:
                            status_injury = bold_elem.text_content().strip()
                            # Split "Status - Injury"
                            if ' - ' in status_injury:
                                status, injury_type = status_injury.split(' - ', 1)
                            else:
                                status = status_injury

                        # Extract date
                        date_match = re.search(r'\((.*?)\)', status_text)
                        if date_match:
                            date_updated = date_match.group(1).strip()

                        # Only add if we have meaningful data
                        if player_name and team_name:
                            injury = {
                                'team': team_name,
                                'player': player_name,
                                'position': position,
                                'status': status.strip(),
                                'injury_type': injury_type.strip(),
                                'date_updated': date_updated,
                                'scraped_at': datetime.now().isoformat()
                            }
                            injuries.append(injury)

                    except Exception as e:
                        print(f"Warning: Error parsing injury row: {e}")
                        continue

            except Exception as e:
                print(f"Warning: Error processing team container: {e}")
                continue

        browser.close()

    print(f"\n[OK] Scraped {len(injuries)} injuries from {len(set(i['team'] for i in injuries))} teams")
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
            'teams': len(set(i['team'] for i in injuries)),
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

    print(f"Total Injuries: {len(injuries)}")
    print(f"Teams Affected: {len(set(i['team'] for i in injuries))}\n")

    print("By Status:")
    for status, inj_list in sorted(by_status.items(), key=lambda x: -len(x[1])):
        print(f"  {status}: {len(inj_list)}")

    print()

    # Show recent major injuries (Out/Doubtful)
    major_injuries = [i for i in injuries if i.get('status') in ['Out', 'Doubtful']]
    if major_injuries:
        print(f"Major Injuries (Out/Doubtful): {len(major_injuries)}")
        for inj in major_injuries[:10]:
            print(f"  - {inj['player']} ({inj['team']}) - {inj['status']} ({inj['injury_type']})")
        if len(major_injuries) > 10:
            print(f"  ... and {len(major_injuries) - 10} more")
        print()

    # Show teams with multiple injuries
    by_team = {}
    for injury in injuries:
        team = injury.get('team', 'Unknown')
        by_team.setdefault(team, []).append(injury)

    teams_multiple = {t: inj for t, inj in by_team.items() if len(inj) >= 3}
    if teams_multiple:
        print(f"Teams with 3+ Injuries ({len(teams_multiple)} teams):")
        for team, inj_list in sorted(teams_multiple.items(), key=lambda x: -len(x[1]))[:10]:
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

    if not injuries:
        print("\n[WARN] No injuries found - scraper may need updating")
        return 1

    # Save to file
    save_injuries(injuries, args.output)

    # Print summary
    print_injury_summary(injuries)

    print("\n" + "="*80)
    print("SCRAPING COMPLETE")
    print("="*80)

    return 0


if __name__ == '__main__':
    exit(main())
