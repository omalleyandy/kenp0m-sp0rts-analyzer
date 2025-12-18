#!/usr/bin/env python3
"""Scrape College Basketball lines from overtime.ag using browser automation.

This script uses Playwright to fetch betting lines through the browser,
which bypasses the server's blocking of direct API requests.

Usage:
    uv run python scripts/scrapers/scrape_overtime_cbb.py
    uv run python scripts/scrapers/scrape_overtime_cbb.py -o lines.json
    uv run python scripts/scrapers/scrape_overtime_cbb.py --include-extra
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from playwright.async_api import async_playwright


async def scrape_college_basketball_lines(
    include_extra: bool = False,
) -> list[dict]:
    """Scrape college basketball lines from overtime.ag.

    Args:
        include_extra: Also fetch "College Extra" games (smaller schools)

    Returns:
        List of game dictionaries with betting lines.
    """
    games = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Navigate to establish session
        await page.goto("https://overtime.ag/sports", timeout=30000)
        await asyncio.sleep(2)

        # Fetch College Basketball lines via browser's fetch API
        subtypes = ["College Basketball"]
        if include_extra:
            subtypes.append("College Extra")

        for subtype in subtypes:
            result = await page.evaluate(
                """
                async (sportSubType) => {
                    try {
                        const r = await fetch(
                            '/sports/Api/Offering.asmx/GetSportOffering',
                            {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({
                                    sportType: 'Basketball',
                                    sportSubType: sportSubType,
                                    wagerType: 'Straight Bet',
                                    hoursAdjustment: 0,
                                    periodNumber: null,
                                    gameNum: null,
                                    parentGameNum: null,
                                    teaserName: '',
                                    requestMode: null
                                })
                            }
                        );
                        return await r.json();
                    } catch(e) {
                        return {error: e.toString()};
                    }
                }
            """,
                subtype,
            )

            if "error" in result:
                print(f"[WARN] Error fetching {subtype}: {result['error']}")
                continue

            if "d" in result and "Data" in result["d"]:
                all_lines = result["d"]["Data"].get("GameLines", [])

                # Filter for full game lines only (not halves/quarters)
                for gl in all_lines:
                    if gl.get("PeriodDescription") != "Game":
                        continue

                    away = gl.get("Team1ID", "").strip()
                    home = gl.get("Team2ID", "").strip()

                    if not away or not home:
                        continue

                    games.append(
                        {
                            "game_date": gl.get("GameDate", ""),
                            "game_time": gl.get("GameDateTimeString", "")
                            .split(" ")[-1]
                            if gl.get("GameDateTimeString")
                            else "",
                            "away_team": away,
                            "home_team": home,
                            "spread": gl.get("Spread2"),
                            "total": gl.get("TotalPoints1"),
                            "away_ml": gl.get("MoneyLine1"),
                            "home_ml": gl.get("MoneyLine2"),
                            "away_rot": gl.get("Team1RotNum"),
                            "home_rot": gl.get("Team2RotNum"),
                            "category": subtype.lower().replace(" ", "_"),
                        }
                    )

        await browser.close()

    return games


def save_lines(games: list[dict], output_path: Path) -> None:
    """Save games to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "scraped_at": datetime.now().isoformat(),
        "source": "overtime.ag",
        "game_count": len(games),
        "games": games,
    }

    with output_path.open("w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape College Basketball lines from overtime.ag"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/overtime_cbb_lines.json",
        help="Output JSON file (default: data/overtime_cbb_lines.json)",
    )
    parser.add_argument(
        "--include-extra",
        action="store_true",
        help="Include College Extra games (smaller schools)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("OVERTIME.AG - COLLEGE BASKETBALL LINES SCRAPER")
    print("=" * 80)

    # Scrape lines
    print("\nFetching lines (this may take a few seconds)...")
    games = asyncio.run(scrape_college_basketball_lines(args.include_extra))

    if not games:
        print("\n[WARN] No games found")
        return 1

    print(f"\n[OK] Found {len(games)} games\n")

    # Display games
    print(f"{'MATCHUP':<50} {'SPREAD':>8} {'TOTAL':>8} {'ML':>14}")
    print("-" * 85)

    for g in games:
        away = g["away_team"][:22]
        home = g["home_team"][:22]
        matchup = f"{away} @ {home}"
        spread = f"{g['spread']:+.1f}" if g["spread"] is not None else "N/A"
        total = f"{g['total']:.1f}" if g["total"] is not None else "N/A"
        ml = (
            f"{g['away_ml']}/{g['home_ml']}"
            if g["away_ml"] and g["home_ml"]
            else "N/A"
        )
        print(f"{matchup:<50} {spread:>8} {total:>8} {ml:>14}")

    # Save to file
    output_path = Path(args.output)
    save_lines(games, output_path)
    print(f"\n[OK] Saved to {output_path}")

    print("\n" + "=" * 80)
    return 0


if __name__ == "__main__":
    exit(main())
