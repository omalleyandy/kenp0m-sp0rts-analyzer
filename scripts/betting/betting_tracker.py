#!/usr/bin/env python3
"""Betting Tracker for NCAA CBB - Track picks, open bets, and daily results.

This script provides a complete betting workflow:
1. Generate betting slips (Excel) from KenPom analysis
2. Track open bets from overtime.ag
3. Track daily figures (P&L) from overtime.ag
4. Analyze end-of-day results

Usage:
    # Generate betting slip for today
    uv run python scripts/betting/betting_tracker.py slip

    # Track open bets
    uv run python scripts/betting/betting_tracker.py open-bets

    # Get daily figures
    uv run python scripts/betting/betting_tracker.py daily-figures

    # Analyze results (run at end of day)
    uv run python scripts/betting/betting_tracker.py analyze --date 2025-12-18

    # Full workflow
    uv run python scripts/betting/betting_tracker.py all
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from playwright.async_api import async_playwright

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.utils import get_overtime_credentials, normalize_team_name


# ============================================================================
# BETTING SLIP GENERATION
# ============================================================================


def generate_betting_slip(
    date: str | None = None,
    min_edge: float = 3.0,
    output_dir: Path | None = None,
) -> Path:
    """Generate Excel betting slip from KenPom analysis.

    Args:
        date: Game date in YYYY-MM-DD format (default: today)
        min_edge: Minimum edge to include in slip
        output_dir: Output directory (default: reports/bet_slips/)

    Returns:
        Path to generated Excel file.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "reports" / "bet_slips"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Format date for filename (YYYYMMDD)
    date_suffix = date.replace("-", "")
    output_path = output_dir / f"NCAA CBB Betting Slip {date_suffix}.xlsx"

    print(f"[INFO] Generating betting slip for {date}...")

    # Load Vegas lines
    vegas_path = (
        Path(__file__).parent.parent.parent / "data" / "overtime_cbb_lines.json"
    )
    if not vegas_path.exists():
        print("[ERROR] No Vegas lines found. Run scrape_overtime_cbb.py first.")
        return None

    with open(vegas_path) as f:
        vegas_data = json.load(f)

    # Build Vegas lookup
    vegas_lines = {}
    for g in vegas_data["games"]:
        away = normalize_team_name(g["away_team"])
        home = normalize_team_name(g["home_team"])
        vegas_lines[(away, home)] = {
            "spread": g["spread"],
            "total": g["total"],
            "away_ml": g["away_ml"],
            "home_ml": g["home_ml"],
            "away_team": g["away_team"],
            "home_team": g["home_team"],
        }

    # Get KenPom predictions
    api = KenPomAPI()
    fanmatch = api.get_fanmatch(date)

    if not fanmatch.data:
        print(f"[ERROR] No games found for {date}")
        return None

    # Analyze games and find value picks
    spread_picks = []
    total_picks = []

    for game in fanmatch.data:
        visitor = game.get("Visitor", "")
        home = game.get("Home", "")

        visitor_pred = game["VisitorPred"]
        home_pred = game["HomePred"]
        home_wp = game["HomeWP"]

        # Calculate KenPom spread (positive = home underdog)
        kenpom_spread = home_pred - visitor_pred
        kp_spread = -kenpom_spread  # Convert to spread notation
        kenpom_total = visitor_pred + home_pred

        # Look up Vegas
        norm_away = normalize_team_name(visitor)
        norm_home = normalize_team_name(home)
        vegas = vegas_lines.get((norm_away, norm_home), {})

        if not vegas:
            continue

        v_spread = vegas.get("spread")
        v_total = vegas.get("total")

        if v_spread is not None:
            spread_edge = kp_spread - v_spread

            if abs(spread_edge) >= min_edge:
                # Determine pick
                if spread_edge < 0:
                    # Vegas gives home MORE points - bet HOME
                    if v_spread > 0:
                        pick = f"{home} +{v_spread}"
                        pick_team = home
                    else:
                        pick = f"{home} {v_spread}"
                        pick_team = home
                else:
                    # KenPom gives home MORE credit - bet AWAY
                    if v_spread > 0:
                        pick = f"{visitor} -{v_spread}"
                        pick_team = visitor
                    else:
                        pick = f"{visitor} +{abs(v_spread)}"
                        pick_team = visitor

                spread_picks.append(
                    {
                        "matchup": f"{visitor} @ {home}",
                        "pick": pick,
                        "pick_team": pick_team,
                        "kp_spread": kp_spread,
                        "vegas_spread": v_spread,
                        "edge": spread_edge,
                        "home_wp": home_wp,
                        "bet_type": "SPREAD",
                    }
                )

        if v_total is not None:
            total_edge = kenpom_total - v_total

            if abs(total_edge) >= 5:
                if total_edge > 0:
                    pick = f"OVER {v_total}"
                else:
                    pick = f"UNDER {v_total}"

                total_picks.append(
                    {
                        "matchup": f"{visitor} @ {home}",
                        "pick": pick,
                        "kp_total": kenpom_total,
                        "vegas_total": v_total,
                        "edge": total_edge,
                        "bet_type": "TOTAL",
                    }
                )

    # Create Excel workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Betting Slip"

    # Styles
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    value_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    thin_border = Border(
        left=Side(style="thin"),
        right=Side(style="thin"),
        top=Side(style="thin"),
        bottom=Side(style="thin"),
    )

    # Title
    ws.merge_cells("A1:H1")
    ws["A1"] = f"NCAA CBB BETTING SLIP - {date}"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A1"].alignment = Alignment(horizontal="center")

    # Generated timestamp
    ws.merge_cells("A2:H2")
    ws["A2"] = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ws["A2"].alignment = Alignment(horizontal="center")

    # Spread Picks Section
    row = 4
    ws.merge_cells(f"A{row}:H{row}")
    ws[f"A{row}"] = "SPREAD VALUE PICKS"
    ws[f"A{row}"].font = Font(bold=True, size=12)
    ws[f"A{row}"].fill = PatternFill(
        start_color="FFC000", end_color="FFC000", fill_type="solid"
    )

    row += 1
    headers = [
        "Matchup",
        "Pick",
        "KP Spread",
        "Vegas",
        "Edge",
        "Home WP%",
        "Placed",
        "Result",
    ]
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=row, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = thin_border
        cell.alignment = Alignment(horizontal="center")

    row += 1
    for pick in sorted(spread_picks, key=lambda x: abs(x["edge"]), reverse=True):
        ws.cell(row=row, column=1, value=pick["matchup"]).border = thin_border
        ws.cell(row=row, column=2, value=pick["pick"]).border = thin_border
        ws.cell(row=row, column=2).fill = value_fill
        ws.cell(row=row, column=3, value=f"{pick['kp_spread']:+.1f}").border = thin_border
        ws.cell(
            row=row, column=4, value=f"{pick['vegas_spread']:+.1f}"
        ).border = thin_border
        ws.cell(row=row, column=5, value=f"{pick['edge']:+.1f}").border = thin_border
        ws.cell(row=row, column=6, value=f"{pick['home_wp']}%").border = thin_border
        ws.cell(row=row, column=7, value="").border = thin_border  # Placed checkbox
        ws.cell(row=row, column=8, value="").border = thin_border  # Result
        row += 1

    if not spread_picks:
        ws.cell(row=row, column=1, value="No spread value picks today")
        row += 1

    # Total Picks Section
    row += 2
    ws.merge_cells(f"A{row}:H{row}")
    ws[f"A{row}"] = "TOTAL VALUE PICKS"
    ws[f"A{row}"].font = Font(bold=True, size=12)
    ws[f"A{row}"].fill = PatternFill(
        start_color="70AD47", end_color="70AD47", fill_type="solid"
    )

    row += 1
    headers = [
        "Matchup",
        "Pick",
        "KP Total",
        "Vegas",
        "Edge",
        "",
        "Placed",
        "Result",
    ]
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=row, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = thin_border
        cell.alignment = Alignment(horizontal="center")

    row += 1
    for pick in sorted(total_picks, key=lambda x: abs(x["edge"]), reverse=True):
        ws.cell(row=row, column=1, value=pick["matchup"]).border = thin_border
        ws.cell(row=row, column=2, value=pick["pick"]).border = thin_border
        ws.cell(row=row, column=2).fill = value_fill
        ws.cell(row=row, column=3, value=f"{pick['kp_total']:.1f}").border = thin_border
        ws.cell(row=row, column=4, value=f"{pick['vegas_total']}").border = thin_border
        ws.cell(row=row, column=5, value=f"{pick['edge']:+.1f}").border = thin_border
        ws.cell(row=row, column=6, value="").border = thin_border
        ws.cell(row=row, column=7, value="").border = thin_border  # Placed checkbox
        ws.cell(row=row, column=8, value="").border = thin_border  # Result
        row += 1

    if not total_picks:
        ws.cell(row=row, column=1, value="No total value picks today")
        row += 1

    # Summary Section
    row += 2
    ws[f"A{row}"] = "SUMMARY"
    ws[f"A{row}"].font = Font(bold=True)
    row += 1
    ws[f"A{row}"] = f"Spread Picks: {len(spread_picks)}"
    row += 1
    ws[f"A{row}"] = f"Total Picks: {len(total_picks)}"
    row += 1
    ws[f"A{row}"] = f"Min Edge Threshold: {min_edge} pts"

    # Adjust column widths
    ws.column_dimensions["A"].width = 35
    ws.column_dimensions["B"].width = 20
    ws.column_dimensions["C"].width = 12
    ws.column_dimensions["D"].width = 12
    ws.column_dimensions["E"].width = 10
    ws.column_dimensions["F"].width = 12
    ws.column_dimensions["G"].width = 10
    ws.column_dimensions["H"].width = 10

    # Save
    wb.save(output_path)
    print(f"[OK] Betting slip saved: {output_path}")

    return output_path


# ============================================================================
# OPEN BETS TRACKING
# ============================================================================


def get_user_data_dir() -> Path:
    """Get persistent browser data directory for overtime.ag session."""
    data_dir = Path.home() / ".cache" / "kenp0m_sp0rts_analyzer" / "overtime_browser"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


async def fetch_open_bets(headless: bool = False) -> dict:
    """Fetch open bets from overtime.ag.

    Uses a persistent browser context to maintain login session.
    First run requires manual login in the browser window.

    Args:
        headless: Run in headless mode. Use False for first login.

    Returns:
        Dictionary with open bets data.
    """
    print("[INFO] Fetching open bets from overtime.ag...")
    print("[INFO] Using persistent browser session.")

    if not headless:
        print("[INFO] Browser window will open. Log in if prompted.")

    user_data_dir = get_user_data_dir()

    async with async_playwright() as p:
        # Use persistent context to preserve login cookies
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            headless=headless,
            viewport={"width": 1280, "height": 800},
        )
        page = context.pages[0] if context.pages else await context.new_page()

        try:
            # Navigate to sports section first
            await page.goto("https://overtime.ag/sports", timeout=30000)
            await asyncio.sleep(3)

            # Navigate to open bets page
            await page.goto("https://overtime.ag/sports#/openBets", timeout=30000)
            await asyncio.sleep(3)

            # Try to get open bets via page content scraping
            result = await page.evaluate(
                """
                () => {
                    // Look for bet information on the page
                    const pageText = document.body.innerText;

                    // Check for "No open bets" message
                    if (pageText.includes('No open bets') ||
                        pageText.includes('no open wagers')) {
                        return {status: 'empty', message: 'No open bets found'};
                    }

                    // Check for login prompt
                    if (pageText.includes('Sign In') ||
                        pageText.includes('Log In') ||
                        pageText.includes('Please login')) {
                        return {status: 'not_logged_in', message: 'Login required'};
                    }

                    // Try to find bet elements
                    const betElements = document.querySelectorAll(
                        '[class*="bet"], [class*="wager"], [class*="open"]'
                    );
                    const bets = [];
                    betElements.forEach(el => {
                        const text = el.innerText.trim();
                        if (text && text.length > 10) {
                            bets.push(text.substring(0, 200));
                        }
                    });

                    if (bets.length > 0) {
                        return {status: 'found', bets: bets};
                    }

                    // Return page content summary
                    return {
                        status: 'unknown',
                        content: pageText.substring(0, 1000)
                    };
                }
            """
            )

            await context.close()
            return result

        except Exception as e:
            await context.close()
            return {"error": str(e)}


def display_open_bets(data: dict) -> None:
    """Display open bets in a formatted table."""
    if "error" in data:
        print(f"[ERROR] {data['error']}")
        return

    print("\n" + "=" * 80)
    print("OPEN BETS")
    print("=" * 80)

    status = data.get("status", "unknown")

    if status == "empty":
        print("\n[INFO] No open bets found")
        return

    if status == "not_logged_in":
        print("\n[WARN] Not logged in to overtime.ag")
        print("       Run with visible browser to log in:")
        print("       uv run python scripts/betting/betting_tracker.py open-bets")
        return

    if status == "found":
        bets = data.get("bets", [])
        print(f"\n[INFO] Found {len(bets)} bet entries:")
        for i, bet in enumerate(bets, 1):
            print(f"\n  [{i}] {bet}")
        return

    # Legacy API format
    if "d" in data and data["d"]:
        bets = data["d"].get("Data", []) if isinstance(data["d"], dict) else data["d"]
        for bet in bets:
            print(f"\nBet ID: {bet.get('BetId', 'N/A')}")
            print(f"  Type: {bet.get('BetType', 'N/A')}")
            print(f"  Risk: ${bet.get('RiskAmount', 0):.2f}")
            print(f"  Win: ${bet.get('WinAmount', 0):.2f}")
            print(f"  Status: {bet.get('Status', 'N/A')}")
            legs = bet.get("Legs", [])
            for leg in legs:
                print(f"    -> {leg.get('Description', 'N/A')}")
        return

    # Unknown format - show raw content
    content = data.get("content", "No content available")
    print(f"\n[INFO] Page content:\n{content[:500]}...")


# ============================================================================
# DAILY FIGURES TRACKING
# ============================================================================


async def fetch_daily_figures(date: str | None = None, headless: bool = False) -> dict:
    """Fetch daily P&L figures from overtime.ag.

    Uses a persistent browser context to maintain login session.

    Args:
        date: Date in YYYY-MM-DD format (default: today)
        headless: Run in headless mode. Use False for first login.

    Returns:
        Dictionary with daily figures data.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    print(f"[INFO] Fetching daily figures for {date}...")
    print("[INFO] Using persistent browser session.")

    user_data_dir = get_user_data_dir()

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            headless=headless,
            viewport={"width": 1280, "height": 800},
        )
        page = context.pages[0] if context.pages else await context.new_page()

        try:
            # Navigate to daily figures page
            await page.goto("https://overtime.ag/sports#/dailyFigures", timeout=30000)
            await asyncio.sleep(3)

            # Try to scrape daily figures from page
            result = await page.evaluate(
                """
                () => {
                    const pageText = document.body.innerText;

                    // Check for login prompt
                    if (pageText.includes('Sign In') ||
                        pageText.includes('Please login')) {
                        return {status: 'not_logged_in'};
                    }

                    // Look for P&L data patterns
                    const figures = {};

                    // Try to find balance/P&L elements
                    const balanceEl = document.querySelector('[class*="balance"]');
                    if (balanceEl) {
                        figures.balance = balanceEl.innerText;
                    }

                    // Look for daily summary table
                    const tables = document.querySelectorAll('table');
                    const tableData = [];
                    tables.forEach(table => {
                        tableData.push(table.innerText.substring(0, 500));
                    });

                    if (tableData.length > 0) {
                        figures.tables = tableData;
                    }

                    // Get page content summary
                    figures.content = pageText.substring(0, 1500);
                    figures.status = 'scraped';

                    return figures;
                }
            """
            )

            await context.close()
            return result

        except Exception as e:
            await context.close()
            return {"error": str(e)}


def display_daily_figures(data: dict, date: str) -> None:
    """Display daily figures."""
    if "error" in data:
        print(f"[ERROR] {data['error']}")
        return

    print("\n" + "=" * 80)
    print(f"DAILY FIGURES - {date}")
    print("=" * 80)

    status = data.get("status", "unknown")

    if status == "not_logged_in":
        print("\n[WARN] Not logged in to overtime.ag")
        print("       Run with visible browser to log in first.")
        return

    if status == "scraped":
        if data.get("balance"):
            print(f"\nBalance: {data['balance']}")

        if data.get("tables"):
            print("\nData Tables Found:")
            for i, table in enumerate(data["tables"], 1):
                print(f"\n  Table {i}:")
                print(f"  {table[:300]}...")

        if data.get("content"):
            print("\nPage Summary:")
            content = data["content"]
            # Look for P&L patterns in content
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if any(kw in line.lower() for kw in ["p&l", "profit", "loss", "win", "total"]):
                    if len(line) < 100:
                        print(f"  {line}")
        return

    # Legacy API format
    if "d" in data and data["d"]:
        figures = data["d"]
        if isinstance(figures, dict):
            print(f"\nNet P&L: ${figures.get('NetPL', 0):.2f}")
            print(f"Total Risked: ${figures.get('TotalRisked', 0):.2f}")
            print(f"Total Won: ${figures.get('TotalWon', 0):.2f}")
            print(f"Bets Placed: {figures.get('BetsPlaced', 0)}")
            print(f"Wins: {figures.get('Wins', 0)}")
            print(f"Losses: {figures.get('Losses', 0)}")
        else:
            print(f"\nRaw data: {figures}")
        return

    print("[INFO] No daily figures available")


# ============================================================================
# END-OF-DAY ANALYSIS
# ============================================================================


def analyze_results(date: str | None = None) -> dict:
    """Analyze betting results for a given date.

    Compares KenPom predictions with actual results.

    Args:
        date: Date in YYYY-MM-DD format (default: today)

    Returns:
        Dictionary with analysis results.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    date_suffix = date.replace("-", "")

    print(f"\n[INFO] Analyzing results for {date}...")

    # Load betting slip
    slip_path = (
        Path(__file__).parent.parent.parent
        / "reports"
        / "bet_slips"
        / f"NCAA CBB Betting Slip {date_suffix}.xlsx"
    )

    if not slip_path.exists():
        print(f"[WARN] No betting slip found for {date}")
        print(f"       Expected: {slip_path}")
        return {"error": "No betting slip found"}

    # Load results from tracking database (if available)
    results_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "results"
        / f"results_{date_suffix}.json"
    )

    results = {"date": date, "slip_path": str(slip_path), "picks": [], "summary": {}}

    if results_path.exists():
        with open(results_path) as f:
            actual_results = json.load(f)

        # Analyze each pick
        wins = 0
        losses = 0
        pushes = 0

        for pick in actual_results.get("picks", []):
            if pick.get("result") == "WIN":
                wins += 1
            elif pick.get("result") == "LOSS":
                losses += 1
            else:
                pushes += 1

            results["picks"].append(pick)

        total = wins + losses
        win_pct = (wins / total * 100) if total > 0 else 0

        results["summary"] = {
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "total": total,
            "win_percentage": win_pct,
        }

        print("\n" + "=" * 80)
        print(f"RESULTS ANALYSIS - {date}")
        print("=" * 80)
        print(f"\nRecord: {wins}-{losses}-{pushes}")
        print(f"Win %: {win_pct:.1f}%")

        print("\nPick-by-Pick:")
        for pick in results["picks"]:
            status = pick.get("result", "PENDING")
            emoji = "[WIN]" if status == "WIN" else "[LOSS]" if status == "LOSS" else "[--]"
            print(f"  {emoji} {pick.get('pick', 'N/A')}")
            if pick.get("notes"):
                print(f"       Notes: {pick['notes']}")

    else:
        print(f"[INFO] No results file found. Create one at:")
        print(f"       {results_path}")
        print("\nExpected format:")
        print(
            """
{
  "date": "2025-12-18",
  "picks": [
    {"pick": "Indiana St. +5", "result": "WIN", "final_margin": 3},
    {"pick": "Temple/Davidson OVER 144", "result": "LOSS", "final_total": 138}
  ]
}
"""
        )

    return results


# ============================================================================
# MAIN CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="NCAA CBB Betting Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Slip command
    slip_parser = subparsers.add_parser("slip", help="Generate betting slip")
    slip_parser.add_argument("--date", "-d", help="Date (YYYY-MM-DD, default: today)")
    slip_parser.add_argument(
        "--min-edge", type=float, default=3.0, help="Minimum edge (default: 3.0)"
    )

    # Open bets command
    subparsers.add_parser("open-bets", help="Fetch open bets from overtime.ag")

    # Daily figures command
    figures_parser = subparsers.add_parser(
        "daily-figures", help="Fetch daily P&L figures"
    )
    figures_parser.add_argument("--date", "-d", help="Date (YYYY-MM-DD, default: today)")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze results")
    analyze_parser.add_argument("--date", "-d", help="Date (YYYY-MM-DD, default: today)")

    # All command
    all_parser = subparsers.add_parser("all", help="Run full workflow")
    all_parser.add_argument("--date", "-d", help="Date (YYYY-MM-DD, default: today)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    print("=" * 80)
    print("NCAA CBB BETTING TRACKER")
    print("=" * 80)

    if args.command == "slip":
        generate_betting_slip(date=args.date, min_edge=args.min_edge)

    elif args.command == "open-bets":
        data = asyncio.run(fetch_open_bets())
        display_open_bets(data)

    elif args.command == "daily-figures":
        date = args.date or datetime.now().strftime("%Y-%m-%d")
        data = asyncio.run(fetch_daily_figures(date))
        display_daily_figures(data, date)

    elif args.command == "analyze":
        analyze_results(date=args.date)

    elif args.command == "all":
        date = args.date or datetime.now().strftime("%Y-%m-%d")
        print("\n[1/4] Generating betting slip...")
        generate_betting_slip(date=date)

        print("\n[2/4] Fetching open bets...")
        bets = asyncio.run(fetch_open_bets())
        display_open_bets(bets)

        print("\n[3/4] Fetching daily figures...")
        figures = asyncio.run(fetch_daily_figures(date))
        display_daily_figures(figures, date)

        print("\n[4/4] Analyzing results...")
        analyze_results(date=date)

    print("\n" + "=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
