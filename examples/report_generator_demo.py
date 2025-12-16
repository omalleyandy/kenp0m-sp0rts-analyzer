"""Demo: Automated Matchup Report Generation.

This script demonstrates how to generate comprehensive matchup reports
in multiple formats (Markdown, HTML, JSON) using the MatchupReportGenerator.

The report synthesizes insights from all 7 analytical modules:
- Four Factors (4 dimensions)
- Point Distribution (3 dimensions)
- Defensive Analysis (3 dimensions)
- Size & Athleticism (2 dimensions)
- Experience & Chemistry (3 dimensions)
"""

import os
from pathlib import Path

from kenp0m_sp0rts_analyzer.report_generator import MatchupReportGenerator


def main():
    """Generate matchup reports in multiple formats."""
    # Initialize report generator
    api_key = os.getenv("KENPOM_API_KEY")
    if not api_key:
        print("Error: KENPOM_API_KEY environment variable not set")
        return

    generator = MatchupReportGenerator(api_key)

    # Example matchup: Top teams from 2025 season
    team1 = "Duke"
    team2 = "North Carolina"
    season = 2025

    print(f"Generating matchup report: {team1} vs {team2} ({season})")
    print("=" * 60)

    # Generate Markdown report
    print("\n[1/3] Generating Markdown report...")
    md_report = generator.generate_report(team1, team2, season, format="markdown")

    # Save to file
    output_dir = Path("examples/reports")
    output_dir.mkdir(exist_ok=True)

    md_file = output_dir / f"{team1}_vs_{team2}_{season}.md"
    md_file.write_text(md_report)
    print(f"✓ Saved: {md_file}")

    # Display preview
    print("\nMarkdown Preview (first 500 chars):")
    print("-" * 60)
    print(md_report[:500] + "...")
    print("-" * 60)

    # Generate HTML report
    print("\n[2/3] Generating HTML report...")
    html_report = generator.generate_report(team1, team2, season, format="html")

    html_file = output_dir / f"{team1}_vs_{team2}_{season}.html"
    html_file.write_text(html_report)
    print(f"✓ Saved: {html_file}")

    # Generate JSON report
    print("\n[3/3] Generating JSON report...")
    json_report = generator.generate_report(team1, team2, season, format="json")

    json_file = output_dir / f"{team1}_vs_{team2}_{season}.json"
    json_file.write_text(json_report)
    print(f"✓ Saved: {json_file}")

    print("\n" + "=" * 60)
    print("Report generation complete!")
    print(f"All reports saved to: {output_dir}")
    print("\nNext steps:")
    print(f"  - View Markdown: cat {md_file}")
    print(f"  - Open HTML: open {html_file}")
    print(f"  - Parse JSON: cat {json_file} | python -m json.tool")


def generate_multiple_matchups():
    """Generate reports for multiple matchups."""
    api_key = os.getenv("KENPOM_API_KEY")
    if not api_key:
        print("Error: KENPOM_API_KEY environment variable not set")
        return

    generator = MatchupReportGenerator(api_key)

    # Top 2025 matchups
    matchups = [
        ("Duke", "North Carolina"),
        ("Kansas", "Kentucky"),
        ("Purdue", "Illinois"),
        ("Houston", "Alabama"),
        ("UConn", "Gonzaga")
    ]

    output_dir = Path("examples/reports/batch")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Batch Report Generation")
    print("=" * 60)

    for i, (team1, team2) in enumerate(matchups, 1):
        print(f"\n[{i}/{len(matchups)}] {team1} vs {team2}")

        try:
            # Generate markdown report
            report = generator.generate_report(team1, team2, 2025, format="markdown")

            # Save to file
            filename = f"{team1}_vs_{team2}_2025.md"
            filepath = output_dir / filename
            filepath.write_text(report)

            print(f"  ✓ Generated: {filename}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "=" * 60)
    print(f"Batch generation complete! Reports saved to: {output_dir}")


def custom_matchup_interactive():
    """Interactive mode: user enters teams."""
    api_key = os.getenv("KENPOM_API_KEY")
    if not api_key:
        print("Error: KENPOM_API_KEY environment variable not set")
        return

    generator = MatchupReportGenerator(api_key)

    print("\n" + "=" * 60)
    print("Interactive Matchup Report Generator")
    print("=" * 60)

    team1 = input("\nEnter Team 1: ").strip()
    team2 = input("Enter Team 2: ").strip()
    season = input("Enter Season (default: 2025): ").strip() or "2025"
    format_choice = input("Format (markdown/html/json, default: markdown): ").strip() or "markdown"

    print(f"\nGenerating {format_choice.upper()} report...")

    try:
        report = generator.generate_report(
            team1,
            team2,
            int(season),
            format=format_choice  # type: ignore
        )

        # Display report
        print("\n" + "=" * 60)
        print("REPORT")
        print("=" * 60)
        print(report)
        print("=" * 60)

        # Save option
        save = input("\nSave to file? (y/n): ").strip().lower()
        if save == 'y':
            output_dir = Path("examples/reports")
            output_dir.mkdir(exist_ok=True)

            ext = "md" if format_choice == "markdown" else format_choice
            filename = f"{team1}_vs_{team2}_{season}.{ext}"
            filepath = output_dir / filename

            filepath.write_text(report)
            print(f"✓ Saved: {filepath}")

    except Exception as e:
        print(f"\n✗ Error generating report: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == "batch":
            generate_multiple_matchups()
        elif mode == "interactive":
            custom_matchup_interactive()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python report_generator_demo.py [batch|interactive]")
    else:
        main()
