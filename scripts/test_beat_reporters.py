"""Test script to verify beat reporter database loads correctly."""

import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

# Direct import to avoid package dependencies
from kenp0m_sp0rts_analyzer.injury_monitor import twitter_monitor


def main():
    """Test loading and displaying beat reporter database."""
    print("="*80)
    print("BEAT REPORTER DATABASE TEST")
    print("="*80)
    print()

    # Load beat reporters
    reporters = twitter_monitor.load_beat_reporters()

    if not reporters:
        print("ERROR: No reporters loaded!")
        return

    # Summary statistics
    print(f"Total teams: {len(reporters)}")
    total_reporters = sum(len(r) for r in reporters.values())
    print(f"Total reporters: {total_reporters}")
    print()

    # Count by tier
    tier_1 = sum(1 for team_reporters in reporters.values()
                 for r in team_reporters if r.tier == 1)
    tier_2 = sum(1 for team_reporters in reporters.values()
                 for r in team_reporters if r.tier == 2)
    tier_3 = sum(1 for team_reporters in reporters.values()
                 for r in team_reporters if r.tier == 3)

    print(f"Tier 1 (Most Reliable): {tier_1}")
    print(f"Tier 2 (Reliable): {tier_2}")
    print(f"Tier 3 (Sometimes Useful): {tier_3}")
    print()

    # Sample teams
    print("="*80)
    print("SAMPLE TEAMS")
    print("="*80)
    print()

    sample_teams = ['Duke', 'Kentucky', 'Kansas', 'UConn', 'Gonzaga']

    for team in sample_teams:
        if team in reporters:
            print(f"{team}:")
            for reporter in reporters[team]:
                print(f"  {reporter.handle} - {reporter.name} ({reporter.outlet}) [Tier {reporter.tier}]")
            print()

    # All teams covered
    print("="*80)
    print("ALL TEAMS COVERED")
    print("="*80)
    print()

    # Group by conference (approximate based on common teams)
    conferences = {
        'ACC': ['Duke', 'North Carolina', 'Virginia', 'Miami', 'Pittsburgh',
                'Louisville', 'Clemson', 'NC State', 'Wake Forest', 'Syracuse'],
        'SEC': ['Kentucky', 'Tennessee', 'Auburn', 'Alabama', 'Arkansas',
                'Florida', 'Texas A&M', 'Missouri', 'Ole Miss', 'Mississippi State'],
        'Big Ten': ['Purdue', 'Indiana', 'Illinois', 'Michigan State', 'Michigan',
                    'Wisconsin', 'Ohio State', 'Iowa', 'Maryland', 'Northwestern'],
        'Big 12': ['Kansas', 'Baylor', 'Texas', 'Texas Tech', 'Iowa State',
                   'Kansas State', 'TCU', 'Oklahoma State', 'West Virginia', 'BYU'],
        'Big East': ['UConn', 'Villanova', 'Creighton', 'Marquette', 'Xavier',
                     'Providence', "St. John's", 'Georgetown', 'Butler', 'Seton Hall'],
        'Pac-12': ['Arizona', 'UCLA', 'USC', 'Oregon', 'Colorado', 'Washington'],
        'Other': ['Gonzaga', 'Houston', 'Memphis', 'San Diego State', 'Dayton', 'VCU']
    }

    for conf, teams in conferences.items():
        covered = [t for t in teams if t in reporters]
        if covered:
            print(f"{conf}: {', '.join(covered)}")

    print()
    print("="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
