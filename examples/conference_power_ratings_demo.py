"""Quick demo of conference power ratings feature."""

import os
from pathlib import Path

from dotenv import load_dotenv

from kenp0m_sp0rts_analyzer.conference_analytics import ConferenceAnalytics

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"[OK] Loaded .env from {env_path}")
else:
    print(f"[ERROR] No .env file found at {env_path}")

# Check if API key is available
api_key = os.getenv("KENPOM_API_KEY")
if api_key:
    print(f"[OK] API key loaded (length: {len(api_key)})")
else:
    print("[ERROR] No KENPOM_API_KEY found in environment")
    exit(1)

# Test the feature
print("\n" + "=" * 60)
print("Conference Power Ratings Demo - 2025 Season")
print("=" * 60 + "\n")

analytics = ConferenceAnalytics()

# Get power ratings
print("Fetching conference power ratings...")
try:
    ratings = analytics.calculate_conference_power_ratings(2025)

    if len(ratings) == 0:
        print("[ERROR] No data returned - check API authentication")
        # Debug: Check if API client is working
        print("\nDebug: Testing API client directly...")
        from kenp0m_sp0rts_analyzer.api_client import KenPomAPI

        test_api = KenPomAPI()
        print(f"API client initialized: {test_api}")
        print(f"API key present: {test_api.api_key is not None}")

        # Try to get conferences
        conferences = test_api.get_conferences(year=2025)
        print(f"Conferences response: {conferences}")
        print(f"Number of conferences: {len(conferences.data)}")

        exit(1)
except Exception as e:
    print(f"[ERROR] Exception occurred: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print(f"[OK] Retrieved data for {len(ratings)} conferences\n")

# Display top 10 conferences
print("=" * 60)
print("TOP 10 CONFERENCES BY POWER RATING")
print("=" * 60)
top_10 = ratings.head(10)
for _idx, row in top_10.iterrows():
    print(
        f"{row['power_rank']:2d}. {row['conf_short']:4s} - {row['power_score']:5.1f} pts"
    )
    print(f"    {row['conference']}")
    print(
        f"    Teams: {row['num_teams']:2d} | "
        f"Avg AdjEM: {row['avg_adj_em']:+6.2f} | "
        f"Top Team: {row['top_team_adj_em']:+6.2f}"
    )
    print(
        f"    Top 25: {row['top_25_teams']:2d} | "
        f"Bids: {row['estimated_bids']:2d} | "
        f"Above Avg: {row['above_average']:2d}"
    )
    print()

# Analyze a specific conference tournament outlook
print("\n" + "=" * 60)
print("SEC TOURNAMENT OUTLOOK")
print("=" * 60)
sec_outlook = analytics.get_conference_tournament_outlook("SEC", 2025)
print(f"\nSEC Teams: {len(sec_outlook)}\n")

locks = sec_outlook[sec_outlook["Bubble"] == "Lock"]
bubbles = sec_outlook[sec_outlook["Bubble"] == "Bubble"]

print(f"Tournament Locks ({len(locks)}):")
for _idx, team in locks.iterrows():
    print(
        f"  {team['Team']:25s} - AdjEM: {team['AdjEM']:+6.2f}, "
        f"Rank: {team['RankAdjEM']:3d}, "
        f"Prob: {team['NCAA_Probability']:.0%}"
    )

if len(bubbles) > 0:
    print(f"\nBubble Teams ({len(bubbles)}):")
    for _idx, team in bubbles.iterrows():
        print(
            f"  {team['Team']:25s} - AdjEM: {team['AdjEM']:+6.2f}, "
            f"Rank: {team['RankAdjEM']:3d}, "
            f"Prob: {team['NCAA_Probability']:.0%}"
        )

print("\n" + "=" * 60)
print("[OK] Conference analytics feature working perfectly!")
print("=" * 60)
