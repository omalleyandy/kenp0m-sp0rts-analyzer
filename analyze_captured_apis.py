"""Analyze captured API responses from overtime.ag."""
import json
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from kenp0m_sp0rts_analyzer.overtime_timing import get_monitoring_dir


def main():
    """Analyze captured API responses."""
    db_path = get_monitoring_dir() / "overtime_odds.db"

    if not db_path.exists():
        print("[ERROR] No database found at:", db_path)
        return 1

    print("=" * 70)
    print("ANALYZING CAPTURED API RESPONSES")
    print("=" * 70)
    print(f"Database: {db_path}\n")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get snapshot count
    cursor.execute("SELECT COUNT(*) FROM api_snapshots")
    snapshot_count = cursor.fetchone()[0]
    print(f"Total snapshots: {snapshot_count}")

    # Get unique endpoints
    cursor.execute("""
        SELECT endpoint, COUNT(*) as count,
               MAX(game_count) as max_games,
               MAX(captured_at) as latest
        FROM api_snapshots
        GROUP BY endpoint
        ORDER BY count DESC
    """)

    endpoints = cursor.fetchall()
    print(f"\nUnique endpoints: {len(endpoints)}\n")

    print(f"{'ENDPOINT':<50} {'CAPTURES':>10} {'GAMES':>8} {'LATEST':<20}")
    print("-" * 95)

    for endpoint, count, max_games, latest in endpoints:
        # Shorten endpoint for display
        endpoint_short = endpoint[:48] if endpoint else "None"
        latest_short = latest[:19] if latest else "None"
        print(f"{endpoint_short:<50} {count:>10} {max_games or 0:>8} {latest_short:<20}")

    # Analyze the most recent snapshots
    print("\n" + "=" * 70)
    print("RECENT API RESPONSES (Last 5)")
    print("=" * 70)

    cursor.execute("""
        SELECT captured_at, endpoint, response_size, game_count, raw_json
        FROM api_snapshots
        ORDER BY captured_at DESC
        LIMIT 5
    """)

    recent = cursor.fetchall()

    for i, (captured_at, endpoint, response_size, game_count, raw_json) in enumerate(recent, 1):
        print(f"\n[{i}] {captured_at}")
        print(f"    Endpoint: {endpoint[:60]}")
        print(f"    Size: {response_size:,} bytes | Games: {game_count}")

        # Try to parse and show structure
        if raw_json:
            try:
                data = json.loads(raw_json)

                # Show top-level keys
                if isinstance(data, dict):
                    keys = list(data.keys())[:10]
                    print(f"    Keys: {', '.join(keys)}")

                    # Look for game-related data
                    if 'd' in data and isinstance(data['d'], dict):
                        d_keys = list(data['d'].keys())[:10]
                        print(f"    d.Keys: {', '.join(d_keys)}")

                        # Check for GameLines
                        if 'Data' in data['d']:
                            data_obj = data['d']['Data']
                            if isinstance(data_obj, dict):
                                data_keys = list(data_obj.keys())[:10]
                                print(f"    d.Data.Keys: {', '.join(data_keys)}")

                                # Show GameLines if present
                                if 'GameLines' in data_obj:
                                    game_lines = data_obj['GameLines']
                                    if isinstance(game_lines, list):
                                        print(f"    -> GameLines: {len(game_lines)} items")
                                        if game_lines:
                                            first_game = game_lines[0]
                                            if isinstance(first_game, dict):
                                                game_keys = list(first_game.keys())[:15]
                                                print(f"       Sample game keys: {', '.join(game_keys)}")
                                                # Show team names if present
                                                team1 = first_game.get('Team1', 'N/A')
                                                team2 = first_game.get('Team2', 'N/A')
                                                print(f"       Sample: {team1} vs {team2}")

                elif isinstance(data, list):
                    print(f"    Array: {len(data)} items")

            except json.JSONDecodeError:
                print("    [ERROR] Invalid JSON")
            except Exception as e:
                print(f"    [ERROR] {e}")

    # Check game tracking
    print("\n" + "=" * 70)
    print("GAME TRACKING")
    print("=" * 70)

    cursor.execute("SELECT COUNT(*) FROM game_tracking")
    game_count = cursor.fetchone()[0]
    print(f"Games tracked: {game_count}")

    if game_count > 0:
        cursor.execute("""
            SELECT game_id, first_seen_at, game_time, home_team, away_team
            FROM game_tracking
            ORDER BY first_seen_at DESC
            LIMIT 10
        """)

        print(f"\n{'MATCHUP':<40} {'FIRST SEEN':<20} {'GAME TIME':<20}")
        print("-" * 85)

        for game_id, first_seen, game_time, home, away in cursor.fetchall():
            matchup = f"{away} @ {home}" if away and home else game_id
            matchup = matchup[:38]
            first_seen_short = first_seen[:19] if first_seen else "N/A"
            game_time_short = game_time[:19] if game_time else "N/A"
            print(f"{matchup:<40} {first_seen_short:<20} {game_time_short:<20}")

    conn.close()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
