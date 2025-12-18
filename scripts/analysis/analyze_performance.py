"""Generate performance analytics from historical odds database.

This script analyzes our prediction performance against Vegas lines
to identify strengths, weaknesses, and improvement opportunities.

Usage:
    # Show overall model performance
    uv run python analyze_performance.py
    
    # Performance for specific date range
    uv run python analyze_performance.py --start 2025-12-01 --end 2025-12-31
    
    # CLV (Closing Line Value) analysis
    uv run python analyze_performance.py --clv
    
    # ATS record by conference
    uv run python analyze_performance.py --by-conference
    
    # Export detailed report
    uv run python analyze_performance.py --export
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def get_model_summary(db, model_name: str = "kenpom_basic", 
                      start_date: str = None, end_date: str = None) -> dict:
    """Get comprehensive model performance summary."""
    return db.get_model_performance(
        model_name=model_name,
        start_date=start_date,
        end_date=end_date
    )


def analyze_ats_by_spread_size(db, start_date: str = None, end_date: str = None) -> dict:
    """Analyze ATS performance by spread size buckets."""
    conn = db.connect()
    cursor = conn.cursor()
    
    query = '''
        SELECT 
            CASE 
                WHEN ABS(os.spread) <= 3 THEN 'Pick-em (0-3)'
                WHEN ABS(os.spread) <= 7 THEN 'Small (3.5-7)'
                WHEN ABS(os.spread) <= 14 THEN 'Medium (7.5-14)'
                ELSE 'Large (14.5+)'
            END as spread_bucket,
            COUNT(*) as total_games,
            SUM(CASE WHEN g.actual_spread < os.spread THEN 1 ELSE 0 END) as home_covers,
            SUM(CASE WHEN g.actual_spread > os.spread THEN 1 ELSE 0 END) as away_covers,
            SUM(CASE WHEN g.actual_spread = os.spread THEN 1 ELSE 0 END) as pushes,
            AVG(g.actual_spread - os.spread) as avg_ats_margin
        FROM games g
        JOIN odds_snapshots os ON g.game_id = os.game_id AND os.snapshot_type = 'close'
        WHERE g.status = 'final' AND os.spread IS NOT NULL
    '''
    
    params = []
    if start_date:
        query += ' AND g.game_date >= ?'
        params.append(start_date)
    if end_date:
        query += ' AND g.game_date <= ?'
        params.append(end_date)
        
    query += ' GROUP BY spread_bucket ORDER BY MIN(ABS(os.spread))'
    
    cursor.execute(query, params)
    
    results = {}
    for row in cursor.fetchall():
        bucket = row['spread_bucket']
        total = row['total_games']
        home_pct = (row['home_covers'] / total * 100) if total > 0 else 0
        
        results[bucket] = {
            'total': total,
            'home_covers': row['home_covers'],
            'away_covers': row['away_covers'],
            'pushes': row['pushes'],
            'home_cover_pct': round(home_pct, 1),
            'avg_ats_margin': round(row['avg_ats_margin'] or 0, 2),
        }
    
    return results


def analyze_totals_accuracy(db, start_date: str = None, end_date: str = None) -> dict:
    """Analyze over/under accuracy by total size."""
    conn = db.connect()
    cursor = conn.cursor()
    
    query = '''
        SELECT 
            CASE 
                WHEN os.total < 130 THEN 'Low (<130)'
                WHEN os.total < 145 THEN 'Medium (130-145)'
                WHEN os.total < 160 THEN 'High (145-160)'
                ELSE 'Very High (160+)'
            END as total_bucket,
            COUNT(*) as total_games,
            SUM(CASE WHEN g.actual_total > os.total THEN 1 ELSE 0 END) as overs,
            SUM(CASE WHEN g.actual_total < os.total THEN 1 ELSE 0 END) as unders,
            SUM(CASE WHEN g.actual_total = os.total THEN 1 ELSE 0 END) as pushes,
            AVG(g.actual_total - os.total) as avg_total_margin
        FROM games g
        JOIN odds_snapshots os ON g.game_id = os.game_id AND os.snapshot_type = 'close'
        WHERE g.status = 'final' AND os.total IS NOT NULL
    '''
    
    params = []
    if start_date:
        query += ' AND g.game_date >= ?'
        params.append(start_date)
    if end_date:
        query += ' AND g.game_date <= ?'
        params.append(end_date)
        
    query += ' GROUP BY total_bucket ORDER BY MIN(os.total)'
    
    cursor.execute(query, params)
    
    results = {}
    for row in cursor.fetchall():
        bucket = row['total_bucket']
        total = row['total_games']
        over_pct = (row['overs'] / total * 100) if total > 0 else 0
        
        results[bucket] = {
            'total': total,
            'overs': row['overs'],
            'unders': row['unders'],
            'pushes': row['pushes'],
            'over_pct': round(over_pct, 1),
            'avg_margin': round(row['avg_total_margin'] or 0, 2),
        }
    
    return results


def analyze_line_movement(db, start_date: str = None, end_date: str = None) -> dict:
    """Analyze outcomes based on line movement direction."""
    conn = db.connect()
    cursor = conn.cursor()
    
    query = '''
        WITH line_moves AS (
            SELECT 
                g.game_id,
                g.actual_spread,
                open_os.spread as open_spread,
                close_os.spread as close_spread,
                close_os.spread - open_os.spread as movement
            FROM games g
            JOIN odds_snapshots open_os ON g.game_id = open_os.game_id 
                AND open_os.snapshot_type = 'open'
            JOIN odds_snapshots close_os ON g.game_id = close_os.game_id 
                AND close_os.snapshot_type = 'close'
            WHERE g.status = 'final'
        )
        SELECT 
            CASE 
                WHEN movement < -1 THEN 'Moved to Home (Sharp on Home)'
                WHEN movement > 1 THEN 'Moved to Away (Sharp on Away)'
                ELSE 'Stable (< 1 pt move)'
            END as movement_type,
            COUNT(*) as total_games,
            SUM(CASE WHEN actual_spread < close_spread THEN 1 ELSE 0 END) as home_covers
        FROM line_moves
        GROUP BY movement_type
    '''
    
    cursor.execute(query)
    
    results = {}
    for row in cursor.fetchall():
        move_type = row['movement_type']
        total = row['total_games']
        cover_rate = (row['home_covers'] / total * 100) if total > 0 else 0
        
        results[move_type] = {
            'total': total,
            'home_covers': row['home_covers'],
            'home_cover_pct': round(cover_rate, 1),
        }
    
    return results


def generate_daily_summary(db, target_date: str) -> dict:
    """Generate summary for a specific date."""
    games = db.get_games_by_date(target_date)
    
    final_games = [g for g in games if g.get('status') == 'final']
    
    if not final_games:
        return {'date': target_date, 'games': 0, 'error': 'No final games'}
    
    summary = {
        'date': target_date,
        'games': len(final_games),
        'home_covers': 0,
        'away_covers': 0,
        'pushes': 0,
        'overs': 0,
        'unders': 0,
        'total_pushes': 0,
    }
    
    for g in final_games:
        closing_spread = g.get('closing_spread')
        closing_total = g.get('closing_total')
        actual_spread = g.get('actual_spread')
        actual_total = g.get('actual_total')
        
        if closing_spread is not None and actual_spread is not None:
            if actual_spread < closing_spread:
                summary['home_covers'] += 1
            elif actual_spread > closing_spread:
                summary['away_covers'] += 1
            else:
                summary['pushes'] += 1
                
        if closing_total is not None and actual_total is not None:
            if actual_total > closing_total:
                summary['overs'] += 1
            elif actual_total < closing_total:
                summary['unders'] += 1
            else:
                summary['total_pushes'] += 1
    
    return summary


def print_performance_report(db, start_date: str = None, end_date: str = None):
    """Print comprehensive performance report."""
    
    print("\n" + "=" * 70)
    print("BETTING PERFORMANCE ANALYTICS")
    print("=" * 70)
    
    if start_date and end_date:
        print(f"Period: {start_date} to {end_date}")
    elif start_date:
        print(f"Period: {start_date} to present")
    else:
        print("Period: All time")
    print()
    
    # ATS by spread size
    print("\n📊 ATS RESULTS BY SPREAD SIZE")
    print("-" * 60)
    print(f"{'Spread Size':<20} {'Games':>8} {'Home%':>8} {'Away%':>8} {'Push':>6}")
    print("-" * 60)
    
    spread_analysis = analyze_ats_by_spread_size(db, start_date, end_date)
    for bucket, stats in spread_analysis.items():
        away_pct = 100 - stats['home_cover_pct'] - (stats['pushes'] / stats['total'] * 100 if stats['total'] > 0 else 0)
        push_pct = stats['pushes'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"{bucket:<20} {stats['total']:>8} {stats['home_cover_pct']:>7.1f}% {away_pct:>7.1f}% {push_pct:>5.1f}%")
    
    # Totals analysis
    print("\n📊 OVER/UNDER RESULTS BY TOTAL SIZE")
    print("-" * 60)
    print(f"{'Total Range':<20} {'Games':>8} {'Over%':>8} {'Under%':>8} {'Avg Diff':>10}")
    print("-" * 60)
    
    totals_analysis = analyze_totals_accuracy(db, start_date, end_date)
    for bucket, stats in totals_analysis.items():
        under_pct = 100 - stats['over_pct'] - (stats['pushes'] / stats['total'] * 100 if stats['total'] > 0 else 0)
        print(f"{bucket:<20} {stats['total']:>8} {stats['over_pct']:>7.1f}% {under_pct:>7.1f}% {stats['avg_margin']:>+10.1f}")
    
    # Line movement analysis
    print("\n📈 LINE MOVEMENT ANALYSIS")
    print("-" * 60)
    line_analysis = analyze_line_movement(db, start_date, end_date)
    for move_type, stats in line_analysis.items():
        print(f"{move_type}: {stats['total']} games, "
              f"Home covers {stats['home_cover_pct']:.1f}%")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate performance analytics from historical data",
    )
    
    parser.add_argument(
        "--start", "-s",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", "-e", 
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="kenpom_basic",
        help="Model name for prediction analysis"
    )
    parser.add_argument(
        "--daily",
        type=str,
        default=None,
        help="Show summary for specific date"
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export report to JSON file"
    )
    
    args = parser.parse_args()
    
    from kenp0m_sp0rts_analyzer.historical_odds_db import get_db
    
    db = get_db()
    
    if args.daily:
        summary = generate_daily_summary(db, args.daily)
        print(f"\n📅 Summary for {summary['date']}:")
        print(f"   Games: {summary['games']}")
        print(f"   ATS: Home {summary['home_covers']} | Away {summary['away_covers']} | Push {summary['pushes']}")
        print(f"   O/U: Over {summary['overs']} | Under {summary['unders']} | Push {summary['total_pushes']}")
    else:
        print_performance_report(db, args.start, args.end)
        
        # Model performance if we have predictions
        model_perf = get_model_summary(db, args.model, args.start, args.end)
        if 'error' not in model_perf:
            print(f"\n🎯 MODEL PERFORMANCE: {args.model}")
            print("-" * 60)
            print(f"   Spread Record: {model_perf['spread']['record']} ({model_perf['spread']['win_pct']}%)")
            print(f"   Totals Record: {model_perf['totals']['record']} ({model_perf['totals']['win_pct']}%)")
            print(f"   Avg CLV Spread: {model_perf['clv']['avg_spread']:+.2f}")
            print(f"   Avg CLV Total: {model_perf['clv']['avg_total']:+.2f}")
    
    if args.export:
        report = {
            'generated_at': datetime.now().isoformat(),
            'period': {'start': args.start, 'end': args.end},
            'spread_by_size': analyze_ats_by_spread_size(db, args.start, args.end),
            'totals_by_size': analyze_totals_accuracy(db, args.start, args.end),
            'line_movement': analyze_line_movement(db, args.start, args.end),
        }
        
        with open(args.export, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📁 Report exported to: {args.export}")
    
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
