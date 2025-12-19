"""
Update Prediction with Actual Result - XGBoost ML Training Pipeline
====================================================================
After the game completes, run this to:
1. Record actual scores
2. Calculate prediction error
3. Determine ATS/Total result
4. Export data for XGBoost retraining

Usage:
    python update_result.py <game_id> <home_score> <away_score>
    
Example:
    python update_result.py 2025-12-19_SetonHall_Providence 78 71
"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Database path
DB_PATH = Path(__file__).parent / "data" / "predictions_tracker.db"


def update_result(game_id: str, home_score: int, away_score: int):
    """Update prediction with actual game result."""
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Get the prediction
    row = conn.execute(
        "SELECT * FROM predictions WHERE game_id = ?", (game_id,)
    ).fetchone()
    
    if not row:
        print(f"❌ Game '{game_id}' not found in database!")
        print("\nAvailable games:")
        for r in conn.execute("SELECT game_id, home_team, away_team FROM predictions"):
            print(f"  - {r['game_id']}: {r['away_team']} @ {r['home_team']}")
        conn.close()
        return
    
    # Calculate actuals
    actual_margin = home_score - away_score
    actual_total = home_score + away_score
    
    vegas_spread = row['vegas_spread']
    vegas_total = row['vegas_total']
    pred_margin = row['predicted_margin']
    pred_total = row['predicted_total']
    
    # Determine spread result
    # vegas_spread is negative when home is favored
    # Home covers if actual_margin > -vegas_spread
    cover_line = -vegas_spread
    if actual_margin > cover_line:
        spread_result = "WIN"  # We picked home (Providence), they covered
    elif actual_margin < cover_line:
        spread_result = "LOSS"
    else:
        spread_result = "PUSH"
    
    # Determine total result (we predicted UNDER)
    if actual_total < vegas_total:
        total_result = "WIN"  # UNDER hit
    elif actual_total > vegas_total:
        total_result = "LOSS"  # OVER hit
    else:
        total_result = "PUSH"
    
    # Update database
    conn.execute("""
        UPDATE predictions SET
            actual_home_score = ?,
            actual_away_score = ?,
            actual_margin = ?,
            actual_total = ?,
            spread_result = ?,
            total_result = ?
        WHERE game_id = ?
    """, (home_score, away_score, actual_margin, actual_total,
          spread_result, total_result, game_id))
    conn.commit()
    
    # Calculate errors
    margin_error = abs(pred_margin - actual_margin)
    total_error = abs(pred_total - actual_total)
    
    # Display results
    print()
    print("=" * 72)
    print("  🏀 GAME RESULT UPDATE - XGBOOST ML TRAINING DATA")
    print("=" * 72)
    print()
    print(f"  Game: {row['away_team']} @ {row['home_team']}")
    print(f"  Date: {row['game_date']} | {row['game_time']}")
    print()
    
    print("  ┌" + "─" * 68 + "┐")
    print("  │  FINAL SCORE                                                        │")
    print("  ├" + "─" * 68 + "┤")
    print(f"  │    {row['away_team']:20} {away_score:3}                                      │")
    print(f"  │    {row['home_team']:20} {home_score:3}                                      │")
    print(f"  │    FINAL MARGIN: {row['home_team']} {actual_margin:+d}                                │")
    print(f"  │    FINAL TOTAL:  {actual_total}                                              │")
    print("  └" + "─" * 68 + "┘")
    print()
    
    print("  ┌" + "─" * 68 + "┐")
    print("  │  PREDICTION vs ACTUAL                                               │")
    print("  ├" + "─" * 68 + "┤")
    print(f"  │    Predicted Margin:  {pred_margin:+.1f}                                        │")
    print(f"  │    Actual Margin:     {actual_margin:+d}                                           │")
    print(f"  │    MARGIN ERROR:      {margin_error:.1f} points                                   │")
    print("  ├" + "─" * 68 + "┤")
    print(f"  │    Predicted Total:   {pred_total:.1f}                                         │")
    print(f"  │    Actual Total:      {actual_total}                                            │")
    print(f"  │    TOTAL ERROR:       {total_error:.1f} points                                    │")
    print("  └" + "─" * 68 + "┘")
    print()
    
    # Betting results
    spread_emoji = "✅" if spread_result == "WIN" else "❌" if spread_result == "LOSS" else "➖"
    total_emoji = "✅" if total_result == "WIN" else "❌" if total_result == "LOSS" else "➖"
    
    print("  ╔" + "═" * 68 + "╗")
    print("  ║  BETTING RESULTS                                                    ║")
    print("  ╠" + "═" * 68 + "╣")
    print(f"  ║    SPREAD: {row['home_team']} {vegas_spread} → {spread_result:4} {spread_emoji}                          ║")
    print(f"  ║    TOTAL:  UNDER {vegas_total}    → {total_result:4} {total_emoji}                          ║")
    print("  ╚" + "═" * 68 + "╝")
    print()
    
    # Show ML training data
    features = json.loads(row['features_json'])
    print("  ┌" + "─" * 68 + "┐")
    print("  │  ML TRAINING DATA (for XGBoost)                                     │")
    print("  ├" + "─" * 68 + "┤")
    print("  │  Features (X):                                                      │")
    for k, v in features.items():
        print(f"  │    {k:28}: {v:+8.3f}                          │")
    print("  ├" + "─" * 68 + "┤")
    print(f"  │  Target (y_margin): {actual_margin:+d}                                           │")
    print(f"  │  Target (y_total):  {actual_total}                                            │")
    print("  └" + "─" * 68 + "┘")
    print()
    
    # Get cumulative stats
    stats = conn.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN spread_result = 'WIN' THEN 1 ELSE 0 END) as spread_w,
            SUM(CASE WHEN spread_result = 'LOSS' THEN 1 ELSE 0 END) as spread_l,
            SUM(CASE WHEN total_result = 'WIN' THEN 1 ELSE 0 END) as total_w,
            SUM(CASE WHEN total_result = 'LOSS' THEN 1 ELSE 0 END) as total_l,
            AVG(ABS(predicted_margin - actual_margin)) as mae_margin,
            AVG(ABS(predicted_total - actual_total)) as mae_total
        FROM predictions WHERE actual_margin IS NOT NULL
    """).fetchone()
    
    if stats['total'] > 0:
        spread_pct = stats['spread_w'] / (stats['spread_w'] + stats['spread_l']) if (stats['spread_w'] + stats['spread_l']) > 0 else 0
        total_pct = stats['total_w'] / (stats['total_w'] + stats['total_l']) if (stats['total_w'] + stats['total_l']) > 0 else 0
        
        print("  ┌" + "─" * 68 + "┐")
        print("  │  CUMULATIVE MODEL PERFORMANCE                                       │")
        print("  ├" + "─" * 68 + "┤")
        print(f"  │    Total Games Tracked:  {stats['total']}                                        │")
        print(f"  │    Spread Record:        {stats['spread_w']}-{stats['spread_l']} ({spread_pct:.1%})                               │")
        print(f"  │    Total Record:         {stats['total_w']}-{stats['total_l']} ({total_pct:.1%})                               │")
        print(f"  │    MAE (Margin):         {stats['mae_margin']:.1f} points                                  │")
        print(f"  │    MAE (Total):          {stats['mae_total']:.1f} points                                  │")
        print("  ├" + "─" * 68 + "┤")
        profitable = spread_pct > 0.524
        print(f"  │    PROFITABLE (>52.4%):  {'✅ YES' if profitable else '❌ NO'}                                      │")
        print("  └" + "─" * 68 + "┘")
    
    print()
    print("=" * 72)
    
    conn.close()
    
    return {
        'margin_error': margin_error,
        'total_error': total_error,
        'spread_result': spread_result,
        'total_result': total_result,
    }


def export_training_data():
    """Export all completed predictions for XGBoost training."""
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    rows = conn.execute("""
        SELECT * FROM predictions WHERE actual_margin IS NOT NULL
    """).fetchall()
    
    training_data = []
    for row in rows:
        features = json.loads(row['features_json'])
        features['y_margin'] = row['actual_margin']
        features['y_total'] = row['actual_total']
        training_data.append(features)
    
    conn.close()
    
    # Save to JSON
    output_path = DB_PATH.parent / "xgboost_training_data.json"
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Exported {len(training_data)} games to {output_path}")
    return training_data


if __name__ == "__main__":
    if len(sys.argv) == 4:
        game_id = sys.argv[1]
        home_score = int(sys.argv[2])
        away_score = int(sys.argv[3])
        update_result(game_id, home_score, away_score)
    elif len(sys.argv) == 2 and sys.argv[1] == "export":
        export_training_data()
    else:
        print("Usage:")
        print("  Update result:  python update_result.py <game_id> <home_score> <away_score>")
        print("  Export data:    python update_result.py export")
        print()
        print("Example:")
        print("  python update_result.py 2025-12-19_SetonHall_Providence 78 71")
