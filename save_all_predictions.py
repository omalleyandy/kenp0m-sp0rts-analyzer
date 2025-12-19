"""Save all 13 NCAA predictions to database for ML tracking."""

import json
import sqlite3
import math
from datetime import date, datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "predictions_tracker.db"

# Team ratings (KenPom-style estimates)
TEAM_RATINGS = {
    'Wisconsin': {'rank': 28, 'adj_em': 18.5, 'adj_oe': 112.8, 'adj_de': 94.3, 'tempo': 63.5, 'luck': 0.012, 'sos': 9.2},
    'Villanova': {'rank': 52, 'adj_em': 13.2, 'adj_oe': 109.5, 'adj_de': 96.3, 'tempo': 66.8, 'luck': -0.005, 'sos': 8.8},
    'Providence': {'rank': 45, 'adj_em': 14.8, 'adj_oe': 111.2, 'adj_de': 96.4, 'tempo': 67.2, 'luck': 0.015, 'sos': 8.5},
    'Seton Hall': {'rank': 78, 'adj_em': 9.5, 'adj_oe': 106.8, 'adj_de': 97.3, 'tempo': 68.5, 'luck': -0.008, 'sos': 7.2},
    'BYU': {'rank': 10, 'adj_em': 26.5, 'adj_oe': 115.8, 'adj_de': 89.3, 'tempo': 67.2, 'luck': 0.022, 'sos': 7.8},
    'UCLA': {'rank': 24, 'adj_em': 19.8, 'adj_oe': 112.8, 'adj_de': 93.0, 'tempo': 66.5, 'luck': 0.008, 'sos': 6.5},
    'Washington': {'rank': 95, 'adj_em': 7.2, 'adj_oe': 105.5, 'adj_de': 98.3, 'tempo': 69.8, 'luck': -0.015, 'sos': 5.8},
    'Saint Marys CA': {'rank': 18, 'adj_em': 22.1, 'adj_oe': 113.5, 'adj_de': 91.4, 'tempo': 61.2, 'luck': 0.018, 'sos': 4.2},
    'Florida Atlantic': {'rank': 82, 'adj_em': 8.8, 'adj_oe': 105.2, 'adj_de': 96.4, 'tempo': 67.5, 'luck': -0.012, 'sos': 3.5},
    'South Dakota State': {'rank': 85, 'adj_em': 8.2, 'adj_oe': 107.5, 'adj_de': 99.3, 'tempo': 68.2, 'luck': 0.005, 'sos': 2.1},
    'Belmont': {'rank': 92, 'adj_em': 7.5, 'adj_oe': 108.2, 'adj_de': 100.7, 'tempo': 69.5, 'luck': -0.002, 'sos': 1.8},
    'Akron': {'rank': 75, 'adj_em': 10.2, 'adj_oe': 106.8, 'adj_de': 96.6, 'tempo': 68.8, 'luck': 0.010, 'sos': 3.2},
    'Tulsa': {'rank': 110, 'adj_em': 5.5, 'adj_oe': 103.8, 'adj_de': 98.3, 'tempo': 67.2, 'luck': -0.018, 'sos': 4.5},
    'Drexel': {'rank': 135, 'adj_em': 2.8, 'adj_oe': 102.5, 'adj_de': 99.7, 'tempo': 66.5, 'luck': 0.008, 'sos': 2.2},
    'Cal Irvine': {'rank': 125, 'adj_em': 3.8, 'adj_oe': 104.2, 'adj_de': 100.4, 'tempo': 65.8, 'luck': -0.005, 'sos': 1.5},
    'UC San Diego': {'rank': 145, 'adj_em': 1.5, 'adj_oe': 101.8, 'adj_de': 100.3, 'tempo': 66.2, 'luck': 0.012, 'sos': 1.2},
    'Wisc Milwaukee': {'rank': 180, 'adj_em': -2.5, 'adj_oe': 99.8, 'adj_de': 102.3, 'tempo': 68.5, 'luck': -0.022, 'sos': 2.8},
    'Sacred Heart': {'rank': 225, 'adj_em': -6.2, 'adj_oe': 98.5, 'adj_de': 104.7, 'tempo': 69.2, 'luck': 0.005, 'sos': -1.5},
    'Dartmouth': {'rank': 240, 'adj_em': -7.8, 'adj_oe': 97.2, 'adj_de': 105.0, 'tempo': 68.8, 'luck': -0.008, 'sos': -2.2},
    'Mt. St. Marys': {'rank': 195, 'adj_em': -4.2, 'adj_oe': 100.5, 'adj_de': 104.7, 'tempo': 67.5, 'luck': 0.002, 'sos': 0.5},
    'Eastern Michigan': {'rank': 285, 'adj_em': -12.5, 'adj_oe': 95.8, 'adj_de': 108.3, 'tempo': 70.2, 'luck': -0.025, 'sos': 1.8},
    'Western Kentucky': {'rank': 155, 'adj_em': 0.5, 'adj_oe': 102.8, 'adj_de': 102.3, 'tempo': 69.5, 'luck': 0.015, 'sos': 2.5},
    'Seattle U': {'rank': 175, 'adj_em': -1.8, 'adj_oe': 101.2, 'adj_de': 103.0, 'tempo': 67.8, 'luck': -0.010, 'sos': 1.2},
    'Abilene Christian': {'rank': 212, 'adj_em': -3.8, 'adj_oe': 100.5, 'adj_de': 104.3, 'tempo': 68.5, 'luck': 0.008, 'sos': -0.8},
    'Cal Poly SLO': {'rank': 310, 'adj_em': -15.2, 'adj_oe': 93.5, 'adj_de': 108.7, 'tempo': 67.2, 'luck': -0.015, 'sos': -1.2},
    'San Diego': {'rank': 265, 'adj_em': -10.5, 'adj_oe': 96.8, 'adj_de': 107.3, 'tempo': 68.8, 'luck': 0.005, 'sos': 0.2},
}

GAMES = [
    {'time': '11:00 AM', 'away': 'Dartmouth', 'home': 'Sacred Heart', 'spread': -1, 'total': 155},
    {'time': '5:00 PM', 'away': 'South Dakota State', 'home': 'Wisc Milwaukee', 'spread': 0, 'total': 153},
    {'time': '6:30 PM', 'away': 'Seton Hall', 'home': 'Providence', 'spread': -2.5, 'total': 153.5},
    {'time': '7:00 PM', 'away': 'Mt. St. Marys', 'home': 'Drexel', 'spread': -5, 'total': 141},
    {'time': '7:00 PM', 'away': 'Eastern Michigan', 'home': 'Akron', 'spread': -17, 'total': 161},
    {'time': '7:30 PM', 'away': 'Tulsa', 'home': 'Western Kentucky', 'spread': 2.5, 'total': 158},
    {'time': '8:00 PM', 'away': 'Villanova', 'home': 'Wisconsin', 'spread': -5.5, 'total': 150.5},
    {'time': '9:00 PM', 'away': 'Belmont', 'home': 'Cal Irvine', 'spread': -1.5, 'total': 147},
    {'time': '9:30 PM', 'away': 'Abilene Christian', 'home': 'BYU', 'spread': -32, 'total': 146},
    {'time': '10:00 PM', 'away': 'Cal Poly SLO', 'home': 'UCLA', 'spread': -26, 'total': 160.5},
    {'time': '10:00 PM', 'away': 'San Diego', 'home': 'UC San Diego', 'spread': -13.5, 'total': 157},
    {'time': '10:00 PM', 'away': 'Florida Atlantic', 'home': 'Saint Marys CA', 'spread': -13.5, 'total': 144},
    {'time': '11:30 PM', 'away': 'Seattle U', 'home': 'Washington', 'spread': -8, 'total': 147.5},
]


def main():
    DB_PATH.parent.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DROP TABLE IF EXISTS predictions')
    conn.execute('''
        CREATE TABLE predictions (
            game_id TEXT PRIMARY KEY,
            game_date TEXT,
            game_time TEXT,
            away_team TEXT,
            home_team TEXT,
            vegas_spread REAL,
            vegas_total REAL,
            predicted_margin REAL,
            predicted_total REAL,
            home_win_prob REAL,
            spread_edge REAL,
            total_edge REAL,
            spread_pick TEXT,
            total_pick TEXT,
            features_json TEXT,
            actual_home_score INTEGER,
            actual_away_score INTEGER,
            actual_margin INTEGER,
            actual_total INTEGER,
            spread_result TEXT,
            total_result TEXT,
            created_at TEXT
        )
    ''')
    
    count = 0
    for game in GAMES:
        home = TEAM_RATINGS.get(game['home'])
        away = TEAM_RATINGS.get(game['away'])
        if not home or not away:
            continue
        
        # Create features
        features = {
            'adj_em_diff': home['adj_em'] - away['adj_em'],
            'adj_oe_diff': home['adj_oe'] - away['adj_oe'],
            'adj_de_diff': home['adj_de'] - away['adj_de'],
            'tempo_avg': (home['tempo'] + away['tempo']) / 2,
            'tempo_diff': home['tempo'] - away['tempo'],
            'luck_diff': home['luck'] - away['luck'],
            'sos_diff': home['sos'] - away['sos'],
            'home_advantage': 1.0,
            'rank_diff': away['rank'] - home['rank'],
        }
        features['em_tempo_interaction'] = features['adj_em_diff'] * features['tempo_avg']
        
        # Prediction
        em_diff = home['adj_em'] - away['adj_em']
        avg_tempo = (home['tempo'] + away['tempo']) / 2
        tempo_factor = avg_tempo / 67.5
        hca = 3.75
        
        pred_margin = em_diff * tempo_factor + hca
        avg_oe = (home['adj_oe'] + away['adj_oe']) / 2
        pred_total = (avg_oe / 100) * (avg_tempo * 2)
        win_prob = 1 / (1 + math.exp(-0.15 * pred_margin))
        
        spread_edge = pred_margin - (-game['spread'])
        total_edge = pred_total - game['total']
        
        if spread_edge >= 2.0:
            spread_pick = f"{game['home']} {game['spread']}"
        elif spread_edge <= -2.0:
            spread_pick = f"{game['away']} +{-game['spread']}"
        else:
            spread_pick = 'PASS'
        
        if total_edge >= 3.0:
            total_pick = f"OVER {game['total']}"
        elif total_edge <= -3.0:
            total_pick = f"UNDER {game['total']}"
        else:
            total_pick = 'PASS'
        
        game_id = f"{date.today().isoformat()}_{game['away'].replace(' ', '')}_{game['home'].replace(' ', '')}"
        
        conn.execute('''
            INSERT INTO predictions
            (game_id, game_date, game_time, away_team, home_team, vegas_spread, vegas_total,
             predicted_margin, predicted_total, home_win_prob, spread_edge, total_edge,
             spread_pick, total_pick, features_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (game_id, str(date.today()), game['time'], game['away'], game['home'],
              game['spread'], game['total'], round(pred_margin, 1), round(pred_total, 1),
              round(win_prob, 3), round(spread_edge, 1), round(total_edge, 1),
              spread_pick, total_pick, json.dumps(features), datetime.now().isoformat()))
        count += 1
    
    conn.commit()
    print(f"✅ Saved {count} predictions to {DB_PATH}")
    print()
    
    # Show summary
    print("=" * 85)
    print("  PREDICTIONS SAVED FOR ML TRACKING")
    print("=" * 85)
    
    rows = conn.execute('''
        SELECT game_time, away_team, home_team, spread_pick, total_pick, spread_edge, total_edge 
        FROM predictions ORDER BY game_time
    ''').fetchall()
    
    print()
    print(f"{'Time':<10} {'Away':<20} {'Home':<20} {'Spread Pick':<25} {'Total Pick':<15}")
    print("-" * 85)
    for r in rows:
        print(f"{r[0]:<10} {r[1]:<20} {r[2]:<20} {r[3]:<25} {r[4]:<15}")
    
    conn.close()
    print()
    print("=" * 85)


if __name__ == "__main__":
    main()
