"""
XGBoost Prediction Tracker - Live Game Analysis
================================================
Picks a game, makes XGBoost prediction, stores for later validation.
After game completes, update with actual result for model training.

Usage:
    # Make prediction before game
    python xgboost_tracker.py predict
    
    # Update with actual result after game
    python xgboost_tracker.py update --home-score 78 --away-score 71
"""

import json
import math
import sqlite3
import sys
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "predictions_tracker.db"


@dataclass
class TeamStats:
    """KenPom-style team statistics."""
    name: str
    rank: int
    adj_em: float
    adj_oe: float
    adj_de: float
    adj_tempo: float
    luck: float = 0.0
    sos: float = 0.0
    pythag: float = 0.0


@dataclass
class GamePrediction:
    """Complete prediction for a game."""
    game_id: str
    game_date: str
    game_time: str
    away_team: str
    home_team: str
    
    # Vegas lines
    vegas_spread: float  # Negative = home favored
    vegas_total: float
    
    # Model prediction
    predicted_margin: float  # Positive = home wins
    predicted_total: float
    home_win_prob: float
    confidence_interval: tuple[float, float]
    
    # Edge analysis
    spread_edge: float
    total_edge: float
    recommended_play: str
    
    # Feature values (for model training)
    features: dict
    
    # Results (filled in after game)
    actual_home_score: Optional[int] = None
    actual_away_score: Optional[int] = None
    actual_margin: Optional[int] = None
    actual_total: Optional[int] = None
    spread_result: Optional[str] = None  # "WIN", "LOSS", "PUSH"
    total_result: Optional[str] = None   # "WIN", "LOSS", "PUSH"
    
    # Metadata
    created_at: str = ""
    updated_at: str = ""


class XGBoostFeatureEngineer:
    """Create features for XGBoost model."""
    
    FEATURE_NAMES = [
        "adj_em_diff",
        "adj_oe_diff", 
        "adj_de_diff",
        "tempo_avg",
        "tempo_diff",
        "pythag_diff",
        "luck_diff",
        "sos_diff",
        "home_advantage",
        "em_tempo_interaction",
    ]
    
    @staticmethod
    def create_features(home: TeamStats, away: TeamStats) -> dict:
        """Create feature vector for prediction."""
        features = {}
        
        # Core efficiency differences
        features["adj_em_diff"] = home.adj_em - away.adj_em
        features["adj_oe_diff"] = home.adj_oe - away.adj_oe
        features["adj_de_diff"] = home.adj_de - away.adj_de
        
        # Tempo features
        features["tempo_avg"] = (home.adj_tempo + away.adj_tempo) / 2
        features["tempo_diff"] = home.adj_tempo - away.adj_tempo
        
        # Strength/luck
        features["pythag_diff"] = home.pythag - away.pythag
        features["luck_diff"] = home.luck - away.luck
        features["sos_diff"] = home.sos - away.sos
        
        # Home advantage
        features["home_advantage"] = 1.0
        
        # Interaction
        features["em_tempo_interaction"] = features["adj_em_diff"] * features["tempo_avg"]
        
        return features


class PredictionTracker:
    """Track predictions and results in SQLite database."""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
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
                    confidence_lower REAL,
                    confidence_upper REAL,
                    spread_edge REAL,
                    total_edge REAL,
                    recommended_play TEXT,
                    features_json TEXT,
                    actual_home_score INTEGER,
                    actual_away_score INTEGER,
                    actual_margin INTEGER,
                    actual_total INTEGER,
                    spread_result TEXT,
                    total_result TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    games_tracked INTEGER,
                    spread_wins INTEGER,
                    spread_losses INTEGER,
                    spread_pushes INTEGER,
                    total_wins INTEGER,
                    total_losses INTEGER,
                    total_pushes INTEGER,
                    mae_margin REAL,
                    mae_total REAL,
                    clv_spread REAL,
                    clv_total REAL
                )
            """)
    
    def save_prediction(self, pred: GamePrediction):
        """Save prediction to database."""
        pred.created_at = datetime.now().isoformat()
        pred.updated_at = pred.created_at
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO predictions
                (game_id, game_date, game_time, away_team, home_team,
                 vegas_spread, vegas_total, predicted_margin, predicted_total,
                 home_win_prob, confidence_lower, confidence_upper,
                 spread_edge, total_edge, recommended_play, features_json,
                 actual_home_score, actual_away_score, actual_margin, actual_total,
                 spread_result, total_result, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pred.game_id, pred.game_date, pred.game_time,
                pred.away_team, pred.home_team,
                pred.vegas_spread, pred.vegas_total,
                pred.predicted_margin, pred.predicted_total,
                pred.home_win_prob,
                pred.confidence_interval[0], pred.confidence_interval[1],
                pred.spread_edge, pred.total_edge, pred.recommended_play,
                json.dumps(pred.features),
                pred.actual_home_score, pred.actual_away_score,
                pred.actual_margin, pred.actual_total,
                pred.spread_result, pred.total_result,
                pred.created_at, pred.updated_at
            ))
    
    def update_result(self, game_id: str, home_score: int, away_score: int):
        """Update prediction with actual game result."""
        with sqlite3.connect(self.db_path) as conn:
            # Get prediction
            row = conn.execute(
                "SELECT vegas_spread, vegas_total, predicted_margin, predicted_total "
                "FROM predictions WHERE game_id = ?",
                (game_id,)
            ).fetchone()
            
            if not row:
                print(f"Game {game_id} not found!")
                return None
            
            vegas_spread, vegas_total, pred_margin, pred_total = row
            
            actual_margin = home_score - away_score
            actual_total = home_score + away_score
            
            # Determine spread result
            # vegas_spread is negative when home is favored
            # Home covers if: actual_margin > -vegas_spread
            home_cover_line = -vegas_spread
            if actual_margin > home_cover_line:
                spread_result = "WIN"  # Home covered
            elif actual_margin < home_cover_line:
                spread_result = "LOSS"  # Away covered
            else:
                spread_result = "PUSH"
            
            # Determine total result (based on our prediction)
            if pred_total > vegas_total:  # We predicted OVER
                if actual_total > vegas_total:
                    total_result = "WIN"
                elif actual_total < vegas_total:
                    total_result = "LOSS"
                else:
                    total_result = "PUSH"
            else:  # We predicted UNDER
                if actual_total < vegas_total:
                    total_result = "WIN"
                elif actual_total > vegas_total:
                    total_result = "LOSS"
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
                    total_result = ?,
                    updated_at = ?
                WHERE game_id = ?
            """, (
                home_score, away_score, actual_margin, actual_total,
                spread_result, total_result,
                datetime.now().isoformat(), game_id
            ))
            
            return {
                "game_id": game_id,
                "predicted_margin": pred_margin,
                "actual_margin": actual_margin,
                "margin_error": abs(pred_margin - actual_margin),
                "predicted_total": pred_total,
                "actual_total": actual_total,
                "total_error": abs(pred_total - actual_total),
                "spread_result": spread_result,
                "total_result": total_result,
            }
    
    def get_performance_summary(self) -> dict:
        """Get overall model performance metrics."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT 
                    COUNT(*) as total_games,
                    SUM(CASE WHEN spread_result = 'WIN' THEN 1 ELSE 0 END) as spread_wins,
                    SUM(CASE WHEN spread_result = 'LOSS' THEN 1 ELSE 0 END) as spread_losses,
                    SUM(CASE WHEN spread_result = 'PUSH' THEN 1 ELSE 0 END) as spread_pushes,
                    SUM(CASE WHEN total_result = 'WIN' THEN 1 ELSE 0 END) as total_wins,
                    SUM(CASE WHEN total_result = 'LOSS' THEN 1 ELSE 0 END) as total_losses,
                    AVG(ABS(predicted_margin - actual_margin)) as mae_margin,
                    AVG(ABS(predicted_total - actual_total)) as mae_total
                FROM predictions
                WHERE actual_margin IS NOT NULL
            """).fetchone()
            
            if not rows or rows[0] == 0:
                return {"message": "No completed predictions yet"}
            
            total, sw, sl, sp, tw, tl, mae_m, mae_t = rows
            
            spread_record = f"{sw}-{sl}" + (f"-{sp}" if sp else "")
            total_record = f"{tw}-{tl}"
            
            spread_pct = sw / (sw + sl) if (sw + sl) > 0 else 0
            total_pct = tw / (tw + tl) if (tw + tl) > 0 else 0
            
            return {
                "total_games": total,
                "spread_record": spread_record,
                "spread_percentage": f"{spread_pct:.1%}",
                "total_record": total_record,
                "total_percentage": f"{total_pct:.1%}",
                "mae_margin": round(mae_m, 1) if mae_m else None,
                "mae_total": round(mae_t, 1) if mae_t else None,
                "profitable": spread_pct > 0.524,  # Need 52.4% to beat -110 vig
            }
    
    def export_for_training(self) -> list[dict]:
        """Export completed predictions for model retraining."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM predictions
                WHERE actual_margin IS NOT NULL
            """).fetchall()
            
            return [dict(row) for row in rows]


def calculate_prediction(home: TeamStats, away: TeamStats, vegas_spread: float, vegas_total: float) -> GamePrediction:
    """Calculate XGBoost-style prediction for a game."""
    
    # Create features
    engineer = XGBoostFeatureEngineer()
    features = engineer.create_features(home, away)
    
    # Calculate predicted margin using KenPom methodology
    em_diff = home.adj_em - away.adj_em
    avg_tempo = (home.adj_tempo + away.adj_tempo) / 2
    tempo_factor = avg_tempo / 67.5
    hca = 3.75  # Home court advantage
    
    predicted_margin = em_diff * tempo_factor + hca
    
    # Calculate predicted total
    avg_oe = (home.adj_oe + away.adj_oe) / 2
    predicted_total = (avg_oe / 100) * (avg_tempo * 2)
    
    # Win probability (logistic)
    home_win_prob = 1 / (1 + math.exp(-0.15 * predicted_margin))
    
    # Confidence interval (simple approximation)
    margin_std = 10.0  # Typical college basketball margin std
    ci_lower = predicted_margin - 1.35 * margin_std / 2
    ci_upper = predicted_margin + 1.35 * margin_std / 2
    
    # Edge calculations
    spread_edge = predicted_margin - (-vegas_spread)
    total_edge = predicted_total - vegas_total
    
    # Determine recommended play
    if abs(spread_edge) >= 2.0:
        if spread_edge > 0:
            rec_play = f"{home.name} {vegas_spread:+.1f}"
        else:
            rec_play = f"{away.name} +{-vegas_spread:.1f}"
    elif abs(total_edge) >= 3.0:
        rec_play = f"{'OVER' if total_edge > 0 else 'UNDER'} {vegas_total}"
    else:
        rec_play = "PASS"
    
    # Create game ID
    game_id = f"{date.today().isoformat()}_{away.name.replace(' ', '')}_{home.name.replace(' ', '')}"
    
    return GamePrediction(
        game_id=game_id,
        game_date=str(date.today()),
        game_time="6:30 PM ET",
        away_team=away.name,
        home_team=home.name,
        vegas_spread=vegas_spread,
        vegas_total=vegas_total,
        predicted_margin=round(predicted_margin, 1),
        predicted_total=round(predicted_total, 1),
        home_win_prob=round(home_win_prob, 3),
        confidence_interval=(round(ci_lower, 1), round(ci_upper, 1)),
        spread_edge=round(spread_edge, 1),
        total_edge=round(total_edge, 1),
        recommended_play=rec_play,
        features=features,
    )


def main():
    """Main entry point."""
    print("=" * 70)
    print("  XGBoost Prediction Tracker")
    print(f"  {date.today().strftime('%A, %B %d, %Y')}")
    print("=" * 70)
    
    # Define today's selected game: Providence vs Seton Hall
    # KenPom-style stats (estimated)
    providence = TeamStats(
        name="Providence",
        rank=45,
        adj_em=14.8,
        adj_oe=111.2,
        adj_de=96.4,
        adj_tempo=67.2,
        luck=0.015,
        sos=8.5,
        pythag=0.72,
    )
    
    seton_hall = TeamStats(
        name="Seton Hall",
        rank=78,
        adj_em=9.5,
        adj_oe=106.8,
        adj_de=97.3,
        adj_tempo=68.5,
        luck=-0.008,
        sos=7.2,
        pythag=0.65,
    )
    
    # Vegas lines from overtime.ag
    vegas_spread = -2.5  # Providence favored by 2.5
    vegas_total = 153.5
    
    # Calculate prediction
    prediction = calculate_prediction(
        home=providence,
        away=seton_hall,
        vegas_spread=vegas_spread,
        vegas_total=vegas_total,
    )
    
    # Display prediction
    print(f"\n  🏀 SELECTED GAME")
    print("  " + "─" * 66)
    print(f"  {prediction.away_team} @ {prediction.home_team}")
    print(f"  Time: {prediction.game_time}")
    print("  " + "─" * 66)
    
    print(f"\n  TEAM PROFILES:")
    print(f"  Away: #{seton_hall.rank} {seton_hall.name} (AdjEM: {seton_hall.adj_em:+.1f})")
    print(f"  Home: #{providence.rank} {providence.name} (AdjEM: {providence.adj_em:+.1f})")
    
    print(f"\n  XGBOOST FEATURES:")
    for name, value in prediction.features.items():
        print(f"    {name:25}: {value:+.2f}")
    
    print(f"\n  PREDICTION:")
    print(f"    Predicted Margin:  {prediction.home_team} {prediction.predicted_margin:+.1f}")
    print(f"    Predicted Total:   {prediction.predicted_total:.1f}")
    print(f"    Win Probability:   {prediction.home_win_prob:.1%}")
    print(f"    Confidence:        ({prediction.confidence_interval[0]:+.1f}, {prediction.confidence_interval[1]:+.1f})")
    
    print(f"\n  VEGAS COMPARISON:")
    print(f"    Vegas Spread:      {prediction.home_team} {prediction.vegas_spread}")
    print(f"    Vegas Total:       {prediction.vegas_total}")
    
    print(f"\n  EDGE ANALYSIS:")
    print(f"    Spread Edge:       {prediction.spread_edge:+.1f} points")
    print(f"    Total Edge:        {prediction.total_edge:+.1f} points")
    
    print(f"\n  ✅ RECOMMENDED PLAY: {prediction.recommended_play}")
    
    # Save to database
    tracker = PredictionTracker()
    tracker.save_prediction(prediction)
    
    print(f"\n  📊 Prediction saved to database: {tracker.db_path}")
    print(f"  Game ID: {prediction.game_id}")
    
    print("\n  " + "─" * 66)
    print("  After the game, update with actual result:")
    print(f"  python xgboost_tracker.py update {prediction.game_id} <home_score> <away_score>")
    print("  " + "─" * 66)
    
    # Show current performance
    perf = tracker.get_performance_summary()
    if "total_games" in perf:
        print(f"\n  MODEL PERFORMANCE (Historical):")
        print(f"    Total Games:       {perf['total_games']}")
        print(f"    Spread Record:     {perf['spread_record']} ({perf['spread_percentage']})")
        print(f"    Total Record:      {perf['total_record']} ({perf['total_percentage']})")
        if perf.get('mae_margin'):
            print(f"    MAE Margin:        {perf['mae_margin']} points")
        print(f"    Profitable:        {'✅ YES' if perf.get('profitable') else '❌ NO'}")
    else:
        print(f"\n  No historical predictions yet. This is your first tracked game!")
    
    print("\n" + "=" * 70)
    
    return prediction


def update_result(game_id: str, home_score: int, away_score: int):
    """Update a prediction with actual result."""
    tracker = PredictionTracker()
    result = tracker.update_result(game_id, home_score, away_score)
    
    if result:
        print("\n" + "=" * 70)
        print("  RESULT UPDATE")
        print("=" * 70)
        print(f"\n  Game ID: {result['game_id']}")
        print(f"\n  Predicted Margin: {result['predicted_margin']:+.1f}")
        print(f"  Actual Margin:    {result['actual_margin']:+d}")
        print(f"  Margin Error:     {result['margin_error']:.1f} points")
        print(f"\n  Predicted Total:  {result['predicted_total']:.1f}")
        print(f"  Actual Total:     {result['actual_total']}")
        print(f"  Total Error:      {result['total_error']:.1f} points")
        print(f"\n  Spread Result:    {result['spread_result']} {'✅' if result['spread_result'] == 'WIN' else '❌' if result['spread_result'] == 'LOSS' else '➖'}")
        print(f"  Total Result:     {result['total_result']} {'✅' if result['total_result'] == 'WIN' else '❌' if result['total_result'] == 'LOSS' else '➖'}")
        
        # Show updated performance
        perf = tracker.get_performance_summary()
        print(f"\n  UPDATED MODEL PERFORMANCE:")
        print(f"    Spread Record:  {perf['spread_record']} ({perf['spread_percentage']})")
        print(f"    MAE Margin:     {perf['mae_margin']} points")
        print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "update":
        if len(sys.argv) == 5:
            game_id = sys.argv[2]
            home_score = int(sys.argv[3])
            away_score = int(sys.argv[4])
            update_result(game_id, home_score, away_score)
        else:
            print("Usage: python xgboost_tracker.py update <game_id> <home_score> <away_score>")
    else:
        main()
