#!/usr/bin/env python3
"""Production testing script for KenPom analyzer.

Tests the full integrated prediction system with XGBoost and ensemble modeling.
"""

from datetime import date
from kenp0m_sp0rts_analyzer import IntegratedPredictor
from kenp0m_sp0rts_analyzer.kenpom import KenPomService


def test_integrated_predictor():
    """Test the IntegratedPredictor with real data."""
    print("\n" + "="*80)
    print("PRODUCTION TEST: Integrated Predictor System")
    print("="*80 + "\n")

    # Initialize predictor
    print("[1/5] Initializing IntegratedPredictor...")
    try:
        predictor = IntegratedPredictor()
        print("   [OK] IntegratedPredictor initialized")
    except Exception as e:
        print(f"   [ERROR] Failed to initialize: {e}")
        return False

    # Check KenPom service
    print("\n[2/5] Checking KenPom service and database...")
    try:
        service = KenPomService()
        latest_ratings = service.get_latest_ratings()
        print(f"   [OK] Database has {len(latest_ratings)} teams")
        print(f"   [OK] Latest data from: {latest_ratings[0].snapshot_date if latest_ratings else 'N/A'}")
    except Exception as e:
        print(f"   [ERROR] KenPom service issue: {e}")
        return False

    # Test game prediction
    print("\n[3/5] Testing game prediction (Duke vs UNC)...")
    try:
        result = predictor.predict_game(
            home_team="Duke",
            away_team="North Carolina"
        )
        print(f"   [OK] Predicted margin: {result.predicted_margin:+.1f}")
        print(f"   [OK] Predicted total: {result.predicted_total:.1f}")
        print(f"   [OK] Win probability: {result.win_probability:.1%}")
        print(f"   [OK] Model version: {result.model_version}")
    except Exception as e:
        print(f"   [ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test with Vegas lines
    print("\n[4/5] Testing edge detection vs Vegas lines...")
    try:
        result_with_lines = predictor.predict_game(
            home_team="Duke",
            away_team="North Carolina",
            vegas_spread=-12.5,  # Duke favored by 12.5
            vegas_total=155.0
        )
        print(f"   [OK] Vegas spread: -{result_with_lines.vegas_spread}")
        print(f"   [OK] Edge vs spread: {result_with_lines.edge_vs_spread:+.1f}")
        print(f"   [OK] Has spread edge: {result_with_lines.has_spread_edge}")
        print(f"   [OK] Edge vs total: {result_with_lines.edge_vs_total:+.1f}")
        print(f"   [OK] Has total edge: {result_with_lines.has_total_edge}")
    except Exception as e:
        print(f"   [ERROR] Edge detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test batch predictions
    print("\n[5/5] Testing batch predictions...")
    try:
        games = [
            ("Duke", "North Carolina"),
            ("Kansas", "Kentucky"),
            ("Gonzaga", "UCLA")
        ]
        results = predictor.predict_batch(games)
        print(f"   [OK] Processed {len(results)} games")
        for i, result in enumerate(results, 1):
            print(f"   [OK] Game {i}: {result.home_team} vs {result.away_team} -> {result.predicted_margin:+.1f}")
    except Exception as e:
        print(f"   [ERROR] Batch prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80 + "\n")
    return True


def check_xgboost_models():
    """Check available XGBoost models."""
    from pathlib import Path

    print("\n" + "="*80)
    print("XGBOOST MODEL CHECK")
    print("="*80 + "\n")

    model_dir = Path("data/xgboost_models")
    if not model_dir.exists():
        print("[ERROR] Model directory not found!")
        return False

    models = list(model_dir.glob("*.json"))
    print(f"Found {len(models)} model files:\n")
    for model in models:
        size_kb = model.stat().st_size / 1024
        print(f"   - {model.name} ({size_kb:.1f} KB)")

    return len(models) > 0


def check_database_status():
    """Check database tables and record counts."""
    import sqlite3

    print("\n" + "="*80)
    print("DATABASE STATUS CHECK")
    print("="*80 + "\n")

    db_path = "data/kenpom.db"
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        print(f"Database has {len(tables)} tables:\n")

        for table in sorted(tables):
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"   - {table:25s} {count:>6d} rows")

        conn.close()
        return True

    except Exception as e:
        print(f"[ERROR] Database check failed: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("KENPOM SPORTS ANALYZER - PRODUCTION TEST SUITE")
    print("="*80)

    # Run all checks
    checks_passed = []

    checks_passed.append(("XGBoost Models", check_xgboost_models()))
    checks_passed.append(("Database Status", check_database_status()))
    checks_passed.append(("Integrated Predictor", test_integrated_predictor()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80 + "\n")

    for name, passed in checks_passed:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {name}")

    all_passed = all(p for _, p in checks_passed)

    if all_passed:
        print("\n" + "="*80)
        print("SYSTEM READY FOR PRODUCTION!")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("SYSTEM NOT READY - PLEASE FIX ERRORS")
        print("="*80 + "\n")

    exit(0 if all_passed else 1)
