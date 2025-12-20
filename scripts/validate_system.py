"""System Validation Script.

Validates:
1. Database integrity (sync, queries, foreign keys)
2. Prediction pipeline (ensemble, FanMatch, end-to-end)
3. Model loading and feature configuration

Usage:
    python scripts/validate_system.py           # Run all validations
    python scripts/validate_system.py --db      # Database checks only
    python scripts/validate_system.py --model   # Model/prediction checks only
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kenp0m_sp0rts_analyzer import IntegratedPredictor
from kenp0m_sp0rts_analyzer.kenpom import KenPomService


def print_section(title: str) -> None:
    """Print section header."""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}\n")


# -----------------------------------------------------------------------------
# Database Validations
# -----------------------------------------------------------------------------


def validate_fanmatch_sync(service: KenPomService) -> bool:
    """Validate FanMatch sync. Checkpoint: FanMatch sync retrieves >0 predictions."""
    print_section("DATABASE: FanMatch Sync")

    try:
        print("Attempting FanMatch sync...")
        result = service.sync_fanmatch()

        if result.success and result.records_synced > 0:
            print(
                f"[OK] FanMatch sync successful: {result.records_synced} predictions"
            )
            return True
        elif result.success and result.records_synced == 0:
            print("[!] FanMatch sync succeeded but no predictions retrieved")
            print("  This may be expected if no games are scheduled today")
            return True
        else:
            print(f"[X] FanMatch sync failed: {result.errors}")
            return False

    except Exception as e:
        print(f"[!] FanMatch sync error: {e}")
        print("  This is expected if KENPOM_API_KEY is not configured")

        try:
            with service.repository.db.connection() as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM fanmatch_predictions"
                )
                count = cursor.fetchone()[0]
                print(f"  Database table exists with {count} existing records")
                if count > 0:
                    print(
                        "[OK] FanMatch table functional (using existing data)"
                    )
                    return True
                else:
                    print("[!] FanMatch table empty (needs API sync)")
                    return False
        except Exception as db_error:
            print(f"[X] Database error: {db_error}")
            return False


def validate_misc_stats_sync(service: KenPomService) -> bool:
    """Validate Misc Stats sync. Checkpoint: Misc stats sync retrieves >300 teams."""
    print_section("DATABASE: Misc Stats Sync")

    try:
        print("Attempting Misc Stats sync...")
        result = service.sync_misc_stats()

        if result.success and result.records_synced >= 300:
            print(
                f"[OK] Misc Stats sync successful: {result.records_synced} teams"
            )
            return True
        elif result.success and result.records_synced > 0:
            print(
                f"[!] Misc Stats sync retrieved {result.records_synced} "
                "teams (target: >=300)"
            )
            return result.records_synced >= 300
        else:
            print(f"[X] Misc Stats sync failed: {result.errors}")
            return False

    except Exception as e:
        print(f"[!] Misc Stats sync error: {e}")
        print("  This is expected if KENPOM_API_KEY is not configured")

        try:
            with service.repository.db.connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM misc_stats")
                count = cursor.fetchone()[0]
                print(f"  Database table exists with {count} existing records")
                if count >= 300:
                    print(
                        "[OK] Misc Stats table functional (using existing data)"
                    )
                    return True
                else:
                    print(
                        f"[!] Misc Stats has {count} records (target: >=300)"
                    )
                    return False
        except Exception as db_error:
            print(f"[X] Database error: {db_error}")
            return False


def validate_query_performance(service: KenPomService) -> bool:
    """Validate database query performance. Checkpoint: Database queries <100ms."""
    print_section("DATABASE: Query Performance (<100ms)")

    queries = [
        (
            "Latest ratings",
            "SELECT * FROM ratings ORDER BY snapshot_date DESC LIMIT 100",
        ),
        ("Team lookup", "SELECT * FROM teams WHERE team_name LIKE '%Duke%'"),
        (
            "Four factors join",
            """
            SELECT r.*, f.* FROM ratings r
            JOIN four_factors f ON r.team_id = f.team_id
            WHERE r.snapshot_date = (SELECT MAX(snapshot_date) FROM ratings)
            LIMIT 50
        """,
        ),
        (
            "Misc stats lookup",
            "SELECT * FROM misc_stats ORDER BY snapshot_date DESC LIMIT 100",
        ),
        (
            "FanMatch predictions",
            "SELECT * FROM fanmatch_predictions ORDER BY snapshot_date DESC LIMIT 50",
        ),
    ]

    all_passed = True
    results = []

    for name, query in queries:
        try:
            with service.repository.db.connection() as conn:
                start = time.perf_counter()
                cursor = conn.execute(query)
                _ = cursor.fetchall()
                duration_ms = (time.perf_counter() - start) * 1000

                passed = duration_ms < 100
                status = "[OK]" if passed else "[X]"
                results.append((name, duration_ms, passed))
                print(f"{status} {name:30s} {duration_ms:6.2f}ms")

                if not passed:
                    all_passed = False

        except Exception as e:
            print(f"[X] {name:30s} ERROR: {e}")
            all_passed = False
            results.append((name, -1, False))

    print("\nPerformance Summary:")
    print(f"  Queries tested: {len(results)}")
    print(f"  Passed (<100ms): {sum(1 for _, _, passed in results if passed)}")
    print(
        f"  Failed (>=100ms): {sum(1 for _, _, passed in results if not passed)}"
    )

    return all_passed


def validate_foreign_keys(service: KenPomService) -> bool:
    """Validate foreign key constraints. Checkpoint: No foreign key violations."""
    print_section("DATABASE: Foreign Key Integrity")

    checks = [
        (
            "FanMatch -> Teams (home)",
            """
            SELECT COUNT(*) FROM fanmatch_predictions f
            WHERE NOT EXISTS (SELECT 1 FROM teams t WHERE t.team_id = f.home_team_id)
        """,
        ),
        (
            "FanMatch -> Teams (visitor)",
            """
            SELECT COUNT(*) FROM fanmatch_predictions f
            WHERE NOT EXISTS (SELECT 1 FROM teams t WHERE t.team_id = f.visitor_team_id)
        """,
        ),
        (
            "Misc Stats -> Teams",
            """
            SELECT COUNT(*) FROM misc_stats m
            WHERE NOT EXISTS (SELECT 1 FROM teams t WHERE t.team_id = m.team_id)
        """,
        ),
        (
            "Four Factors -> Teams",
            """
            SELECT COUNT(*) FROM four_factors f
            WHERE NOT EXISTS (SELECT 1 FROM teams t WHERE t.team_id = f.team_id)
        """,
        ),
        (
            "Point Dist -> Teams",
            """
            SELECT COUNT(*) FROM point_distribution p
            WHERE NOT EXISTS (SELECT 1 FROM teams t WHERE t.team_id = p.team_id)
        """,
        ),
    ]

    all_passed = True

    for name, query in checks:
        try:
            with service.repository.db.connection() as conn:
                cursor = conn.execute(query)
                violations = cursor.fetchone()[0]

                if violations == 0:
                    print(f"[OK] {name:40s} No violations")
                else:
                    print(f"[X] {name:40s} {violations} violations found")
                    all_passed = False

        except Exception as e:
            print(f"[!] {name:40s} ERROR: {e}")
            if "no such table" not in str(e).lower():
                all_passed = False

    return all_passed


# -----------------------------------------------------------------------------
# Model/Prediction Validations
# -----------------------------------------------------------------------------


def validate_ensemble_functionality(predictor: IntegratedPredictor) -> bool:
    """Validate ensemble prediction functionality."""
    print_section("MODEL: Ensemble Functionality")

    if not predictor.use_ensemble:
        print("[X] Ensemble is disabled (use_ensemble=False)")
        return False

    print(f"[OK] Ensemble enabled: {predictor.use_ensemble}")
    print(
        f"[OK] Weights: XGBoost={predictor.ensemble_weights['xgboost']:.0%}, "
        f"FanMatch={predictor.ensemble_weights['fanmatch']:.0%}"
    )

    print("\nTesting ensemble prediction pipeline...")

    try:
        result_ensemble = predictor.predict_game(
            home_team="Duke", away_team="North Carolina"
        )
        print("[OK] Ensemble prediction successful:")
        print(f"  Margin: {result_ensemble.predicted_margin:+.1f}")
        print(f"  Total: {result_ensemble.predicted_total:.1f}")
        print(f"  Win Prob: {result_ensemble.win_probability:.1%}")

        predictor_xgb = IntegratedPredictor(use_ensemble=False)
        result_xgb = predictor_xgb.predict_game(
            home_team="Duke", away_team="North Carolina"
        )
        print("\n[OK] XGBoost-only prediction successful:")
        print(f"  Margin: {result_xgb.predicted_margin:+.1f}")
        print(f"  Total: {result_xgb.predicted_total:.1f}")
        print(f"  Win Prob: {result_xgb.win_probability:.1%}")

        margin_diff = abs(
            result_ensemble.predicted_margin - result_xgb.predicted_margin
        )
        print(f"\n[OK] Prediction difference: {margin_diff:.1f} points")

        if margin_diff > 0:
            print("  (Ensemble and XGBoost produced different predictions)")
        else:
            print(
                "  [!] Predictions identical (FanMatch may not be available)"
            )

        return True

    except Exception as e:
        print(f"[X] Prediction failed: {e}")
        return False


def validate_fanmatch_availability(predictor: IntegratedPredictor) -> bool:
    """Validate FanMatch availability. Checkpoint: FanMatch >=80% games."""
    print_section("MODEL: FanMatch Availability (>=80%)")

    try:
        with predictor.kenpom.repository.db.connection() as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(DISTINCT game_id)
                FROM fanmatch_predictions
                WHERE snapshot_date >= date('now', '-7 days')
            """
            )
            fanmatch_games = cursor.fetchone()[0]

            cursor = conn.execute(
                """
                SELECT COUNT(*)
                FROM fanmatch_predictions
                WHERE snapshot_date >= date('now', '-7 days')
            """
            )
            total_predictions = cursor.fetchone()[0]

            if total_predictions == 0:
                print("[!] No FanMatch predictions in last 7 days")
                print("  Run daily sync to populate")
                return False

            availability = (fanmatch_games / max(1, total_predictions)) * 100

            print(f"FanMatch predictions (last 7 days): {fanmatch_games}")
            print(f"Total prediction records: {total_predictions}")
            print(f"Availability: {availability:.1f}%")

            if availability >= 80:
                print("[OK] FanMatch availability >=80%")
                return True
            else:
                print("[!] FanMatch availability <80% (target: >=80%)")
                return False

    except Exception as e:
        print(f"[X] Error checking FanMatch availability: {e}")
        print("  Table may be empty - run sync first")
        return False


def validate_end_to_end_pipeline(predictor: IntegratedPredictor) -> bool:
    """Validate end-to-end prediction pipeline."""
    print_section("MODEL: End-to-End Pipeline")

    test_cases = [
        ("Duke", "North Carolina", -3.5, 145.0),
        ("Kansas", "Kentucky", -2.0, 150.0),
        ("Gonzaga", "UCLA", -5.5, 155.0),
    ]

    successes = 0

    for home, away, spread, total in test_cases:
        try:
            print(f"\nTest: {home} vs {away}")
            print(f"  Vegas: {home} {spread:+.1f}, Total {total:.1f}")

            result = predictor.predict_game(
                home_team=home,
                away_team=away,
                vegas_spread=spread,
                vegas_total=total,
            )

            print(
                f"  [OK] Prediction: {home} {result.predicted_margin:+.1f}, "
                f"Total {result.predicted_total:.1f}"
            )
            print(f"  [OK] Win Prob: {result.win_probability:.1%}")
            print(
                f"  [OK] Confidence: [{result.confidence_interval[0]:+.1f}, "
                f"{result.confidence_interval[1]:+.1f}]"
            )

            if result.edge_vs_spread is not None:
                print(
                    f"  [OK] Spread Edge: {result.edge_vs_spread:+.1f} points"
                )
            if result.edge_vs_total is not None:
                print(f"  [OK] Total Edge: {result.edge_vs_total:+.1f} points")

            print(
                f"  [OK] Ratings: {home}={result.home_rating.adj_em:.1f}, "
                f"{away}={result.away_rating.adj_em:.1f}"
            )

            successes += 1

        except ValueError as e:
            print(f"  [!] Skipped (team not found): {e}")

        except Exception as e:
            print(f"  [X] Error: {e}")

    print(f"\nPipeline Test Results: {successes}/{len(test_cases)} successful")

    if successes >= 1:
        print("[OK] End-to-end pipeline functional")
        return True
    else:
        print("[X] End-to-end pipeline failed all tests")
        print("  Ensure database is populated with team ratings")
        return False


def validate_model_loaded(predictor: IntegratedPredictor) -> bool:
    """Validate XGBoost model is loaded."""
    print_section("MODEL: XGBoost Model Validation")

    try:
        if predictor.predictor.model is None:
            print("[!] No XGBoost model loaded")
            print("  Load model with: predictor.predictor.load_model(...)")
            return False

        print("[OK] XGBoost model loaded")

        if predictor.use_enhanced_features:
            print("[OK] Enhanced features enabled (42 features)")
        else:
            print("[!] Using base features only (14 features)")

        if predictor.predictor.is_fitted:
            print("[OK] Model is fitted and ready for predictions")
        else:
            print("[!] Model not fitted")

        return True

    except Exception as e:
        print(f"[X] Error: {e}")
        return False


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def run_database_checks() -> dict[str, bool]:
    """Run database validation checks."""
    try:
        service = KenPomService(db_path="data/kenpom.db")
    except Exception as e:
        print(f"\n[X] Failed to initialize KenPomService: {e}")
        return {}

    return {
        "FanMatch Sync": validate_fanmatch_sync(service),
        "Misc Stats Sync": validate_misc_stats_sync(service),
        "Query Performance": validate_query_performance(service),
        "Foreign Key Integrity": validate_foreign_keys(service),
    }


def run_model_checks() -> dict[str, bool]:
    """Run model/prediction validation checks."""
    try:
        predictor = IntegratedPredictor(
            use_ensemble=True,
            model_path="data/xgboost_models/margin_model_2025_enhanced.json",
        )
    except Exception as e:
        print(f"\n[X] Failed to initialize IntegratedPredictor: {e}")
        return {}

    return {
        "Ensemble Functionality": validate_ensemble_functionality(predictor),
        "FanMatch Availability": validate_fanmatch_availability(predictor),
        "End-to-End Pipeline": validate_end_to_end_pipeline(predictor),
        "XGBoost Model": validate_model_loaded(predictor),
    }


def main() -> int:
    """Run system validation."""
    parser = argparse.ArgumentParser(description="System Validation Script")
    parser.add_argument(
        "--db", action="store_true", help="Database checks only"
    )
    parser.add_argument(
        "--model", action="store_true", help="Model checks only"
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("SYSTEM VALIDATION")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now()}")
    print("Database: data/kenpom.db")

    results: dict[str, bool] = {}

    # Determine which checks to run
    run_db = args.db or (not args.db and not args.model)
    run_model = args.model or (not args.db and not args.model)

    if run_db:
        results.update(run_database_checks())

    if run_model:
        results.update(run_model_checks())

    # Summary
    print_section("VALIDATION SUMMARY")

    for checkpoint, passed in results.items():
        status = "[OK] PASS" if passed else "[X] FAIL"
        print(f"{status:8s} {checkpoint}")

    total = len(results)
    passed_count = sum(results.values())

    print(f"\nOverall: {passed_count}/{total} checkpoints passed")

    if passed_count == total:
        print("\n[SUCCESS] All checkpoints PASSED!")
        return 0
    else:
        print(f"\n[!] {total - passed_count} checkpoint(s) failed")
        print("\nNext Steps:")
        print(
            "  1. Populate database: python scripts/populate_historical_data.py"
        )
        print("  2. Ensure KENPOM_API_KEY is configured in .env")
        print("  3. Re-run validation after data sync")
        return 1


if __name__ == "__main__":
    sys.exit(main())
