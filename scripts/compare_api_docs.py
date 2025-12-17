#!/usr/bin/env python3
"""Compare discovered API endpoints with existing implementation.

This script compares the reverse-engineered API documentation with
the current api_client.py implementation to identify:
- Missing endpoints
- Missing parameters
- Undocumented response fields
- Implementation gaps

Usage:
    python scripts/compare_api_docs.py [path_to_reverse_engineering_results.json]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI


def load_reverse_engineering_results(json_file: Path) -> dict[str, Any]:
    """Load reverse engineering results from JSON file."""
    with open(json_file, encoding="utf-8") as f:
        return json.load(f)


def get_implemented_endpoints() -> dict[str, dict[str, Any]]:
    """Get currently implemented endpoints from api_client.py."""
    implemented = {}

    # Known endpoints from api_client.py
    endpoints = {
        "ratings": {
            "method": "get_ratings",
            "parameters": ["year", "team_id", "conference", "y", "c"],
        },
        "archive": {
            "method": "get_archive",
            "parameters": [
                "archive_date",
                "year",
                "preseason",
                "team_id",
                "conference",
                "d",
            ],
        },
        "teams": {
            "method": "get_teams",
            "parameters": ["year", "conference"],
        },
        "conferences": {
            "method": "get_conferences",
            "parameters": ["year"],
        },
        "fanmatch": {
            "method": "get_fanmatch",
            "parameters": ["game_date", "d"],
        },
        "four-factors": {
            "method": "get_four_factors",
            "parameters": [
                "year",
                "team_id",
                "conference",
                "conf_only",
                "y",
                "c",
            ],
        },
        "misc-stats": {
            "method": "get_misc_stats",
            "parameters": [
                "year",
                "team_id",
                "conference",
                "conf_only",
                "y",
                "c",
            ],
        },
        "height": {
            "method": "get_height",
            "parameters": ["year", "team_id", "conference", "y", "c"],
        },
        "pointdist": {
            "method": "get_point_distribution",
            "parameters": [
                "year",
                "team_id",
                "conference",
                "conf_only",
                "y",
                "c",
            ],
        },
    }

    return endpoints


def normalize_endpoint_name(name: str) -> str:
    """Normalize endpoint name for comparison (handles hyphens, underscores, etc.)."""
    return name.lower().replace("-", "").replace("_", "").replace(".", "")


def compare_endpoints(
    discovered: list[dict[str, Any]], implemented: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Compare discovered endpoints with implemented ones."""
    comparison = {
        "missing_endpoints": [],
        "missing_parameters": {},
        "extra_parameters": {},
        "fully_covered": [],
        "partially_covered": [],
    }

    # Normalize endpoint names for comparison
    discovered_names = {normalize_endpoint_name(e["name"]): e["name"] for e in discovered}
    implemented_names = {
        normalize_endpoint_name(name): name for name in implemented.keys()
    }

    # Find missing endpoints (using original names for reporting)
    discovered_normalized = set(discovered_names.keys())
    implemented_normalized = set(implemented_names.keys())
    missing_normalized = discovered_normalized - implemented_normalized
    comparison["missing_endpoints"] = sorted(
        [discovered_names[n] for n in missing_normalized]
    )

    # Find fully covered endpoints (using original names)
    fully_covered_normalized = discovered_normalized & implemented_normalized
    comparison["fully_covered"] = sorted(
        [discovered_names[n] for n in fully_covered_normalized]
    )

    # Compare parameters for each endpoint
    for endpoint in discovered:
        discovered_name = endpoint["name"]
        normalized_name = normalize_endpoint_name(discovered_name)
        
        # Find matching implemented endpoint
        if normalized_name in implemented_normalized:
            implemented_name = implemented_names[normalized_name]
            if implemented_name in implemented:
                discovered_params = {p["name"] for p in endpoint.get("parameters", [])}
                implemented_params = set(implemented[implemented_name]["parameters"])

                # Normalize parameter names (remove aliases, handle variations)
                discovered_params_normalized = {
                    p.lower().replace("_", "").replace("-", "") for p in discovered_params
                }
                implemented_params_normalized = {
                    p.lower().replace("_", "").replace("-", "") for p in implemented_params
                }

                missing = discovered_params_normalized - implemented_params_normalized
                extra = implemented_params_normalized - discovered_params_normalized

                if missing:
                    comparison["missing_parameters"][discovered_name] = sorted(missing)
                    if discovered_name not in comparison["partially_covered"]:
                        comparison["partially_covered"].append(discovered_name)
                elif extra:
                    comparison["extra_parameters"][discovered_name] = sorted(extra)
                else:
                    if discovered_name not in comparison["fully_covered"]:
                        comparison["fully_covered"].append(discovered_name)

    return comparison


def print_comparison_report(comparison: dict[str, Any]) -> None:
    """Print a formatted comparison report."""
    print("\n" + "=" * 80)
    print("API IMPLEMENTATION COMPARISON REPORT")
    print("=" * 80)

    # Missing endpoints
    if comparison["missing_endpoints"]:
        print("\n[!] MISSING ENDPOINTS (discovered but not implemented):")
        for endpoint in comparison["missing_endpoints"]:
            print(f"  - {endpoint}")
    else:
        print("\n[OK] All discovered endpoints are implemented")

    # Missing parameters
    if comparison["missing_parameters"]:
        print("\n[WARNING] MISSING PARAMETERS:")
        for endpoint, params in comparison["missing_parameters"].items():
            print(f"  - {endpoint}:")
            for param in params:
                print(f"    * {param}")
    else:
        print("\n[OK] All discovered parameters are implemented")

    # Extra parameters (implemented but not in docs)
    if comparison["extra_parameters"]:
        print("\n[INFO] EXTRA PARAMETERS (implemented but not in docs):")
        for endpoint, params in comparison["extra_parameters"].items():
            print(f"  - {endpoint}:")
            for param in params:
                print(f"    * {param}")

    # Fully covered
    if comparison["fully_covered"]:
        print(f"\n[OK] FULLY COVERED ENDPOINTS ({len(comparison['fully_covered'])}):")
        for endpoint in sorted(comparison["fully_covered"]):
            print(f"  - {endpoint}")

    # Partially covered
    if comparison["partially_covered"]:
        print(f"\n[WARNING] PARTIALLY COVERED ENDPOINTS ({len(comparison['partially_covered'])}):")
        for endpoint in sorted(comparison["partially_covered"]):
            print(f"  - {endpoint}")

    print("\n" + "=" * 80)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare reverse-engineered API docs with implementation"
    )
    parser.add_argument(
        "results_file",
        type=Path,
        nargs="?",
        help="Path to reverse engineering results JSON file",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/api_reverse_engineering"),
        help="Directory containing reverse engineering results",
    )

    args = parser.parse_args()

    # Find latest results file if not specified
    if not args.results_file:
        if args.reports_dir.exists():
            json_files = sorted(
                args.reports_dir.glob("api_reverse_engineering_*.json"),
                reverse=True,
            )
            if json_files:
                args.results_file = json_files[0]
                print(f"Using latest results file: {args.results_file}")
            else:
                print(
                    f"No results files found in {args.reports_dir}. "
                    "Run reverse_engineer_api_docs.py first."
                )
                sys.exit(1)
        else:
            print(
                f"Results directory {args.reports_dir} does not exist. "
                "Run reverse_engineer_api_docs.py first."
            )
            sys.exit(1)

    if not args.results_file.exists():
        print(f"Results file not found: {args.results_file}")
        sys.exit(1)

    # Load results
    print(f"Loading results from {args.results_file}...")
    results = load_reverse_engineering_results(args.results_file)

    # Get implemented endpoints
    implemented = get_implemented_endpoints()

    # Compare
    comparison = compare_endpoints(
        results["endpoints_discovered"], implemented
    )

    # Print report
    print_comparison_report(comparison)

    # Save comparison report
    report_file = args.results_file.parent / "comparison_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[SAVED] Comparison report saved to {report_file}")


if __name__ == "__main__":
    main()

