#!/usr/bin/env python3
"""CLI script to reverse engineer KenPom API documentation.

This script uses Chrome DevTools Protocol to access the KenPom API
documentation page, monitor network requests, and extract endpoint
information.

Usage:
    uv run python scripts/reverse_engineer_api_docs.py
    uv run python scripts/reverse_engineer_api_docs.py --headless
    uv run python scripts/reverse_engineer_api_docs.py --email user@example.com --password pass
    
    Or with regular python (if in virtual environment):
    python scripts/reverse_engineer_api_docs.py --email user@example.com --password pass
"""

import asyncio
import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import directly from the module file to avoid __init__.py dependency chain
import importlib.util
import sys

# Create a mock package structure first
if "kenp0m_sp0rts_analyzer" not in sys.modules:
    from types import ModuleType
    mock_package = ModuleType("kenp0m_sp0rts_analyzer")
    mock_package.__path__ = [str(src_path / "kenp0m_sp0rts_analyzer")]
    sys.modules["kenp0m_sp0rts_analyzer"] = mock_package

spec = importlib.util.spec_from_file_location(
    "kenp0m_sp0rts_analyzer.api_docs_reverse_engineer",
    src_path / "kenp0m_sp0rts_analyzer" / "api_docs_reverse_engineer.py"
)
api_docs_module = importlib.util.module_from_spec(spec)
# Set proper module attributes BEFORE loading
api_docs_module.__name__ = "kenp0m_sp0rts_analyzer.api_docs_reverse_engineer"
api_docs_module.__package__ = "kenp0m_sp0rts_analyzer"
api_docs_module.__file__ = str(src_path / "kenp0m_sp0rts_analyzer" / "api_docs_reverse_engineer.py")
# Register in sys.modules before execution so dataclasses can find it
sys.modules["kenp0m_sp0rts_analyzer.api_docs_reverse_engineer"] = api_docs_module
spec.loader.exec_module(api_docs_module)

APIDocsReverseEngineer = api_docs_module.APIDocsReverseEngineer
reverse_engineer_api_docs = api_docs_module.reverse_engineer_api_docs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def print_results_summary(results) -> None:
    """Print a summary of reverse engineering results."""
    print("\n" + "=" * 80)
    print("API REVERSE ENGINEERING RESULTS")
    print("=" * 80)

    print(f"\n📊 Endpoints Discovered: {len(results.endpoints_discovered)}")
    for endpoint in results.endpoints_discovered:
        print(f"  • {endpoint.name} (source: {endpoint.source})")
        if endpoint.parameters:
            print(f"    Parameters: {len(endpoint.parameters)}")
        if endpoint.response_fields:
            print(f"    Response Fields: {len(endpoint.response_fields)}")
        if endpoint.examples:
            print(f"    Examples: {len(endpoint.examples)}")

    print(f"\n🌐 Network Requests Captured: {len(results.network_requests)}")
    api_requests = [r for r in results.network_requests if "api.php" in r.url]
    print(f"  • API Requests: {len(api_requests)}")
    for req in api_requests[:5]:  # Show first 5
        print(f"    - {req.method} {req.url}")
    if len(api_requests) > 5:
        print(f"    ... and {len(api_requests) - 5} more")

    print(f"\n📄 Documentation Tables: {len(results.documentation_tables)}")
    print(f"📝 Page Text Length: {len(results.page_text)} characters")

    print(f"\n💾 Output Directory: {Path('reports/api_reverse_engineering').absolute()}")
    print("\n" + "=" * 80)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reverse engineer KenPom API documentation using Chrome DevTools"
    )
    parser.add_argument(
        "--email",
        type=str,
        default=os.getenv("KENPOM_EMAIL"),
        help="KenPom email (or set KENPOM_EMAIL env var)",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=os.getenv("KENPOM_PASSWORD"),
        help="KenPom password (or set KENPOM_PASSWORD env var)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output files",
    )

    args = parser.parse_args()

    if not args.email or not args.password:
        logger.error(
            "Email and password required. Set KENPOM_EMAIL and KENPOM_PASSWORD "
            "environment variables or use --email and --password flags."
        )
        sys.exit(1)

    try:
        logger.info("Starting API documentation reverse engineering...")
        logger.info(f"Browser mode: {'headless' if args.headless else 'visible'}")

        async with APIDocsReverseEngineer(
            email=args.email,
            password=args.password,
            headless=args.headless,
            save_output=not args.no_save,
        ) as engineer:
            results = await engineer.reverse_engineer()

        print_results_summary(results)

        logger.info("Reverse engineering complete!")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during reverse engineering: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

