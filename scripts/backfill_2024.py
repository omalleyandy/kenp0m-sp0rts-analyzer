"""Backfill historical KenPom ratings for 2024 season.

This script loads historical ratings data from the KenPom archive API
for the 2024 college basketball season (Nov 1, 2023 - Apr 15, 2024).

The ArchiveLoader includes built-in rate limiting (10 requests/minute)
to respect API limits.
"""

from datetime import date
from kenp0m_sp0rts_analyzer.kenpom import ArchiveLoader

# Initialize the archive loader
loader = ArchiveLoader()

print("="*60)
print("KenPom Historical Data Backfill - 2024 Season")
print("="*60)
print("\nDate range: Nov 1, 2023 -> Apr 15, 2024")
print("This will take ~3-5 minutes due to API rate limiting (10 req/min)")
print("\nStarting backfill...\n")

# Backfill the 2024 season
# The backfill method automatically:
# - Skips offseason months (Apr-Oct)
# - Handles rate limiting (10 req/min)
# - Skips already-loaded dates (skip_existing=True)
# - Records errors without failing completely
result = loader.backfill(
    start_date=date(2023, 11, 1),
    end_date=date(2024, 4, 15),
    skip_existing=True
)

print("\n" + "="*60)
print("Backfill Complete!")
print("="*60)
print(f"\nResults:")
print(f"  Dates filled: {result.dates_filled}")
print(f"  Dates skipped: {result.dates_skipped} (already in database)")
print(f"  Dates requested: {result.dates_requested}")
print(f"  Errors: {len(result.errors)}")
print(f"  Duration: {result.duration_seconds:.1f}s ({result.duration_seconds/60:.1f} minutes)")

if result.errors:
    print(f"\nErrors encountered ({len(result.errors)} total):")
    # Show first 5 errors
    for i, error in enumerate(result.errors[:5], 1):
        print(f"  {i}. {error}")
    if len(result.errors) > 5:
        print(f"  ... and {len(result.errors) - 5} more errors")

print("\n" + "="*60)
print("Next steps:")
print("  1. Verify data with: uv run python -c \"from kenp0m_sp0rts_analyzer.kenpom import KenPomRepository; repo = KenPomRepository(); dates = repo.get_available_dates(); print(f'Total dates: {len(dates)}')\"")
print("  2. Continue to Phase 4: Daily sync automation")
print("="*60)
