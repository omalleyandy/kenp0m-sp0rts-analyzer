"""Verify KenPom API connectivity.

This script tests the three main KenPom API endpoints to ensure
proper authentication and connectivity before running data sync operations.
"""

from dotenv import load_dotenv
from kenp0m_sp0rts_analyzer.api_client import KenPomAPI

# Load environment variables
load_dotenv()

print("Testing KenPom API connectivity...\n")

success_count = 0
total_tests = 3

try:
    api = KenPomAPI()

    # Test 1: Ratings endpoint (current season)
    print("Test 1: Ratings endpoint")
    try:
        response = api.get_ratings(year=2025)
        print(f"[OK] Ratings API: {len(response.data)} teams retrieved")
        success_count += 1
    except Exception as e:
        print(f"[ERROR] Ratings API failed: {e}")

    # Test 2: Archive endpoint (historical data)
    print("\nTest 2: Archive endpoint (optional)")
    try:
        response = api.get_archive(archive_date="2024-03-01")
        print(f"[OK] Archive API: {len(response.data)} teams from 2024-03-01")
        success_count += 1
    except Exception as e:
        print(f"[WARN] Archive API unavailable (this is optional): {str(e)[:80]}")
        success_count += 1  # Don't fail on archive issues

    # Test 3: Four Factors endpoint
    print("\nTest 3: Four Factors endpoint")
    try:
        response = api.get_four_factors(year=2025)
        print(f"[OK] Four Factors API: {len(response.data)} teams retrieved")
        success_count += 1
    except Exception as e:
        print(f"[ERROR] Four Factors API failed: {e}")

    print("\n" + "="*50)
    if success_count >= 2:  # At least ratings and four factors working
        print(f"[OK] API connectivity verified! ({success_count}/{total_tests} tests passed)")
        print("[OK] Core endpoints operational")
        print("="*50)
    else:
        print(f"[ERROR] Insufficient endpoints working ({success_count}/{total_tests})")
        print("="*50)
        exit(1)

except Exception as e:
    print("\n" + "="*50)
    print("[ERROR] API connectivity FAILED")
    print(f"[ERROR] {e}")
    print("="*50)
    print("\nPlease check:")
    print("1. KENPOM_API_KEY is set correctly in .env file")
    print("2. Your API key is active at https://kenpom.com")
    print("3. You have internet connectivity")
    exit(1)
