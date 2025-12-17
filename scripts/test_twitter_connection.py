"""Test Twitter API connection with X credentials."""

import os
import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

import tweepy


def test_twitter_connection():
    """Test Twitter API connection."""
    print("="*80)
    print("TWITTER API CONNECTION TEST")
    print("="*80)
    print()

    # Get bearer token from environment
    bearer_token = os.getenv('X_BEARER_TOKEN')

    if not bearer_token:
        print("[ERROR] X_BEARER_TOKEN not set!")
        print()
        print("Set it with:")
        print("  $env:X_BEARER_TOKEN='your-token'  # PowerShell")
        return False

    print(f"[OK] X_BEARER_TOKEN found (length: {len(bearer_token)} chars)")
    print()

    # Try to connect to Twitter API
    print("Testing Twitter API v2 connection...")
    try:
        client = tweepy.Client(bearer_token=bearer_token)

        # Test: Look up a well-known account (Duke Basketball)
        print("  Looking up @DukeMBB account...")
        response = client.get_user(username='DukeMBB', user_fields=['id', 'username', 'name'])

        if response.data:
            user = response.data
            print(f"  [OK] Found account: {user.name} (@{user.username})")
            print(f"  [OK] User ID: {user.id}")
            print()
            print("[OK] Twitter API connection successful!")
            print()
            return True
        else:
            print("  [ERROR] Could not find account")
            return False

    except tweepy.errors.Unauthorized as e:
        print(f"  [ERROR] Unauthorized: {e}")
        print()
        print("  Possible issues:")
        print("  - Bearer token is invalid or expired")
        print("  - App doesn't have proper permissions")
        return False

    except tweepy.errors.TooManyRequests as e:
        print(f"  [ERROR] Rate limited: {e}")
        print()
        print("  Too many requests - wait a few minutes and try again")
        return False

    except Exception as e:
        print(f"  [ERROR] Unexpected error: {e}")
        return False


def test_recent_tweets():
    """Test fetching recent tweets from a beat reporter."""
    print("="*80)
    print("RECENT TWEETS TEST")
    print("="*80)
    print()

    bearer_token = os.getenv('X_BEARER_TOKEN')
    if not bearer_token:
        print("[SKIP] X_BEARER_TOKEN not set")
        return False

    try:
        client = tweepy.Client(bearer_token=bearer_token)

        # Get recent tweets from Duke Basketball
        print("Fetching recent tweets from @DukeMBB...")
        response = client.get_users_tweets(
            id='20356470',  # Duke MBB user ID
            max_results=5,
            tweet_fields=['created_at', 'text']
        )

        if response.data:
            print(f"[OK] Found {len(response.data)} recent tweets")
            print()
            for i, tweet in enumerate(response.data, 1):
                print(f"Tweet {i}:")
                print(f"  Time: {tweet.created_at}")
                print(f"  Text: {tweet.text[:100]}...")
                print()
            return True
        else:
            print("[WARN] No tweets found (might be rate limited)")
            return False

    except Exception as e:
        print(f"[ERROR] {e}")
        return False


if __name__ == "__main__":
    # Test connection
    connection_ok = test_twitter_connection()

    if connection_ok:
        print()
        # Test fetching tweets
        test_recent_tweets()

        print()
        print("="*80)
        print("ALL TESTS COMPLETE")
        print("="*80)
        print()
        print("[OK] Twitter API is working!")
        print("[OK] Ready to run injury monitor")
    else:
        print()
        print("="*80)
        print("CONNECTION FAILED")
        print("="*80)
        print()
        print("Fix the bearer token and try again.")
