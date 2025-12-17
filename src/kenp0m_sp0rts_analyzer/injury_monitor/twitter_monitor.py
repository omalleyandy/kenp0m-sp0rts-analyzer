"""Real-time Twitter monitoring for NCAA basketball injury news.

This module monitors beat reporters on Twitter for injury updates and
late scratch information that can create betting edges.
"""

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import tweepy


# Injury-related keywords to monitor
INJURY_KEYWORDS = [
    # Status keywords
    "injury", "injured", "hurt", "out", "doubtful", "questionable", "probable",
    "game-time decision", "will not play", "ruled out", "available",
    "not available", "scratch", "scratched", "DNP", "inactive", "active",

    # Body parts
    "ankle", "knee", "back", "shoulder", "wrist", "hand", "foot", "leg",
    "hamstring", "quad", "groin", "hip", "concussion", "illness", "sick",
    "flu", "covid", "protocol",

    # Participation
    "practice", "shootaround", "warmups", "warming up", "limited",
    "full participant", "did not practice", "did not participate",

    # Lineup
    "starting lineup", "starting five", "will start", "won't start",
    "replaced by", "backup", "coming off bench"
]

# Exclusion keywords (avoid false positives)
EXCLUSION_KEYWORDS = [
    "fantasy", "draft kings", "fanduel",  # Fantasy sports
    "tomorrow", "yesterday", "last week",  # Past/future references
    "if he", "if they", "could be",  # Speculation
]


@dataclass
class InjuryTweet:
    """Structured injury information from a tweet."""

    tweet_id: str
    timestamp: datetime
    author: str
    author_handle: str
    text: str

    # Extracted info
    player_names: list[str]
    teams_mentioned: list[str]
    status_keywords: list[str]
    injury_keywords: list[str]

    # Confidence scoring
    confidence: float
    is_definitive: bool  # "OUT" vs "questionable"

    # Context
    game_context: str | None = None
    raw_url: str | None = None


@dataclass
class BeatReporter:
    """Beat reporter information."""

    handle: str
    name: str
    teams: list[str]
    tier: int  # 1 = most reliable, 2 = reliable, 3 = sometimes useful
    outlet: str


class TwitterInjuryMonitor:
    """Monitor Twitter for real-time injury news."""

    def __init__(
        self,
        bearer_token: str,
        beat_reporters: dict[str, list[BeatReporter]],
        injury_callback: Callable[[InjuryTweet], None] | None = None
    ):
        """Initialize Twitter injury monitor.

        Args:
            bearer_token: Twitter API v2 bearer token
            beat_reporters: Dict mapping team names to list of beat reporters
            injury_callback: Function to call when injury detected
        """
        self.client = tweepy.Client(bearer_token=bearer_token)
        self.beat_reporters = beat_reporters
        self.injury_callback = injury_callback

        # Build user ID mapping
        self.reporter_handles = set()
        self.reporter_info = {}
        for team, reporters in beat_reporters.items():
            for reporter in reporters:
                self.reporter_handles.add(reporter.handle.lstrip('@'))
                self.reporter_info[reporter.handle] = reporter

        # Cache for recent tweets (avoid duplicates)
        self.recent_tweets = set()
        self.recent_tweets_window = timedelta(hours=24)

    def start_monitoring(self, teams_to_monitor: list[str] | None = None):
        """Start monitoring Twitter for injury news.

        Args:
            teams_to_monitor: List of team names to monitor, or None for all
        """
        print(f"🔍 Starting Twitter injury monitor...")
        print(f"📊 Monitoring {len(self.reporter_handles)} beat reporters")

        if teams_to_monitor:
            print(f"🏀 Focusing on teams: {', '.join(teams_to_monitor)}")

        # Get user IDs for the reporters we want to monitor
        user_ids = self._get_user_ids(teams_to_monitor)

        if not user_ids:
            print("❌ No reporters found to monitor!")
            return

        print(f"✅ Monitoring {len(user_ids)} reporter accounts")

        # Start streaming
        self._stream_tweets(user_ids)

    def _get_user_ids(
        self,
        teams_to_monitor: list[str] | None = None
    ) -> list[str]:
        """Get Twitter user IDs for beat reporters.

        Args:
            teams_to_monitor: Teams to focus on, or None for all

        Returns:
            List of Twitter user IDs to monitor
        """
        handles_to_monitor = []

        if teams_to_monitor:
            # Only monitor reporters for specific teams
            for team in teams_to_monitor:
                if team in self.beat_reporters:
                    for reporter in self.beat_reporters[team]:
                        handles_to_monitor.append(reporter.handle.lstrip('@'))
        else:
            # Monitor all reporters
            handles_to_monitor = list(self.reporter_handles)

        # Look up user IDs
        user_ids = []
        batch_size = 100  # Twitter API limit

        for i in range(0, len(handles_to_monitor), batch_size):
            batch = handles_to_monitor[i:i + batch_size]

            try:
                response = self.client.get_users(
                    usernames=batch,
                    user_fields=['id', 'username', 'name']
                )

                if response.data:
                    for user in response.data:
                        user_ids.append(user.id)

                time.sleep(1)  # Rate limit protection

            except Exception as e:
                print(f"⚠️ Error looking up users: {e}")
                continue

        return user_ids

    def _stream_tweets(self, user_ids: list[str]):
        """Stream tweets from monitored reporters.

        Args:
            user_ids: List of Twitter user IDs to monitor
        """
        # Note: Twitter API v2 streaming is complex
        # For production, use tweepy.StreamingClient or filtered stream
        # For now, we'll use polling (search recent tweets every minute)

        print("🔄 Starting tweet polling (checking every 60 seconds)...")

        last_check = datetime.now() - timedelta(minutes=5)

        while True:
            try:
                # Search recent tweets from our reporters
                for user_id in user_ids:
                    tweets = self._get_recent_tweets_from_user(
                        user_id,
                        since=last_check
                    )

                    for tweet in tweets:
                        self._process_tweet(tweet)

                # Update last check time
                last_check = datetime.now()

                # Wait before next poll
                print(f"⏸️  Sleeping 60s... (checked at {last_check.strftime('%I:%M:%S %p')})")
                time.sleep(60)

            except KeyboardInterrupt:
                print("\n⛔ Monitoring stopped by user")
                break
            except Exception as e:
                print(f"⚠️ Error in monitoring loop: {e}")
                time.sleep(60)
                continue

    def _get_recent_tweets_from_user(
        self,
        user_id: str,
        since: datetime
    ) -> list:
        """Get recent tweets from a specific user.

        Args:
            user_id: Twitter user ID
            since: Only get tweets after this time

        Returns:
            List of recent tweets
        """
        try:
            response = self.client.get_users_tweets(
                id=user_id,
                max_results=10,
                start_time=since,
                tweet_fields=['created_at', 'text', 'author_id'],
                expansions=['author_id'],
                user_fields=['username', 'name']
            )

            if not response.data:
                return []

            # Build tweets with author info
            tweets = []
            users = {user.id: user for user in response.includes['users']}

            for tweet in response.data:
                author = users.get(tweet.author_id)
                tweets.append({
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'author_id': tweet.author_id,
                    'author_username': author.username if author else None,
                    'author_name': author.name if author else None
                })

            return tweets

        except Exception as e:
            print(f"⚠️ Error fetching tweets for user {user_id}: {e}")
            return []

    def _process_tweet(self, tweet: dict):
        """Process a tweet for injury information.

        Args:
            tweet: Tweet data dict
        """
        tweet_id = tweet['id']

        # Skip if already processed
        if tweet_id in self.recent_tweets:
            return

        # Check if tweet contains injury keywords
        text_lower = tweet['text'].lower()

        # Quick filter: must contain at least one injury keyword
        if not any(keyword in text_lower for keyword in INJURY_KEYWORDS):
            return

        # Check exclusions
        if any(exclusion in text_lower for exclusion in EXCLUSION_KEYWORDS):
            return

        # Extract injury information
        injury_tweet = self._extract_injury_info(tweet)

        if injury_tweet and injury_tweet.player_names:
            # Add to processed cache
            self.recent_tweets.add(tweet_id)

            # Clean up old tweets from cache
            if len(self.recent_tweets) > 1000:
                self.recent_tweets = set(list(self.recent_tweets)[-500:])

            # Log detection
            print(f"\n🚨 INJURY DETECTED:")
            print(f"   Tweet: {tweet['text'][:100]}...")
            print(f"   Players: {', '.join(injury_tweet.player_names)}")
            print(f"   Status: {', '.join(injury_tweet.status_keywords)}")
            print(f"   Confidence: {injury_tweet.confidence:.2f}")

            # Call callback if provided
            if self.injury_callback:
                self.injury_callback(injury_tweet)

    def _extract_injury_info(self, tweet: dict) -> InjuryTweet | None:
        """Extract structured injury information from tweet.

        Args:
            tweet: Tweet data dict

        Returns:
            InjuryTweet object or None if no valid info extracted
        """
        text = tweet['text']
        text_lower = text.lower()

        # Extract status keywords
        status_keywords = []
        for keyword in ["out", "doubtful", "questionable", "probable",
                       "ruled out", "will not play", "scratch", "DNP",
                       "game-time decision", "available"]:
            if keyword in text_lower:
                status_keywords.append(keyword)

        # Extract injury keywords
        injury_keywords = []
        for keyword in ["ankle", "knee", "back", "shoulder", "wrist",
                       "hamstring", "concussion", "illness", "protocol"]:
            if keyword in text_lower:
                injury_keywords.append(keyword)

        # Extract player names (simple heuristic - proper nouns)
        # In production, use NER (Named Entity Recognition)
        player_names = self._extract_player_names(text)

        # Extract team names
        teams_mentioned = self._extract_team_names(text)

        # Calculate confidence
        confidence = self._calculate_confidence(
            status_keywords,
            injury_keywords,
            player_names,
            tweet
        )

        # Is this definitive (OUT) or uncertain (questionable)?
        is_definitive = any(
            keyword in status_keywords
            for keyword in ["out", "ruled out", "will not play", "scratch", "DNP"]
        )

        return InjuryTweet(
            tweet_id=tweet['id'],
            timestamp=tweet['created_at'],
            author=tweet.get('author_name', 'Unknown'),
            author_handle=tweet.get('author_username', 'unknown'),
            text=text,
            player_names=player_names,
            teams_mentioned=teams_mentioned,
            status_keywords=status_keywords,
            injury_keywords=injury_keywords,
            confidence=confidence,
            is_definitive=is_definitive,
            raw_url=f"https://twitter.com/user/status/{tweet['id']}"
        )

    def _extract_player_names(self, text: str) -> list[str]:
        """Extract player names from tweet text.

        This is a simple heuristic version. For production, use spaCy NER.

        Args:
            text: Tweet text

        Returns:
            List of potential player names
        """
        # Simple pattern: Capital Letter followed by lowercase, then Capital Letter
        # Examples: "Mikel Brown", "Cooper Flagg", "Ryan Nembhard"
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        matches = re.findall(pattern, text)

        # Filter out common false positives
        false_positives = ['The', 'This', 'That', 'North Carolina', 'Michigan State']
        matches = [m for m in matches if m not in false_positives]

        return matches

    def _extract_team_names(self, text: str) -> list[str]:
        """Extract team names from tweet text.

        Args:
            text: Tweet text

        Returns:
            List of team names mentioned
        """
        # Check against known team names
        teams_found = []

        # Common team names to check
        common_teams = [
            'Duke', 'Kentucky', 'Kansas', 'North Carolina', 'Louisville',
            'Tennessee', 'UConn', 'Gonzaga', 'Villanova', 'Michigan State',
            'Arizona', 'UCLA', 'Michigan', 'Indiana', 'Purdue'
        ]

        for team in common_teams:
            if team.lower() in text.lower():
                teams_found.append(team)

        return teams_found

    def _calculate_confidence(
        self,
        status_keywords: list[str],
        injury_keywords: list[str],
        player_names: list[str],
        tweet: dict
    ) -> float:
        """Calculate confidence score for injury detection.

        Args:
            status_keywords: Status keywords found
            injury_keywords: Injury keywords found
            player_names: Player names extracted
            tweet: Original tweet data

        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5  # Base confidence

        # Boost for definitive status
        if any(k in status_keywords for k in ["out", "ruled out", "will not play"]):
            confidence += 0.2

        # Boost for player name found
        if player_names:
            confidence += 0.15

        # Boost for injury type mentioned
        if injury_keywords:
            confidence += 0.10

        # Boost for tier 1 reporter
        author_handle = f"@{tweet.get('author_username', '')}"
        if author_handle in self.reporter_info:
            reporter = self.reporter_info[author_handle]
            if reporter.tier == 1:
                confidence += 0.15
            elif reporter.tier == 2:
                confidence += 0.10

        return min(confidence, 1.0)


def load_beat_reporters() -> dict[str, list[BeatReporter]]:
    """Load beat reporter database from JSON file.

    Returns:
        Dict mapping team names to list of beat reporters
    """
    # Find data directory (go up from src/kenp0m_sp0rts_analyzer/injury_monitor)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    json_path = project_root / "data" / "beat_reporters.json"

    if not json_path.exists():
        print(f"Warning: Beat reporters file not found at {json_path}")
        return {}

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert JSON to BeatReporter objects
    reporters_by_team = {}

    for conference_name, teams in data['conferences'].items():
        for team_name, reporters in teams.items():
            reporters_by_team[team_name] = [
                BeatReporter(
                    handle=r['handle'],
                    name=r['name'],
                    teams=[team_name],  # Single team for now
                    tier=r['tier'],
                    outlet=r['outlet']
                )
                for r in reporters
            ]

    return reporters_by_team


if __name__ == "__main__":
    # Example usage
    import os

    # Load Twitter API token from environment
    bearer_token = os.getenv('TWITTER_BEARER_TOKEN')

    if not bearer_token:
        print("❌ TWITTER_BEARER_TOKEN not set!")
        print("Get your token from: https://developer.twitter.com/")
        exit(1)

    # Load beat reporters
    beat_reporters = load_beat_reporters()

    # Define callback for when injury is detected
    def on_injury_detected(injury_tweet: InjuryTweet):
        print(f"\n{'='*80}")
        print(f"🚨 INJURY ALERT 🚨")
        print(f"{'='*80}")
        print(f"Player(s): {', '.join(injury_tweet.player_names)}")
        print(f"Status: {', '.join(injury_tweet.status_keywords)}")
        print(f"Injury: {', '.join(injury_tweet.injury_keywords)}")
        print(f"Confidence: {injury_tweet.confidence:.0%}")
        print(f"Source: @{injury_tweet.author_handle}")
        print(f"Tweet: {injury_tweet.text}")
        print(f"URL: {injury_tweet.raw_url}")
        print(f"{'='*80}\n")

    # Start monitoring
    monitor = TwitterInjuryMonitor(
        bearer_token=bearer_token,
        beat_reporters=beat_reporters,
        injury_callback=on_injury_detected
    )

    # Monitor specific teams (or None for all)
    teams_to_watch = ['Duke', 'North Carolina', 'Kentucky', 'Kansas', 'Louisville', 'Tennessee']

    monitor.start_monitoring(teams_to_watch)