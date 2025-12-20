"""Utility functions for KenPom Sports Analyzer."""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def setup_logging(level: str | None = None) -> None:
    """Configure logging for the application.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR). Defaults to INFO.
    """
    log_level = level or os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_env_config() -> dict[str, str | None]:
    """Load environment configuration from .env file.

    Returns:
        Dictionary with configuration values.
    """
    # Look for .env in current dir and parent directories
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    return {
        "email": os.getenv("KENPOM_EMAIL"),
        "password": os.getenv("KENPOM_PASSWORD"),
        "cache_dir": os.getenv("KENPOM_CACHE_DIR"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
    }


def get_credentials() -> tuple[str, str]:
    """Get KenPom credentials from environment variables.

    Returns:
        Tuple of (email, password).

    Raises:
        ValueError: If credentials are not configured.
    """
    config = load_env_config()
    email = config["email"]
    password = config["password"]

    if not email or not password:
        raise ValueError(
            "KenPom credentials not found. "
            "Set KENPOM_EMAIL and KENPOM_PASSWORD environment variables."
        )

    return email, password


def get_overtime_credentials() -> tuple[str | None, str | None]:
    """Get Overtime.ag credentials from environment variables.

    Returns:
        Tuple of (username, password). May be (None, None) if not configured.
    """
    load_dotenv()
    username = os.getenv("OVERTIME_USER")
    password = os.getenv("OVERTIME_PASSWORD")
    return username, password


@lru_cache(maxsize=1)
def get_cache_dir() -> Path:
    """Get the cache directory for storing data.

    Returns:
        Path to cache directory.
    """
    cache_dir_env = os.getenv("KENPOM_CACHE_DIR")
    if cache_dir_env:
        cache_dir = Path(cache_dir_env)
    else:
        cache_dir = Path.home() / ".cache" / "kenp0m_sp0rts_analyzer"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@lru_cache(maxsize=1)
def _load_team_aliases() -> dict[str, str]:
    """Load team aliases from JSON file.

    Returns:
        Dictionary mapping alias -> canonical KenPom name.
    """
    aliases_path = (
        Path(__file__).parent.parent.parent / "data" / "team_aliases.json"
    )

    if not aliases_path.exists():
        logger.warning(f"Team aliases file not found: {aliases_path}")
        return {}

    try:
        import json

        with open(aliases_path) as f:
            data = json.load(f)
        return data.get("aliases", {})
    except Exception as e:
        logger.warning(f"Failed to load team aliases: {e}")
        return {}


def normalize_team_name(team: str) -> str:
    """Normalize team name for consistent lookups.

    Maps common abbreviations and name variations to official KenPom team names.
    Uses data/team_aliases.json for comprehensive mappings with fallback rules.

    Args:
        team: Raw team name input.

    Returns:
        Normalized team name matching KenPom API format.
    """
    normalized = team.strip()

    # 1. Check exact match in aliases (case-sensitive first)
    aliases = _load_team_aliases()
    if normalized in aliases:
        return aliases[normalized]

    # 2. Check case-insensitive match
    lower_name = normalized.lower()
    for alias, canonical in aliases.items():
        if alias.lower() == lower_name:
            return canonical

    # 3. Apply "State" -> "St." fallback rule
    # But NOT for teams where "State" is part of a different pattern
    state_exceptions = {
        "penn state": "Penn St.",
        "ohio state": "Ohio St.",
        "michigan state": "Michigan St.",
        "iowa state": "Iowa St.",
        "kansas state": "Kansas St.",
        "oregon state": "Oregon St.",
        "washington state": "Washington St.",
        "florida state": "Florida St.",
        "nc state": "N.C. State",
        "north carolina state": "N.C. State",
    }

    if lower_name in state_exceptions:
        return state_exceptions[lower_name]

    # General "State" -> "St." conversion
    if " state" in lower_name and not lower_name.endswith(" st."):
        # Convert "X State" to "X St."
        result = normalized.replace(" State", " St.").replace(" state", " St.")
        return result

    # 4. Handle common patterns that might slip through
    # "St." at start usually means "Saint" in KenPom for some teams
    if normalized.startswith("St. ") and "Joseph" in normalized:
        return "Saint Joseph's"

    return normalized


def normalize_team_name_bidirectional(
    team: str, source: str = "kenpom"
) -> tuple[str, bool]:
    """Normalize team name with match confidence indicator.

    Args:
        team: Raw team name input.
        source: Source format ("kenpom", "overtime", "espn").

    Returns:
        Tuple of (normalized_name, was_exact_match).
    """
    original = team.strip()
    normalized = normalize_team_name(original)
    was_exact = original == normalized or original in _load_team_aliases()
    return normalized, was_exact


def calculate_expected_score(
    adj_em1: float,
    adj_em2: float,
    adj_tempo1: float,
    adj_tempo2: float,
    home_advantage: float = 3.5,
    is_home_team1: bool = False,
    neutral_site: bool = True,
) -> dict[str, Any]:
    """Calculate expected game score based on efficiency metrics.

    Args:
        adj_em1: Team 1 adjusted efficiency margin.
        adj_em2: Team 2 adjusted efficiency margin.
        adj_tempo1: Team 1 adjusted tempo.
        adj_tempo2: Team 2 adjusted tempo.
        home_advantage: Home court advantage in points (default 3.5).
        is_home_team1: Whether team 1 is the home team.
        neutral_site: Whether game is at neutral site.

    Returns:
        Dictionary with prediction details.
    """
    # Average national efficiency (roughly 100 points per 100 possessions)
    national_avg_efficiency = 100.0

    # Expected tempo is average of both teams
    expected_tempo = (adj_tempo1 + adj_tempo2) / 2

    # Possessions per game (tempo is per 40 min, games are 40 min)
    possessions = expected_tempo

    # Calculate raw efficiency margin
    raw_margin = adj_em1 - adj_em2

    # Apply home court advantage
    if not neutral_site:
        if is_home_team1:
            raw_margin += home_advantage
        else:
            raw_margin -= home_advantage

    # Scale margin to actual points (margin is per 100 possessions)
    predicted_margin = raw_margin * (possessions / 100)

    # Estimate total (both teams combined)
    avg_offensive_efficiency = national_avg_efficiency + (adj_em1 + adj_em2) / 4
    predicted_total = 2 * avg_offensive_efficiency * (possessions / 100)

    return {
        "predicted_margin": round(predicted_margin, 1),
        "predicted_total": round(predicted_total, 1),
        "expected_possessions": round(possessions, 1),
        "team1_projected": round((predicted_total + predicted_margin) / 2, 1),
        "team2_projected": round((predicted_total - predicted_margin) / 2, 1),
    }


def format_record(wins: int, losses: int) -> str:
    """Format a win-loss record string.

    Args:
        wins: Number of wins.
        losses: Number of losses.

    Returns:
        Formatted record string (e.g., '25-6').
    """
    return f"{wins}-{losses}"


def parse_score(score_str: str) -> tuple[int, int]:
    """Parse a score string into team scores.

    Args:
        score_str: Score string (e.g., '85-72' or '85 - 72').

    Returns:
        Tuple of (winner_score, loser_score).

    Raises:
        ValueError: If score string cannot be parsed.
    """
    parts = score_str.replace(" ", "").split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid score format: {score_str}")

    try:
        return int(parts[0]), int(parts[1])
    except ValueError as e:
        raise ValueError(f"Invalid score format: {score_str}") from e
