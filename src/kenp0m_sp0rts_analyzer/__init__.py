"""KenPom Sports Analyzer - NCAA Basketball analytics with Billy Walters methodology."""

from .analysis import (
    analyze_matchup,
    analyze_team_trends,
    calculate_tournament_seed_line,
    find_value_games,
    get_conference_standings,
)
from .client import KenPomClient
from .models import (
    FourFactors,
    HomeCourtAdvantage,
    MatchupAnalysis,
    ScoutingReport,
    TeamEfficiency,
    TeamScheduleGame,
)
from .utils import get_credentials, normalize_team_name, setup_logging

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Client
    "KenPomClient",
    # Analysis
    "analyze_matchup",
    "analyze_team_trends",
    "calculate_tournament_seed_line",
    "find_value_games",
    "get_conference_standings",
    # Models
    "FourFactors",
    "HomeCourtAdvantage",
    "MatchupAnalysis",
    "ScoutingReport",
    "TeamEfficiency",
    "TeamScheduleGame",
    # Utils
    "get_credentials",
    "normalize_team_name",
    "setup_logging",
]
