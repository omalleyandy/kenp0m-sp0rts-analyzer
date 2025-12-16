"""KenPom Sports Analyzer - NCAA Division I Men's Basketball advanced analytics."""

from .analysis import (
    analyze_matchup,
    analyze_team_trends,
    calculate_tournament_seed_line,
    find_value_games,
    get_conference_standings,
)
from .client import KenPomClient
from .defensive_analysis import (
    DefensiveAnalyzer,
    DefensiveMatchup,
    DefensiveProfile,
)
from .experience_chemistry_analysis import (
    ExperienceChemistryAnalyzer,
    ExperienceMatchup,
    ExperienceProfile,
    TournamentReadiness,
)
from .four_factors_matchup import (
    FactorMatchup,
    FourFactorsAnalysis,
    FourFactorsMatchup,
)
from .models import (
    FourFactors,
    HomeCourtAdvantage,
    MatchupAnalysis,
    ScoutingReport,
    TeamEfficiency,
    TeamScheduleGame,
)
from .point_distribution_analysis import (
    PointDistributionAnalyzer,
    ScoringStyleMatchup,
    ScoringStyleProfile,
)
from .size_athleticism_analysis import (
    PositionMatchup,
    ReboundingCorrelation,
    SizeAthleticismAnalyzer,
    SizeMatchup,
    SizeProfile,
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
    # Four Factors Matchup (TIER 1)
    "FactorMatchup",
    "FourFactorsAnalysis",
    "FourFactorsMatchup",
    # Point Distribution Analysis (TIER 1)
    "PointDistributionAnalyzer",
    "ScoringStyleMatchup",
    "ScoringStyleProfile",
    # Defensive Analysis (TIER 1)
    "DefensiveAnalyzer",
    "DefensiveMatchup",
    "DefensiveProfile",
    # Size & Athleticism Analysis (TIER 2)
    "SizeAthleticismAnalyzer",
    "SizeProfile",
    "SizeMatchup",
    "PositionMatchup",
    "ReboundingCorrelation",
    # Experience & Chemistry Analysis (TIER 2)
    "ExperienceChemistryAnalyzer",
    "ExperienceProfile",
    "ExperienceMatchup",
    "TournamentReadiness",
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

# Optional browser/scraper exports (require optional dependencies)
try:
    from .browser import (  # noqa: F401
        BrowserConfig,
        StealthBrowser,
        create_stealth_browser,
    )
    from .scraper import KenPomScraper, scrape_kenpom  # noqa: F401

    __all__.extend(
        [
            # Browser automation
            "BrowserConfig",
            "StealthBrowser",
            "create_stealth_browser",
            # Scraper
            "KenPomScraper",
            "scrape_kenpom",
        ]
    )
except ImportError:
    # Browser dependencies not installed
    pass
