"""KenPom Sports Analyzer - NCAA Division I Men's Basketball analytics.

Combines KenPom data with XGBoost ML for game predictions and edge detection.
"""

__version__ = "1.0.0"

# Core integrated predictor
from .integrated_predictor import IntegratedPredictor, GameAnalysis

# KenPom module
from .kenpom import (
    KenPomService,
    ArchiveLoader,
    BatchScheduler,
    RealtimeMonitor,
    DatabaseManager,
    KenPomRepository,
    DataValidator,
    TeamRating,
    FourFactors as KenPomFourFactors,
    PointDistribution,
    GamePrediction,
    AccuracyReport,
    MatchupData,
    KenPomError,
    TeamNotFoundError,
)

# XGBoost prediction
from .prediction import (
    XGBoostGamePredictor,
    XGBoostFeatureEngineer,
    PredictionResult,
)

# API client
from .api_client import KenPomAPI

# Analysis
from .analysis import (
    analyze_matchup,
    analyze_team_trends,
    calculate_tournament_seed_line,
    find_value_games,
    get_conference_standings,
)

# Comprehensive matchup
from .comprehensive_matchup_analysis import (
    ComprehensiveMatchupAnalyzer,
    ComprehensiveMatchupReport,
    DimensionScore,
    MatchupWeights,
)

# Four Factors
from .four_factors_matchup import (
    FourFactorsMatchup,
    FourFactorsAnalysis,
    FactorMatchup,
)

# Tournament simulation
from .tournament_simulator import (
    TournamentSimulator,
    TournamentTeam,
    TournamentProbabilities,
    GameSimulation,
    UpsetPick,
    BracketRecommendation,
)

# Report generation
from .report_generator import MatchupReportGenerator, MatchupReport

# Luck regression
from .luck_regression import LuckRegressionAnalyzer

# Legacy client
from .client import KenPomClient

# Models
from .models import (
    FourFactors,
    HomeCourtAdvantage,
    MatchupAnalysis,
    TeamEfficiency,
)

# Utils
from .utils import get_credentials, normalize_team_name, setup_logging

__all__ = [
    "__version__",
    # Core
    "IntegratedPredictor",
    "GameAnalysis",
    # KenPom Module
    "KenPomService",
    "ArchiveLoader",
    "BatchScheduler",
    "RealtimeMonitor",
    "DatabaseManager",
    "KenPomRepository",
    "DataValidator",
    "TeamRating",
    "KenPomFourFactors",
    "PointDistribution",
    "GamePrediction",
    "AccuracyReport",
    "MatchupData",
    "KenPomError",
    "TeamNotFoundError",
    # XGBoost
    "XGBoostGamePredictor",
    "XGBoostFeatureEngineer",
    "PredictionResult",
    # API
    "KenPomAPI",
    # Analysis
    "analyze_matchup",
    "analyze_team_trends",
    "calculate_tournament_seed_line",
    "find_value_games",
    "get_conference_standings",
    # Comprehensive
    "ComprehensiveMatchupAnalyzer",
    "ComprehensiveMatchupReport",
    "DimensionScore",
    "MatchupWeights",
    # Four Factors
    "FourFactorsMatchup",
    "FourFactorsAnalysis",
    "FactorMatchup",
    # Tournament
    "TournamentSimulator",
    "TournamentTeam",
    "TournamentProbabilities",
    "GameSimulation",
    "UpsetPick",
    "BracketRecommendation",
    # Reports
    "MatchupReportGenerator",
    "MatchupReport",
    # Luck
    "LuckRegressionAnalyzer",
    # Legacy
    "KenPomClient",
    # Models
    "FourFactors",
    "HomeCourtAdvantage",
    "MatchupAnalysis",
    "TeamEfficiency",
    # Utils
    "get_credentials",
    "normalize_team_name",
    "setup_logging",
]

# Optional imports
try:
    from .overtime_scraper import OvertimeScraper
    __all__.append("OvertimeScraper")
except ImportError:
    pass
