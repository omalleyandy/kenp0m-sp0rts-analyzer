"""Comprehensive matchup analysis combining all 7 analyzer modules.

This module provides a unified interface for analyzing basketball matchups across
all dimensions: efficiency, tempo, four factors, scoring styles, defensive schemes,
size/athleticism, and experience/chemistry.

The analyzer produces weighted composite scores and strategic recommendations,
with configurable weights for different contexts (regular season vs NCAA Tournament).

Example:
    ```python
    from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
    from kenp0m_sp0rts_analyzer.comprehensive_matchup_analysis import (
        ComprehensiveMatchupAnalyzer
    )

    api = KenPomAPI()
    analyzer = ComprehensiveMatchupAnalyzer(api)

    # Regular season analysis
    report = analyzer.analyze_matchup("Duke", "North Carolina", 2025)
    print(report.generate_text_report())

    # NCAA Tournament analysis (different weights)
    report = analyzer.analyze_matchup(
        "Duke", "North Carolina", 2025,
        tournament_context=True
    )
    print(report.generate_text_report())
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .analysis import analyze_matchup
from .api_client import KenPomAPI
from .defensive_analysis import DefensiveAnalyzer, DefensiveMatchup
from .experience_chemistry_analysis import (
    ExperienceChemistryAnalyzer,
    ExperienceMatchup,
)
from .four_factors_matchup import FourFactorsMatchup
from .point_distribution_analysis import (
    PointDistributionAnalyzer,
    ScoringStyleMatchup,
)
from .prediction import GamePredictor, PredictionResult
from .size_athleticism_analysis import SizeAthleticismAnalyzer, SizeMatchup
from .tempo_analysis import PaceMatchupAnalysis, TempoMatchupAnalyzer


@dataclass
class DimensionScore:
    """Score for a single analysis dimension.

    Attributes:
        dimension: Name of the analysis dimension
        score: Normalized score (-10 to +10, positive favors team1)
        confidence: Confidence level (0.0 to 1.0)
        key_insight: Brief description of main finding
        details: Additional context or details
    """

    dimension: str
    score: float
    confidence: float
    key_insight: str
    details: str = ""


@dataclass
class MatchupWeights:
    """Weights for different analysis dimensions.

    Attributes:
        efficiency: Weight for efficiency/ratings analysis
        four_factors: Weight for Four Factors analysis
        tempo: Weight for tempo/pace analysis
        point_distribution: Weight for scoring style analysis
        defensive: Weight for defensive scheme analysis
        size: Weight for size/athleticism analysis
        experience: Weight for experience/chemistry analysis
    """

    efficiency: float = 0.25
    four_factors: float = 0.20
    tempo: float = 0.10
    point_distribution: float = 0.15
    defensive: float = 0.15
    size: float = 0.10
    experience: float = 0.05

    def __post_init__(self):
        """Validate that weights sum to 1.0."""
        total = (
            self.efficiency
            + self.four_factors
            + self.tempo
            + self.point_distribution
            + self.defensive
            + self.size
            + self.experience
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    @classmethod
    def tournament_weights(cls) -> MatchupWeights:
        """Create weight profile for NCAA Tournament context.

        In tournament play:
        - Experience matters more (clutch performance, tournament savvy)
        - Four Factors slightly more important (execution under pressure)
        - Tempo slightly less important (teams adjust in tournament)

        Returns:
            MatchupWeights configured for tournament analysis
        """
        return cls(
            efficiency=0.25,
            four_factors=0.22,  # Increased (execution matters)
            tempo=0.08,  # Decreased (less predictable in tourney)
            point_distribution=0.14,  # Slightly decreased
            defensive=0.14,  # Slightly decreased
            size=0.10,  # Same (still important)
            experience=0.07,  # Increased (tournament experience crucial)
        )


@dataclass
class ComprehensiveMatchupReport:
    """Complete matchup analysis report combining all dimensions.

    Attributes:
        team1: First team name
        team2: Second team name
        season: Season year
        tournament_context: Whether this is tournament analysis
        weights: Weight configuration used
        efficiency_score: Efficiency dimension score
        four_factors_score: Four Factors dimension score
        tempo_score: Tempo dimension score
        point_distribution_score: Point distribution dimension score
        defensive_score: Defensive dimension score
        size_score: Size dimension score
        experience_score: Experience dimension score
        composite_score: Overall weighted composite score
        prediction: ML prediction result (if available)
        key_factors: Ranked list of most important matchup factors
        team1_game_plan: Strategic recommendations for team1
        team2_game_plan: Strategic recommendations for team2
        x_factors: Potential swing factors in the matchup
        overall_assessment: Summary assessment with reasoning
    """

    team1: str
    team2: str
    season: int
    tournament_context: bool
    weights: MatchupWeights

    # Individual dimension scores
    efficiency_score: DimensionScore
    four_factors_score: DimensionScore
    tempo_score: DimensionScore
    point_distribution_score: DimensionScore
    defensive_score: DimensionScore
    size_score: DimensionScore
    experience_score: DimensionScore

    # Composite and prediction
    composite_score: float
    prediction: PredictionResult | None

    # Strategic synthesis
    key_factors: list[tuple[str, float]]
    team1_game_plan: list[str]
    team2_game_plan: list[str]
    x_factors: list[str]
    overall_assessment: str

    # Raw analyzer outputs (for detailed inspection)
    raw_four_factors: object = field(default=None, repr=False)
    raw_tempo: PaceMatchupAnalysis | None = field(default=None, repr=False)
    raw_point_distribution: ScoringStyleMatchup | None = field(default=None, repr=False)
    raw_defensive: DefensiveMatchup | None = field(default=None, repr=False)
    raw_size: SizeMatchup | None = field(default=None, repr=False)
    raw_experience: ExperienceMatchup | None = field(default=None, repr=False)

    def generate_text_report(self, detailed: bool = True) -> str:
        """Generate comprehensive text-based matchup report.

        Args:
            detailed: Include detailed breakdowns for each dimension

        Returns:
            Formatted text report suitable for LLM/MCP consumption
        """
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append(
            f"{self.team1} vs {self.team2} - COMPREHENSIVE MATCHUP ANALYSIS"
        )
        lines.append("=" * 80)
        lines.append("")

        # Context
        context = "NCAA Tournament" if self.tournament_context else "Regular Season"
        lines.append(f"Season: {self.season} ({context})")
        lines.append("")

        # Overall prediction
        lines.append("OVERALL MATCHUP ASSESSMENT")
        lines.append("-" * 40)

        if self.composite_score > 0:
            favorite = self.team1
            margin = abs(self.composite_score)
        else:
            favorite = self.team2
            margin = abs(self.composite_score)

        lines.append(
            f"Projected Advantage: {favorite} +{margin:.1f} "
            f"(Composite Score: {self.composite_score:+.2f})"
        )

        if self.prediction:
            lines.append(
                f"ML Prediction: {self.prediction.team1_name if self.prediction.predicted_margin > 0 else self.prediction.team2_name} "
                f"by {abs(self.prediction.predicted_margin):.1f}"
            )
            lines.append(
                f"Win Probability: {self.team1} {self.prediction.team1_win_prob:.1%} | "
                f"{self.team2} {(1 - self.prediction.team1_win_prob):.1%}"
            )
            lines.append(f"Confidence Interval: {self.prediction.confidence_interval}")

        lines.append("")
        lines.append(self.overall_assessment)
        lines.append("")

        # Key factors
        lines.append("KEY MATCHUP FACTORS (Ranked by Impact)")
        lines.append("-" * 40)
        for i, (factor, impact) in enumerate(self.key_factors[:5], 1):
            lines.append(f"{i}. {factor} (Impact: {impact:+.2f})")
        lines.append("")

        # Dimension scores
        if detailed:
            lines.append("DIMENSIONAL BREAKDOWN")
            lines.append("-" * 40)
            dimensions = [
                ("Efficiency & Ratings", self.efficiency_score, self.weights.efficiency),
                ("Four Factors", self.four_factors_score, self.weights.four_factors),
                ("Tempo & Pace", self.tempo_score, self.weights.tempo),
                (
                    "Point Distribution",
                    self.point_distribution_score,
                    self.weights.point_distribution,
                ),
                ("Defensive Schemes", self.defensive_score, self.weights.defensive),
                ("Size & Athleticism", self.size_score, self.weights.size),
                (
                    "Experience & Chemistry",
                    self.experience_score,
                    self.weights.experience,
                ),
            ]

            for name, score, weight in dimensions:
                weighted = score.score * weight * 10
                lines.append(
                    f"{name:25s} | Score: {score.score:+5.1f} | "
                    f"Weight: {weight:4.0%} | Weighted: {weighted:+5.1f}"
                )
                lines.append(f"  → {score.key_insight}")
                if score.details:
                    lines.append(f"     {score.details}")
            lines.append("")

        # Strategic game plans
        lines.append(f"STRATEGIC GAME PLAN: {self.team1}")
        lines.append("-" * 40)
        for i, rec in enumerate(self.team1_game_plan, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

        lines.append(f"STRATEGIC GAME PLAN: {self.team2}")
        lines.append("-" * 40)
        for i, rec in enumerate(self.team2_game_plan, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

        # X-Factors
        if self.x_factors:
            lines.append("X-FACTORS & SWING ELEMENTS")
            lines.append("-" * 40)
            for factor in self.x_factors:
                lines.append(f"• {factor}")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)


class ComprehensiveMatchupAnalyzer:
    """Unified analyzer combining all 7 matchup analysis dimensions.

    This class orchestrates comprehensive matchup analysis by integrating:
    1. Efficiency & Ratings (from analysis.py)
    2. Tempo & Pace (TempoMatchupAnalyzer)
    3. Four Factors (FourFactorsMatchup)
    4. Point Distribution (PointDistributionAnalyzer)
    5. Defensive Schemes (DefensiveAnalyzer)
    6. Size & Athleticism (SizeAthleticismAnalyzer)
    7. Experience & Chemistry (ExperienceChemistryAnalyzer)

    Key capabilities:
    - Multi-dimensional matchup analysis
    - Weighted composite scoring
    - Strategic synthesis and recommendations
    - ML-enhanced predictions
    - Configurable weights for different contexts
    """

    def __init__(
        self,
        api: KenPomAPI | None = None,
        predictor: GamePredictor | None = None,
    ):
        """Initialize comprehensive matchup analyzer.

        Args:
            api: Optional KenPomAPI instance
            predictor: Optional trained GamePredictor instance
        """
        self.api = api or KenPomAPI()
        self.predictor = predictor

        # Initialize all 7 analyzers
        self.four_factors = FourFactorsMatchup(self.api)
        self.tempo = TempoMatchupAnalyzer(self.api)
        self.point_dist = PointDistributionAnalyzer(self.api)
        self.defensive = DefensiveAnalyzer(self.api)
        self.size = SizeAthleticismAnalyzer(self.api)
        self.experience = ExperienceChemistryAnalyzer(self.api)

    def analyze_matchup(
        self,
        team1: str,
        team2: str,
        season: int,
        tournament_context: bool = False,
        custom_weights: MatchupWeights | None = None,
        neutral_site: bool = True,
    ) -> ComprehensiveMatchupReport:
        """Perform comprehensive matchup analysis across all dimensions.

        Args:
            team1: First team name
            team2: Second team name
            season: Season year (e.g., 2025)
            tournament_context: Use tournament weight profile
            custom_weights: Optional custom weight configuration
            neutral_site: Whether game is at neutral site (for prediction)

        Returns:
            ComprehensiveMatchupReport with full analysis

        Raises:
            ValueError: If team not found
        """
        # Determine weights
        if custom_weights:
            weights = custom_weights
        elif tournament_context:
            weights = MatchupWeights.tournament_weights()
        else:
            weights = MatchupWeights()

        # Run all analyzers
        basic_analysis = analyze_matchup(
            team1, team2, season, neutral_site=neutral_site
        )
        ff_analysis = self.four_factors.analyze_matchup(team1, team2, season)
        tempo_analysis = self.tempo.analyze_matchup(team1, team2, season)
        pd_analysis = self.point_dist.analyze_matchup(team1, team2, season)
        def_analysis = self.defensive.analyze_matchup(team1, team2, season)
        size_analysis = self.size.analyze_matchup(team1, team2, season)
        exp_analysis = self.experience.analyze_matchup(team1, team2, season)

        # Generate dimension scores
        efficiency_score = self._score_efficiency(basic_analysis, team1, team2)
        four_factors_score = self._score_four_factors(ff_analysis, team1, team2)
        tempo_score = self._score_tempo(tempo_analysis, team1, team2)
        point_dist_score = self._score_point_distribution(pd_analysis, team1, team2)
        defensive_score = self._score_defensive(def_analysis, team1, team2)
        size_score = self._score_size(size_analysis, team1, team2)
        experience_score = self._score_experience(exp_analysis, team1, team2)

        # Calculate composite score
        composite = self._calculate_composite_score(
            [
                (efficiency_score, weights.efficiency),
                (four_factors_score, weights.four_factors),
                (tempo_score, weights.tempo),
                (point_dist_score, weights.point_distribution),
                (defensive_score, weights.defensive),
                (size_score, weights.size),
                (experience_score, weights.experience),
            ]
        )

        # Get ML prediction if predictor available
        prediction = None
        if self.predictor:
            try:
                team1_stats = basic_analysis.team1_efficiency.__dict__
                team2_stats = basic_analysis.team2_efficiency.__dict__
                prediction = self.predictor.predict_with_confidence(
                    team1_stats, team2_stats, neutral_site=neutral_site
                )
            except Exception:
                # Predictor not fitted or other issue
                pass

        # Strategic synthesis
        key_factors = self._rank_key_factors(
            [
                efficiency_score,
                four_factors_score,
                tempo_score,
                point_dist_score,
                defensive_score,
                size_score,
                experience_score,
            ]
        )

        team1_plan = self._generate_game_plan(
            team1,
            team2,
            efficiency_score,
            ff_analysis,
            tempo_analysis,
            pd_analysis,
            def_analysis,
            size_analysis,
        )

        team2_plan = self._generate_game_plan(
            team2,
            team1,
            DimensionScore(
                "efficiency", -efficiency_score.score, efficiency_score.confidence, ""
            ),
            ff_analysis,
            tempo_analysis,
            pd_analysis,
            def_analysis,
            size_analysis,
        )

        x_factors = self._identify_x_factors(
            tempo_analysis, size_analysis, def_analysis, pd_analysis
        )

        overall = self._generate_overall_assessment(
            team1, team2, composite, key_factors, prediction
        )

        return ComprehensiveMatchupReport(
            team1=team1,
            team2=team2,
            season=season,
            tournament_context=tournament_context,
            weights=weights,
            efficiency_score=efficiency_score,
            four_factors_score=four_factors_score,
            tempo_score=tempo_score,
            point_distribution_score=point_dist_score,
            defensive_score=defensive_score,
            size_score=size_score,
            experience_score=experience_score,
            composite_score=composite,
            prediction=prediction,
            key_factors=key_factors,
            team1_game_plan=team1_plan,
            team2_game_plan=team2_plan,
            x_factors=x_factors,
            overall_assessment=overall,
            raw_four_factors=ff_analysis,
            raw_tempo=tempo_analysis,
            raw_point_distribution=pd_analysis,
            raw_defensive=def_analysis,
            raw_size=size_analysis,
            raw_experience=exp_analysis,
        )

    def _score_efficiency(
        self, analysis: object, team1: str, team2: str
    ) -> DimensionScore:
        """Convert efficiency analysis to normalized score."""
        # Extract AdjEM difference
        team1_em = analysis.team1_efficiency.adj_em
        team2_em = analysis.team2_efficiency.adj_em
        em_diff = team1_em - team2_em

        # Normalize to -10 to +10 scale (±20 AdjEM = ±10 score)
        score = max(-10.0, min(10.0, em_diff / 2.0))

        winner = team1 if em_diff > 0 else team2
        insight = (
            f"{winner} efficiency advantage "
            f"({abs(em_diff):.1f} AdjEM differential)"
        )

        return DimensionScore(
            dimension="Efficiency & Ratings",
            score=round(score, 2),
            confidence=0.9,  # Efficiency is highly predictive
            key_insight=insight,
            details=f"{team1}: {team1_em:.1f} AdjEM | {team2}: {team2_em:.1f} AdjEM",
        )

    def _score_four_factors(
        self, analysis: object, team1: str, team2: str
    ) -> DimensionScore:
        """Convert Four Factors analysis to normalized score."""
        # Use overall advantage
        advantage = analysis.overall_advantage

        # Extract advantage magnitude from string (e.g., "Duke +2.5")
        if "+" in advantage:
            parts = advantage.split("+")
            adv_team = parts[0].strip()
            magnitude = float(parts[1].strip())

            if adv_team == team1:
                score = magnitude
            else:
                score = -magnitude
        else:
            score = 0.0

        # Normalize to -10 to +10 scale
        score = max(-10.0, min(10.0, score))

        insight = f"Four Factors: {analysis.overall_advantage}"

        return DimensionScore(
            dimension="Four Factors",
            score=round(score, 2),
            confidence=0.85,
            key_insight=insight,
            details=f"Most important: {analysis.most_important_factor}",
        )

    def _score_tempo(
        self, analysis: PaceMatchupAnalysis, team1: str, team2: str
    ) -> DimensionScore:
        """Convert tempo analysis to normalized score."""
        # Use tempo control factor (-1 to +1, positive = team1 controls)
        control = analysis.tempo_control_factor

        # Scale to -10 to +10
        score = control * 10.0

        controller = team1 if control > 0 else team2
        insight = f"{controller} controls tempo (expected: {analysis.expected_tempo:.1f} poss)"

        return DimensionScore(
            dimension="Tempo & Pace",
            score=round(score, 2),
            confidence=0.7,  # Tempo less predictive than efficiency
            key_insight=insight,
            details=f"Style mismatch: {analysis.style_mismatch_score:.1f}/10",
        )

    def _score_point_distribution(
        self, analysis: ScoringStyleMatchup, team1: str, team2: str
    ) -> DimensionScore:
        """Convert point distribution analysis to normalized score."""
        # Use style mismatch score (0-10)
        # Positive mismatch favors team with better matchup
        mismatch = analysis.style_mismatch_score

        # Determine which team has better matchup based on exploitable areas
        team1_exploits = len(
            [e for e in analysis.team1_exploitable_areas if "No clear" not in e]
        )
        team2_exploits = len(
            [e for e in analysis.team2_exploitable_areas if "No clear" not in e]
        )

        if team1_exploits > team2_exploits:
            score = mismatch
        elif team2_exploits > team1_exploits:
            score = -mismatch
        else:
            score = 0.0

        insight = analysis.key_matchup_factor

        return DimensionScore(
            dimension="Point Distribution",
            score=round(score, 2),
            confidence=0.75,
            key_insight=insight,
            details=analysis.recommended_strategy,
        )

    def _score_defensive(
        self, analysis: DefensiveMatchup, team1: str, team2: str
    ) -> DimensionScore:
        """Convert defensive analysis to normalized score."""
        # Use defensive advantage score (0-10, 5 = neutral)
        adv_score = analysis.defensive_advantage_score

        # Convert to -10 to +10 scale (5 = 0)
        score = (adv_score - 5.0) * 2.0

        insight = f"{analysis.better_defense} defensive advantage"

        return DimensionScore(
            dimension="Defensive Schemes",
            score=round(score, 2),
            confidence=0.8,
            key_insight=insight,
            details=analysis.matchup_recommendation,
        )

    def _score_size(
        self, analysis: SizeMatchup, team1: str, team2: str
    ) -> DimensionScore:
        """Convert size analysis to normalized score."""
        # Use size advantage score (0-10, 5 = neutral)
        adv_score = analysis.size_advantage_score

        # Convert to -10 to +10 scale
        score = (adv_score - 5.0) * 2.0

        insight = f"{analysis.better_size_team} size advantage"

        return DimensionScore(
            dimension="Size & Athleticism",
            score=round(score, 2),
            confidence=0.7,
            key_insight=insight,
            details=analysis.rebounding_prediction,
        )

    def _score_experience(
        self, analysis: ExperienceMatchup, team1: str, team2: str
    ) -> DimensionScore:
        """Convert experience analysis to normalized score."""
        # Use experience advantage score (0-10, 5 = neutral)
        adv_score = analysis.experience_advantage_score

        # Convert to -10 to +10 scale
        score = (adv_score - 5.0) * 2.0

        insight = f"{analysis.more_experienced_team} experience edge"

        return DimensionScore(
            dimension="Experience & Chemistry",
            score=round(score, 2),
            confidence=0.65,  # Experience matters more in tournament
            key_insight=insight,
            details=f"Continuity: {analysis.continuity_advantage}",
        )

    def _calculate_composite_score(
        self, dimension_weights: list[tuple[DimensionScore, float]]
    ) -> float:
        """Calculate weighted composite score across all dimensions.

        Args:
            dimension_weights: List of (DimensionScore, weight) tuples

        Returns:
            Composite score from -10 to +10
        """
        total = 0.0
        for dimension, weight in dimension_weights:
            total += dimension.score * weight

        # Scale by 10 to get -10 to +10 range
        return round(total * 10.0, 2)

    def _rank_key_factors(
        self, dimensions: list[DimensionScore]
    ) -> list[tuple[str, float]]:
        """Rank matchup factors by absolute impact.

        Args:
            dimensions: List of dimension scores

        Returns:
            List of (factor_name, impact_score) sorted by impact
        """
        factors = [
            (f"{dim.dimension}: {dim.key_insight}", abs(dim.score)) for dim in dimensions
        ]

        # Sort by absolute impact
        factors.sort(key=lambda x: x[1], reverse=True)

        return factors

    def _generate_game_plan(
        self,
        team: str,
        opponent: str,
        efficiency: DimensionScore,
        four_factors: object,
        tempo: PaceMatchupAnalysis,
        point_dist: ScoringStyleMatchup,
        defense: DefensiveMatchup,
        size: SizeMatchup,
    ) -> list[str]:
        """Generate strategic game plan recommendations for a team.

        Args:
            team: Team to generate plan for
            opponent: Opponent team
            efficiency: Efficiency dimension score
            four_factors: Four Factors analysis
            tempo: Tempo analysis
            point_dist: Point distribution analysis
            defense: Defensive analysis
            size: Size analysis

        Returns:
            List of strategic recommendations
        """
        plan = []

        # Four Factors-based strategies
        if hasattr(four_factors, "team1_advantages"):
            # Get team's advantages
            is_team1 = team == four_factors.team1_profile.team_name
            advantages = (
                four_factors.team1_advantages
                if is_team1
                else four_factors.team2_advantages
            )

            for adv in advantages[:2]:  # Top 2 advantages
                plan.append(f"Exploit {adv.factor.value} advantage")

        # Tempo strategy
        if tempo.faster_pace_benefits == team:
            plan.append(f"Push tempo to {tempo.expected_tempo + 3:.0f}+ possessions")
        elif tempo.slower_pace_benefits == team:
            plan.append(f"Slow tempo to {tempo.expected_tempo - 3:.0f} possessions")

        # Point distribution strategy
        if point_dist.recommended_strategy.startswith(team):
            strategy = point_dist.recommended_strategy.replace(f"{team} should ", "")
            plan.append(strategy.capitalize())

        # Defensive keys
        defense_keys = (
            defense.team1_defensive_keys
            if team == defense.team1_defense.team_name
            else defense.team2_defensive_keys
        )
        if defense_keys and "Play fundamental defense" not in defense_keys[0]:
            plan.append(defense_keys[0])

        # Size strategy
        if size.better_size_team == team:
            plan.append(size.strategic_recommendation)

        return plan[:5]  # Return top 5 recommendations

    def _identify_x_factors(
        self,
        tempo: PaceMatchupAnalysis,
        size: SizeMatchup,
        defense: DefensiveMatchup,
        point_dist: ScoringStyleMatchup,
    ) -> list[str]:
        """Identify potential swing factors in the matchup.

        Args:
            tempo: Tempo analysis
            size: Size analysis
            defense: Defensive analysis
            point_dist: Point distribution analysis

        Returns:
            List of X-factors that could swing the game
        """
        x_factors = []

        # Tempo variance
        if tempo.style_mismatch_score > 7.0:
            x_factors.append(
                f"Tempo control (extreme style clash: {tempo.style_mismatch_score:.1f}/10)"
            )

        # Rebounding battle
        if size.better_size_team != "neutral":
            x_factors.append(f"Rebounding battle ({size.rebounding_prediction})")

        # Three-point variance
        if abs(point_dist.three_point_advantage) > 5.0:
            x_factors.append(
                f"Three-point shooting variance (high volatility matchup)"
            )

        # Turnover margin
        if defense.pressure_defense_advantage:
            x_factors.append(
                f"Turnover margin ({defense.pressure_defense_advantage} applies pressure)"
            )

        return x_factors

    def _generate_overall_assessment(
        self,
        team1: str,
        team2: str,
        composite: float,
        key_factors: list[tuple[str, float]],
        prediction: PredictionResult | None,
    ) -> str:
        """Generate overall matchup assessment with reasoning.

        Args:
            team1: First team name
            team2: Second team name
            composite: Composite score
            key_factors: Ranked key factors
            prediction: ML prediction (if available)

        Returns:
            Overall assessment string
        """
        if composite > 2.0:
            favorite = team1
            margin = composite
            level = "significant"
        elif composite < -2.0:
            favorite = team2
            margin = abs(composite)
            level = "significant"
        elif composite > 0:
            favorite = team1
            margin = composite
            level = "slight"
        elif composite < 0:
            favorite = team2
            margin = abs(composite)
            level = "slight"
        else:
            return f"Evenly matched game. {key_factors[0][0]} could be decisive."

        assessment = f"{favorite} has a {level} advantage ({margin:.1f} composite score). "

        # Add top 2 reasons
        reasons = [f.split(":")[0] for f, _ in key_factors[:2]]
        assessment += f"Key advantages: {reasons[0]} and {reasons[1]}. "

        # Add prediction validation
        if prediction:
            pred_favorite = (
                prediction.team1_name
                if prediction.predicted_margin > 0
                else prediction.team2_name
            )
            if pred_favorite == favorite:
                assessment += "ML model agrees with composite analysis."
            else:
                assessment += (
                    f"Note: ML model favors {pred_favorite} "
                    f"(model sees factors composite may underweight)."
                )

        return assessment
