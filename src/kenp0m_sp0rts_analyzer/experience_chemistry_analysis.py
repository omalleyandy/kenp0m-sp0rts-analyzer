"""Experience and chemistry analysis for basketball games.

This module analyzes intangible factors including team experience, bench depth,
and continuity. It predicts late-game execution, tournament readiness, and
identifies teams with veteran advantages or chemistry benefits.

Experience Classifications:
- Very Experienced: >2.5 rating (mostly juniors/seniors)
- Experienced: >2.0 rating
- Average: 1.5-2.0 rating
- Young: <1.5 rating (mostly freshmen/sophomores)

Example:
    ```python
    from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
    from kenp0m_sp0rts_analyzer.experience_chemistry_analysis import (
        ExperienceChemistryAnalyzer
    )

    api = KenPomAPI()
    analyzer = ExperienceChemistryAnalyzer(api)

    # Analyze Duke's experience profile
    profile = analyzer.get_experience_profile("Duke", 2025)
    print(f"Experience: {profile.experience_level}")
    print(f"Continuity: {profile.continuity:.1f}%")

    # Assess tournament readiness
    readiness = analyzer.assess_tournament_readiness("Duke", 2025)
    print(f"Readiness: {readiness.readiness_tier}")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .api_client import KenPomAPI
from .utils import normalize_team_name


@dataclass
class ExperienceProfile:
    """Team's experience and chemistry profile.

    Attributes:
        team_name: Name of the team
        season: Season year (e.g., 2025)
        experience_rating: Experience rating (0-4 scale: Fr=0, Sr=3, 5th=4)
        experience_rank: National rank for experience
        experience_level: Experience classification
        bench_strength: Bench depth rating
        bench_rank: National rank for bench strength
        bench_classification: Bench depth classification
        continuity: Percentage of minutes returning
        continuity_rank: National rank for continuity
        continuity_level: Continuity classification
        intangibles_score: Composite score (0-10)
        primary_strength: Main intangible advantage
        primary_weakness: Main intangible weakness
    """

    team_name: str
    season: int

    # Experience
    experience_rating: float
    experience_rank: int
    experience_level: Literal[
        "very_experienced", "experienced", "average", "young", "very_young"
    ]

    # Bench depth
    bench_strength: float
    bench_rank: int
    bench_classification: Literal[
        "elite_depth", "good_depth", "average_depth", "thin_bench"
    ]

    # Team continuity
    continuity: float
    continuity_rank: int
    continuity_level: Literal[
        "high_continuity", "moderate_continuity", "low_continuity", "rebuild"
    ]

    # Overall intangibles score
    intangibles_score: float
    primary_strength: str
    primary_weakness: str


@dataclass
class ExperienceMatchup:
    """Experience and chemistry matchup analysis.

    Attributes:
        team1_profile: Experience profile for team 1
        team2_profile: Experience profile for team 2
        experience_advantage: Which team has experience edge
        experience_gap: Experience rating difference
        experience_impact: Expected impact description
        bench_advantage: Which team has deeper bench
        bench_gap: Bench strength difference
        bench_impact: Expected bench impact
        continuity_advantage: Which team has more continuity
        continuity_gap: Continuity percentage difference
        continuity_impact: Expected continuity impact
        better_intangibles: Team with better overall intangibles
        intangibles_advantage_score: Advantage score (0-10, 5 = neutral)
        late_game_execution: Late-game edge prediction
        tournament_readiness: Tournament preparation comparison
        adverse_conditions: Adversity handling prediction
        team1_keys: Strategic keys for team 1
        team2_keys: Strategic keys for team 2
        matchup_recommendation: Overall recommendation
    """

    team1_profile: ExperienceProfile
    team2_profile: ExperienceProfile

    # Experience advantage
    experience_advantage: str
    experience_gap: float
    experience_impact: str

    # Bench depth advantage
    bench_advantage: str
    bench_gap: float
    bench_impact: str

    # Continuity advantage
    continuity_advantage: str
    continuity_gap: float
    continuity_impact: str

    # Overall intangibles
    better_intangibles: str
    intangibles_advantage_score: float

    # Situational predictions
    late_game_execution: str
    tournament_readiness: str
    adverse_conditions: str

    # Strategic insights
    team1_keys: list[str]
    team2_keys: list[str]
    matchup_recommendation: str


@dataclass
class TournamentReadiness:
    """Assess tournament readiness based on experience factors.

    Attributes:
        team_name: Name of the team
        season: Season year
        experience_score: How experienced for tournament (0-10)
        late_game_poise: Close game execution ability (0-10)
        depth_for_neutral_site: Bench ready for tournament grind (0-10)
        tournament_readiness_score: Overall readiness (0-10)
        readiness_tier: Tier classification
        biggest_concern: Main concern for tournament
    """

    team_name: str
    season: int

    # Tournament-critical factors
    experience_score: float
    late_game_poise: float
    depth_for_neutral_site: float

    # Overall readiness
    tournament_readiness_score: float
    readiness_tier: Literal["elite", "strong", "average", "questionable"]
    biggest_concern: str


class ExperienceChemistryAnalyzer:
    """Analyze experience, bench depth, and team chemistry.

    This class provides comprehensive intangibles analysis for basketball
    matchups, identifying experience advantages, depth benefits, and chemistry
    factors that affect performance.

    Experience classifications (0-4 scale):
    - Very Experienced: >2.5 (mostly juniors/seniors)
    - Experienced: >2.0
    - Average: 1.5-2.0
    - Young: <1.5 (mostly freshmen/sophomores)

    Continuity classifications (percentage):
    - High Continuity: >70% minutes returning
    - Moderate Continuity: >50% minutes returning
    - Low Continuity: >30% minutes returning
    - Rebuild: <30% minutes returning

    Key capabilities:
    - Experience profile generation
    - Bench depth analysis
    - Continuity assessment
    - Tournament readiness evaluation
    - Late-game execution predictions
    """

    # Experience thresholds (0-4 scale)
    VERY_EXPERIENCED = 2.5
    EXPERIENCED = 2.0
    YOUNG = 1.5
    VERY_YOUNG = 1.0

    # Continuity thresholds (percentage)
    HIGH_CONTINUITY = 70.0
    MODERATE_CONTINUITY = 50.0
    LOW_CONTINUITY = 30.0

    # Bench strength thresholds
    ELITE_DEPTH = 3.0
    GOOD_DEPTH = 1.0
    AVERAGE_DEPTH = -1.0

    def __init__(self, api: KenPomAPI | None = None):
        """Initialize Experience & Chemistry analyzer.

        Args:
            api: Optional KenPomAPI instance. Creates one if not provided.
        """
        self.api = api or KenPomAPI()

    def get_experience_profile(self, team: str, season: int) -> ExperienceProfile:
        """Generate experience and chemistry profile.

        Args:
            team: Team name
            season: Season year (e.g., 2025)

        Returns:
            ExperienceProfile with complete intangibles breakdown

        Raises:
            ValueError: If team not found
        """
        # Normalize team name for consistent lookup
        team = normalize_team_name(team)

        data = self.api.get_height(year=season)
        team_data = next((t for t in data.data if t["TeamName"] == team), None)

        if not team_data:
            raise ValueError(f"Team '{team}' not found in {season} season")

        # Classify experience level
        exp = team_data["Exp"]
        exp_level: Literal[
            "very_experienced", "experienced", "average", "young", "very_young"
        ]
        if exp >= self.VERY_EXPERIENCED:
            exp_level = "very_experienced"
        elif exp >= self.EXPERIENCED:
            exp_level = "experienced"
        elif exp >= self.YOUNG:
            exp_level = "average"
        elif exp >= self.VERY_YOUNG:
            exp_level = "young"
        else:
            exp_level = "very_young"

        # Classify continuity
        continuity = team_data["Continuity"]
        cont_level: Literal[
            "high_continuity", "moderate_continuity", "low_continuity", "rebuild"
        ]
        if continuity >= self.HIGH_CONTINUITY:
            cont_level = "high_continuity"
        elif continuity >= self.MODERATE_CONTINUITY:
            cont_level = "moderate_continuity"
        elif continuity >= self.LOW_CONTINUITY:
            cont_level = "low_continuity"
        else:
            cont_level = "rebuild"

        # Classify bench depth
        bench = team_data["Bench"]
        bench_class: Literal["elite_depth", "good_depth", "average_depth", "thin_bench"]
        if bench >= self.ELITE_DEPTH:
            bench_class = "elite_depth"
        elif bench >= self.GOOD_DEPTH:
            bench_class = "good_depth"
        elif bench >= self.AVERAGE_DEPTH:
            bench_class = "average_depth"
        else:
            bench_class = "thin_bench"

        # Calculate composite intangibles score (0-10 scale)
        exp_score = (exp / 4.0) * 10.0  # Max 10 points
        cont_score = (continuity / 100.0) * 10.0  # Max 10 points
        bench_score = ((bench + 5.0) / 10.0) * 10.0  # Normalize (-5 to +5 → 0 to 10)

        intangibles = exp_score * 0.4 + cont_score * 0.3 + bench_score * 0.3

        # Identify strength/weakness
        scores = [
            (exp_score, "Experience", "Inexperience"),
            (cont_score, "Team chemistry/continuity", "Lack of continuity"),
            (bench_score, "Bench depth", "Thin bench"),
        ]
        strength = max(scores, key=lambda x: x[0])[1]
        weakness = min(scores, key=lambda x: x[0])[2]

        return ExperienceProfile(
            team_name=team,
            season=season,
            experience_rating=exp,
            experience_rank=team_data["ExpRank"],
            experience_level=exp_level,
            bench_strength=bench,
            bench_rank=team_data["BenchRank"],
            bench_classification=bench_class,
            continuity=continuity,
            continuity_rank=team_data["RankContinuity"],
            continuity_level=cont_level,
            intangibles_score=round(intangibles, 2),
            primary_strength=strength,
            primary_weakness=weakness,
        )

    def analyze_matchup(self, team1: str, team2: str, season: int) -> ExperienceMatchup:
        """Analyze experience and chemistry matchup.

        Args:
            team1: First team name
            team2: Second team name
            season: Season year (e.g., 2025)

        Returns:
            ExperienceMatchup with complete intangibles analysis

        Raises:
            ValueError: If team not found
        """
        profile1 = self.get_experience_profile(team1, season)
        profile2 = self.get_experience_profile(team2, season)

        # Experience advantage
        exp_gap = profile1.experience_rating - profile2.experience_rating
        exp_adv = team1 if exp_gap > 0.3 else (team2 if exp_gap < -0.3 else "neutral")
        exp_impact = self._predict_experience_impact(exp_gap, team1, team2)

        # Bench advantage
        bench_gap = profile1.bench_strength - profile2.bench_strength
        bench_adv = (
            team1 if bench_gap > 1.0 else (team2 if bench_gap < -1.0 else "neutral")
        )
        bench_impact = self._predict_bench_impact(bench_gap, team1, team2)

        # Continuity advantage
        cont_gap = profile1.continuity - profile2.continuity
        cont_adv = (
            team1 if cont_gap > 15.0 else (team2 if cont_gap < -15.0 else "neutral")
        )
        cont_impact = self._predict_continuity_impact(cont_gap, team1, team2)

        # Overall intangibles
        intangibles_score = self._calculate_intangibles_advantage(profile1, profile2)
        better_intangibles = (
            team1
            if intangibles_score > 5.5
            else (team2 if intangibles_score < 4.5 else "neutral")
        )

        # Situational predictions
        late_game = self._predict_late_game_execution(profile1, profile2, team1, team2)
        tournament = self._predict_tournament_readiness(
            profile1, profile2, team1, team2
        )
        adversity = self._predict_adversity_handling(profile1, profile2, team1, team2)

        # Strategic keys
        team1_keys = self._generate_experience_keys(profile1, profile2)
        team2_keys = self._generate_experience_keys(profile2, profile1)
        recommendation = self._generate_matchup_recommendation(
            profile1, profile2, intangibles_score
        )

        return ExperienceMatchup(
            team1_profile=profile1,
            team2_profile=profile2,
            experience_advantage=exp_adv,
            experience_gap=round(exp_gap, 2),
            experience_impact=exp_impact,
            bench_advantage=bench_adv,
            bench_gap=round(bench_gap, 2),
            bench_impact=bench_impact,
            continuity_advantage=cont_adv,
            continuity_gap=round(cont_gap, 2),
            continuity_impact=cont_impact,
            better_intangibles=better_intangibles,
            intangibles_advantage_score=round(intangibles_score, 2),
            late_game_execution=late_game,
            tournament_readiness=tournament,
            adverse_conditions=adversity,
            team1_keys=team1_keys,
            team2_keys=team2_keys,
            matchup_recommendation=recommendation,
        )

    def assess_tournament_readiness(
        self, team: str, season: int
    ) -> TournamentReadiness:
        """Assess team's readiness for NCAA Tournament.

        Tournament success correlates with:
        1. Experience (veterans handle pressure better)
        2. Bench depth (6 games in 3 weeks)
        3. Late-game poise (tight tournament games)

        Args:
            team: Team name
            season: Season year

        Returns:
            TournamentReadiness assessment
        """
        profile = self.get_experience_profile(team, season)

        # Experience score (0-10)
        # Very experienced teams (~3.0+) excel in March
        exp_score = min(profile.experience_rating / 3.0 * 10.0, 10.0)

        # Late-game poise (experience × continuity)
        # Teams with experienced returners handle late-game situations better
        late_game = exp_score * 0.6 + (profile.continuity / 100.0) * 10.0 * 0.4

        # Depth for neutral site (bench strength matters in tournament)
        depth = ((profile.bench_strength + 5.0) / 10.0) * 10.0

        # Overall readiness (weighted composite)
        readiness = exp_score * 0.4 + late_game * 0.3 + depth * 0.3

        # Classify readiness tier
        tier: Literal["elite", "strong", "average", "questionable"]
        if readiness >= 8.0:
            tier = "elite"
            concern = "None - well-prepared for tournament run"
        elif readiness >= 6.5:
            tier = "strong"
            concern = profile.primary_weakness
        elif readiness >= 5.0:
            tier = "average"
            concern = f"{profile.primary_weakness} - could struggle in close games"
        else:
            tier = "questionable"
            concern = (
                f"Major concerns: {profile.primary_weakness}, limited March experience"
            )

        return TournamentReadiness(
            team_name=team,
            season=season,
            experience_score=round(exp_score, 2),
            late_game_poise=round(late_game, 2),
            depth_for_neutral_site=round(depth, 2),
            tournament_readiness_score=round(readiness, 2),
            readiness_tier=tier,
            biggest_concern=concern,
        )

    def _predict_experience_impact(self, gap: float, team1: str, team2: str) -> str:
        """Predict impact of experience gap."""
        if abs(gap) < 0.3:
            return "Experience is relatively even"

        more_exp = team1 if gap > 0 else team2
        if abs(gap) > 1.0:
            return f"{more_exp} has major experience advantage - expect better late-game execution"
        else:
            return f"{more_exp} has experience edge - slight advantage in close games"

    def _predict_bench_impact(self, gap: float, team1: str, team2: str) -> str:
        """Predict impact of bench depth gap."""
        if abs(gap) < 1.0:
            return "Bench depth is similar"

        deeper = team1 if gap > 0 else team2
        if abs(gap) > 2.5:
            return f"{deeper} has significant depth advantage - can sustain performance over 40 minutes"
        else:
            return f"{deeper} has bench edge - advantage in foul trouble or fatigue situations"

    def _predict_continuity_impact(self, gap: float, team1: str, team2: str) -> str:
        """Predict impact of continuity gap."""
        if abs(gap) < 15.0:
            return "Team continuity is comparable"

        more_cont = team1 if gap > 0 else team2
        if abs(gap) > 30.0:
            return f"{more_cont} has major continuity advantage - team chemistry should be superior"
        else:
            return f"{more_cont} has continuity edge - better system familiarity"

    def _calculate_intangibles_advantage(
        self, team1: ExperienceProfile, team2: ExperienceProfile
    ) -> float:
        """Calculate overall intangibles advantage (0-10 scale, 5 = neutral)."""
        # Direct comparison of composite scores
        score = 5.0 + (team1.intangibles_score - team2.intangibles_score) / 2.0
        return max(0.0, min(10.0, score))

    def _predict_late_game_execution(
        self,
        p1: ExperienceProfile,
        p2: ExperienceProfile,
        t1: str,
        t2: str,
    ) -> str:
        """Predict late-game execution edge."""
        # Experience + continuity = late-game poise
        p1_late_game = p1.experience_rating * 0.6 + (p1.continuity / 100.0) * 4.0 * 0.4
        p2_late_game = p2.experience_rating * 0.6 + (p2.continuity / 100.0) * 4.0 * 0.4

        diff = p1_late_game - p2_late_game

        if diff > 0.6:
            return f"{t1} has edge in close games (experience + continuity)"
        elif diff < -0.6:
            return f"{t2} has edge in close games (experience + continuity)"
        else:
            return "Late-game execution should be comparable"

    def _predict_tournament_readiness(
        self,
        p1: ExperienceProfile,
        p2: ExperienceProfile,
        t1: str,
        t2: str,
    ) -> str:
        """Predict which team is better prepared for tournament."""
        t1_readiness = self.assess_tournament_readiness(t1, p1.season)
        t2_readiness = self.assess_tournament_readiness(t2, p2.season)

        diff = (
            t1_readiness.tournament_readiness_score
            - t2_readiness.tournament_readiness_score
        )

        if diff > 1.5:
            return f"{t1} ({t1_readiness.readiness_tier}) more tournament-ready than {t2} ({t2_readiness.readiness_tier})"
        elif diff < -1.5:
            return f"{t2} ({t2_readiness.readiness_tier}) more tournament-ready than {t1} ({t1_readiness.readiness_tier})"
        else:
            return f"Both teams similarly prepared for tournament ({t1_readiness.readiness_tier}/{t2_readiness.readiness_tier})"

    def _predict_adversity_handling(
        self,
        p1: ExperienceProfile,
        p2: ExperienceProfile,
        t1: str,
        t2: str,
    ) -> str:
        """Predict how teams handle adversity (deficits, foul trouble, etc.)."""
        # Experience + bench depth = adversity handling
        p1_adversity = (
            p1.experience_rating * 0.5 + ((p1.bench_strength + 5.0) / 10.0) * 4.0 * 0.5
        )
        p2_adversity = (
            p2.experience_rating * 0.5 + ((p2.bench_strength + 5.0) / 10.0) * 4.0 * 0.5
        )

        diff = p1_adversity - p2_adversity

        if diff > 0.6:
            return f"{t1} better equipped to handle foul trouble, injuries, deficits"
        elif diff < -0.6:
            return f"{t2} better equipped to handle foul trouble, injuries, deficits"
        else:
            return "Both teams similarly resilient to adversity"

    def _generate_experience_keys(
        self, team: ExperienceProfile, opponent: ExperienceProfile
    ) -> list[str]:
        """Generate strategic keys based on experience profile."""
        keys = []

        # Leverage strengths
        if team.experience_level in ["very_experienced", "experienced"]:
            keys.append("Leverage veteran poise in late-game situations")

        if team.bench_classification in ["elite_depth", "good_depth"]:
            keys.append("Use depth advantage - play fast, substitute freely")

        if team.continuity_level == "high_continuity":
            keys.append(
                "Execute complex sets - team chemistry allows sophisticated offense"
            )

        # Exploit opponent weaknesses
        if opponent.experience_level in ["young", "very_young"]:
            keys.append(f"Apply pressure - force {opponent.team_name} into mistakes")

        if opponent.bench_classification == "thin_bench":
            keys.append(f"Attack {opponent.team_name} starters - force foul trouble")

        return (
            keys
            if keys
            else ["Play fundamental basketball - experience edge is minimal"]
        )

    def _generate_matchup_recommendation(
        self,
        team1: ExperienceProfile,
        team2: ExperienceProfile,
        advantage_score: float,
    ) -> str:
        """Generate strategic matchup recommendation."""
        if advantage_score > 6.5:
            better_team = team1.team_name
            margin = "significant"
        elif advantage_score < 3.5:
            better_team = team2.team_name
            margin = "significant"
        else:
            return "Experience and intangibles are evenly matched"

        return (
            f"{better_team} has {margin} intangibles advantage "
            f"(score: {advantage_score:.1f}/10) - expect better execution in crucial moments"
        )
