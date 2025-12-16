"""Point Distribution analysis for basketball games.

This module analyzes scoring styles and identifies matchup advantages based on
how teams score (3pt, 2pt, free throws) and how they defend each scoring method.

The analysis helps identify:
- Offensive scoring style (perimeter, interior, balanced)
- Defensive weaknesses
- Exploitable matchups
- Strategic recommendations

Example:
    ```python
    from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
    from kenp0m_sp0rts_analyzer.point_distribution_analysis import (
        PointDistributionAnalyzer
    )

    api = KenPomAPI()
    analyzer = PointDistributionAnalyzer(api)

    # Analyze Duke's scoring style
    profile = analyzer.get_scoring_profile("Duke", 2025)
    print(f"Style: {profile.style}")
    print(f"Strength: {profile.primary_strength}")

    # Analyze matchup
    matchup = analyzer.analyze_matchup("Duke", "North Carolina", 2025)
    print(f"Strategy: {matchup.recommended_strategy}")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .api_client import KenPomAPI


@dataclass
class ScoringStyleProfile:
    """Team's scoring style breakdown.

    Attributes:
        team_name: Name of the team
        season: Season year (e.g., 2025)
        ft_pct: Percentage of points from free throws
        fg2_pct: Percentage of points from 2-point field goals
        fg3_pct: Percentage of points from 3-point field goals
        def_ft_pct: Percentage of opponent points from free throws
        def_fg2_pct: Percentage of opponent points from 2-point field goals
        def_fg3_pct: Percentage of opponent points from 3-point field goals
        ft_rank: National rank for free throw percentage
        fg2_rank: National rank for 2-point percentage
        fg3_rank: National rank for 3-point percentage
        style: Overall scoring style classification
        primary_strength: Description of team's main offensive strength
        defensive_weakness: Description of team's main defensive weakness
    """

    team_name: str
    season: int

    # Offensive distribution
    ft_pct: float  # % of points from free throws
    fg2_pct: float  # % of points from 2-point FGs
    fg3_pct: float  # % of points from 3-point FGs

    # Defensive distribution (points allowed)
    def_ft_pct: float
    def_fg2_pct: float
    def_fg3_pct: float

    # Rankings
    ft_rank: int
    fg2_rank: int
    fg3_rank: int

    # Style classification
    style: Literal["perimeter", "balanced", "interior"]
    primary_strength: str
    defensive_weakness: str


@dataclass
class ScoringStyleMatchup:
    """Matchup analysis between two scoring styles.

    Attributes:
        team1_profile: Scoring profile for team 1
        team2_profile: Scoring profile for team 2
        three_point_advantage: 3pt advantage (-10 to +10, positive = team1)
        two_point_advantage: 2pt advantage (-10 to +10, positive = team1)
        free_throw_advantage: FT advantage (-10 to +10, positive = team1)
        style_mismatch_score: Overall style mismatch (0-10, higher = better)
        team1_exploitable_areas: Areas where team1 can exploit team2
        team2_exploitable_areas: Areas where team2 can exploit team1
        key_matchup_factor: Most important factor in this matchup
        recommended_strategy: Strategic recommendation for the matchup
    """

    team1_profile: ScoringStyleProfile
    team2_profile: ScoringStyleProfile

    # Advantage scores (-10 to +10, positive favors team1)
    three_point_advantage: float
    two_point_advantage: float
    free_throw_advantage: float

    # Overall scoring style mismatch (0-10 scale)
    style_mismatch_score: float

    # Strategic insights
    team1_exploitable_areas: list[str]
    team2_exploitable_areas: list[str]
    key_matchup_factor: str
    recommended_strategy: str


class PointDistributionAnalyzer:
    """Analyze scoring styles and identify matchup advantages.

    This class provides detailed point distribution analysis for basketball
    matchups, identifying how teams score and where defensive weaknesses exist.

    Style classifications:
    - Perimeter: >40% of points from 3-pointers
    - Interior: >55% of points from 2-pointers
    - Balanced: Neither threshold met

    Key capabilities:
    - Classify team scoring styles
    - Identify defensive weaknesses
    - Find exploitable matchups
    - Generate strategic recommendations
    """

    # Thresholds for style classification
    PERIMETER_THRESHOLD = 40.0  # >40% from 3pt = perimeter team
    INTERIOR_THRESHOLD = 55.0  # >55% from 2pt = interior team

    def __init__(self, api: KenPomAPI | None = None):
        """Initialize Point Distribution analyzer.

        Args:
            api: Optional KenPomAPI instance. Creates one if not provided.
        """
        self.api = api or KenPomAPI()

    def get_scoring_profile(self, team: str, season: int) -> ScoringStyleProfile:
        """Generate scoring style profile for a team.

        Args:
            team: Team name
            season: Season year (e.g., 2025)

        Returns:
            ScoringStyleProfile with complete scoring breakdown

        Raises:
            ValueError: If team not found
        """
        # Fetch data
        data = self.api.get_point_distribution(year=season)
        team_data = next((t for t in data.data if t["TeamName"] == team), None)

        if not team_data:
            raise ValueError(f"Team '{team}' not found in {season} season")

        # Classify style
        fg3_pct = team_data["OffFg3"]
        fg2_pct = team_data["OffFg2"]

        style: Literal["perimeter", "balanced", "interior"]
        if fg3_pct >= self.PERIMETER_THRESHOLD:
            style = "perimeter"
            strength = f"Three-point heavy ({fg3_pct:.1f}% from 3pt)"
        elif fg2_pct >= self.INTERIOR_THRESHOLD:
            style = "interior"
            strength = f"Interior oriented ({fg2_pct:.1f}% from 2pt)"
        else:
            style = "balanced"
            strength = "Balanced scoring attack"

        # Identify defensive weakness
        def_fg3 = team_data["DefFg3"]
        def_fg2 = team_data["DefFg2"]
        def_ft = team_data["DefFt"]

        weaknesses = [
            (def_fg3, "three-point defense"),
            (def_fg2, "interior defense"),
            (def_ft, "free throw prevention"),
        ]
        defensive_weakness = max(weaknesses, key=lambda x: x[0])[1]

        return ScoringStyleProfile(
            team_name=team,
            season=season,
            ft_pct=team_data["OffFt"],
            fg2_pct=fg2_pct,
            fg3_pct=fg3_pct,
            def_ft_pct=def_ft,
            def_fg2_pct=def_fg2,
            def_fg3_pct=def_fg3,
            ft_rank=team_data["RankOffFt"],
            fg2_rank=team_data["RankOffFg2"],
            fg3_rank=team_data["RankOffFg3"],
            style=style,
            primary_strength=strength,
            defensive_weakness=defensive_weakness,
        )

    def analyze_matchup(
        self, team1: str, team2: str, season: int
    ) -> ScoringStyleMatchup:
        """Analyze scoring style matchup between two teams.

        Args:
            team1: First team name
            team2: Second team name
            season: Season year (e.g., 2025)

        Returns:
            ScoringStyleMatchup with complete matchup analysis

        Raises:
            ValueError: If team not found
        """
        profile1 = self.get_scoring_profile(team1, season)
        profile2 = self.get_scoring_profile(team2, season)

        # Calculate advantages (positive = team1 advantage)

        # 3pt advantage: team1 offense vs team2 defense
        three_pt_adv = profile1.fg3_pct - profile2.def_fg3_pct

        # 2pt advantage
        two_pt_adv = profile1.fg2_pct - profile2.def_fg2_pct

        # FT advantage
        ft_adv = profile1.ft_pct - profile2.def_ft_pct

        # Calculate style mismatch score (0-10)
        # Higher = better matchup for exploiting opponent weakness
        mismatch_score = self._calculate_style_mismatch(profile1, profile2)

        # Generate strategic insights
        team1_exploits = self._identify_exploitable_areas(profile1, profile2)
        team2_exploits = self._identify_exploitable_areas(profile2, profile1)

        # Determine key matchup factor
        advantages = [
            (abs(three_pt_adv), "three-point shooting", three_pt_adv > 0),
            (abs(two_pt_adv), "interior scoring", two_pt_adv > 0),
            (abs(ft_adv), "free throw drawing", ft_adv > 0),
        ]
        key_factor_magnitude, key_factor, team1_has_advantage = max(
            advantages, key=lambda x: x[0]
        )
        winner = team1 if team1_has_advantage else team2
        key_matchup = f"{key_factor.title()} advantage to {winner}"

        # Generate strategy recommendation
        strategy = self._generate_strategy_recommendation(
            profile1, profile2, three_pt_adv, two_pt_adv, ft_adv
        )

        return ScoringStyleMatchup(
            team1_profile=profile1,
            team2_profile=profile2,
            three_point_advantage=round(three_pt_adv, 2),
            two_point_advantage=round(two_pt_adv, 2),
            free_throw_advantage=round(ft_adv, 2),
            style_mismatch_score=round(mismatch_score, 2),
            team1_exploitable_areas=team1_exploits,
            team2_exploitable_areas=team2_exploits,
            key_matchup_factor=key_matchup,
            recommended_strategy=strategy,
        )

    def _calculate_style_mismatch(
        self, offense: ScoringStyleProfile, defense: ScoringStyleProfile
    ) -> float:
        """Calculate style mismatch score (0-10).

        Higher score means offense's strength aligns well with defense's weakness.

        Args:
            offense: Offensive team's profile
            defense: Defensive team's profile

        Returns:
            Mismatch score from 0-10
        """
        # Find offense's strongest scoring method
        scoring_methods = [
            (offense.fg3_pct, defense.def_fg3_pct, "3pt"),
            (offense.fg2_pct, defense.def_fg2_pct, "2pt"),
            (offense.ft_pct, defense.def_ft_pct, "FT"),
        ]

        # Score each matchup (offense strength vs defense weakness)
        scores = []
        for off_strength, def_weakness, _method in scoring_methods:
            # Normalize to 0-10 scale
            # If offense excels (high %) and defense struggles (high % allowed)
            matchup_score = (off_strength / 10.0) + (def_weakness / 10.0)
            scores.append(matchup_score)

        # Return max mismatch score
        return min(max(scores), 10.0)

    def _identify_exploitable_areas(
        self, team: ScoringStyleProfile, opponent: ScoringStyleProfile
    ) -> list[str]:
        """Identify areas where team can exploit opponent's defense.

        Args:
            team: Offensive team's profile
            opponent: Defensive team's profile

        Returns:
            List of exploitable areas with specific percentages
        """
        exploits = []

        # Check each scoring method against opponent defense
        if team.fg3_pct > 35.0 and opponent.def_fg3_pct > 33.0:
            exploits.append(
                f"Three-point shooting ({team.fg3_pct:.1f}% offense vs "
                f"{opponent.def_fg3_pct:.1f}% defense allowed)"
            )

        if team.fg2_pct > 50.0 and opponent.def_fg2_pct > 48.0:
            exploits.append(
                f"Interior scoring ({team.fg2_pct:.1f}% offense vs "
                f"{opponent.def_fg2_pct:.1f}% defense allowed)"
            )

        if team.ft_pct > 20.0 and opponent.def_ft_pct > 22.0:
            exploits.append(
                f"Free throw drawing ({team.ft_pct:.1f}% offense vs "
                f"{opponent.def_ft_pct:.1f}% defense allowed)"
            )

        return exploits if exploits else ["No clear exploitable areas"]

    def _generate_strategy_recommendation(
        self,
        team1: ScoringStyleProfile,
        team2: ScoringStyleProfile,
        three_adv: float,
        two_adv: float,
        ft_adv: float,
    ) -> str:
        """Generate strategic game plan recommendation.

        Args:
            team1: First team's profile
            team2: Second team's profile
            three_adv: 3-point advantage
            two_adv: 2-point advantage
            ft_adv: Free throw advantage

        Returns:
            Strategic recommendation string
        """
        # Find biggest advantage
        advantages = [
            (three_adv, "emphasize three-point shooting", team1.team_name),
            (two_adv, "attack the paint", team1.team_name),
            (ft_adv, "draw fouls and get to the line", team1.team_name),
            (-three_adv, "emphasize three-point shooting", team2.team_name),
            (-two_adv, "attack the paint", team2.team_name),
            (-ft_adv, "draw fouls and get to the line", team2.team_name),
        ]

        magnitude, strategy, team = max(advantages, key=lambda x: abs(x[0]))

        return f"{team} should {strategy} (advantage: {abs(magnitude):.1f}%)"
