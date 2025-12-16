"""Defensive analysis for basketball games.

This module provides advanced defensive analysis, identifying defensive schemes,
strengths, weaknesses, and matchup advantages.

The analysis categorizes defenses into:
- Rim Protection: Elite interior defense with high block rates
- Pressure: Aggressive defense with high steal rates
- Versatile: Elite at multiple defensive dimensions
- Balanced: Solid fundamental defense

Example:
    ```python
    from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
    from kenp0m_sp0rts_analyzer.defensive_analysis import DefensiveAnalyzer

    api = KenPomAPI()
    analyzer = DefensiveAnalyzer(api)

    # Analyze Duke's defensive scheme
    profile = analyzer.get_defensive_profile("Duke", 2025)
    print(f"Scheme: {profile.defensive_scheme}")
    print(f"Strength: {profile.primary_strength}")

    # Analyze defensive matchup
    matchup = analyzer.analyze_matchup("Duke", "North Carolina", 2025)
    print(f"Better defense: {matchup.better_defense}")
    print(f"Advantage: {matchup.defensive_advantage_score:.1f}/10")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .api_client import KenPomAPI


@dataclass
class DefensiveProfile:
    """Team's defensive identity and scheme.

    Attributes:
        team_name: Name of the team
        season: Season year (e.g., 2025)
        opp_fg3_pct: Opponent 3-point percentage allowed
        opp_fg3_rank: National rank for opponent 3pt%
        opp_fg2_pct: Opponent 2-point percentage allowed
        opp_fg2_rank: National rank for opponent 2pt%
        block_pct: Block percentage
        block_rank: National rank for block percentage
        stl_rate: Steal rate
        stl_rank: National rank for steal rate
        opp_nst_rate: Opponent non-steal turnover rate
        nst_rank: National rank for opponent NST rate
        opp_assist_rate: Opponent assist rate allowed
        assist_rank: National rank for opponent assist rate
        defensive_scheme: Overall defensive scheme classification
        primary_strength: Description of team's main defensive strength
        primary_weakness: Description of team's main defensive weakness
    """

    team_name: str
    season: int

    # Perimeter defense
    opp_fg3_pct: float  # Opponent 3pt %
    opp_fg3_rank: int

    # Interior defense
    opp_fg2_pct: float  # Opponent 2pt %
    opp_fg2_rank: int
    block_pct: float  # Block percentage
    block_rank: int

    # Pressure defense
    stl_rate: float  # Steal rate
    stl_rank: int
    opp_nst_rate: float  # Opponent non-steal turnovers
    nst_rank: int

    # Ball movement allowed
    opp_assist_rate: float
    assist_rank: int

    # Scheme classification
    defensive_scheme: Literal["rim_protection", "pressure", "balanced", "versatile"]
    primary_strength: str
    primary_weakness: str


@dataclass
class DefensiveMatchup:
    """Defensive matchup analysis.

    Attributes:
        team1_defense: Defensive profile for team 1
        team2_defense: Defensive profile for team 2
        perimeter_defense_advantage: Which team has better perimeter defense
        interior_defense_advantage: Which team has better interior defense
        pressure_defense_advantage: Which team applies more pressure
        better_defense: Overall better defensive team
        defensive_advantage_score: Advantage score (0-10, 5 = neutral)
        team1_defensive_keys: Key defensive strategies for team 1
        team2_defensive_keys: Key defensive strategies for team 2
        matchup_recommendation: Overall strategic recommendation
    """

    team1_defense: DefensiveProfile
    team2_defense: DefensiveProfile

    # Defensive advantages
    perimeter_defense_advantage: str
    interior_defense_advantage: str
    pressure_defense_advantage: str

    # Overall defensive edge
    better_defense: str
    defensive_advantage_score: float  # 0-10 scale (5 = neutral)

    # Strategic insights
    team1_defensive_keys: list[str]
    team2_defensive_keys: list[str]
    matchup_recommendation: str


class DefensiveAnalyzer:
    """Advanced defensive analysis and matchup evaluation.

    This class provides comprehensive defensive analysis for basketball matchups,
    identifying defensive schemes, strengths, and strategic opportunities.

    Defensive scheme classifications:
    - Rim Protection: High block rate (>10%) + low opponent 2pt% (<46%)
    - Pressure: High steal rate (>9%)
    - Versatile: Elite at both perimeter and interior defense
    - Balanced: Solid fundamental defense across all areas

    Key capabilities:
    - Classify defensive schemes
    - Identify defensive strengths and weaknesses
    - Compare defensive matchups
    - Generate defensive game plans
    """

    # Classification thresholds
    HIGH_BLOCK_RATE = 10.0  # Top rim protection
    HIGH_STEAL_RATE = 9.0  # Aggressive pressure
    LOW_OPP_FG3 = 31.0  # Elite perimeter defense
    LOW_OPP_FG2 = 46.0  # Elite interior defense

    def __init__(self, api: KenPomAPI | None = None):
        """Initialize Defensive analyzer.

        Args:
            api: Optional KenPomAPI instance. Creates one if not provided.
        """
        self.api = api or KenPomAPI()

    def get_defensive_profile(self, team: str, season: int) -> DefensiveProfile:
        """Generate comprehensive defensive profile.

        Args:
            team: Team name
            season: Season year (e.g., 2025)

        Returns:
            DefensiveProfile with complete defensive breakdown

        Raises:
            ValueError: If team not found
        """
        data = self.api.get_misc_stats(year=season)
        team_data = next((t for t in data.data if t["TeamName"] == team), None)

        if not team_data:
            raise ValueError(f"Team '{team}' not found in {season} season")

        # Extract defensive metrics
        opp_fg3 = team_data["OppFG3Pct"]
        opp_fg2 = team_data["OppFG2Pct"]
        blocks = team_data["BlockPct"]
        steals = team_data["StlRate"]

        # Classify defensive scheme
        scheme, strength, weakness = self._classify_defensive_scheme(
            opp_fg3, opp_fg2, blocks, steals
        )

        return DefensiveProfile(
            team_name=team,
            season=season,
            opp_fg3_pct=opp_fg3,
            opp_fg3_rank=team_data["RankOppFG3Pct"],
            opp_fg2_pct=opp_fg2,
            opp_fg2_rank=team_data["RankOppFG2Pct"],
            block_pct=blocks,
            block_rank=team_data["RankBlockPct"],
            stl_rate=steals,
            stl_rank=team_data["RankStlRate"],
            opp_nst_rate=team_data["OppNSTRate"],
            nst_rank=team_data["RankOppNSTRate"],
            opp_assist_rate=team_data["OppARate"],
            assist_rank=team_data["RankOppARate"],
            defensive_scheme=scheme,
            primary_strength=strength,
            primary_weakness=weakness,
        )

    def analyze_matchup(self, team1: str, team2: str, season: int) -> DefensiveMatchup:
        """Analyze defensive matchup between two teams.

        Args:
            team1: First team name
            team2: Second team name
            season: Season year (e.g., 2025)

        Returns:
            DefensiveMatchup with complete defensive analysis

        Raises:
            ValueError: If team not found
        """
        profile1 = self.get_defensive_profile(team1, season)
        profile2 = self.get_defensive_profile(team2, season)

        # Determine advantages
        perimeter_adv = team1 if profile1.opp_fg3_pct < profile2.opp_fg3_pct else team2

        interior_adv = team1 if profile1.opp_fg2_pct < profile2.opp_fg2_pct else team2

        pressure_adv = team1 if profile1.stl_rate > profile2.stl_rate else team2

        # Calculate overall defensive advantage (0-10 scale)
        defense_score = self._calculate_defensive_advantage(profile1, profile2)
        better_defense = team1 if defense_score > 5.0 else team2

        # Generate defensive keys
        team1_keys = self._generate_defensive_keys(profile1, profile2)
        team2_keys = self._generate_defensive_keys(profile2, profile1)

        # Matchup recommendation
        recommendation = self._generate_matchup_recommendation(
            profile1, profile2, defense_score
        )

        return DefensiveMatchup(
            team1_defense=profile1,
            team2_defense=profile2,
            perimeter_defense_advantage=perimeter_adv,
            interior_defense_advantage=interior_adv,
            pressure_defense_advantage=pressure_adv,
            better_defense=better_defense,
            defensive_advantage_score=round(defense_score, 2),
            team1_defensive_keys=team1_keys,
            team2_defensive_keys=team2_keys,
            matchup_recommendation=recommendation,
        )

    def _classify_defensive_scheme(
        self, opp_fg3: float, opp_fg2: float, blocks: float, steals: float
    ) -> tuple[
        Literal["rim_protection", "pressure", "balanced", "versatile"], str, str
    ]:
        """Classify defensive scheme and identify strengths/weaknesses.

        Args:
            opp_fg3: Opponent 3-point percentage
            opp_fg2: Opponent 2-point percentage
            blocks: Block percentage
            steals: Steal rate

        Returns:
            Tuple of (scheme, strength description, weakness description)
        """
        scheme: Literal["rim_protection", "pressure", "balanced", "versatile"]
        # Rim protection scheme
        if blocks >= self.HIGH_BLOCK_RATE and opp_fg2 <= self.LOW_OPP_FG2:
            scheme = "rim_protection"
            strength = f"Elite rim protection ({blocks:.1f}% blocks)"
            weakness = "Perimeter defense" if opp_fg3 > 32.0 else "None apparent"

        # Pressure scheme
        elif steals >= self.HIGH_STEAL_RATE:
            scheme = "pressure"
            strength = f"Aggressive pressure ({steals:.1f}% steal rate)"
            weakness = "Fouling" if opp_fg3 > 33.0 else "Can be beaten backdoor"

        # Versatile (good at everything)
        elif (
            opp_fg3 <= self.LOW_OPP_FG3
            and opp_fg2 <= self.LOW_OPP_FG2
            and steals >= 8.0
        ):
            scheme = "versatile"
            strength = "Multi-dimensional elite defense"
            weakness = "None - complete defense"

        # Balanced
        else:
            scheme = "balanced"
            strength = "Solid fundamental defense"

            # Identify weakest area
            weaknesses = [
                (opp_fg3, "three-point defense"),
                (opp_fg2, "interior defense"),
                (10.0 - steals, "pressure defense"),
            ]
            weakness = max(weaknesses, key=lambda x: x[0])[1]

        return scheme, strength, weakness

    def _calculate_defensive_advantage(
        self, team1: DefensiveProfile, team2: DefensiveProfile
    ) -> float:
        """Calculate overall defensive advantage (0-10 scale, 5 = neutral).

        Args:
            team1: First team's defensive profile
            team2: Second team's defensive profile

        Returns:
            Advantage score from 0-10 (5 = neutral, >5 = team1 advantage)
        """
        # Compare each defensive dimension
        perimeter_diff = team2.opp_fg3_pct - team1.opp_fg3_pct
        interior_diff = team2.opp_fg2_pct - team1.opp_fg2_pct
        pressure_diff = team1.stl_rate - team2.stl_rate

        # Normalize to 0-10 scale
        # Positive differences favor team1
        score = 5.0  # Start neutral
        score += perimeter_diff * 0.5  # 3pt defense weight
        score += interior_diff * 0.5  # 2pt defense weight
        score += pressure_diff * 0.2  # Steal rate weight

        return max(0.0, min(10.0, score))

    def _generate_defensive_keys(
        self, team: DefensiveProfile, opponent: DefensiveProfile
    ) -> list[str]:
        """Generate defensive game plan keys.

        Args:
            team: Defensive team's profile
            opponent: Offensive team's profile

        Returns:
            List of strategic defensive keys
        """
        keys = []

        # Leverage strengths
        if team.defensive_scheme == "rim_protection":
            keys.append(f"Protect the rim - force {opponent.team_name} outside")

        elif team.defensive_scheme == "pressure":
            keys.append("Apply full-court pressure - force turnovers")

        # Exploit opponent weaknesses
        if opponent.opp_fg3_pct > 33.0:
            keys.append(f"Attack {opponent.team_name}'s perimeter defense")

        if opponent.opp_fg2_pct > 48.0:
            keys.append(f"Exploit {opponent.team_name}'s interior defense")

        return keys if keys else ["Play fundamental defense"]

    def _generate_matchup_recommendation(
        self,
        team1: DefensiveProfile,
        team2: DefensiveProfile,
        advantage_score: float,
    ) -> str:
        """Generate strategic matchup recommendation.

        Args:
            team1: First team's defensive profile
            team2: Second team's defensive profile
            advantage_score: Overall advantage score (0-10)

        Returns:
            Strategic recommendation string
        """
        if advantage_score > 6.5:
            better_team = team1.team_name
            margin = "significant"
        elif advantage_score < 3.5:
            better_team = team2.team_name
            margin = "significant"
        else:
            return "Defensive matchup is evenly matched"

        return (
            f"{better_team} has a {margin} defensive advantage "
            f"(score: {advantage_score:.1f}/10)"
        )
