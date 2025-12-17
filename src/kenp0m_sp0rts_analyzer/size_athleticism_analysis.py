"""Size and athleticism analysis for basketball games.

This module analyzes physical matchups and identifies size advantages at both
the team level and position-specific level. It predicts rebounding battles,
paint scoring opportunities, and provides size-based strategic recommendations.

Size Classifications:
- Elite Size: AvgHgt > 79"
- Above Average: AvgHgt > 77.5"
- Average: 76" < AvgHgt < 77.5"
- Undersized: AvgHgt < 76"

Example:
    ```python
    from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
    from kenp0m_sp0rts_analyzer.size_athleticism_analysis import (
        SizeAthleticismAnalyzer
    )

    api = KenPomAPI()
    analyzer = SizeAthleticismAnalyzer(api)

    # Analyze Duke's size profile
    profile = analyzer.get_size_profile("Duke", 2025)
    print(f"Size: {profile.size_profile}")
    print(f"Effective Height: {profile.eff_height}")

    # Analyze matchup
    matchup = analyzer.analyze_matchup("Duke", "North Carolina", 2025)
    print(f"Better Size: {matchup.better_size_team}")
    print(f"Rebounding: {matchup.rebounding_prediction}")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .api_client import KenPomAPI
from .utils import normalize_team_name


@dataclass
class SizeProfile:
    """Team's size and physical profile.

    Attributes:
        team_name: Name of the team
        season: Season year (e.g., 2025)
        avg_height: Average team height in inches
        avg_height_rank: National rank for average height
        eff_height: Effective height (minutes-weighted)
        eff_height_rank: National rank for effective height
        pg_height: Point guard height (Hgt1)
        sg_height: Shooting guard height (Hgt2)
        sf_height: Small forward height (Hgt3)
        pf_height: Power forward height (Hgt4)
        c_height: Center height (Hgt5)
        pg_height_rank: National rank for PG height
        sg_height_rank: National rank for SG height
        sf_height_rank: National rank for SF height
        pf_height_rank: National rank for PF height
        c_height_rank: National rank for C height
        size_profile: Overall size classification
        biggest_advantage: Area of biggest size advantage
        biggest_weakness: Area of biggest size weakness
    """

    team_name: str
    season: int

    # Overall height
    avg_height: float
    avg_height_rank: int
    eff_height: float
    eff_height_rank: int

    # Position-specific heights
    pg_height: float
    sg_height: float
    sf_height: float
    pf_height: float
    c_height: float

    # Position ranks
    pg_height_rank: int
    sg_height_rank: int
    sf_height_rank: int
    pf_height_rank: int
    c_height_rank: int

    # Size classification
    size_profile: Literal["elite_size", "above_average", "average", "undersized"]
    biggest_advantage: str
    biggest_weakness: str


@dataclass
class PositionMatchup:
    """Position-specific size matchup.

    Attributes:
        position: Position code (PG, SG, SF, PF, C)
        position_name: Full position name
        team1_height: Team 1's height at this position
        team2_height: Team 2's height at this position
        height_advantage: Height difference (positive = team1 taller)
        advantage_inches: Absolute advantage in inches
        advantage_classification: Classification of advantage magnitude
        predicted_impact: Strategic impact of height difference
    """

    position: Literal["PG", "SG", "SF", "PF", "C"]
    position_name: str
    team1_height: float
    team2_height: float
    height_advantage: float
    advantage_inches: float
    advantage_classification: Literal[
        "massive", "significant", "moderate", "minimal", "neutral"
    ]
    predicted_impact: str


@dataclass
class SizeMatchup:
    """Complete size matchup analysis.

    Attributes:
        team1_profile: Size profile for team 1
        team2_profile: Size profile for team 2
        overall_height_advantage: Effective height difference
        better_size_team: Which team has better overall size
        size_advantage_score: Size advantage score (0-10, 5 = neutral)
        pg_matchup: Point guard position matchup
        sg_matchup: Shooting guard position matchup
        sf_matchup: Small forward position matchup
        pf_matchup: Power forward position matchup
        c_matchup: Center position matchup
        frontcourt_advantage: Which team has frontcourt size edge
        backcourt_advantage: Which team has backcourt size edge
        rebounding_prediction: Expected rebounding battle outcome
        paint_scoring_prediction: Expected interior scoring outcome
        strategic_recommendation: How to exploit size advantages
    """

    team1_profile: SizeProfile
    team2_profile: SizeProfile

    # Overall size advantage
    overall_height_advantage: float
    better_size_team: str
    size_advantage_score: float

    # Position matchups
    pg_matchup: PositionMatchup
    sg_matchup: PositionMatchup
    sf_matchup: PositionMatchup
    pf_matchup: PositionMatchup
    c_matchup: PositionMatchup

    # Strategic insights
    frontcourt_advantage: str
    backcourt_advantage: str
    rebounding_prediction: str
    paint_scoring_prediction: str
    strategic_recommendation: str


@dataclass
class ReboundingCorrelation:
    """Correlation between height and rebounding performance.

    Attributes:
        team_name: Name of the team
        eff_height: Effective height in inches
        or_pct: Offensive rebounding percentage
        dr_pct: Defensive rebounding percentage
        height_rebounding_score: How well height translates to boards
        rebounding_efficiency: Efficiency classification
    """

    team_name: str
    eff_height: float
    or_pct: float
    dr_pct: float
    height_rebounding_score: float
    rebounding_efficiency: Literal["excellent", "good", "average", "poor"]


class SizeAthleticismAnalyzer:
    """Analyze physical matchups and size advantages.

    This class provides detailed size analysis for basketball matchups,
    identifying position-specific height advantages, rebounding predictions,
    and size-based strategic opportunities.

    Size classifications (based on average height):
    - Elite Size: >79"
    - Above Average: >77.5"
    - Average: 76-77.5"
    - Undersized: <76"

    Position matchup impact thresholds:
    - Massive: >3" advantage
    - Significant: >2" advantage
    - Moderate: >1" advantage
    - Minimal: >0.5" advantage

    Key capabilities:
    - Team size profile generation
    - Position-by-position matchup analysis
    - Rebounding battle predictions
    - Paint scoring predictions
    - Height-to-rebounding correlation
    """

    # Classification thresholds (inches)
    ELITE_SIZE_THRESHOLD = 79.0
    ABOVE_AVG_THRESHOLD = 77.5
    UNDERSIZED_THRESHOLD = 76.0

    # Position matchup impact thresholds (inches)
    MASSIVE_ADVANTAGE = 3.0
    SIGNIFICANT_ADVANTAGE = 2.0
    MODERATE_ADVANTAGE = 1.0
    MINIMAL_ADVANTAGE = 0.5

    def __init__(self, api: KenPomAPI | None = None):
        """Initialize Size & Athleticism analyzer.

        Args:
            api: Optional KenPomAPI instance. Creates one if not provided.
        """
        self.api = api or KenPomAPI()

    def get_size_profile(self, team: str, season: int) -> SizeProfile:
        """Generate size profile for a team.

        Args:
            team: Team name
            season: Season year (e.g., 2025)

        Returns:
            SizeProfile with complete size breakdown

        Raises:
            ValueError: If team not found
        """
        # Normalize team name for consistent lookup
        team = normalize_team_name(team)

        data = self.api.get_height(year=season)
        team_data = next((t for t in data.data if t["TeamName"] == team), None)

        if not team_data:
            raise ValueError(f"Team '{team}' not found in {season} season")

        # Classify overall size
        avg_hgt = team_data["AvgHgt"]
        size_profile: Literal["elite_size", "above_average", "average", "undersized"]
        if avg_hgt >= self.ELITE_SIZE_THRESHOLD:
            size_profile = "elite_size"
        elif avg_hgt >= self.ABOVE_AVG_THRESHOLD:
            size_profile = "above_average"
        elif avg_hgt >= self.UNDERSIZED_THRESHOLD:
            size_profile = "average"
        else:
            size_profile = "undersized"

        # Identify biggest advantage/weakness
        frontcourt_avg = (team_data["Hgt3"] + team_data["Hgt4"] + team_data["Hgt5"]) / 3
        backcourt_avg = (team_data["Hgt1"] + team_data["Hgt2"]) / 2

        if frontcourt_avg > 79.0 and backcourt_avg < 74.0:
            biggest_advantage = "Frontcourt"
            biggest_weakness = "Perimeter size"
        elif backcourt_avg > 74.5 and frontcourt_avg < 78.0:
            biggest_advantage = "Backcourt"
            biggest_weakness = "Interior size"
        elif avg_hgt >= self.ABOVE_AVG_THRESHOLD:
            biggest_advantage = "Balanced"
            biggest_weakness = "None"
        else:
            biggest_advantage = "None"
            biggest_weakness = "Overall size"

        return SizeProfile(
            team_name=team,
            season=season,
            avg_height=avg_hgt,
            avg_height_rank=team_data["AvgHgtRank"],
            eff_height=team_data["HgtEff"],
            eff_height_rank=team_data["HgtEffRank"],
            pg_height=team_data["Hgt1"],
            sg_height=team_data["Hgt2"],
            sf_height=team_data["Hgt3"],
            pf_height=team_data["Hgt4"],
            c_height=team_data["Hgt5"],
            pg_height_rank=team_data["Hgt1Rank"],
            sg_height_rank=team_data["Hgt2Rank"],
            sf_height_rank=team_data["Hgt3Rank"],
            pf_height_rank=team_data["Hgt4Rank"],
            c_height_rank=team_data["Hgt5Rank"],
            size_profile=size_profile,
            biggest_advantage=biggest_advantage,
            biggest_weakness=biggest_weakness,
        )

    def analyze_matchup(self, team1: str, team2: str, season: int) -> SizeMatchup:
        """Analyze complete size matchup between two teams.

        Args:
            team1: First team name
            team2: Second team name
            season: Season year (e.g., 2025)

        Returns:
            SizeMatchup with complete size analysis

        Raises:
            ValueError: If team not found
        """
        profile1 = self.get_size_profile(team1, season)
        profile2 = self.get_size_profile(team2, season)

        # Overall size advantage (use effective height)
        overall_advantage = profile1.eff_height - profile2.eff_height
        size_score = self._calculate_size_advantage_score(profile1, profile2)
        better_size = (
            team1 if size_score > 5.0 else (team2 if size_score < 5.0 else "neutral")
        )

        # Analyze each position
        pg = self._analyze_position_matchup(
            "PG",
            "Point Guard",
            profile1.pg_height,
            profile2.pg_height,
            team1,
            team2,
        )
        sg = self._analyze_position_matchup(
            "SG",
            "Shooting Guard",
            profile1.sg_height,
            profile2.sg_height,
            team1,
            team2,
        )
        sf = self._analyze_position_matchup(
            "SF",
            "Small Forward",
            profile1.sf_height,
            profile2.sf_height,
            team1,
            team2,
        )
        pf = self._analyze_position_matchup(
            "PF",
            "Power Forward",
            profile1.pf_height,
            profile2.pf_height,
            team1,
            team2,
        )
        c = self._analyze_position_matchup(
            "C", "Center", profile1.c_height, profile2.c_height, team1, team2
        )

        # Determine frontcourt/backcourt advantages
        frontcourt_advantage = self._determine_court_advantage(
            [sf, pf, c], team1, team2
        )
        backcourt_advantage = self._determine_court_advantage([pg, sg], team1, team2)

        # Generate predictions
        rebounding_pred = self._predict_rebounding_battle(
            profile1, profile2, team1, team2
        )
        paint_scoring_pred = self._predict_paint_scoring(
            profile1, profile2, team1, team2
        )
        strategy = self._generate_size_strategy(profile1, profile2, team1, team2)

        return SizeMatchup(
            team1_profile=profile1,
            team2_profile=profile2,
            overall_height_advantage=round(overall_advantage, 2),
            better_size_team=better_size,
            size_advantage_score=round(size_score, 2),
            pg_matchup=pg,
            sg_matchup=sg,
            sf_matchup=sf,
            pf_matchup=pf,
            c_matchup=c,
            frontcourt_advantage=frontcourt_advantage,
            backcourt_advantage=backcourt_advantage,
            rebounding_prediction=rebounding_pred,
            paint_scoring_prediction=paint_scoring_pred,
            strategic_recommendation=strategy,
        )

    def _analyze_position_matchup(
        self,
        pos: Literal["PG", "SG", "SF", "PF", "C"],
        pos_name: str,
        team1_height: float,
        team2_height: float,
        _team1_name: str,
        _team2_name: str,
    ) -> PositionMatchup:
        """Analyze height matchup at a specific position.

        Args:
            pos: Position code
            pos_name: Full position name
            team1_height: Team 1's height at this position
            team2_height: Team 2's height at this position
            _team1_name: Team 1 name (unused)
            _team2_name: Team 2 name (unused)

        Returns:
            PositionMatchup with height advantage analysis
        """
        advantage = team1_height - team2_height
        adv_inches = abs(advantage)

        # Classify advantage
        classification: Literal[
            "massive", "significant", "moderate", "minimal", "neutral"
        ]
        if adv_inches >= self.MASSIVE_ADVANTAGE:
            classification = "massive"
        elif adv_inches >= self.SIGNIFICANT_ADVANTAGE:
            classification = "significant"
        elif adv_inches >= self.MODERATE_ADVANTAGE:
            classification = "moderate"
        elif adv_inches >= self.MINIMAL_ADVANTAGE:
            classification = "minimal"
        else:
            classification = "neutral"

        # Predict impact based on position and advantage
        impact = self._predict_position_impact(pos, adv_inches, advantage > 0)

        return PositionMatchup(
            position=pos,
            position_name=pos_name,
            team1_height=round(team1_height, 1),
            team2_height=round(team2_height, 1),
            height_advantage=round(advantage, 2),
            advantage_inches=round(adv_inches, 2),
            advantage_classification=classification,
            predicted_impact=impact,
        )

    def _predict_position_impact(
        self, position: str, inches: float, team1_taller: bool
    ) -> str:
        """Predict strategic impact of height advantage at position.

        Args:
            position: Position code (PG, SG, SF, PF, C)
            inches: Height advantage in inches
            team1_taller: Whether team 1 is taller

        Returns:
            Strategic impact description
        """
        if inches < 0.5:
            return "Minimal impact - even matchup"

        taller_team = "Team 1" if team1_taller else "Team 2"

        if position in ["PF", "C"]:  # Frontcourt
            if inches >= 3.0:
                return f"{taller_team} massive advantage: Dominate paint, rebounding, rim protection"
            elif inches >= 2.0:
                return f"{taller_team} significant edge: Control boards, alter shots inside"
            else:
                return (
                    f"{taller_team} moderate edge: Slight rebounding/interior advantage"
                )

        elif position == "SF":  # Wing
            if inches >= 2.5:
                return f"{taller_team} can exploit: Post up smaller defender, defensive versatility"
            else:
                return f"{taller_team} slight edge: Rebounding, mid-range defense"

        else:  # Backcourt (PG, SG)
            if inches >= 3.0:
                return f"{taller_team} rare advantage: See over pressure, disrupt passing lanes"
            elif inches >= 1.5:
                return f"{taller_team} can exploit: Post up on switches, rebounding from guard spot"
            else:
                return (
                    f"{taller_team} minimal impact: Speed/quickness may offset height"
                )

        return "Even matchup"

    def _calculate_size_advantage_score(
        self, team1: SizeProfile, team2: SizeProfile
    ) -> float:
        """Calculate overall size advantage (0-10 scale, 5 = neutral).

        Args:
            team1: First team's size profile
            team2: Second team's size profile

        Returns:
            Size advantage score from 0-10
        """
        # Use effective height (more important than average)
        eff_height_diff = team1.eff_height - team2.eff_height

        # Convert to 0-10 scale (each inch ≈ 1 point)
        score = 5.0 + eff_height_diff

        return max(0.0, min(10.0, score))

    def _determine_court_advantage(
        self, position_matchups: list[PositionMatchup], team1: str, team2: str
    ) -> str:
        """Determine which team has advantage in frontcourt or backcourt.

        Args:
            position_matchups: List of position matchups (frontcourt or backcourt)
            team1: Team 1 name
            team2: Team 2 name

        Returns:
            Team name with advantage or "neutral"
        """
        team1_advantages = sum(
            1 for pm in position_matchups if pm.height_advantage > 0.5
        )
        team2_advantages = sum(
            1 for pm in position_matchups if pm.height_advantage < -0.5
        )

        if team1_advantages > team2_advantages:
            return team1
        elif team2_advantages > team1_advantages:
            return team2
        else:
            return "neutral"

    def _predict_rebounding_battle(
        self, team1: SizeProfile, team2: SizeProfile, team1_name: str, team2_name: str
    ) -> str:
        """Predict rebounding battle outcome based on size.

        Args:
            team1: First team's size profile
            team2: Second team's size profile
            team1_name: Team 1 name
            team2_name: Team 2 name

        Returns:
            Rebounding prediction string
        """
        # Weight frontcourt height more heavily
        team1_rebound_score = (
            team1.eff_height * 0.6 + (team1.pf_height + team1.c_height) / 2 * 0.4
        )
        team2_rebound_score = (
            team2.eff_height * 0.6 + (team2.pf_height + team2.c_height) / 2 * 0.4
        )

        diff = team1_rebound_score - team2_rebound_score

        if diff > 2.0:
            return f"{team1_name} should dominate the glass (frontcourt size advantage)"
        elif diff < -2.0:
            return f"{team2_name} should dominate the glass (frontcourt size advantage)"
        elif diff > 1.0:
            return f"{team1_name} slight rebounding edge"
        elif diff < -1.0:
            return f"{team2_name} slight rebounding edge"
        else:
            return "Even rebounding battle expected"

    def _predict_paint_scoring(
        self, team1: SizeProfile, team2: SizeProfile, team1_name: str, team2_name: str
    ) -> str:
        """Predict interior scoring based on size matchup.

        Args:
            team1: First team's size profile
            team2: Second team's size profile
            team1_name: Team 1 name
            team2_name: Team 2 name

        Returns:
            Paint scoring prediction string
        """
        # Interior size = average of SF, PF, C
        team1_interior = (team1.sf_height + team1.pf_height + team1.c_height) / 3
        team2_interior = (team2.sf_height + team2.pf_height + team2.c_height) / 3

        diff = team1_interior - team2_interior

        if diff > 2.5:
            return f"{team1_name} should attack the paint relentlessly"
        elif diff < -2.5:
            return f"{team2_name} should attack the paint relentlessly"
        elif diff > 1.0:
            return f"{team1_name} has interior scoring advantage"
        elif diff < -1.0:
            return f"{team2_name} has interior scoring advantage"
        else:
            return "Balanced interior scoring opportunities"

    def _generate_size_strategy(
        self, team1: SizeProfile, team2: SizeProfile, team1_name: str, team2_name: str
    ) -> str:
        """Generate strategic recommendation based on size matchup.

        Args:
            team1: First team's size profile
            team2: Second team's size profile
            team1_name: Team 1 name
            team2_name: Team 2 name

        Returns:
            Strategic recommendation string
        """
        eff_diff = team1.eff_height - team2.eff_height

        if eff_diff > 2.0:
            return (
                f"{team1_name} should play big: Slow tempo, pound the paint, "
                f"crash the offensive glass, protect the rim"
            )
        elif eff_diff < -2.0:
            return (
                f"{team2_name} should play big: Slow tempo, pound the paint, "
                f"crash the offensive glass, protect the rim"
            )
        elif eff_diff > 1.0:
            return f"{team1_name} should leverage size advantage selectively in half-court sets"
        elif eff_diff < -1.0:
            return f"{team2_name} should leverage size advantage selectively in half-court sets"
        else:
            # Size is even - look for specific position mismatches
            if abs(team1.pg_height - team2.pg_height) > 2.0:
                taller_pg = (
                    team1_name if team1.pg_height > team2.pg_height else team2_name
                )
                return f"Even overall size, but {taller_pg} should exploit PG height mismatch"
            else:
                return "Size is even - exploit speed, skill, and execution advantages instead"

    def correlate_height_rebounding(
        self, team: str, season: int, or_pct: float, dr_pct: float
    ) -> ReboundingCorrelation:
        """Analyze how well team converts height advantage into rebounds.

        Args:
            team: Team name
            season: Season year
            or_pct: Offensive rebounding % (from Four Factors)
            dr_pct: Defensive rebounding % (from Four Factors)

        Returns:
            ReboundingCorrelation showing height-to-rebounding efficiency
        """
        profile = self.get_size_profile(team, season)

        # Calculate expected rebounding based on height
        # Rough baseline: 78" effective height → ~30% OR, ~70% DR
        expected_or = 20.0 + (profile.eff_height - 78.0) * 2.0  # ~2% per inch
        expected_dr = 65.0 + (profile.eff_height - 78.0) * 1.5  # ~1.5% per inch

        # Calculate how actual rebounding compares to expected
        or_efficiency = (
            (or_pct - expected_or) / expected_or * 100
        )  # % above/below expected
        dr_efficiency = (dr_pct - expected_dr) / expected_dr * 100

        # Combined score
        height_rebounding_score = (or_efficiency + dr_efficiency) / 2

        # Classify efficiency
        efficiency: Literal["excellent", "good", "average", "poor"]
        if height_rebounding_score > 10.0:
            efficiency = "excellent"
        elif height_rebounding_score > 0.0:
            efficiency = "good"
        elif height_rebounding_score > -10.0:
            efficiency = "average"
        else:
            efficiency = "poor"

        return ReboundingCorrelation(
            team_name=team,
            eff_height=profile.eff_height,
            or_pct=or_pct,
            dr_pct=dr_pct,
            height_rebounding_score=round(height_rebounding_score, 2),
            rebounding_efficiency=efficiency,
        )
