# TIER 2 Implementation Plan: Height & Experience Analysis

**High-Impact Analytics Modules for Physical and Intangible Matchup Factors**

---

## Overview

This plan details implementation of TIER 2 analyzer modules that leverage the `height` API endpoint:

1. **Size & Athleticism Analyzer** - Physical matchups, position-specific height advantages
2. **Experience & Chemistry Analyzer** - Veteran edge, bench depth, team continuity

**Effort**: ~4-6 hours total (2-3 hours each module)
**Impact**: High - Addresses physical matchups and intangible factors
**Dependencies**: Existing API client with `height` endpoint
**Complements**: TIER 1 modules (Four Factors, Point Distribution, Defensive Analysis)

---

## Available Data (height endpoint)

### Height Metrics
```python
AvgHgt: float           # Average team height in inches
HgtEff: float           # Effective height (weighted by minutes played)
Hgt5: float             # Center (5) position height
Hgt4: float             # Power Forward (4) position height
Hgt3: float             # Small Forward (3) position height
Hgt2: float             # Shooting Guard (2) position height
Hgt1: float             # Point Guard (1) position height
```

### Experience & Chemistry Metrics
```python
Exp: float              # Experience rating (0-4 scale: Fr=0, So=1, Jr=2, Sr=3, 5th=4)
Bench: float            # Bench strength rating
Continuity: float       # Team continuity rating (% of minutes returning)
```

### Key Insights from Data
- **Position-specific heights** allow granular matchup analysis
- **Effective height** accounts for playing time (more valuable than raw average)
- **Experience** correlates with late-game execution, tournament success
- **Continuity** indicates team chemistry and system familiarity
- **Bench strength** reveals depth advantages in close games

---

## Module 1: Size & Athleticism Analyzer

### File Structure
```
src/kenp0m_sp0rts_analyzer/
├── size_athleticism_analysis.py  (NEW)

examples/
├── size_matchup_demo.py           (NEW)

tests/
├── test_size_athleticism.py       (NEW)
```

### Implementation Details

#### Data Classes
```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class SizeProfile:
    """Team's size and physical profile."""
    team_name: str
    season: int

    # Overall height
    avg_height: float       # Average team height in inches
    avg_height_rank: int
    eff_height: float       # Effective height (minutes-weighted)
    eff_height_rank: int

    # Position-specific heights (1=PG, 5=C)
    pg_height: float        # Point guard height (Hgt1)
    sg_height: float        # Shooting guard height (Hgt2)
    sf_height: float        # Small forward height (Hgt3)
    pf_height: float        # Power forward height (Hgt4)
    c_height: float         # Center height (Hgt5)

    # Position ranks
    pg_height_rank: int
    sg_height_rank: int
    sf_height_rank: int
    pf_height_rank: int
    c_height_rank: int

    # Size classification
    size_profile: Literal["elite_size", "above_average", "average", "undersized"]
    biggest_advantage: str  # "Frontcourt", "Backcourt", "Balanced", "None"
    biggest_weakness: str   # "Interior size", "Perimeter size", "Overall size", "None"

@dataclass
class PositionMatchup:
    """Position-specific size matchup."""
    position: Literal["PG", "SG", "SF", "PF", "C"]
    position_name: str      # "Point Guard", "Shooting Guard", etc.
    team1_height: float
    team2_height: float
    height_advantage: float  # Positive = team1 taller
    advantage_inches: float  # Absolute advantage in inches
    advantage_classification: Literal["massive", "significant", "moderate", "minimal", "neutral"]
    predicted_impact: str    # Strategic impact of height difference

@dataclass
class SizeMatchup:
    """Complete size matchup analysis."""
    team1_profile: SizeProfile
    team2_profile: SizeProfile

    # Overall size advantage
    overall_height_advantage: float     # Effective height difference
    better_size_team: str              # "team1", "team2", "neutral"
    size_advantage_score: float        # 0-10 scale (5 = neutral)

    # Position matchups
    pg_matchup: PositionMatchup
    sg_matchup: PositionMatchup
    sf_matchup: PositionMatchup
    pf_matchup: PositionMatchup
    c_matchup: PositionMatchup

    # Strategic insights
    frontcourt_advantage: str          # Which team has frontcourt size edge
    backcourt_advantage: str           # Which team has backcourt size edge
    rebounding_prediction: str         # Expected rebounding battle outcome
    paint_scoring_prediction: str      # Expected interior scoring outcome
    strategic_recommendation: str      # How to exploit size advantages

@dataclass
class ReboundingCorrelation:
    """Correlation between height and rebounding performance."""
    team_name: str
    eff_height: float
    or_pct: float           # Offensive rebounding %
    dr_pct: float           # Defensive rebounding %
    height_rebounding_score: float  # How well height translates to boards
    rebounding_efficiency: Literal["excellent", "good", "average", "poor"]
```

#### Main Analyzer Class
```python
class SizeAthleticismAnalyzer:
    """Analyze physical matchups and size advantages."""

    # Classification thresholds (inches)
    ELITE_SIZE_THRESHOLD = 79.0      # Avg height >79" = elite size
    ABOVE_AVG_THRESHOLD = 77.5       # Avg height >77.5" = above average
    UNDERSIZED_THRESHOLD = 76.0      # Avg height <76" = undersized

    # Position matchup impact thresholds (inches)
    MASSIVE_ADVANTAGE = 3.0          # >3" height advantage = massive
    SIGNIFICANT_ADVANTAGE = 2.0      # >2" height advantage = significant
    MODERATE_ADVANTAGE = 1.0         # >1" height advantage = moderate
    MINIMAL_ADVANTAGE = 0.5          # >0.5" height advantage = minimal

    def __init__(self, api_client: KenPomAPI):
        self.api = api_client

    def get_size_profile(self, team: str, season: int) -> SizeProfile:
        """Generate size profile for a team.

        Classifies team size based on average and effective height:
        - Elite Size: AvgHgt > 79"
        - Above Average: AvgHgt > 77.5"
        - Average: 76" < AvgHgt < 77.5"
        - Undersized: AvgHgt < 76"
        """
        data = self.api.get_height(year=season)
        team_data = next((t for t in data.data if t['TeamName'] == team), None)

        if not team_data:
            raise ValueError(f"Team '{team}' not found")

        # Classify overall size
        avg_hgt = team_data['AvgHgt']
        if avg_hgt >= self.ELITE_SIZE_THRESHOLD:
            size_profile = "elite_size"
        elif avg_hgt >= self.ABOVE_AVG_THRESHOLD:
            size_profile = "above_average"
        elif avg_hgt >= self.UNDERSIZED_THRESHOLD:
            size_profile = "average"
        else:
            size_profile = "undersized"

        # Identify biggest advantage/weakness
        frontcourt_avg = (team_data['Hgt3'] + team_data['Hgt4'] + team_data['Hgt5']) / 3
        backcourt_avg = (team_data['Hgt1'] + team_data['Hgt2']) / 2

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
            avg_height_rank=team_data['AvgHgtRank'],
            eff_height=team_data['HgtEff'],
            eff_height_rank=team_data['HgtEffRank'],
            pg_height=team_data['Hgt1'],
            sg_height=team_data['Hgt2'],
            sf_height=team_data['Hgt3'],
            pf_height=team_data['Hgt4'],
            c_height=team_data['Hgt5'],
            pg_height_rank=team_data['Hgt1Rank'],
            sg_height_rank=team_data['Hgt2Rank'],
            sf_height_rank=team_data['Hgt3Rank'],
            pf_height_rank=team_data['Hgt4Rank'],
            c_height_rank=team_data['Hgt5Rank'],
            size_profile=size_profile,
            biggest_advantage=biggest_advantage,
            biggest_weakness=biggest_weakness
        )

    def analyze_matchup(
        self,
        team1: str,
        team2: str,
        season: int
    ) -> SizeMatchup:
        """Analyze complete size matchup between two teams."""

        profile1 = self.get_size_profile(team1, season)
        profile2 = self.get_size_profile(team2, season)

        # Overall size advantage (use effective height)
        overall_advantage = profile1.eff_height - profile2.eff_height
        size_score = self._calculate_size_advantage_score(profile1, profile2)
        better_size = team1 if size_score > 5.0 else team2 if size_score < 5.0 else "neutral"

        # Analyze each position
        pg = self._analyze_position_matchup("PG", "Point Guard",
                                           profile1.pg_height, profile2.pg_height, team1, team2)
        sg = self._analyze_position_matchup("SG", "Shooting Guard",
                                           profile1.sg_height, profile2.sg_height, team1, team2)
        sf = self._analyze_position_matchup("SF", "Small Forward",
                                           profile1.sf_height, profile2.sf_height, team1, team2)
        pf = self._analyze_position_matchup("PF", "Power Forward",
                                           profile1.pf_height, profile2.pf_height, team1, team2)
        c = self._analyze_position_matchup("C", "Center",
                                          profile1.c_height, profile2.c_height, team1, team2)

        # Determine frontcourt/backcourt advantages
        frontcourt_advantage = self._determine_court_advantage(
            [sf, pf, c], team1, team2
        )
        backcourt_advantage = self._determine_court_advantage(
            [pg, sg], team1, team2
        )

        # Generate predictions
        rebounding_pred = self._predict_rebounding_battle(profile1, profile2, team1, team2)
        paint_scoring_pred = self._predict_paint_scoring(profile1, profile2, team1, team2)
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
            strategic_recommendation=strategy
        )

    def _analyze_position_matchup(
        self,
        pos: str,
        pos_name: str,
        team1_height: float,
        team2_height: float,
        team1_name: str,
        team2_name: str
    ) -> PositionMatchup:
        """Analyze height matchup at a specific position."""

        advantage = team1_height - team2_height
        adv_inches = abs(advantage)

        # Classify advantage
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
            predicted_impact=impact
        )

    def _predict_position_impact(
        self,
        position: str,
        inches: float,
        team1_taller: bool
    ) -> str:
        """Predict strategic impact of height advantage at position."""

        if inches < 0.5:
            return "Minimal impact - even matchup"

        taller_team = "Team 1" if team1_taller else "Team 2"

        if position in ["PF", "C"]:  # Frontcourt
            if inches >= 3.0:
                return f"{taller_team} massive advantage: Dominate paint, rebounding, rim protection"
            elif inches >= 2.0:
                return f"{taller_team} significant edge: Control boards, alter shots inside"
            else:
                return f"{taller_team} moderate edge: Slight rebounding/interior advantage"

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
                return f"{taller_team} minimal impact: Speed/quickness may offset height"

    def _calculate_size_advantage_score(
        self,
        team1: SizeProfile,
        team2: SizeProfile
    ) -> float:
        """Calculate overall size advantage (0-10 scale, 5 = neutral)."""

        # Use effective height (more important than average)
        eff_height_diff = team1.eff_height - team2.eff_height

        # Convert to 0-10 scale (each inch ≈ 1 point)
        score = 5.0 + eff_height_diff

        return max(0.0, min(10.0, score))

    def _determine_court_advantage(
        self,
        position_matchups: list[PositionMatchup],
        team1: str,
        team2: str
    ) -> str:
        """Determine which team has advantage in frontcourt or backcourt."""

        team1_advantages = sum(1 for pm in position_matchups if pm.height_advantage > 0.5)
        team2_advantages = sum(1 for pm in position_matchups if pm.height_advantage < -0.5)

        if team1_advantages > team2_advantages:
            return team1
        elif team2_advantages > team1_advantages:
            return team2
        else:
            return "neutral"

    def _predict_rebounding_battle(
        self,
        team1: SizeProfile,
        team2: SizeProfile,
        team1_name: str,
        team2_name: str
    ) -> str:
        """Predict rebounding battle outcome based on size."""

        # Weight frontcourt height more heavily
        team1_rebound_score = (team1.eff_height * 0.6 +
                               (team1.pf_height + team1.c_height) / 2 * 0.4)
        team2_rebound_score = (team2.eff_height * 0.6 +
                               (team2.pf_height + team2.c_height) / 2 * 0.4)

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
        self,
        team1: SizeProfile,
        team2: SizeProfile,
        team1_name: str,
        team2_name: str
    ) -> str:
        """Predict interior scoring based on size matchup."""

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
        self,
        team1: SizeProfile,
        team2: SizeProfile,
        team1_name: str,
        team2_name: str
    ) -> str:
        """Generate strategic recommendation based on size matchup."""

        eff_diff = team1.eff_height - team2.eff_height

        if eff_diff > 2.0:
            return (f"{team1_name} should play big: Slow tempo, pound the paint, "
                   f"crash the offensive glass, protect the rim")
        elif eff_diff < -2.0:
            return (f"{team2_name} should play big: Slow tempo, pound the paint, "
                   f"crash the offensive glass, protect the rim")
        elif eff_diff > 1.0:
            return f"{team1_name} should leverage size advantage selectively in half-court sets"
        elif eff_diff < -1.0:
            return f"{team2_name} should leverage size advantage selectively in half-court sets"
        else:
            # Size is even - look for specific position mismatches
            if abs(team1.pg_height - team2.pg_height) > 2.0:
                taller_pg = team1_name if team1.pg_height > team2.pg_height else team2_name
                return f"Even overall size, but {taller_pg} should exploit PG height mismatch"
            else:
                return "Size is even - exploit speed, skill, and execution advantages instead"

    def correlate_height_rebounding(
        self,
        team: str,
        season: int,
        or_pct: float,
        dr_pct: float
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
        or_efficiency = (or_pct - expected_or) / expected_or * 100  # % above/below expected
        dr_efficiency = (dr_pct - expected_dr) / expected_dr * 100

        # Combined score
        height_rebounding_score = (or_efficiency + dr_efficiency) / 2

        # Classify efficiency
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
            rebounding_efficiency=efficiency
        )
```

---

## Module 2: Experience & Chemistry Analyzer

### File Structure
```
src/kenp0m_sp0rts_analyzer/
├── experience_chemistry_analysis.py  (NEW)

examples/
├── experience_matchup_demo.py         (NEW)

tests/
├── test_experience_chemistry.py       (NEW)
```

### Implementation Details

#### Data Classes
```python
@dataclass
class ExperienceProfile:
    """Team's experience and chemistry profile."""
    team_name: str
    season: int

    # Experience
    experience_rating: float    # 0-4 scale (Fr=0, Sr=3, 5th=4)
    experience_rank: int
    experience_level: Literal["very_experienced", "experienced", "average", "young", "very_young"]

    # Bench depth
    bench_strength: float
    bench_rank: int
    bench_classification: Literal["elite_depth", "good_depth", "average_depth", "thin_bench"]

    # Team continuity
    continuity: float          # % of minutes returning
    continuity_rank: int
    continuity_level: Literal["high_continuity", "moderate_continuity", "low_continuity", "rebuild"]

    # Overall intangibles score
    intangibles_score: float   # 0-10 composite of exp + continuity + bench
    primary_strength: str
    primary_weakness: str

@dataclass
class ExperienceMatchup:
    """Experience and chemistry matchup analysis."""
    team1_profile: ExperienceProfile
    team2_profile: ExperienceProfile

    # Experience advantage
    experience_advantage: str        # Which team has experience edge
    experience_gap: float            # Rating difference
    experience_impact: str           # Expected impact description

    # Bench depth advantage
    bench_advantage: str             # Which team has deeper bench
    bench_gap: float
    bench_impact: str

    # Continuity advantage
    continuity_advantage: str
    continuity_gap: float
    continuity_impact: str

    # Overall intangibles
    better_intangibles: str
    intangibles_advantage_score: float  # 0-10 scale

    # Situational predictions
    late_game_execution: str         # Who has edge in close games
    tournament_readiness: str        # March Madness preparation
    adverse_conditions: str          # How teams handle adversity

    # Strategic insights
    team1_keys: list[str]
    team2_keys: list[str]
    matchup_recommendation: str

@dataclass
class TournamentReadiness:
    """Assess tournament readiness based on experience factors."""
    team_name: str
    season: int

    # Tournament-critical factors
    experience_score: float          # How experienced for tournament
    late_game_poise: float           # Close game execution ability
    depth_for_neutral_site: float    # Bench ready for tournament grind

    # Overall readiness
    tournament_readiness_score: float  # 0-10 composite
    readiness_tier: Literal["elite", "strong", "average", "questionable"]
    biggest_concern: str
```

#### Main Analyzer Class
```python
class ExperienceChemistryAnalyzer:
    """Analyze experience, bench depth, and team chemistry."""

    # Experience thresholds (0-4 scale)
    VERY_EXPERIENCED = 2.5    # >2.5 = mostly juniors/seniors
    EXPERIENCED = 2.0         # >2.0 = experienced
    YOUNG = 1.5               # <1.5 = mostly freshmen/sophomores
    VERY_YOUNG = 1.0          # <1.0 = very young team

    # Continuity thresholds (percentage)
    HIGH_CONTINUITY = 70.0    # >70% minutes returning
    MODERATE_CONTINUITY = 50.0  # >50% minutes returning
    LOW_CONTINUITY = 30.0     # >30% minutes returning (rebuild if less)

    # Bench strength thresholds
    ELITE_DEPTH = 3.0         # Bench rating >3.0
    GOOD_DEPTH = 1.0          # Bench rating >1.0
    AVERAGE_DEPTH = -1.0      # Bench rating >-1.0

    def __init__(self, api_client: KenPomAPI):
        self.api = api_client

    def get_experience_profile(self, team: str, season: int) -> ExperienceProfile:
        """Generate experience and chemistry profile."""

        data = self.api.get_height(year=season)
        team_data = next((t for t in data.data if t['TeamName'] == team), None)

        if not team_data:
            raise ValueError(f"Team '{team}' not found")

        # Classify experience level
        exp = team_data['Exp']
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
        continuity = team_data['Continuity']
        if continuity >= self.HIGH_CONTINUITY:
            cont_level = "high_continuity"
        elif continuity >= self.MODERATE_CONTINUITY:
            cont_level = "moderate_continuity"
        elif continuity >= self.LOW_CONTINUITY:
            cont_level = "low_continuity"
        else:
            cont_level = "rebuild"

        # Classify bench depth
        bench = team_data['Bench']
        if bench >= self.ELITE_DEPTH:
            bench_class = "elite_depth"
        elif bench >= self.GOOD_DEPTH:
            bench_class = "good_depth"
        elif bench >= self.AVERAGE_DEPTH:
            bench_class = "average_depth"
        else:
            bench_class = "thin_bench"

        # Calculate composite intangibles score
        # Normalize to 0-10 scale
        exp_score = (exp / 4.0) * 10.0         # Max 10 points
        cont_score = (continuity / 100.0) * 10.0  # Max 10 points
        bench_score = ((bench + 5.0) / 10.0) * 10.0  # Normalize bench (-5 to +5 → 0 to 10)

        intangibles = (exp_score * 0.4 + cont_score * 0.3 + bench_score * 0.3)

        # Identify strength/weakness
        scores = [
            (exp_score, "Experience", "Inexperience"),
            (cont_score, "Team chemistry/continuity", "Lack of continuity"),
            (bench_score, "Bench depth", "Thin bench")
        ]
        strength = max(scores, key=lambda x: x[0])[1]
        weakness = min(scores, key=lambda x: x[0])[2]

        return ExperienceProfile(
            team_name=team,
            season=season,
            experience_rating=exp,
            experience_rank=team_data['ExpRank'],
            experience_level=exp_level,
            bench_strength=bench,
            bench_rank=team_data['BenchRank'],
            bench_classification=bench_class,
            continuity=continuity,
            continuity_rank=team_data['RankContinuity'],
            continuity_level=cont_level,
            intangibles_score=round(intangibles, 2),
            primary_strength=strength,
            primary_weakness=weakness
        )

    def analyze_matchup(
        self,
        team1: str,
        team2: str,
        season: int
    ) -> ExperienceMatchup:
        """Analyze experience and chemistry matchup."""

        profile1 = self.get_experience_profile(team1, season)
        profile2 = self.get_experience_profile(team2, season)

        # Experience advantage
        exp_gap = profile1.experience_rating - profile2.experience_rating
        exp_adv = team1 if exp_gap > 0.3 else (team2 if exp_gap < -0.3 else "neutral")
        exp_impact = self._predict_experience_impact(exp_gap, team1, team2)

        # Bench advantage
        bench_gap = profile1.bench_strength - profile2.bench_strength
        bench_adv = team1 if bench_gap > 1.0 else (team2 if bench_gap < -1.0 else "neutral")
        bench_impact = self._predict_bench_impact(bench_gap, team1, team2)

        # Continuity advantage
        cont_gap = profile1.continuity - profile2.continuity
        cont_adv = team1 if cont_gap > 15.0 else (team2 if cont_gap < -15.0 else "neutral")
        cont_impact = self._predict_continuity_impact(cont_gap, team1, team2)

        # Overall intangibles
        intangibles_score = self._calculate_intangibles_advantage(profile1, profile2)
        better_intangibles = team1 if intangibles_score > 5.5 else (
            team2 if intangibles_score < 4.5 else "neutral"
        )

        # Situational predictions
        late_game = self._predict_late_game_execution(profile1, profile2, team1, team2)
        tournament = self._predict_tournament_readiness(profile1, profile2, team1, team2)
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
            matchup_recommendation=recommendation
        )

    def assess_tournament_readiness(
        self,
        team: str,
        season: int
    ) -> TournamentReadiness:
        """Assess team's readiness for NCAA Tournament.

        Tournament success correlates with:
        1. Experience (veterans handle pressure better)
        2. Bench depth (6 games in 3 weeks)
        3. Late-game poise (tight tournament games)
        """

        profile = self.get_experience_profile(team, season)

        # Experience score (0-10)
        # Very experienced teams (~3.0+) excel in March
        exp_score = min(profile.experience_rating / 3.0 * 10.0, 10.0)

        # Late-game poise (experience × continuity)
        # Teams with experienced returners handle late-game situations better
        late_game = (exp_score * 0.6 + (profile.continuity / 100.0) * 10.0 * 0.4)

        # Depth for neutral site (bench strength matters in tournament)
        depth = ((profile.bench_strength + 5.0) / 10.0) * 10.0

        # Overall readiness (weighted composite)
        readiness = (exp_score * 0.4 + late_game * 0.3 + depth * 0.3)

        # Classify readiness tier
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
            concern = f"Major concerns: {profile.primary_weakness}, limited March experience"

        return TournamentReadiness(
            team_name=team,
            season=season,
            experience_score=round(exp_score, 2),
            late_game_poise=round(late_game, 2),
            depth_for_neutral_site=round(depth, 2),
            tournament_readiness_score=round(readiness, 2),
            readiness_tier=tier,
            biggest_concern=concern
        )

    # Helper methods (similar structure to TIER 1)
    def _predict_experience_impact(
        self, gap: float, team1: str, team2: str
    ) -> str:
        """Predict impact of experience gap."""
        if abs(gap) < 0.3:
            return "Experience is relatively even"

        more_exp = team1 if gap > 0 else team2
        if abs(gap) > 1.0:
            return f"{more_exp} has major experience advantage - expect better late-game execution"
        else:
            return f"{more_exp} has experience edge - slight advantage in close games"

    def _predict_bench_impact(
        self, gap: float, team1: str, team2: str
    ) -> str:
        """Predict impact of bench depth gap."""
        if abs(gap) < 1.0:
            return "Bench depth is similar"

        deeper = team1 if gap > 0 else team2
        if abs(gap) > 2.5:
            return f"{deeper} has significant depth advantage - can sustain performance over 40 minutes"
        else:
            return f"{deeper} has bench edge - advantage in foul trouble or fatigue situations"

    def _predict_continuity_impact(
        self, gap: float, team1: str, team2: str
    ) -> str:
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
        self, p1: ExperienceProfile, p2: ExperienceProfile, t1: str, t2: str
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
        self, p1: ExperienceProfile, p2: ExperienceProfile, t1: str, t2: str
    ) -> str:
        """Predict which team is better prepared for tournament."""
        t1_readiness = self.assess_tournament_readiness(t1, p1.season)
        t2_readiness = self.assess_tournament_readiness(t2, p2.season)

        diff = t1_readiness.tournament_readiness_score - t2_readiness.tournament_readiness_score

        if diff > 1.5:
            return f"{t1} ({t1_readiness.readiness_tier}) more tournament-ready than {t2} ({t2_readiness.readiness_tier})"
        elif diff < -1.5:
            return f"{t2} ({t2_readiness.readiness_tier}) more tournament-ready than {t1} ({t1_readiness.readiness_tier})"
        else:
            return f"Both teams similarly prepared for tournament ({t1_readiness.readiness_tier}/{t2_readiness.readiness_tier})"

    def _predict_adversity_handling(
        self, p1: ExperienceProfile, p2: ExperienceProfile, t1: str, t2: str
    ) -> str:
        """Predict how teams handle adversity (deficits, foul trouble, etc.)."""
        # Experience + bench depth = adversity handling
        p1_adversity = p1.experience_rating * 0.5 + ((p1.bench_strength + 5.0) / 10.0) * 4.0 * 0.5
        p2_adversity = p2.experience_rating * 0.5 + ((p2.bench_strength + 5.0) / 10.0) * 4.0 * 0.5

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
            keys.append(f"Leverage veteran poise in late-game situations")

        if team.bench_classification in ["elite_depth", "good_depth"]:
            keys.append("Use depth advantage - play fast, substitute freely")

        if team.continuity_level == "high_continuity":
            keys.append("Execute complex sets - team chemistry allows sophisticated offense")

        # Exploit opponent weaknesses
        if opponent.experience_level in ["young", "very_young"]:
            keys.append(f"Apply pressure - force {opponent.team_name} into mistakes")

        if opponent.bench_classification == "thin_bench":
            keys.append(f"Attack {opponent.team_name} starters - force foul trouble")

        return keys if keys else ["Play fundamental basketball - experience edge is minimal"]

    def _generate_matchup_recommendation(
        self,
        team1: ExperienceProfile,
        team2: ExperienceProfile,
        advantage_score: float
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
```

---

## Integration with Comprehensive Matchup Analysis

### Update `comprehensive_matchup_demo.py`

Add TIER 2 analyzers to the comprehensive demo:

```python
# Initialize all analyzers (TIER 1 + TIER 2)
api = KenPomAPI()
four_factors = FourFactorsMatchup(api)
point_dist = PointDistributionAnalyzer(api)
defensive = DefensiveAnalyzer(api)
size_athleticism = SizeAthleticismAnalyzer(api)  # NEW
experience = ExperienceChemistryAnalyzer(api)    # NEW

# Run all analyses
ff_analysis = four_factors.analyze_matchup(team1, team2, 2025)
pd_analysis = point_dist.analyze_matchup(team1, team2, 2025)
def_analysis = defensive.analyze_matchup(team1, team2, 2025)
size_analysis = size_athleticism.analyze_matchup(team1, team2, 2025)  # NEW
exp_analysis = experience.analyze_matchup(team1, team2, 2025)         # NEW

# Add new sections to output:
# [6] SIZE & ATHLETICISM ANALYSIS
# [7] EXPERIENCE & INTANGIBLES ANALYSIS
# [8] COMPREHENSIVE SUMMARY (all 7 dimensions)
```

---

## Success Metrics

### Quantitative
- ✅ API data successfully retrieved from `height` endpoint
- ✅ All classification logic covered by tests (>80% coverage)
- ✅ Demo scripts run without errors
- ✅ Analysis completes in <2 seconds per matchup

### Qualitative
- ✅ Size matchup insights align with game observations
- ✅ Experience factors capture intangible advantages
- ✅ Tournament readiness predictions match historical patterns
- ✅ Integrates seamlessly with TIER 1 modules

---

## Timeline

### Session 1 (2-3 hours): Size & Athleticism Analyzer
1. Create `size_athleticism_analysis.py` with dataclasses and analyzer
2. Create `size_matchup_demo.py` showing position-specific analysis
3. Test with real data, validate size classifications
4. Update `__init__.py` exports

### Session 2 (2-3 hours): Experience & Chemistry Analyzer
1. Create `experience_chemistry_analysis.py` with dataclasses and analyzer
2. Create `experience_matchup_demo.py` showing intangibles analysis
3. Test tournament readiness assessments
4. Update `__init__.py` exports

### Session 3 (1-2 hours): Integration & Testing
1. Update `comprehensive_matchup_demo.py` with TIER 2 modules
2. Add sections for size and experience analysis
3. Create 15-dimensional battle analysis (10 from TIER 1 + 5 from TIER 2)
4. Write unit tests for both modules
5. Update ANALYTICS_ROADMAP.md with completion status
6. Commit and push all changes

---

## Key Insights & Strategic Value

### Size & Athleticism
- **Rebounding Prediction**: Height directly correlates with rebounding percentage
- **Paint Scoring**: Interior size determines post-up and rim-finishing opportunities
- **Defensive Versatility**: Size allows switching and positional flexibility
- **Strategic Tempo**: Bigger teams often prefer slower pace to maximize size advantage

### Experience & Chemistry
- **Tournament Success**: Experienced teams significantly outperform in March Madness
- **Late-Game Execution**: Veterans handle pressure situations better
- **Bench Depth**: Critical in tournament (6 games in 3 weeks)
- **Continuity**: Teams with returning players have chemistry and system familiarity

### Combined TIER 1 + TIER 2 Analysis

**15-Dimensional Matchup Framework**:
1-4: Four Factors (eFG%, TO%, OR%, FT Rate)
5-7: Scoring Styles (3pt, 2pt, FT distribution)
8-10: Defensive Schemes (perimeter, interior, pressure)
11: Overall Size (effective height)
12-15: Intangibles (experience, bench, continuity, late-game)

This comprehensive framework captures:
- **Statistical edges** (Four Factors, scoring)
- **Scheme matchups** (defense, size)
- **Intangible factors** (experience, chemistry)

---

## Next Steps

**Immediate**: Start with Size & Athleticism Analyzer (clearer metrics, easier to validate)
**Following**: Experience & Chemistry Analyzer (more subjective, requires contextual knowledge)
**Integration**: Update comprehensive matchup demo with all 7 analyzers
**Future**: Tempo/Pace Analyzer (TIER 3), Home Court Advantage (TIER 3)
