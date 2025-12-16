# TIER 1 Implementation Plan: Point Distribution & Defensive Analysis

**High-Impact Analytics Modules to Complement Four Factors Matchup**

---

## Overview

This plan details implementation of two high-value analyzer modules that leverage existing API endpoints:

1. **Point Distribution Analyzer** - Scoring style matchups and defensive vulnerabilities
2. **Defensive Analyzer** - Advanced defensive scheme identification and matchup analysis

**Effort**: ~4-6 hours total (2-3 hours each module)
**Impact**: High - Direct enhancement to matchup analysis and predictions
**Dependencies**: Existing API client with `pointdist` and `misc-stats` endpoints

---

## Module 1: Point Distribution Analyzer

### File Structure
```
src/kenp0m_sp0rts_analyzer/
├── point_distribution_analysis.py  (NEW)

examples/
├── point_distribution_demo.py       (NEW)

tests/
├── test_point_distribution.py       (NEW)
```

### Implementation Details

#### Data Classes
```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class ScoringStyleProfile:
    """Team's scoring style breakdown."""
    team_name: str
    season: int

    # Offensive distribution
    ft_pct: float        # % of points from free throws
    fg2_pct: float       # % of points from 2-point FGs
    fg3_pct: float       # % of points from 3-point FGs

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
    """Matchup analysis between two scoring styles."""
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
```

#### Main Analyzer Class
```python
class PointDistributionAnalyzer:
    """Analyze scoring styles and identify matchup advantages."""

    # Thresholds for style classification
    PERIMETER_THRESHOLD = 40.0  # >40% from 3pt = perimeter team
    INTERIOR_THRESHOLD = 55.0   # >55% from 2pt = interior team

    def __init__(self, api_client: KenPomAPI):
        self.api = api_client

    def get_scoring_profile(self, team: str, season: int) -> ScoringStyleProfile:
        """Generate scoring style profile for a team."""

        # Fetch data
        data = self.api.get_point_distribution(year=season)
        team_data = next((t for t in data if t['TeamName'] == team), None)

        if not team_data:
            raise ValueError(f"Team '{team}' not found")

        # Classify style
        fg3_pct = team_data['OffFg3']
        fg2_pct = team_data['OffFg2']

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
        def_fg3 = team_data['DefFg3']
        def_fg2 = team_data['DefFg2']
        def_ft = team_data['DefFt']

        weaknesses = [
            (def_fg3, "three-point defense"),
            (def_fg2, "interior defense"),
            (def_ft, "free throw prevention")
        ]
        defensive_weakness = max(weaknesses, key=lambda x: x[0])[1]

        return ScoringStyleProfile(
            team_name=team,
            season=season,
            ft_pct=team_data['OffFt'],
            fg2_pct=fg2_pct,
            fg3_pct=fg3_pct,
            def_ft_pct=def_ft,
            def_fg2_pct=def_fg2,
            def_fg3_pct=def_fg3,
            ft_rank=team_data['RankOffFt'],
            fg2_rank=team_data['RankOffFg2'],
            fg3_rank=team_data['RankOffFg3'],
            style=style,
            primary_strength=strength,
            defensive_weakness=defensive_weakness
        )

    def analyze_matchup(
        self,
        team1: str,
        team2: str,
        season: int
    ) -> ScoringStyleMatchup:
        """Analyze scoring style matchup between two teams."""

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
            (abs(ft_adv), "free throw drawing", ft_adv > 0)
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
            three_point_advantage=three_pt_adv,
            two_point_advantage=two_pt_adv,
            free_throw_advantage=ft_adv,
            style_mismatch_score=mismatch_score,
            team1_exploitable_areas=team1_exploits,
            team2_exploitable_areas=team2_exploits,
            key_matchup_factor=key_matchup,
            recommended_strategy=strategy
        )

    def _calculate_style_mismatch(
        self,
        offense: ScoringStyleProfile,
        defense: ScoringStyleProfile
    ) -> float:
        """
        Calculate style mismatch score (0-10).

        Higher score = offense's strength aligns with defense's weakness.
        """
        # Find offense's strongest scoring method
        scoring_methods = [
            (offense.fg3_pct, defense.def_fg3_pct, "3pt"),
            (offense.fg2_pct, defense.def_fg2_pct, "2pt"),
            (offense.ft_pct, defense.def_ft_pct, "FT")
        ]

        # Score each matchup (offense strength vs defense weakness)
        scores = []
        for off_strength, def_weakness, method in scoring_methods:
            # Normalize to 0-10 scale
            # If offense excels (high %) and defense struggles (high % allowed)
            matchup_score = (off_strength / 10.0) + (def_weakness / 10.0)
            scores.append(matchup_score)

        # Return max mismatch score
        return min(max(scores), 10.0)

    def _identify_exploitable_areas(
        self,
        team: ScoringStyleProfile,
        opponent: ScoringStyleProfile
    ) -> list[str]:
        """Identify areas where team can exploit opponent's defense."""
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
        ft_adv: float
    ) -> str:
        """Generate strategic game plan recommendation."""
        # Find biggest advantage
        advantages = [
            (three_adv, "emphasize three-point shooting", team1.team_name),
            (two_adv, "attack the paint", team1.team_name),
            (ft_adv, "draw fouls and get to the line", team1.team_name),
            (-three_adv, "emphasize three-point shooting", team2.team_name),
            (-two_adv, "attack the paint", team2.team_name),
            (-ft_adv, "draw fouls and get to the line", team2.team_name)
        ]

        magnitude, strategy, team = max(advantages, key=lambda x: abs(x[0]))

        return f"{team} should {strategy} (advantage: {abs(magnitude):.1f}%)"
```

---

## Module 2: Defensive Analyzer

### File Structure
```
src/kenp0m_sp0rts_analyzer/
├── defensive_analysis.py           (NEW)

examples/
├── defensive_matchup_demo.py        (NEW)

tests/
├── test_defensive_analysis.py       (NEW)
```

### Implementation Details

#### Data Classes
```python
@dataclass
class DefensiveProfile:
    """Team's defensive identity and scheme."""
    team_name: str
    season: int

    # Perimeter defense
    opp_fg3_pct: float       # Opponent 3pt %
    opp_fg3_rank: int

    # Interior defense
    opp_fg2_pct: float       # Opponent 2pt %
    opp_fg2_rank: int
    block_pct: float         # Block percentage
    block_rank: int

    # Pressure defense
    stl_rate: float          # Steal rate
    stl_rank: int
    opp_nst_rate: float      # Opponent non-steal turnovers
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
    """Defensive matchup analysis."""
    team1_defense: DefensiveProfile
    team2_defense: DefensiveProfile

    # Defensive advantages
    perimeter_defense_advantage: str
    interior_defense_advantage: str
    pressure_defense_advantage: str

    # Overall defensive edge
    better_defense: str
    defensive_advantage_score: float  # 0-10 scale

    # Strategic insights
    team1_defensive_keys: list[str]
    team2_defensive_keys: list[str]
    matchup_recommendation: str
```

#### Main Analyzer Class
```python
class DefensiveAnalyzer:
    """Advanced defensive analysis and matchup evaluation."""

    # Classification thresholds
    HIGH_BLOCK_RATE = 10.0      # Top rim protection
    HIGH_STEAL_RATE = 9.0       # Aggressive pressure
    LOW_OPP_FG3 = 31.0          # Elite perimeter defense
    LOW_OPP_FG2 = 46.0          # Elite interior defense

    def __init__(self, api_client: KenPomAPI):
        self.api = api_client

    def get_defensive_profile(self, team: str, season: int) -> DefensiveProfile:
        """Generate comprehensive defensive profile."""

        data = self.api.get_misc_stats(year=season)
        team_data = next((t for t in data if t['TeamName'] == team), None)

        if not team_data:
            raise ValueError(f"Team '{team}' not found")

        # Extract defensive metrics
        opp_fg3 = team_data['OppFG3Pct']
        opp_fg2 = team_data['OppFG2Pct']
        blocks = team_data['BlockPct']
        steals = team_data['StlRate']

        # Classify defensive scheme
        scheme, strength, weakness = self._classify_defensive_scheme(
            opp_fg3, opp_fg2, blocks, steals
        )

        return DefensiveProfile(
            team_name=team,
            season=season,
            opp_fg3_pct=opp_fg3,
            opp_fg3_rank=team_data['RankOppFG3Pct'],
            opp_fg2_pct=opp_fg2,
            opp_fg2_rank=team_data['RankOppFG2Pct'],
            block_pct=blocks,
            block_rank=team_data['RankBlockPct'],
            stl_rate=steals,
            stl_rank=team_data['RankStlRate'],
            opp_nst_rate=team_data['OppNSTRate'],
            nst_rank=team_data['RankOppNSTRate'],
            opp_assist_rate=team_data['OppARate'],
            assist_rank=team_data['RankOppARate'],
            defensive_scheme=scheme,
            primary_strength=strength,
            primary_weakness=weakness
        )

    def _classify_defensive_scheme(
        self,
        opp_fg3: float,
        opp_fg2: float,
        blocks: float,
        steals: float
    ) -> tuple[str, str, str]:
        """Classify defensive scheme and identify strengths/weaknesses."""

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
        elif (opp_fg3 <= self.LOW_OPP_FG3 and
              opp_fg2 <= self.LOW_OPP_FG2 and
              steals >= 8.0):
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
                (10.0 - steals, "pressure defense")
            ]
            weakness = max(weaknesses, key=lambda x: x[0])[1]

        return scheme, strength, weakness

    def analyze_matchup(
        self,
        team1: str,
        team2: str,
        season: int
    ) -> DefensiveMatchup:
        """Analyze defensive matchup between two teams."""

        profile1 = self.get_defensive_profile(team1, season)
        profile2 = self.get_defensive_profile(team2, season)

        # Determine advantages
        perimeter_adv = (
            team1 if profile1.opp_fg3_pct < profile2.opp_fg3_pct
            else team2
        )

        interior_adv = (
            team1 if profile1.opp_fg2_pct < profile2.opp_fg2_pct
            else team2
        )

        pressure_adv = (
            team1 if profile1.stl_rate > profile2.stl_rate
            else team2
        )

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
            defensive_advantage_score=defense_score,
            team1_defensive_keys=team1_keys,
            team2_defensive_keys=team2_keys,
            matchup_recommendation=recommendation
        )

    def _calculate_defensive_advantage(
        self,
        team1: DefensiveProfile,
        team2: DefensiveProfile
    ) -> float:
        """Calculate overall defensive advantage (0-10 scale, 5 = neutral)."""

        # Compare each defensive dimension
        perimeter_diff = team2.opp_fg3_pct - team1.opp_fg3_pct
        interior_diff = team2.opp_fg2_pct - team1.opp_fg2_pct
        pressure_diff = team1.stl_rate - team2.stl_rate

        # Normalize to 0-10 scale
        # Positive differences favor team1
        score = 5.0  # Start neutral
        score += perimeter_diff * 0.5  # 3pt defense weight
        score += interior_diff * 0.5   # 2pt defense weight
        score += pressure_diff * 0.2   # Steal rate weight

        return max(0.0, min(10.0, score))

    def _generate_defensive_keys(
        self,
        team: DefensiveProfile,
        opponent: DefensiveProfile
    ) -> list[str]:
        """Generate defensive game plan keys."""
        keys = []

        # Leverage strengths
        if team.defensive_scheme == "rim_protection":
            keys.append(f"Protect the rim - force {opponent.team_name} outside")

        elif team.defensive_scheme == "pressure":
            keys.append(f"Apply full-court pressure - force turnovers")

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
            return "Defensive matchup is evenly matched"

        return (
            f"{better_team} has a {margin} defensive advantage "
            f"(score: {advantage_score:.1f}/10)"
        )
```

---

## Integration Plan

### 1. Update `__init__.py`
```python
from .point_distribution_analysis import (
    PointDistributionAnalyzer,
    ScoringStyleProfile,
    ScoringStyleMatchup,
)
from .defensive_analysis import (
    DefensiveAnalyzer,
    DefensiveProfile,
    DefensiveMatchup,
)

__all__.extend([
    "PointDistributionAnalyzer",
    "ScoringStyleProfile",
    "ScoringStyleMatchup",
    "DefensiveAnalyzer",
    "DefensiveProfile",
    "DefensiveMatchup",
])
```

### 2. Enhance `FourFactorsMatchup`
Add complementary analysis:
```python
class FourFactorsMatchup:
    def analyze_comprehensive_matchup(
        self,
        team1: str,
        team2: str,
        season: int
    ) -> dict:
        """Complete matchup analysis with all dimensions."""

        # Four Factors
        four_factors = self.analyze_matchup(team1, team2, season)

        # Point Distribution
        point_dist_analyzer = PointDistributionAnalyzer(self.api)
        scoring_styles = point_dist_analyzer.analyze_matchup(team1, team2, season)

        # Defensive Analysis
        def_analyzer = DefensiveAnalyzer(self.api)
        defensive = def_analyzer.analyze_matchup(team1, team2, season)

        return {
            'four_factors': four_factors,
            'scoring_styles': scoring_styles,
            'defensive': defensive
        }
```

### 3. Create Comprehensive Demo
**File**: `examples/comprehensive_matchup_demo.py`

```python
"""Comprehensive matchup analysis demo."""

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.four_factors_matchup import FourFactorsMatchup
from kenp0m_sp0rts_analyzer.point_distribution_analysis import PointDistributionAnalyzer
from kenp0m_sp0rts_analyzer.defensive_analysis import DefensiveAnalyzer

api = KenPomAPI()

matchups = [
    ("Duke", "North Carolina"),
    ("Purdue", "Arizona"),
]

for team1, team2 in matchups:
    print(f"\n{'=' * 80}")
    print(f"{team1} vs {team2} - COMPREHENSIVE MATCHUP ANALYSIS")
    print(f"{'=' * 80}\n")

    # Four Factors
    ff = FourFactorsMatchup(api)
    ff_analysis = ff.analyze_matchup(team1, team2, 2025)
    print(f"[FOUR FACTORS]")
    print(f"Overall Advantage: {ff_analysis.overall_advantage}")
    print(f"Key Factor: {ff_analysis.most_important_factor}\n")

    # Scoring Styles
    pd = PointDistributionAnalyzer(api)
    scoring = pd.analyze_matchup(team1, team2, 2025)
    print(f"[SCORING STYLES]")
    print(f"{team1}: {scoring.team1_profile.style.upper()} offense")
    print(f"{team2}: {scoring.team2_profile.style.upper()} offense")
    print(f"Strategy: {scoring.recommended_strategy}\n")

    # Defense
    da = DefensiveAnalyzer(api)
    defense = da.analyze_matchup(team1, team2, 2025)
    print(f"[DEFENSIVE MATCHUP]")
    print(f"Better Defense: {defense.better_defense}")
    print(f"Defensive Advantage: {defense.defensive_advantage_score:.1f}/10")
    print(f"Recommendation: {defense.matchup_recommendation}")
```

---

## Testing Strategy

### Unit Tests
```python
# tests/test_point_distribution.py
def test_scoring_profile_creation():
    """Test scoring profile generation."""

def test_style_classification():
    """Test perimeter/balanced/interior classification."""

def test_matchup_analysis():
    """Test scoring style matchup analysis."""

# tests/test_defensive_analysis.py
def test_defensive_profile_creation():
    """Test defensive profile generation."""

def test_scheme_classification():
    """Test rim_protection/pressure/balanced/versatile classification."""

def test_defensive_matchup():
    """Test defensive matchup analysis."""
```

### Integration Tests
```python
def test_comprehensive_matchup():
    """Test combining Four Factors + Scoring + Defense."""
```

---

## Success Metrics

### Quantitative
- ✅ API data successfully retrieved and parsed
- ✅ All classification logic covered by tests (>80% coverage)
- ✅ Demo scripts run without errors
- ✅ Analysis completes in <2 seconds per matchup

### Qualitative
- ✅ Strategic insights are actionable and clear
- ✅ Classifications match intuitive understanding of teams
- ✅ Recommendations align with basketball strategy principles
- ✅ Integrates seamlessly with existing `FourFactorsMatchup`

---

## Timeline

### Session 1 (2-3 hours): Point Distribution Analyzer
1. Create `point_distribution_analysis.py` with dataclasses and analyzer
2. Create `point_distribution_demo.py` showing 3-4 matchups
3. Test with real data, validate classifications
4. Update `__init__.py` exports

### Session 2 (2-3 hours): Defensive Analyzer
1. Create `defensive_analysis.py` with dataclasses and analyzer
2. Create `defensive_matchup_demo.py` showing scheme identification
3. Test with real data, validate defensive classifications
4. Update `__init__.py` exports

### Session 3 (1-2 hours): Integration
1. Create `comprehensive_matchup_demo.py` combining all analyzers
2. Update ANALYTICS_ROADMAP.md with completion status
3. Write unit tests for both modules
4. Commit and push all changes

---

## Next Steps

**Immediate**: Start with Point Distribution Analyzer (simpler, clear use case)
**Following**: Defensive Analyzer (builds on similar patterns)
**Integration**: Comprehensive matchup analysis combining all dimensions
**Future**: Height/Experience Analyzer (TIER 2)
