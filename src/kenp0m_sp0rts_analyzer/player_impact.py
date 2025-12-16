"""Player impact modeling for injury analysis and player valuation.

This module provides tools to:
1. Calculate individual player value and contribution to team performance
2. Estimate the impact of player injuries on team ratings
3. Create depth charts with player valuations
4. Adjust game predictions for injured players

The model uses player efficiency metrics (ORtg, eFG%, TS%) combined with
usage rates (Poss%) to estimate how much each player contributes to their
team's Adjusted Efficiency Margin (AdjEM).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class PlayerValue:
    """Player's estimated value and impact on team performance.

    Attributes:
        player_name: Player's full name
        team: Team name
        position: Position (if available)
        possession_pct: Percentage of team possessions used when on floor
        minutes_pct: Estimated percentage of available minutes played
        offensive_rating: Individual offensive rating (points per 100 poss)
        effective_fg_pct: Effective field goal percentage
        true_shooting_pct: True shooting percentage
        estimated_value: Estimated AdjEM points contributed to team
        offensive_contribution: Points contributed to offensive efficiency
        defensive_contribution: Points contributed to defensive efficiency
        replacement_level: Expected value of replacement player
        value_over_replacement: Value above replacement level (VOR)
    """

    player_name: str
    team: str
    position: str
    possession_pct: float
    minutes_pct: float
    offensive_rating: float
    effective_fg_pct: float
    true_shooting_pct: float
    estimated_value: float
    offensive_contribution: float
    defensive_contribution: float
    replacement_level: float
    value_over_replacement: float


@dataclass
class InjuryImpact:
    """Estimated impact of player absence on team performance.

    Attributes:
        player: PlayerValue object for injured player
        team_adj_em_baseline: Team's baseline AdjEM (healthy)
        estimated_adj_em_loss: Estimated loss in AdjEM points
        confidence_interval: (lower, upper) bounds for AdjEM with injury
        adjusted_adj_em: Team AdjEM adjusted for injury
        adjusted_adj_oe: Team AdjOE adjusted for injury
        adjusted_adj_de: Team AdjDE adjusted for injury
        severity: Classification of injury impact
    """

    player: PlayerValue
    team_adj_em_baseline: float
    estimated_adj_em_loss: float
    confidence_interval: tuple[float, float]
    adjusted_adj_em: float
    adjusted_adj_oe: float
    adjusted_adj_de: float
    severity: str  # "Minor", "Moderate", "Major", "Devastating"


class PlayerImpactModel:
    """Model for calculating player value and injury impact.

    This model estimates how much each player contributes to their team's
    performance and how the team's ratings would change if that player
    were absent due to injury.

    Example:
        >>> from kenp0m_sp0rts_analyzer.client import KenPomClient
        >>> from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
        >>>
        >>> # Get player and team data
        >>> client = KenPomClient()
        >>> api = KenPomAPI()
        >>> player_stats_df = client.get_playerstats(season=2025)
        >>> team_ratings = api.get_ratings(year=2025)
        >>>
        >>> # Calculate player value
        >>> model = PlayerImpactModel()
        >>> duke_players = player_stats_df[player_stats_df["Team"] == "Duke"]
        >>> duke_stats = next(r for r in team_ratings.data if r["TeamName"] == "Duke")
        >>>
        >>> star_player = duke_players.iloc[0]
        >>> value = model.calculate_player_value(
        ...     star_player.to_dict(), duke_stats
        ... )
        >>> print(f"{value.player_name}: {value.estimated_value:.2f} AdjEM pts")
        >>>
        >>> # Estimate injury impact
        >>> injury = model.estimate_injury_impact(
        ...     value, duke_stats, injury_severity="out"
        ... )
        >>> print(f"Impact: {injury.estimated_adj_em_loss:.2f} pts ({injury.severity})")
    """

    # Constants for value calculation
    MINUTES_PCT_MULTIPLIER = 1.2  # Estimate minutes from possession %
    REPLACEMENT_VALUE_PCT = 0.70  # Replacement player is 70% as good
    DEFENSIVE_WEIGHT = 0.5  # Defensive impact is harder to isolate

    # Constants for uncertainty
    VALUE_ESTIMATION_ERROR = 0.20  # ±20% uncertainty in value estimate
    REPLACEMENT_VARIABILITY = 0.30  # ±30% uncertainty in replacement
    TOTAL_UNCERTAINTY = 0.35  # Combined uncertainty factor

    # Severity thresholds (AdjEM points lost)
    SEVERITY_MINOR = 1.0
    SEVERITY_MODERATE = 3.0
    SEVERITY_MAJOR = 6.0

    def __init__(self) -> None:
        """Initialize the player impact model."""
        self.player_values: dict[str, PlayerValue] = {}

    def calculate_player_value(
        self,
        player_stats: dict[str, Any],
        team_stats: dict[str, Any],
    ) -> PlayerValue:
        """Calculate player's estimated value to team.

        The value estimation uses:
        1. Usage: possession % and estimated minutes %
        2. Efficiency: ORtg relative to team, eFG%, TS%
        3. Offensive contribution: How much player adds to team offense
        4. Defensive contribution: Estimated from rebounds, steals, blocks
        5. Total value: Combined offensive and defensive contributions
        6. Replacement level: Expected backup performance (70% of value)

        Args:
            player_stats: Player statistics dictionary with keys:
                - Player: Player name
                - Team: Team name
                - Poss%: Possession percentage
                - ORtg: Offensive rating
                - eFG%: Effective FG percentage
                - TS%: True shooting percentage
                - TRB%: Total rebound percentage
                - STL%: Steal percentage (optional)
                - BLK%: Block percentage (optional)
            team_stats: Team statistics dictionary with keys:
                - TeamName: Team name
                - AdjOE: Adjusted offensive efficiency
                - AdjDE: Adjusted defensive efficiency

        Returns:
            PlayerValue object with estimated contributions
        """
        # Extract player metrics
        player_name = player_stats["Player"]
        team = player_stats["Team"]
        poss_pct = float(player_stats["Poss%"])

        # Estimate minutes percentage from possession usage
        # High usage players typically play more minutes
        minutes_pct = min(100.0, poss_pct * self.MINUTES_PCT_MULTIPLIER)

        # Efficiency metrics
        ortg = float(player_stats["ORtg"])
        efg = float(player_stats["eFG%"])
        ts = float(player_stats["TS%"])

        # Team context
        team_oe = float(team_stats["AdjOE"])

        # Offensive contribution
        # How much player's ORtg exceeds team average, weighted by usage
        oe_above_team = (ortg - team_oe) * (poss_pct / 100.0)

        # Defensive contribution (estimated from box score stats)
        # Rebounds, steals, and blocks are proxies for defensive impact
        trb_pct = float(player_stats.get("TRB%", 0))
        stl_pct = float(player_stats.get("STL%", 0))
        blk_pct = float(player_stats.get("BLK%", 0))

        # Weight steals and blocks more heavily (direct defensive plays)
        defensive_impact = (trb_pct + stl_pct * 2 + blk_pct * 2) / 10.0

        # Total estimated value (AdjEM points)
        estimated_value = oe_above_team + defensive_impact

        # Replacement level (backup player is ~70% as efficient)
        replacement_level = estimated_value * self.REPLACEMENT_VALUE_PCT
        vor = estimated_value - replacement_level

        return PlayerValue(
            player_name=player_name,
            team=team,
            position="",  # Position not in kenpompy data
            possession_pct=poss_pct,
            minutes_pct=minutes_pct,
            offensive_rating=ortg,
            effective_fg_pct=efg,
            true_shooting_pct=ts,
            estimated_value=estimated_value,
            offensive_contribution=oe_above_team,
            defensive_contribution=defensive_impact,
            replacement_level=replacement_level,
            value_over_replacement=vor,
        )

    def estimate_injury_impact(
        self,
        player: PlayerValue,
        team_stats: dict[str, Any],
        injury_severity: str = "out",
    ) -> InjuryImpact:
        """Estimate impact of player absence on team ratings.

        The injury impact model adjusts team ratings based on:
        1. Player's value over replacement (VOR)
        2. Injury severity multiplier:
           - "questionable": 50% of value lost
           - "doubtful": 75% of value lost
           - "out": 100% of value lost (full replacement)
        3. Confidence intervals accounting for uncertainty

        Args:
            player: PlayerValue object for injured player
            team_stats: Team's current statistics with keys:
                - AdjEM: Adjusted efficiency margin
                - AdjOE: Adjusted offensive efficiency
                - AdjDE: Adjusted defensive efficiency
            injury_severity: One of "questionable", "doubtful", "out"

        Returns:
            InjuryImpact with adjusted team ratings and confidence intervals

        Raises:
            ValueError: If injury_severity is not recognized
        """
        # Current team ratings
        team_adj_em = float(team_stats["AdjEM"])
        team_adj_oe = float(team_stats["AdjOE"])
        team_adj_de = float(team_stats["AdjDE"])

        # Value multiplier based on injury severity
        severity_multipliers = {
            "out": 1.0,
            "doubtful": 0.75,
            "questionable": 0.5,
        }

        if injury_severity not in severity_multipliers:
            raise ValueError(
                f"Invalid injury_severity '{injury_severity}'. "
                f"Must be one of: {list(severity_multipliers.keys())}"
            )

        value_multiplier = severity_multipliers[injury_severity]

        # Value over replacement is what we actually lose
        # (Starter's value - Backup's value) * severity multiplier
        value_lost = player.value_over_replacement * value_multiplier

        # Adjust team ratings
        adjusted_em = team_adj_em - value_lost
        adjusted_oe = team_adj_oe - (player.offensive_contribution * value_multiplier)

        # Defensive impact is smaller (individual defense matters less)
        adjusted_de = team_adj_de + (
            player.defensive_contribution * value_multiplier * self.DEFENSIVE_WEIGHT
        )

        # Confidence interval
        # Uncertainty comes from:
        # 1. Player value estimation error (±20%)
        # 2. Replacement player variability (±30%)
        # Combined: ~35% uncertainty
        uncertainty = abs(value_lost) * self.TOTAL_UNCERTAINTY
        ci_lower = adjusted_em - uncertainty
        ci_upper = adjusted_em + uncertainty

        # Classify severity based on AdjEM points lost
        if abs(value_lost) < self.SEVERITY_MINOR:
            severity_class = "Minor"
        elif abs(value_lost) < self.SEVERITY_MODERATE:
            severity_class = "Moderate"
        elif abs(value_lost) < self.SEVERITY_MAJOR:
            severity_class = "Major"
        else:
            severity_class = "Devastating"

        return InjuryImpact(
            player=player,
            team_adj_em_baseline=team_adj_em,
            estimated_adj_em_loss=value_lost,
            confidence_interval=(ci_lower, ci_upper),
            adjusted_adj_em=adjusted_em,
            adjusted_adj_oe=adjusted_oe,
            adjusted_adj_de=adjusted_de,
            severity=severity_class,
        )

    def get_team_depth_chart(
        self,
        team_name: str,
        player_stats_df: pd.DataFrame,
        team_stats: dict[str, Any],
    ) -> pd.DataFrame:
        """Create depth chart with player values ranked by contribution.

        Args:
            team_name: Team to analyze
            player_stats_df: DataFrame from client.get_playerstats()
            team_stats: Team statistics dictionary

        Returns:
            DataFrame with players sorted by estimated value, including:
            - Player: Player name
            - yr: Year (Fr, So, Jr, Sr)
            - ht: Height
            - Poss%: Possession percentage
            - ORtg: Offensive rating
            - eFG%: Effective FG percentage
            - EstimatedValue: Calculated value in AdjEM points
            - VOR: Value over replacement
        """
        # Filter to team
        team_players = player_stats_df[player_stats_df["Team"] == team_name].copy()

        if team_players.empty:
            return pd.DataFrame()

        # Calculate values for each player
        values = []
        for _, player_stats in team_players.iterrows():
            try:
                value = self.calculate_player_value(
                    player_stats.to_dict(), team_stats
                )
                values.append(
                    {
                        "Player": value.player_name,
                        "yr": player_stats.get("yr", ""),
                        "ht": player_stats.get("ht", ""),
                        "Poss%": value.possession_pct,
                        "ORtg": value.offensive_rating,
                        "eFG%": value.effective_fg_pct,
                        "EstimatedValue": value.estimated_value,
                        "VOR": value.value_over_replacement,
                    }
                )
            except (KeyError, ValueError) as e:
                # Skip players with missing data
                continue

        if not values:
            return pd.DataFrame()

        # Create DataFrame and sort by value
        depth_chart = pd.DataFrame(values)
        depth_chart = depth_chart.sort_values("EstimatedValue", ascending=False)
        depth_chart = depth_chart.reset_index(drop=True)

        return depth_chart
