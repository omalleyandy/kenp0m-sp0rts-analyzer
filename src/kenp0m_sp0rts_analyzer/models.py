"""Pydantic models for KenPom basketball analytics data."""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class TeamEfficiency(BaseModel):
    """Team efficiency ratings from KenPom."""

    team: str = Field(..., description="Team name")
    conference: str = Field(..., description="Conference abbreviation")
    rank: int = Field(..., description="Overall KenPom ranking")
    adj_efficiency_margin: float = Field(
        ..., alias="adjEM", description="Adjusted efficiency margin (AdjO - AdjD)"
    )
    adj_offense: float = Field(
        ..., alias="adjO", description="Adjusted offensive efficiency"
    )
    adj_offense_rank: int = Field(
        ..., alias="adjO_rank", description="Offensive efficiency rank"
    )
    adj_defense: float = Field(
        ..., alias="adjD", description="Adjusted defensive efficiency"
    )
    adj_defense_rank: int = Field(
        ..., alias="adjD_rank", description="Defensive efficiency rank"
    )
    adj_tempo: float = Field(
        ..., alias="adjT", description="Adjusted tempo (possessions per 40 min)"
    )
    adj_tempo_rank: int = Field(..., alias="adjT_rank", description="Tempo rank")


class FourFactors(BaseModel):
    """Dean Oliver's Four Factors for team analysis."""

    team: str = Field(..., description="Team name")
    conference: str = Field(..., description="Conference abbreviation")

    # Offensive Four Factors
    off_efg_pct: float = Field(
        ..., alias="offEFG", description="Offensive effective FG%"
    )
    off_to_pct: float = Field(
        ..., alias="offTO", description="Offensive turnover percentage"
    )
    off_or_pct: float = Field(
        ..., alias="offOR", description="Offensive rebound percentage"
    )
    off_ft_rate: float = Field(
        ..., alias="offFTRate", description="Offensive free throw rate"
    )

    # Defensive Four Factors
    def_efg_pct: float = Field(
        ..., alias="defEFG", description="Defensive effective FG%"
    )
    def_to_pct: float = Field(
        ..., alias="defTO", description="Defensive turnover percentage"
    )
    def_or_pct: float = Field(
        ..., alias="defOR", description="Defensive rebound percentage (allowed)"
    )
    def_ft_rate: float = Field(
        ..., alias="defFTRate", description="Defensive free throw rate (allowed)"
    )


class TeamScheduleGame(BaseModel):
    """Single game from a team's schedule."""

    date: date = Field(..., description="Game date")
    opponent: str = Field(..., description="Opponent team name")
    result: str = Field(..., description="W or L")
    score: str = Field(..., description="Final score (e.g., '85-72')")
    location: str = Field(..., description="Home, Away, or Neutral")
    opponent_rank: Optional[int] = Field(
        None, description="Opponent's KenPom rank at time of game"
    )
    pace: Optional[float] = Field(None, description="Game pace (possessions)")
    offensive_efficiency: Optional[float] = Field(
        None, alias="oEff", description="Team's offensive efficiency in game"
    )
    defensive_efficiency: Optional[float] = Field(
        None, alias="dEff", description="Team's defensive efficiency in game"
    )


class HomeCourtAdvantage(BaseModel):
    """Home court advantage data for a team."""

    team: str = Field(..., description="Team name")
    conference: str = Field(..., description="Conference abbreviation")
    hca: float = Field(..., description="Home court advantage in points")
    hca_rank: int = Field(..., description="HCA rank among all teams")


class MatchupAnalysis(BaseModel):
    """Analysis of a head-to-head matchup between two teams."""

    team1: str = Field(..., description="First team name")
    team2: str = Field(..., description="Second team name")
    team1_rank: int = Field(..., description="Team 1 KenPom rank")
    team2_rank: int = Field(..., description="Team 2 KenPom rank")

    # Efficiency comparison
    team1_adj_em: float = Field(..., description="Team 1 adjusted efficiency margin")
    team2_adj_em: float = Field(..., description="Team 2 adjusted efficiency margin")
    em_difference: float = Field(..., description="Efficiency margin difference")

    # Predictions
    predicted_winner: str = Field(..., description="Predicted winner")
    predicted_margin: float = Field(..., description="Predicted margin of victory")
    predicted_total: Optional[float] = Field(
        None, description="Predicted total score"
    )

    # Tempo analysis
    team1_tempo: float = Field(..., description="Team 1 adjusted tempo")
    team2_tempo: float = Field(..., description="Team 2 adjusted tempo")
    expected_tempo: float = Field(..., description="Expected game tempo")

    # Style matchup
    pace_advantage: str = Field(
        ..., description="Which team benefits from expected pace"
    )
    neutral_site: bool = Field(default=True, description="Is game at neutral site")


class ScoutingReport(BaseModel):
    """Comprehensive scouting report for a team."""

    team: str = Field(..., description="Team name")
    season: int = Field(..., description="Season year")
    conference: str = Field(..., description="Conference")
    rank: int = Field(..., description="KenPom rank")
    record: str = Field(..., description="Win-loss record")

    # Efficiency
    adj_em: float = Field(..., description="Adjusted efficiency margin")
    adj_offense: float = Field(..., description="Adjusted offensive efficiency")
    adj_defense: float = Field(..., description="Adjusted defensive efficiency")
    adj_tempo: float = Field(..., description="Adjusted tempo")

    # Four Factors
    four_factors: FourFactors = Field(..., description="Four Factors breakdown")

    # Schedule strength
    strength_of_schedule: float = Field(..., alias="sos", description="Overall SOS")
    nc_strength_of_schedule: float = Field(
        ..., alias="ncSOS", description="Non-conference SOS"
    )
    luck: float = Field(..., description="Luck factor")

    # Additional metrics
    experience: Optional[float] = Field(None, description="Experience rating")
    bench: Optional[float] = Field(None, description="Bench minutes percentage")
    continuity: Optional[float] = Field(None, description="Roster continuity")
