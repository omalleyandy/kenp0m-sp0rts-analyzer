"""Pydantic data models for KenPom data.

These models provide type safety, validation, and serialization
for all KenPom data structures used throughout the system.
"""

from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator


class Team(BaseModel):
    """Team information model."""

    team_id: int
    team_name: str
    conference: str | None = None
    coach: str | None = None
    arena: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    class Config:
        from_attributes = True


class TeamRating(BaseModel):
    """Team rating snapshot model with validation.

    Contains all KenPom efficiency metrics for a team at a point in time.
    """

    team_id: int
    team_name: str
    snapshot_date: date
    season: int

    # Core efficiency metrics
    adj_em: float = Field(
        ..., ge=-50, le=50, description="Adjusted Efficiency Margin"
    )
    adj_oe: float = Field(
        ..., ge=70, le=150, description="Adjusted Offensive Efficiency"
    )
    adj_de: float = Field(
        ..., ge=70, le=150, description="Adjusted Defensive Efficiency"
    )
    adj_tempo: float = Field(..., ge=55, le=90, description="Adjusted Tempo")

    # Luck and strength metrics
    luck: float = Field(
        default=0.0, ge=-0.3, le=0.3, description="Luck rating"
    )
    sos: float = Field(default=0.0, description="Strength of Schedule")
    soso: float = Field(
        default=0.0, description="Strength of Schedule (Offense)"
    )
    sosd: float = Field(
        default=0.0, description="Strength of Schedule (Defense)"
    )
    ncsos: float = Field(default=0.0, description="Non-Conference SOS")
    pythag: float = Field(
        default=0.5, ge=0, le=1, description="Pythagorean expectation"
    )

    # Rankings
    rank_adj_em: int | None = Field(default=None, ge=1, le=400)
    rank_adj_oe: int | None = Field(default=None, ge=1, le=400)
    rank_adj_de: int | None = Field(default=None, ge=1, le=400)
    rank_tempo: int | None = Field(default=None, ge=1, le=400)
    rank_sos: int | None = Field(default=None, ge=1, le=400)
    rank_luck: int | None = Field(default=None, ge=1, le=400)

    # Record
    wins: int = Field(default=0, ge=0)
    losses: int = Field(default=0, ge=0)
    conference: str | None = None

    # Average Possession Length
    apl_off: float | None = Field(
        default=None, description="Avg Possession Length (Off)"
    )
    apl_def: float | None = Field(
        default=None, description="Avg Possession Length (Def)"
    )

    @field_validator("adj_em")
    @classmethod
    def validate_adj_em(cls, v: float) -> float:
        """Validate AdjEM is in reasonable range."""
        if v < -50 or v > 50:
            raise ValueError(f"AdjEM {v} outside reasonable range [-50, 50]")
        return round(v, 2)

    @field_validator("adj_tempo")
    @classmethod
    def validate_tempo(cls, v: float) -> float:
        """Validate tempo is in reasonable range."""
        if v < 55 or v > 90:
            raise ValueError(f"Tempo {v} outside reasonable range [55, 90]")
        return round(v, 1)

    class Config:
        from_attributes = True


class FourFactors(BaseModel):
    """Dean Oliver's Four Factors model.

    The four factors are:
    1. Effective FG% (shooting efficiency)
    2. Turnover Rate (ball security)
    3. Offensive Rebounding % (second chances)
    4. Free Throw Rate (getting to the line)
    """

    team_id: int
    snapshot_date: date

    # Offensive Four Factors
    efg_pct_off: float = Field(
        ..., ge=0, le=100, description="Effective FG% (Offense)"
    )
    to_pct_off: float = Field(
        ..., ge=0, le=50, description="Turnover Rate (Offense)"
    )
    or_pct_off: float = Field(
        ..., ge=0, le=60, description="Offensive Rebound % (Offense)"
    )
    ft_rate_off: float = Field(
        ..., ge=0, le=100, description="FT Rate (Offense)"
    )

    # Defensive Four Factors (opponent values)
    efg_pct_def: float = Field(
        ..., ge=0, le=100, description="Effective FG% (Defense)"
    )
    to_pct_def: float = Field(
        ..., ge=0, le=50, description="Turnover Rate (Defense)"
    )
    or_pct_def: float = Field(
        ..., ge=0, le=60, description="Offensive Rebound % (Defense)"
    )
    ft_rate_def: float = Field(
        ..., ge=0, le=100, description="FT Rate (Defense)"
    )

    # Rankings
    rank_efg_off: int | None = None
    rank_efg_def: int | None = None
    rank_to_off: int | None = None
    rank_to_def: int | None = None
    rank_or_off: int | None = None
    rank_or_def: int | None = None
    rank_ft_rate_off: int | None = None
    rank_ft_rate_def: int | None = None

    class Config:
        from_attributes = True


class PointDistribution(BaseModel):
    """Team point distribution model.

    Tracks what percentage of points come from FTs, 2-pointers, and 3-pointers.
    High 3-point reliance indicates higher variance in game outcomes.
    """

    team_id: int
    snapshot_date: date

    # Offensive distribution (% of points)
    ft_pct: float = Field(..., ge=0, le=50, description="% of points from FTs")
    two_pct: float = Field(
        ..., ge=20, le=80, description="% of points from 2-pointers"
    )
    three_pct: float = Field(
        ..., ge=0, le=60, description="% of points from 3-pointers"
    )

    # Defensive distribution (opponent)
    ft_pct_def: float = Field(..., ge=0, le=50)
    two_pct_def: float = Field(..., ge=20, le=80)
    three_pct_def: float = Field(..., ge=0, le=60)

    # Rankings
    rank_three_pct: int | None = None
    rank_two_pct: int | None = None
    rank_ft_pct: int | None = None

    @field_validator("ft_pct", "two_pct", "three_pct")
    @classmethod
    def validate_pct_sum(cls, v: float) -> float:
        """Validate percentage is reasonable."""
        return round(v, 1)

    class Config:
        from_attributes = True


class HeightExperience(BaseModel):
    """Team height and experience model."""

    team_id: int
    snapshot_date: date

    avg_height: float = Field(
        ..., ge=70, le=85, description="Average height in inches"
    )
    effective_height: float = Field(
        ...,
        ge=-10,
        le=10,
        description="Effective height relative score (z-score)",
    )
    experience: float = Field(
        ..., ge=0, le=5, description="Average years of experience"
    )
    bench_minutes: float = Field(
        ..., ge=0, le=100, description="Bench minutes %"
    )
    continuity: float = Field(
        ..., ge=0, le=1, description="Roster continuity (0-1)"
    )

    # Rankings
    rank_height: int | None = None
    rank_experience: int | None = None
    rank_continuity: int | None = None

    class Config:
        from_attributes = True


class FanMatchPrediction(BaseModel):
    """KenPom game prediction for ensemble blending.

    Provides KenPom's own predictions to blend with XGBoost model.
    """

    game_id: str
    snapshot_date: date
    home_team_id: int
    visitor_team_id: int
    home_team_name: str | None = None
    visitor_team_name: str | None = None

    pred_home_score: float = Field(
        ..., ge=30, le=150, description="Predicted home score"
    )
    pred_visitor_score: float = Field(
        ..., ge=30, le=150, description="Predicted visitor score"
    )
    pred_margin: float = Field(
        ..., description="Predicted margin (home - visitor)"
    )
    home_win_prob: float = Field(
        ..., ge=0, le=1, description="Home win probability"
    )
    pred_tempo: float = Field(..., ge=55, le=90, description="Predicted tempo")
    thrill_score: float | None = Field(
        default=None, ge=0, le=100, description="Game excitement score"
    )

    class Config:
        from_attributes = True


class MiscStats(BaseModel):
    """Miscellaneous team statistics (shooting, assists, steals, blocks)."""

    team_id: int
    snapshot_date: date

    # Shooting percentages (0-100 scale)
    fg3_pct_off: float = Field(
        ..., ge=0, le=100, description="3PT FG% (Offense)"
    )
    fg3_pct_def: float = Field(
        ..., ge=0, le=100, description="3PT FG% (Defense)"
    )
    fg2_pct_off: float = Field(
        ..., ge=0, le=100, description="2PT FG% (Offense)"
    )
    fg2_pct_def: float = Field(
        ..., ge=0, le=100, description="2PT FG% (Defense)"
    )
    ft_pct_off: float = Field(..., ge=0, le=100, description="FT% (Offense)")
    ft_pct_def: float = Field(..., ge=0, le=100, description="FT% (Defense)")

    # Advanced metrics
    assist_rate: float = Field(..., ge=0, le=100, description="Assist rate")
    assist_rate_def: float = Field(
        ..., ge=0, le=100, description="Opponent assist rate"
    )
    steal_rate: float = Field(..., ge=0, le=50, description="Steal rate")
    steal_rate_def: float = Field(
        ..., ge=0, le=50, description="Opponent steal rate"
    )
    block_pct_off: float = Field(
        ..., ge=0, le=30, description="Block percentage"
    )
    block_pct_def: float = Field(
        ..., ge=0, le=30, description="Opponent block percentage"
    )

    # Rankings
    rank_fg3_pct: int | None = Field(default=None, ge=1, le=400)
    rank_fg2_pct: int | None = Field(default=None, ge=1, le=400)
    rank_ft_pct: int | None = Field(default=None, ge=1, le=400)
    rank_assist_rate: int | None = Field(default=None, ge=1, le=400)

    class Config:
        from_attributes = True


class GamePrediction(BaseModel):
    """Game prediction model for tracking accuracy.

    Stores both the prediction and eventual result for model evaluation.
    """

    id: int | None = None
    game_date: date
    team1_id: int
    team2_id: int
    team1_name: str | None = None
    team2_name: str | None = None

    # Predictions
    predicted_margin: float = Field(
        ..., description="Predicted margin (team1 - team2)"
    )
    predicted_total: float = Field(
        ..., ge=80, le=200, description="Predicted total"
    )
    win_probability: float = Field(
        ..., ge=0, le=1, description="Team1 win probability"
    )
    confidence_lower: float = Field(
        ..., description="Lower bound of confidence interval"
    )
    confidence_upper: float = Field(
        ..., description="Upper bound of confidence interval"
    )

    # Vegas lines at prediction time
    vegas_spread: float | None = None
    vegas_total: float | None = None

    # Actual results (filled after game)
    actual_margin: float | None = None
    actual_total: float | None = None

    # Calculated metrics (after game)
    prediction_error: float | None = None
    beat_spread: bool | None = None
    clv: float | None = None  # Closing Line Value

    # Metadata
    model_version: str = "v1.0"
    neutral_site: bool = False
    created_at: datetime | None = None
    resolved_at: datetime | None = None

    @field_validator("win_probability")
    @classmethod
    def validate_prob(cls, v: float) -> float:
        """Ensure probability is valid."""
        return round(max(0, min(1, v)), 4)

    class Config:
        from_attributes = True


class GameResult(BaseModel):
    """Game result for updating predictions."""

    prediction_id: int
    actual_margin: float
    actual_total: float
    team1_score: int
    team2_score: int
    resolved_at: datetime = Field(default_factory=datetime.now)


class SyncResult(BaseModel):
    """Result of a data sync operation."""

    success: bool
    endpoint: str
    records_synced: int
    records_skipped: int = 0
    errors: list[str] = []
    duration_seconds: float
    timestamp: datetime = Field(default_factory=datetime.now)


class BackfillResult(BaseModel):
    """Result of historical data backfill."""

    success: bool
    start_date: date
    end_date: date
    dates_requested: int
    dates_filled: int
    dates_skipped: int
    errors: list[str] = []
    duration_seconds: float


class AccuracyReport(BaseModel):
    """Prediction accuracy metrics over a time period."""

    start_date: date
    end_date: date
    games_predicted: int
    games_resolved: int

    # Margin accuracy
    mae_margin: float = Field(..., ge=0, description="Mean Absolute Error")
    rmse_margin: float = Field(
        ..., ge=0, description="Root Mean Squared Error"
    )
    r2_margin: float = Field(..., ge=-1, le=1, description="R² score")

    # Win prediction accuracy
    win_accuracy: float = Field(
        ..., ge=0, le=1, description="Correct winner %"
    )
    brier_score: float = Field(
        ..., ge=0, le=1, description="Probability calibration"
    )

    # Against-the-spread performance
    ats_wins: int = 0
    ats_losses: int = 0
    ats_pushes: int = 0
    ats_percentage: float = Field(default=0.0, ge=0, le=1)

    # Closing Line Value
    avg_clv: float = Field(default=0.0, description="Average CLV in points")
    positive_clv_rate: float = Field(default=0.0, ge=0, le=1)

    @property
    def total_ats_games(self) -> int:
        """Total ATS games (excluding pushes)."""
        return self.ats_wins + self.ats_losses

    @property
    def is_profitable(self) -> bool:
        """Check if ATS performance exceeds breakeven (52.4%)."""
        if self.total_ats_games == 0:
            return False
        return self.ats_percentage > 0.524


class RatingChange(BaseModel):
    """Notification of a team rating change."""

    team_id: int
    team_name: str
    old_rating: float
    new_rating: float
    change: float
    field: str = "adj_em"
    detected_at: datetime = Field(default_factory=datetime.now)

    @property
    def direction(self) -> str:
        """Return 'up' or 'down' based on change direction."""
        return "up" if self.change > 0 else "down"


class MatchupData(BaseModel):
    """Comprehensive data for a matchup between two teams."""

    team1: TeamRating
    team2: TeamRating
    team1_four_factors: FourFactors | None = None
    team2_four_factors: FourFactors | None = None
    team1_point_dist: PointDistribution | None = None
    team2_point_dist: PointDistribution | None = None
    team1_height: HeightExperience | None = None
    team2_height: HeightExperience | None = None

    # Recent history
    team1_recent_games: list[dict] = []
    team2_recent_games: list[dict] = []

    # Head-to-head
    h2h_games: list[dict] = []

    # Venue info
    neutral_site: bool = False
    home_team: int | None = None  # team_id of home team


class DailySnapshot(BaseModel):
    """Collection of all team ratings for a single date."""

    snapshot_date: date
    season: int
    ratings: list[TeamRating]
    record_count: int

    @field_validator("ratings")
    @classmethod
    def validate_ratings_count(cls, v: list[TeamRating]) -> list[TeamRating]:
        """Ensure we have reasonable number of teams."""
        if len(v) < 100:
            raise ValueError(f"Expected 350+ teams, got {len(v)}")
        return v


class SeasonData(BaseModel):
    """Complete data for a season."""

    season: int
    snapshots: list[DailySnapshot]
    preseason_ratings: list[TeamRating] | None = None
    teams: list[Team]

    @property
    def date_range(self) -> tuple[date, date]:
        """Get the date range covered by snapshots."""
        dates = [s.snapshot_date for s in self.snapshots]
        return min(dates), max(dates)

    @property
    def snapshot_count(self) -> int:
        """Number of daily snapshots."""
        return len(self.snapshots)
