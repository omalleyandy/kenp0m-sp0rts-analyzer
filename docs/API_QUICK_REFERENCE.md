# KenPom API Quick Reference

**One-page cheat sheet for common operations**

## Setup

```python
from kenp0m_sp0rts_analyzer.api_client import KenPomAPI

# Initialize (uses KENPOM_API_KEY environment variable)
api = KenPomAPI()
```

## Common Patterns

### Get Current Season Ratings

```python
# All teams
ratings = api.get_ratings(year=2025)

# Specific conference
sec = api.get_ratings(year=2025, conference="SEC")

# Official API naming
big12 = api.get_ratings(y=2025, c="B12")
```

### Get Team History

```python
# Duke's historical data (TeamID: 73)
duke = api.get_ratings(team_id=73)

# Find team by name
duke_info = api.get_team_by_name("Duke", 2025)
print(f"TeamID: {duke_info['TeamID']}")
```

### Game Predictions

```python
from datetime import date

# Using string
games = api.get_fanmatch("2025-03-15")

# Using date object
games = api.get_fanmatch(date(2025, 3, 15))

# Official API parameter
games = api.get_fanmatch(d="2025-03-15")

# Find close games
close = [g for g in games.data if 40 <= g['HomeWP'] <= 60]
```

### Historical Ratings

```python
# Ratings from a specific date
archive = api.get_archive(archive_date="2025-02-15")

# Preseason ratings
preseason = api.get_archive(preseason=True, year=2025)

# Using official API parameter
archive = api.get_archive(d="2025-02-15")

# Compare preseason to final
for team in preseason.data[:10]:
    print(f"{team['TeamName']}: #{team['RankAdjEM']} -> #{team['RankAdjEMFinal']} ({team['RankChg']:+d})")
```

### Four Factors Analysis

```python
# All teams
ff = api.get_four_factors(year=2025)

# Conference only
conf_ff = api.get_four_factors(year=2025, conf_only=True)

# Convert to DataFrame
df = ff.to_dataframe()

# Best shooting teams
best_shooting = df.nsmallest(10, 'RankeFG_Pct')
```

### Team Data

```python
# Get all teams for a season
teams = api.get_teams(year=2025)

# Filter by conference
big_east = api.get_teams(year=2025, conference="BE")

# Get conferences list
conferences = api.get_conferences(year=2025)
```

### Miscellaneous Stats

```python
# Advanced metrics
stats = api.get_misc_stats(year=2025)

# Height and experience
height = api.get_height(year=2025)

# Point distribution
dist = api.get_point_distribution(year=2025)
```

## Parameter Aliases

The API client supports both Python-style and official API parameter names:

| Python Style | Official API | Endpoints |
|--------------|--------------|-----------|
| `year` | `y` | All endpoints with season data |
| `conference` | `c` | Filtering endpoints |
| `archive_date` | `d` | Archive endpoint |
| `game_date` | `d` | Fanmatch endpoint |

**Both work identically:**

```python
# Python style
ratings = api.get_ratings(year=2025, conference="ACC")

# Official API style
ratings = api.get_ratings(y=2025, c="ACC")
```

## Response Handling

### Iterate Over Results

```python
ratings = api.get_ratings(year=2025)

# Direct iteration
for team in ratings:
    print(f"{team['TeamName']}: {team['AdjEM']}")

# Length
print(f"Total teams: {len(ratings)}")
```

### Convert to DataFrame

```python
ratings = api.get_ratings(year=2025)
df = ratings.to_dataframe()

# Now use pandas operations
top_10 = df.nsmallest(10, 'RankAdjEM')
```

### Access Response Data

```python
ratings = api.get_ratings(year=2025)

# Access data list
teams = ratings.data

# Get endpoint used
endpoint = ratings.endpoint  # "ratings"

# Get parameters sent
params = ratings.params  # {"y": 2025}
```

## Common Fields

### Ratings Endpoint

| Field | Description |
|-------|-------------|
| `AdjEM` | Adjusted Efficiency Margin |
| `AdjOE` | Adjusted Offensive Efficiency |
| `AdjDE` | Adjusted Defensive Efficiency |
| `AdjTempo` | Adjusted Tempo |
| `SOS` | Strength of Schedule |
| `Pythag` | Pythagorean Win Expectation |
| `Luck` | Luck Rating |

### Four Factors

| Field | Description |
|-------|-------------|
| `eFG_Pct` | Effective Field Goal % (Offense) |
| `TO_Pct` | Turnover % (Offense) |
| `OR_Pct` | Offensive Rebound % |
| `FT_Rate` | Free Throw Rate (Offense) |
| `DeFG_Pct` | Effective FG% Allowed (Defense) |
| `DTO_Pct` | Turnover % Forced (Defense) |

### Fanmatch (Game Predictions)

| Field | Description |
|-------|-------------|
| `GameID` | Unique game identifier |
| `Visitor` / `Home` | Team names |
| `VisitorPred` / `HomePred` | Predicted scores |
| `HomeWP` | Home team win probability (%) |
| `PredTempo` | Predicted game tempo |
| `ThrillScore` | Expected excitement level |

## Conference Abbreviations

Common conference short names for filtering:

| Code | Conference |
|------|------------|
| `ACC` | Atlantic Coast Conference |
| `B10` | Big Ten |
| `B12` | Big 12 |
| `BE` | Big East |
| `SEC` | Southeastern Conference |
| `P12` | Pac-12 |
| `A10` | Atlantic 10 |
| `WCC` | West Coast Conference |

Get full list: `api.get_conferences(year=2025)`

## Team IDs

Common team IDs for historical queries:

| Team | TeamID |
|------|--------|
| Duke | 73 |
| North Carolina | 153 |
| Kentucky | 96 |
| Kansas | 88 |
| Villanova | 222 |
| Gonzaga | 285 |

Find team ID: `api.get_team_by_name("Duke", 2025)['TeamID']`

## Boolean Fields

These fields are automatically converted from string ("true"/"false") to Python bool (True/False):

- `Preseason` (archive endpoint)
- `ConfOnly` (four-factors, misc-stats, pointdist endpoints)

```python
result = api.get_archive(preseason=True, year=2025)
print(result.data[0]["Preseason"])  # True (bool, not "true" string)
```

## Complete Documentation

See [KENPOM_API.md](KENPOM_API.md) for complete documentation with all parameters, response fields, and examples.
