# Project Scope and Boundaries

**Last Updated**: 2025-12-17
**Status**: Official project definition

---

## What This Project IS

### **NCAA Division I Men's Basketball Analytics**

This project provides advanced analytics for college basketball using Ken Pomeroy's (KenPom) methodology and data.

**Core Focus**:
- NCAA Division I Men's Basketball ONLY
- KenPom efficiency-based analytics
- Indoor arena games (no weather factors)
- Academic/research approach to basketball analytics
- Matchup analysis using advanced metrics
- Predictive modeling for game outcomes
- Tournament simulation and bracket analysis

**Data Sources** (Basketball Only):
- KenPom.com Official API (primary)
- KenPom historical archives
- NCAA basketball schedules and results
- Conference-specific basketball data

---

## What This Project IS NOT

### **âŒ Not an NFL/Football Analytics System**

- This project does **NOT** use Billy Walters methodology
- This project does **NOT** analyze NFL football
- This project does **NOT** incorporate weather factors (basketball is indoors)
- Billy Walters principles and NFL betting analysis are **OUT OF SCOPE**

### **âŒ Not a General Sports Betting Platform**

While the analytics can inform betting decisions:
- Focus is on analytical accuracy, not gambling strategy
- No sports betting-specific features beyond model validation
- CLV tracking and edge detection are for model validation only
- Educational and research purposes

### **âŒ Not a Multi-Sport System**

- **ONLY NCAA Men's Basketball**
- No NFL, NBA, MLB, NHL, soccer, or other sports
- No plans to expand beyond college basketball
- Single sport = better depth and accuracy

---

## Scope Boundaries

### In Scope âœ…

| Feature Category | Included |
|------------------|----------|
| **Sport** | NCAA Division I Men's Basketball ONLY |
| **Data Sources** | KenPom API, KenPom archives, NCAA schedules |
| **Analytics** | Efficiency metrics, Four Factors, tempo, matchups |
| **Predictions** | Game outcomes, spreads, totals using KenPom data |
| **Validation** | Backtesting, cross-validation, accuracy metrics |
| **Environment** | Indoor arenas (weather irrelevant) |
| **Methodology** | Ken Pomeroy's efficiency-based system |
| **Tools** | Python, ML models, statistical analysis |

### Out of Scope âŒ

| Feature Category | Excluded |
|------------------|----------|
| **Other Sports** | NFL, NBA, MLB, NHL, soccer, etc. |
| **Weather** | Not applicable (indoor games) |
| **Billy Walters** | NFL-specific methodology (wrong sport) |
| **NFL Betting** | Football analysis (different sport) |
| **Player Props** | Individual player betting markets |
| **Live Betting** | In-game betting analytics |
| **Venue Factors** | Court dimensions, rim height (standardized) |
| **Outdoor Games** | No outdoor basketball at D-I level |

---

## Technology Stack

### Core Libraries
- **kenpompy**: KenPom data access library
- **pandas/numpy**: Data manipulation and analysis
- **scikit-learn**: Machine learning predictions
- **pydantic**: Data validation and modeling
- **httpx**: HTTP client for API access

### Optional Components
- **Playwright**: Browser automation for web scraping (KenPom)
- **MCP**: Claude AI integration for interactive analysis
- **pytest**: Testing framework

### Python Requirements
- Python 3.11+ required
- Type hints mandatory
- PEP 8 style compliance

---

## Data Philosophy

### Single Source of Truth: KenPom

**Why KenPom Only?**
1. **Consistent Methodology**: All data adjusted using same system
2. **Opponent-Adjusted**: Metrics account for strength of opposition
3. **Proven Track Record**: 20+ years of accurate predictions
4. **Comprehensive Coverage**: All D-I teams, all games
5. **Historical Depth**: Data back to 1999 season

**Forbidden Data Sources**:
- âŒ ESPN stats (not opponent-adjusted)
- âŒ CBS Sports (different methodology)
- âŒ Web searches for "recent games" (introduces bias)
- âŒ Manual game selection (cherry-picking)
- âŒ Non-KenPom box scores (inconsistent adjustments)

### KenPom Metrics Priority

| Priority | Metric Category | Examples |
|----------|-----------------|----------|
| **1** | Efficiency Ratings | AdjO, AdjD, AdjEM |
| **2** | Four Factors | eFG%, TO%, OR%, FTRate |
| **3** | Tempo Metrics | AdjT, Average Possession Length |
| **4** | Strength Metrics | SOS, NCSOS, Pythag |
| **5** | Derived Metrics | Efficiency differentials, tempo impacts |

---

## Basketball-Specific Considerations

### Why Weather Doesn't Matter

**NCAA Basketball = Indoor Sport**
- All Division I games played in climate-controlled arenas
- Temperature: Regulated (~70Â°F / 21Â°C)
- Humidity: Controlled HVAC systems
- Wind: Non-existent indoors
- Precipitation: Not applicable
- Altitude: Only minor factor for visiting teams (e.g., Denver, Wyoming)

**Relevant Environmental Factors**:
- Crowd noise and size (home court advantage)
- Arena size and configuration
- Travel distance and fatigue
- Altitude adjustment (rare cases)

### No Billy Walters Methodology Here

**Why Billy Walters Doesn't Apply**:
- Billy Walters = NFL football betting expert
- NFL â‰  NCAA Basketball (different sport entirely)
- Weather is critical in NFL (outdoor games)
- Football has vastly different dynamics than basketball
- Walters' information edges were NFL-specific

**Correct Attribution**:
- Ken Pomeroy = NCAA Basketball analytics pioneer
- Dean Oliver = Four Factors framework (basketball)
- John Hollinger = Advanced basketball metrics (PER, etc.)
- These are basketball-specific methodologies

---

## Project Evolution

### Original Vision (Maintained)
- NCAA basketball analytics using KenPom data
- Advanced efficiency-based matchup analysis
- Predictive modeling with confidence intervals
- Educational and research focus

### What Changed (Clarified Scope)
- Removed incorrect references to NFL/football
- Clarified that weather is not applicable
- Established single-sport focus (basketball only)
- Documented KenPom as single source of truth

### Future Direction
- Deeper KenPom metric integration
- Enhanced tournament simulation
- Conference-specific analysis
- Historical trend analysis
- Player impact modeling (within KenPom framework)

---

## Why These Boundaries Matter

### 1. Data Consistency
Using only KenPom data ensures all metrics use the same opponent-adjustment methodology. Mixing data sources introduces inconsistencies.

### 2. Model Accuracy
Single-sport focus allows deeper domain expertise and better predictions. Multi-sport systems sacrifice depth for breadth.

### 3. Validation Integrity
When model predictions, historical data, and validation all use KenPom methodology, we can trust the validation results.

### 4. Correct Attribution
Using Ken Pomeroy's methods for basketball (not Billy Walters' NFL methods) ensures we're applying the right expertise to the right sport.

---

## Common Misconceptions (Corrected)

| âŒ Misconception | âœ… Reality |
|-----------------|-----------|
| "This uses Billy Walters methods" | **No** - Uses Ken Pomeroy methods for basketball |
| "Weather affects predictions" | **No** - Basketball is played indoors |
| "System analyzes NFL games" | **No** - NCAA basketball only |
| "Multi-sport betting platform" | **No** - Single sport analytics system |
| "Uses various data sources" | **No** - KenPom only for consistency |

---

## Documentation Standards

### File Naming
All documentation should clearly indicate basketball focus:
- âœ… `KENPOM_ANALYTICS_GUIDE.md`
- âœ… `BASKETBALL_MATCHUP_FRAMEWORK.md`
- âœ… `NCAA_TOURNAMENT_SIMULATOR.md`
- âŒ `SPORTS_BETTING_GUIDE.md` (too generic)
- âŒ `NFL_ANALYSIS.md` (wrong sport)

### Code Comments
Use basketball-specific terminology:
- âœ… "Adjust for home court advantage"
- âœ… "Account for altitude in Denver"
- âŒ "Adjust for weather conditions"
- âŒ "Apply NFL home field advantage"

---

## Summary

**One Sport. One Methodology. One Source of Truth.**

This project is a focused, disciplined approach to NCAA Men's Basketball analytics using Ken Pomeroy's proven efficiency-based methodology. By maintaining strict boundaries around sport (basketball only), data source (KenPom only), and environment (indoor arenas), we ensure:

- Analytical consistency
- Predictive accuracy
- Validation integrity
- Correct domain expertise

**Remember**: Basketball is not football. KenPom is not Billy Walters. Indoor is not outdoor. Focus is power.

---

*This document serves as the official project scope definition and supersedes any conflicting information in other documentation.*
