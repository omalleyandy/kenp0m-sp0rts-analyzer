# KenPom Sports Analyzer Documentation

This directory contains comprehensive documentation for the KenPom Sports Analyzer project. All documentation files are organized by topic below.

---

## 🎯 Project Scope

**IMPORTANT**: Before diving into documentation, read **[../PROJECT_SCOPE.md](../PROJECT_SCOPE.md)** to understand:
- What this project IS (NCAA Basketball analytics with KenPom)
- What this project IS NOT (No NFL, no Billy Walters, no weather factors)
- Official boundaries and data philosophy

---

## 📚 Table of Contents

### Getting Started
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete setup instructions for new users
  - Installing dependencies with `uv`
  - Configuring API keys
  - Running your first analysis
  - Troubleshooting common issues

### API Reference
- **[KENPOM_API.md](KENPOM_API.md)** - Official KenPom API documentation
  - All 9 API endpoints (ratings, four-factors, height, etc.)
  - Request/response formats
  - Authentication details
  - Example requests

- **[API_QUICK_REFERENCE.md](API_QUICK_REFERENCE.md)** - Quick API cheat sheet
  - Common API calls
  - Parameter aliases
  - Response field mappings

- **[KENPOM_DATA_COVERAGE.md](KENPOM_DATA_COVERAGE.md)** - Data availability reference
  - What data is available from KenPom
  - Data update schedules
  - Historical data access

### Analysis Framework
- **[MATCHUP_ANALYSIS_FRAMEWORK.md](MATCHUP_ANALYSIS_FRAMEWORK.md)** - Core analysis methodology
  - 7-module analytical framework
  - 15-dimensional matchup analysis
  - Weighting systems (regular season vs tournament)
  - Integration architecture

- **[TEMPO_PACE_DEEP_DIVE.md](TEMPO_PACE_DEEP_DIVE.md)** - Advanced tempo analysis
  - Average Possession Length (APL) metrics
  - Pace control calculations
  - Offensive/defensive style classifications
  - Tempo impact on predictions

- **[TEMPO_VALIDATION_SUMMARY.md](TEMPO_VALIDATION_SUMMARY.md)** - Tempo model validation
  - Testing methodology
  - Performance metrics
  - Known limitations

### Implementation Plans
- **[ANALYTICS_ROADMAP.md](ANALYTICS_ROADMAP.md)** - Project roadmap overview
  - Feature tiers (TIER 1, TIER 2, TIER 3)
  - Development priorities
  - Timeline and milestones

- **[TIER1_IMPLEMENTATION_PLAN.md](TIER1_IMPLEMENTATION_PLAN.md)** - TIER 1 features
  - Four Factors Analysis
  - Point Distribution Analysis
  - Defensive Analysis
  - Implementation status

- **[TIER2_IMPLEMENTATION_PLAN.md](TIER2_IMPLEMENTATION_PLAN.md)** - TIER 2 features
  - Size & Athleticism Analysis
  - Experience & Chemistry Analysis
  - Intangibles modeling

- **[HIGH_VALUE_FEATURES_PLAN.md](HIGH_VALUE_FEATURES_PLAN.md)** - High-impact features
  - Prioritized feature list
  - Value vs effort analysis
  - Quick wins

---

## 🎯 Quick Navigation by Use Case

### "I'm new and want to get started"
1. Start with **[SETUP_GUIDE.md](SETUP_GUIDE.md)**
2. Review **[API_QUICK_REFERENCE.md](API_QUICK_REFERENCE.md)**
3. Run the example: `uv run python examples/analyze_game_example.py`

### "I want to understand the analysis methodology"
1. Read **[MATCHUP_ANALYSIS_FRAMEWORK.md](MATCHUP_ANALYSIS_FRAMEWORK.md)**
2. Explore **[TEMPO_PACE_DEEP_DIVE.md](TEMPO_PACE_DEEP_DIVE.md)**
3. Review implementation plans (TIER1 & TIER2)

### "I need API documentation"
1. Check **[KENPOM_API.md](KENPOM_API.md)** for complete reference
2. Use **[API_QUICK_REFERENCE.md](API_QUICK_REFERENCE.md)** for quick lookups
3. Review **[KENPOM_DATA_COVERAGE.md](KENPOM_DATA_COVERAGE.md)** for data availability

### "I want to contribute or understand the codebase"
1. Start with **[ANALYTICS_ROADMAP.md](ANALYTICS_ROADMAP.md)**
2. Review **[TIER1_IMPLEMENTATION_PLAN.md](TIER1_IMPLEMENTATION_PLAN.md)** and **[TIER2_IMPLEMENTATION_PLAN.md](TIER2_IMPLEMENTATION_PLAN.md)**
3. Check **[HIGH_VALUE_FEATURES_PLAN.md](HIGH_VALUE_FEATURES_PLAN.md)** for priorities

---

## 📊 Analysis Modules Overview

The KenPom Sports Analyzer provides 7 specialized analysis modules:

| Module | Status | Documentation |
|--------|--------|---------------|
| **Basic Efficiency** | ✅ Complete | Included in examples |
| **Four Factors** | ✅ Complete | TIER1_IMPLEMENTATION_PLAN.md |
| **Point Distribution** | ✅ Complete | TIER1_IMPLEMENTATION_PLAN.md |
| **Defensive Analysis** | ✅ Complete | TIER1_IMPLEMENTATION_PLAN.md |
| **Tempo/Pace** | ✅ Complete | TEMPO_PACE_DEEP_DIVE.md |
| **Size & Athleticism** | ✅ Complete | TIER2_IMPLEMENTATION_PLAN.md |
| **Experience & Chemistry** | ✅ Complete | TIER2_IMPLEMENTATION_PLAN.md |

---

## 🔧 Configuration Files

Located in project root:
- `.env` - API keys and environment variables (create from `.env.example`)
- `CLAUDE.md` - Project-specific instructions for Claude AI
- `pyproject.toml` - Project dependencies and configuration

---

## 📝 Additional Resources

### Code Examples
- `examples/` directory contains working analysis scripts
- `examples/analyze_game_example.py` - Full matchup analysis example
- Other demo scripts for individual modules

### Testing
- `tests/` directory contains comprehensive test suite
- Run tests: `uv run pytest`
- Coverage: `uv run pytest --cov=src/kenp0m_sp0rts_analyzer`

### Source Code
- `src/kenp0m_sp0rts_analyzer/` - Main package
  - `api_client.py` - Official KenPom API client
  - `analysis.py` - Basic matchup analysis
  - `four_factors_matchup.py` - Four Factors module
  - `point_distribution_analysis.py` - Scoring styles
  - `defensive_analysis.py` - Defensive schemes
  - `tempo_analysis.py` - Tempo/pace analysis
  - `size_athleticism_analysis.py` - Physical matchups
  - `experience_chemistry_analysis.py` - Intangibles
  - `comprehensive_matchup_analysis.py` - Integration layer
  - `tournament_simulator.py` - Monte Carlo bracket simulation
  - `prediction.py` - ML prediction engine

---

## 🆘 Getting Help

### Common Issues
- **"KENPOM_API_KEY not found"**: Check [SETUP_GUIDE.md](SETUP_GUIDE.md) for configuration instructions
- **"Team not found"**: Verify team name spelling matches KenPom database
- **API errors**: Review [KENPOM_API.md](KENPOM_API.md) for correct usage

### Support
- GitHub Issues: [Report bugs or request features](https://github.com/omalleyandy/kenp0m-sp0rts-analyzer/issues)
- KenPom API Support: https://kenpom.com/contact.php

---

## 📅 Document Status

| Document | Last Updated | Status |
|----------|--------------|--------|
| SETUP_GUIDE.md | 2025-12-16 | ✅ Current |
| KENPOM_API.md | 2025-12-16 | ✅ Current |
| MATCHUP_ANALYSIS_FRAMEWORK.md | 2025-12-15 | ✅ Current |
| ANALYTICS_ROADMAP.md | 2025-12-15 | ✅ Current |
| All others | 2025-12-15 | ✅ Current |

---

## 🔄 Contributing to Documentation

When adding new documentation:
1. Create markdown file in `docs/`
2. Add entry to this README under appropriate section
3. Update "Document Status" table
4. Link from other relevant docs where appropriate
5. Follow existing documentation style (headers, code blocks, tables)

---

*For the main project README, see [../README.md](../README.md)*
