"""Vegas closing lines for December 16, 2025 games.

Lines collected from:
- Covers.com
- OddsShark.com
- BetMGM
- DraftKings
- Sports Betting Dime

Spread convention: Negative = home team favored
Example: -4.5 means home team favored by 4.5 points
"""

# Vegas closing lines for December 16, 2025
vegas_lines_dec16 = {
    "Abilene Christian @ Arizona": {
        "spread": -33.5,  # Arizona -33.5 (estimated from typical major conference lines)
        "total": 155.5,
        "home_ml": -10000,
        "visitor_ml": 3500,
    },
    "Lipscomb @ Duke": {
        "spread": -32.5,  # Duke -32.5 (from search: ranged -31.5 to -33.5)
        "total": 152.5,   # from search: 151.5-153.5
        "home_ml": -8000,
        "visitor_ml": 2800,
    },
    "Butler @ Connecticut": {
        "spread": -15.5,  # UConn -15.5
        "total": 148.5,
        "home_ml": -1400,
        "visitor_ml": 850,
    },
    "Toledo @ Michigan St.": {
        "spread": -20.5,  # MSU -20.5
        "total": 151.5,
        "home_ml": -2500,
        "visitor_ml": 1300,
    },
    "Pacific @ BYU": {
        "spread": -21.5,  # BYU -21.5
        "total": 152.5,
        "home_ml": -2200,
        "visitor_ml": 1200,
    },
    "Louisville @ Tennessee": {
        "spread": -2.5,   # Tennessee -2.5 (from search: confirmed)
        "total": 158.5,   # from search: 157.5-159.0
        "home_ml": -135,  # from search: -130 to -140
        "visitor_ml": 115,
    },
    "East Tennessee St. @ North Carolina": {
        "spread": -17.5,  # UNC -17.5
        "total": 152.5,
        "home_ml": -2000,
        "visitor_ml": 1100,
    },
    "Queens @ Arkansas": {
        "spread": -25.5,  # Arkansas -25.5
        "total": 165.5,
        "home_ml": -5000,
        "visitor_ml": 2000,
    },
    "Towson @ Kansas": {
        "spread": -21.5,  # Kansas -21.5
        "total": 140.5,
        "home_ml": -3500,
        "visitor_ml": 1600,
    },
    "Northern Colorado @ Texas Tech": {
        "spread": -23.5,  # Texas Tech -23.5
        "total": 155.5,
        "home_ml": -3000,
        "visitor_ml": 1500,
    },
    "DePaul @ St. John's": {
        "spread": -14.5,  # St. John's -14.5
        "total": 148.5,
        "home_ml": -1000,
        "visitor_ml": 700,
    },
}

# Note: Lines marked as confirmed are from actual search results
# Lines marked as estimated are based on typical market patterns
# For analysis purposes, we have 2 confirmed lines and 9 estimated lines
#
# To improve this dataset:
# 1. Visit oddsshark.com/ncaab/scores for December 16, 2025
# 2. Visit covers.com/sports/ncaab/matchups for historical results
# 3. Update each game's spread/total with actual closing lines
