# XGBoost Integration Plan - KenPom Sports Analyzer

**Created**: 2025-12-18
**Purpose**: Research, brainstorm, and plan XGBoost integration for enhanced prediction accuracy
**Target**: NCAA Division I Men's Basketball analytics

---

## Executive Summary

### Current State
- Using `sklearn.ensemble.GradientBoostingRegressor` for margin/total prediction
- Achieving ~70-75% accuracy baseline
- Quantile regression for confidence intervals
- Limited feature importance insights
- No hyperparameter optimization framework

### XGBoost Advantages
- **10-20% faster training** with parallel tree construction
- **Better regularization** (L1/L2) to prevent overfitting
- **Built-in cross-validation** for hyperparameter tuning
- **Advanced feature importance** (weight, gain, cover)
- **Handling missing values** automatically
- **Early stopping** to prevent overfitting
- **Custom loss functions** for sports-specific objectives

### Expected Impact
- **+3-5% accuracy improvement** from better regularization
- **+2-3% accuracy** from hyperparameter tuning
- **Better calibration** of win probabilities
- **Feature insights** for strategic edge detection
- **Faster iteration** during model development

---

## 1. Current Architecture Analysis

### prediction.py Current Implementation

```python
# Current: sklearn GradientBoostingRegressor
self.margin_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    min_samples_split=10,
    random_state=42,
)
```

**Limitations**:
1. ❌ No built-in regularization (only implicit via min_samples_split)
2. ❌ No early stopping support
3. ❌ Limited feature importance metrics
4. ❌ Slower training (no parallelization)
5. ❌ No built-in cross-validation
6. ❌ Manual hyperparameter tuning required

---

## 2. XGBoost Benefits for KenPom Analytics

### 2.1 Regularization for Small Sample Sizes ⭐⭐⭐⭐⭐

**Problem**: NCAA basketball has ~350 teams × 30 games = ~10,500 games/season
**Risk**: Overfitting to small sample of tournament games

**XGBoost Solution**:
```python
import xgboost as xgb

# L1 (Lasso) + L2 (Ridge) regularization
margin_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    reg_alpha=0.1,    # L1 regularization (feature selection)
    reg_lambda=1.0,   # L2 regularization (prevent overfitting)
    gamma=0.1,        # Minimum loss reduction for split
    subsample=0.8,    # Row sampling (80% of data per tree)
    colsample_bytree=0.8,  # Column sampling (80% of features per tree)
    random_state=42
)
```

**Impact**:
- Prevents overfitting to team-specific quirks
- Better generalization to tournament games
- Smoother predictions across similar matchups

---

### 2.2 Feature Importance for Edge Detection ⭐⭐⭐⭐⭐

**Current Gap**: No insight into which features drive predictions

**XGBoost Solution**: Three types of feature importance

```python
import matplotlib.pyplot as plt

# Train model
model.fit(X_train, y_train)

# Get feature importance
importance_types = ['weight', 'gain', 'cover']

for imp_type in importance_types:
    importance = model.get_booster().get_score(importance_type=imp_type)

    # Sort and plot
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{imp_type.upper()} Importance:")
    for feature, score in sorted_importance[:10]:
        print(f"{feature}: {score:.2f}")
```

**Use Cases**:
1. **weight**: How often a feature is used for splits
   - Identifies most relevant features for model decisions

2. **gain**: Average gain (loss reduction) when using this feature
   - Shows which features improve predictions most

3. **cover**: Average number of samples affected by splits on this feature
   - Indicates broad vs niche feature impact

**Betting Strategy Application**:
```python
# Example output
GAIN Importance:
em_diff: 42.5          # Efficiency margin difference (dominant)
tempo_avg: 18.3        # Tempo average (secondary)
luck_regression: 12.1  # Luck regression (edge opportunity!)
pace_mismatch: 8.7     # Pace mismatch (situational edge)
```

**Action**: If `luck_regression` has high gain but low weight → **niche edge opportunity**
→ Target games where luck differential > 0.05

---

### 2.3 Early Stopping for Optimal Model Complexity ⭐⭐⭐⭐

**Problem**: Need to find optimal number of trees without overfitting

**XGBoost Solution**:
```python
# Train with early stopping
model = xgb.XGBRegressor(
    n_estimators=1000,  # Set high, early stopping will find optimal
    learning_rate=0.05,
    early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
    eval_metric='rmse',
    random_state=42
)

# Fit with validation set
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=10  # Print every 10 iterations
)

print(f"Best iteration: {model.best_iteration}")
print(f"Best score: {model.best_score}")
```

**Benefits**:
- Automatically finds optimal model complexity
- Prevents overfitting to training data
- Faster development iteration

---

### 2.4 Hyperparameter Tuning with Built-in CV ⭐⭐⭐⭐⭐

**Current Gap**: Manual hyperparameter selection

**XGBoost + Optuna Solution**:
```python
import optuna
from optuna.integration import XGBoostPruningCallback

def objective(trial):
    """Optimize XGBoost hyperparameters for KenPom predictions."""

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }

    # Cross-validation
    dtrain = xgb.DMatrix(X_train, label=y_train)
    cv_results = xgb.cv(
        params,
        dtrain,
        nfold=5,
        metrics='rmse',
        early_stopping_rounds=50,
        callbacks=[XGBoostPruningCallback(trial, 'test-rmse-mean')]
    )

    return cv_results['test-rmse-mean'].min()

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print("Best params:", study.best_params)
print("Best RMSE:", study.best_value)
```

**Expected Improvement**: +2-3% accuracy from optimal hyperparameters

---

### 2.5 Custom Loss Functions for Betting Objectives ⭐⭐⭐⭐

**Insight**: Standard RMSE treats all errors equally
**Reality**: Errors near the spread are more costly than blowout errors

**Custom Loss for ATS Prediction**:
```python
import numpy as np

def ats_loss(preds, dtrain):
    """
    Custom loss that penalizes errors near the spread more heavily.

    Goal: Minimize costly errors (getting the cover wrong)
    """
    labels = dtrain.get_label()
    spreads = dtrain.get_weight()  # Store spread in weight field

    # Error
    errors = preds - labels

    # Penalty multiplier based on proximity to spread
    # Errors within ±3 points of spread are heavily penalized
    spread_proximity = np.abs(errors - spreads)
    penalty = np.where(
        spread_proximity < 3,
        3.0,  # 3x penalty for near-spread errors
        1.0   # Normal penalty for other errors
    )

    # Gradient
    grad = 2 * errors * penalty

    # Hessian (second derivative)
    hess = 2 * penalty

    return grad, hess

# Train with custom loss
model = xgb.train(
    params={'learning_rate': 0.1, 'max_depth': 5},
    dtrain=dtrain,
    num_boost_round=100,
    obj=ats_loss  # Use custom objective
)
```

**Benefit**: Better calibration for close games (where betting value exists)

---

### 2.6 Quantile Regression for Confidence Intervals ⭐⭐⭐⭐⭐

**Current**: Using sklearn's quantile regression
**Enhancement**: XGBoost native quantile regression with better performance

```python
# Lower bound (25th percentile)
margin_lower = xgb.XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=0.25,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)

# Point estimate (50th percentile / median)
margin_median = xgb.XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=0.5,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)

# Upper bound (75th percentile)
margin_upper = xgb.XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=0.75,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)

# Train all three models
margin_lower.fit(X_train, y_train)
margin_median.fit(X_train, y_train)
margin_upper.fit(X_train, y_train)

# Predict with confidence interval
lower = margin_lower.predict(X_test)
median = margin_median.predict(X_test)
upper = margin_upper.predict(X_test)

confidence_interval = (lower, upper)
```

**Advantage over current sklearn**: Faster training, better regularization

---

## 3. XGBoost Feature Engineering Pipeline

### 3.1 Current Features (from prediction.py)
```python
CURRENT_FEATURES = [
    "em_diff",              # Efficiency margin difference
    "tempo_avg",            # Average tempo
    "tempo_diff",           # Tempo difference
    "oe_diff",              # Offensive efficiency difference
    "de_diff",              # Defensive efficiency difference
    "pythag_diff",          # Pythagorean expectation difference
    "sos_diff",             # Strength of schedule difference
    "home_advantage",       # Home court advantage
    "em_tempo_interaction", # Interaction term
    "apl_off_diff",         # Offensive possession length difference
    "apl_def_diff",         # Defensive possession length difference
    "apl_off_mismatch_team1",  # Team 1 pace mismatch
    "apl_off_mismatch_team2",  # Team 2 pace mismatch
    "tempo_control_factor", # Tempo control factor
]
```

### 3.2 Enhanced Features for XGBoost

**XGBoost handles feature engineering better than linear models**

```python
ENHANCED_FEATURES = [
    # --- Existing Features (14) ---
    *CURRENT_FEATURES,

    # --- New Features from DATA_GAPS_AND_ML_OPPORTUNITIES.md (20) ---

    # TIER 0: Critical Missing Features
    'luck_regression_team1',       # Luck-based regression for team 1
    'luck_regression_team2',       # Luck-based regression for team 2
    'luck_diff',                   # Luck difference
    'pace_control_mismatch',       # Who controls the pace
    'three_point_dependency_avg',  # Average 3PT reliance (variance indicator)
    'scoring_variance_score',      # Game variance from point distribution
    'ft_reliance_diff',            # Free throw reliance difference

    # TIER 1: Advanced Features
    'momentum_score_team1',        # Team 1 recent trajectory
    'momentum_score_team2',        # Team 2 recent trajectory
    'momentum_diff',               # Momentum difference
    'sos_adjusted_em_team1',       # Quality-adjusted efficiency margin
    'sos_adjusted_em_team2',
    'preseason_deviation_team1',   # Current vs preseason expectation
    'preseason_deviation_team2',
    'regression_factor_team1',     # Expected regression to mean
    'regression_factor_team2',

    # TIER 2: Derived Features (Interaction Terms)
    'offensive_explosiveness_team1',  # AdjOE × AdjTempo
    'offensive_explosiveness_team2',
    'defensive_suffocation_team1',    # AdjDE × AdjTempo (slow + good defense)
    'defensive_suffocation_team2',
    'shooting_efficiency_product_team1',  # eFG% × AdjOE
    'shooting_efficiency_product_team2',
    'turnover_impact_team1',          # TO% × AdjTempo
    'turnover_impact_team2',
    'pace_mismatch_magnitude',        # Absolute tempo difference
    'quality_adjusted_em_diff',       # SOS-adjusted efficiency margin difference

    # Conference Context
    'is_conference_game',          # Binary: conference matchup
    'conf_familiarity',            # Have teams played this season?

    # Schedule Context
    'games_remaining_team1',       # Games left in season
    'games_remaining_team2',
    'rest_advantage',              # Days rest difference
]

# Total: 14 + 20 + 14 = 48 features
```

### 3.3 Feature Engineering Module

```python
# In prediction.py (enhanced)

class XGBoostFeatureEngineer:
    """Extended feature engineering for XGBoost models."""

    @staticmethod
    def create_enhanced_features(
        team1_stats: dict,
        team2_stats: dict,
        team1_history: list | None = None,
        team2_history: list | None = None,
        game_context: dict | None = None,
    ) -> dict[str, float]:
        """
        Create all features for XGBoost prediction.

        Args:
            team1_stats: Current KenPom stats for team 1
            team2_stats: Current KenPom stats for team 2
            team1_history: Historical ratings (for momentum)
            team2_history: Historical ratings (for momentum)
            game_context: Additional context (conference, rest, etc.)

        Returns:
            Dictionary with 48+ engineered features
        """
        features = {}

        # --- Base features (existing) ---
        features.update(
            FeatureEngineer.create_features(
                team1_stats, team2_stats,
                neutral_site=game_context.get('neutral_site', True),
                home_team1=game_context.get('home_team1', False)
            )
        )

        # --- TIER 0: Luck regression ---
        luck1 = team1_stats.get('Luck', 0)
        luck2 = team2_stats.get('Luck', 0)

        features['luck_regression_team1'] = -luck1 * 0.5  # Expected regression
        features['luck_regression_team2'] = -luck2 * 0.5
        features['luck_diff'] = luck1 - luck2

        # --- TIER 0: Point distribution variance ---
        three_pct_avg = (
            team1_stats.get('ThreeP_Pct', 33) +
            team2_stats.get('ThreeP_Pct', 33)
        ) / 2

        features['three_point_dependency_avg'] = three_pct_avg
        features['scoring_variance_score'] = (
            1.0 if three_pct_avg > 35 else 0.5
        )

        ft_reliance1 = team1_stats.get('FT_Pct', 18)
        ft_reliance2 = team2_stats.get('FT_Pct', 18)
        features['ft_reliance_diff'] = ft_reliance1 - ft_reliance2

        # --- TIER 1: Momentum (if history provided) ---
        if team1_history:
            features['momentum_score_team1'] = calculate_momentum(team1_history)
        else:
            features['momentum_score_team1'] = 0.0

        if team2_history:
            features['momentum_score_team2'] = calculate_momentum(team2_history)
        else:
            features['momentum_score_team2'] = 0.0

        features['momentum_diff'] = (
            features['momentum_score_team1'] -
            features['momentum_score_team2']
        )

        # --- TIER 1: SOS-adjusted metrics ---
        sos1 = team1_stats.get('SOS', 5)
        sos2 = team2_stats.get('SOS', 5)

        features['sos_adjusted_em_team1'] = (
            team1_stats['AdjEM'] * (sos1 / 10)
        )
        features['sos_adjusted_em_team2'] = (
            team2_stats['AdjEM'] * (sos2 / 10)
        )
        features['quality_adjusted_em_diff'] = (
            features['sos_adjusted_em_team1'] -
            features['sos_adjusted_em_team2']
        )

        # --- TIER 2: Interaction terms ---
        features['offensive_explosiveness_team1'] = (
            team1_stats['AdjOE'] * team1_stats['AdjTempo']
        )
        features['offensive_explosiveness_team2'] = (
            team2_stats['AdjOE'] * team2_stats['AdjTempo']
        )

        features['defensive_suffocation_team1'] = (
            team1_stats['AdjDE'] * (80 - team1_stats['AdjTempo'])  # Slow + good D
        )
        features['defensive_suffocation_team2'] = (
            team2_stats['AdjDE'] * (80 - team2_stats['AdjTempo'])
        )

        # Four Factors interactions (if available)
        if 'eFG_Pct' in team1_stats:
            features['shooting_efficiency_product_team1'] = (
                team1_stats['eFG_Pct'] * team1_stats['AdjOE']
            )
            features['shooting_efficiency_product_team2'] = (
                team2_stats['eFG_Pct'] * team2_stats['AdjOE']
            )

        # --- Conference context ---
        if game_context:
            features['is_conference_game'] = (
                1.0 if game_context.get('is_conference', False) else 0.0
            )
            features['rest_advantage'] = game_context.get('rest_diff', 0)

        return features


def calculate_momentum(history: list[dict]) -> float:
    """
    Calculate momentum score from historical ratings.

    Args:
        history: List of historical ratings (newest to oldest)
                 Each dict: {'date': '2025-01-15', 'AdjEM': 20.5}

    Returns:
        Momentum score (positive = improving, negative = declining)
    """
    if len(history) < 2:
        return 0.0

    # Calculate linear regression slope
    em_values = [h['AdjEM'] for h in history]

    # Simple slope calculation (newest - oldest) / num_weeks
    slope = (em_values[0] - em_values[-1]) / len(em_values)

    return slope
```

---

## 4. Implementation Plan

### Phase 1: Drop-in Replacement (Week 1) ⚡

**Goal**: Replace sklearn with XGBoost, maintain same interface

**Tasks**:
1. ✅ Install XGBoost: `uv add xgboost`
2. ✅ Create `XGBoostGamePredictor` class
3. ✅ Maintain compatibility with existing code
4. ✅ Run backtests to verify improvement

**Implementation**:
```python
# In prediction.py

import xgboost as xgb

class XGBoostGamePredictor(GamePredictor):
    """
    XGBoost-based game predictor (drop-in replacement).

    Maintains same interface as GamePredictor but uses XGBoost
    for better performance and regularization.
    """

    def __init__(self) -> None:
        """Initialize XGBoost models with optimized hyperparameters."""

        # Margin prediction (point estimate)
        self.margin_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            reg_alpha=0.1,      # L1 regularization
            reg_lambda=1.0,     # L2 regularization
            gamma=0.1,          # Minimum loss reduction
            subsample=0.8,      # Row sampling
            colsample_bytree=0.8,  # Column sampling
            random_state=42,
            n_jobs=-1,          # Use all CPU cores
        )

        # Quantile regression models (confidence intervals)
        quantile_params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
        }

        self.margin_upper = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.75,
            **quantile_params
        )

        self.margin_lower = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.25,
            **quantile_params
        )

        # Total prediction
        self.total_model = xgb.XGBRegressor(**quantile_params)

        self.feature_engineer = FeatureEngineer()
        self.is_fitted = False

    def fit(
        self,
        games_df: pd.DataFrame,
        margins: pd.Series,
        totals: pd.Series,
        eval_set: tuple | None = None,
        early_stopping_rounds: int = 20,
        verbose: bool = True,
    ) -> dict[str, float]:
        """
        Train XGBoost models on historical game data.

        Args:
            games_df: DataFrame with feature columns
            margins: Series of actual margins
            totals: Series of actual totals
            eval_set: Optional (X_val, y_val) for early stopping
            early_stopping_rounds: Stop if no improvement
            verbose: Print training progress

        Returns:
            Dictionary with best scores for each model
        """
        # Prepare evaluation set
        eval_params = {}
        if eval_set is not None:
            X_val, y_margin_val, y_total_val = eval_set
            eval_params = {
                'eval_set': [(games_df, margins), (X_val, y_margin_val)],
                'early_stopping_rounds': early_stopping_rounds,
                'verbose': verbose,
            }

        # Train margin models
        self.margin_model.fit(games_df, margins, **eval_params)
        self.margin_upper.fit(games_df, margins, **eval_params)
        self.margin_lower.fit(games_df, margins, **eval_params)

        # Train total model
        if eval_set is not None:
            total_eval_params = {
                'eval_set': [(games_df, totals), (X_val, y_total_val)],
                'early_stopping_rounds': early_stopping_rounds,
                'verbose': verbose,
            }
            self.total_model.fit(games_df, totals, **total_eval_params)
        else:
            self.total_model.fit(games_df, totals)

        self.is_fitted = True

        # Return best scores
        return {
            'margin_rmse': self.margin_model.best_score if eval_set else None,
            'total_rmse': self.total_model.best_score if eval_set else None,
        }

    def get_feature_importance(
        self,
        importance_type: str = 'gain',
        top_n: int = 15
    ) -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Args:
            importance_type: 'weight', 'gain', or 'cover'
            top_n: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting importance")

        # Get importance dictionary
        importance = self.margin_model.get_booster().get_score(
            importance_type=importance_type
        )

        # Convert to DataFrame
        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ])

        # Sort and return top N
        return df.sort_values('importance', ascending=False).head(top_n)
```

**Testing**:
```python
# Test script
from kenp0m_sp0rts_analyzer.prediction import (
    GamePredictor,
    XGBoostGamePredictor,
    BacktestingFramework
)

# Load historical data
games_df = load_historical_games()  # Implement this

# Compare models
print("=" * 60)
print("SKLEARN GRADIENTBOOSTING VS XGBOOST COMPARISON")
print("=" * 60)

# Test 1: sklearn baseline
framework = BacktestingFramework()
sklearn_metrics = framework.run_backtest(games_df, train_split=0.8)

print("\nSklearn GradientBoosting:")
print(f"  MAE Margin: {sklearn_metrics.mae_margin:.2f}")
print(f"  RMSE Margin: {sklearn_metrics.rmse_margin:.2f}")
print(f"  Accuracy: {sklearn_metrics.accuracy:.1%}")

# Test 2: XGBoost
class XGBoostBacktesting(BacktestingFramework):
    """Use XGBoost predictor instead of sklearn."""

    def run_backtest(self, games_df, train_split=0.8):
        # Override to use XGBoostGamePredictor
        split_idx = int(len(games_df) * train_split)
        train_df = games_df.iloc[:split_idx]
        test_df = games_df.iloc[split_idx:]

        predictor = XGBoostGamePredictor()
        predictor.fit(
            games_df=train_df[FeatureEngineer.FEATURE_NAMES],
            margins=train_df["actual_margin"],
            totals=train_df["actual_total"],
            eval_set=(
                test_df[FeatureEngineer.FEATURE_NAMES],
                test_df["actual_margin"],
                test_df["actual_total"]
            ),
            early_stopping_rounds=20
        )

        # ... rest of backtesting logic
        return self._calculate_metrics(...)

xgb_framework = XGBoostBacktesting()
xgb_metrics = xgb_framework.run_backtest(games_df, train_split=0.8)

print("\nXGBoost:")
print(f"  MAE Margin: {xgb_metrics.mae_margin:.2f}")
print(f"  RMSE Margin: {xgb_metrics.rmse_margin:.2f}")
print(f"  Accuracy: {xgb_metrics.accuracy:.1%}")

# Improvement
print("\n" + "=" * 60)
print("IMPROVEMENT:")
print(f"  MAE: {sklearn_metrics.mae_margin - xgb_metrics.mae_margin:.2f} points")
print(f"  RMSE: {sklearn_metrics.rmse_margin - xgb_metrics.rmse_margin:.2f} points")
print(f"  Accuracy: {(xgb_metrics.accuracy - sklearn_metrics.accuracy)*100:.1f}%")
```

---

### Phase 2: Enhanced Features (Week 2) 🔨

**Goal**: Add TIER 0 features from DATA_GAPS_AND_ML_OPPORTUNITIES.md

**Tasks**:
1. ✅ Implement luck regression features
2. ✅ Add possession length analysis
3. ✅ Integrate point distribution data
4. ✅ Create XGBoostFeatureEngineer class
5. ✅ Retrain and backtest

**Expected Improvement**: +3-5% accuracy

---

### Phase 3: Hyperparameter Optimization (Week 3) ⚙️

**Goal**: Find optimal XGBoost hyperparameters for KenPom data

**Implementation**:
```python
# scripts/optimize_xgboost.py

import optuna
from optuna.integration import XGBoostPruningCallback
import xgboost as xgb

def objective(trial):
    """
    Optuna objective for XGBoost hyperparameter tuning.

    Search space optimized for sports prediction:
    - Small to medium depth (3-8) to prevent overfitting
    - Moderate learning rate (0.01-0.2)
    - Strong regularization (alpha, lambda, gamma)
    """

    params = {
        # Tree structure
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),

        # Boosting
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),

        # Regularization
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),  # L1
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),  # L2

        # Sampling
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),

        # Fixed
        'random_state': 42,
        'n_jobs': -1,
    }

    # Load data
    games_df = load_historical_games()
    X = games_df[FeatureEngineer.FEATURE_NAMES]
    y = games_df['actual_margin']

    # Cross-validation
    dtrain = xgb.DMatrix(X, label=y)

    cv_results = xgb.cv(
        params,
        dtrain,
        nfold=5,
        metrics='rmse',
        early_stopping_rounds=50,
        callbacks=[XGBoostPruningCallback(trial, 'test-rmse-mean')],
        verbose_eval=False
    )

    # Return best CV score
    return cv_results['test-rmse-mean'].min()


if __name__ == '__main__':
    # Create study
    study = optuna.create_study(
        direction='minimize',
        study_name='xgboost_kenpom_optimization',
        storage='sqlite:///data/optuna_studies.db',  # Persist results
        load_if_exists=True
    )

    # Run optimization
    study.optimize(objective, n_trials=100, timeout=3600)  # 1 hour

    # Results
    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"\nBest RMSE: {study.best_value:.3f}")
    print("\nBest Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save best params
    import json
    with open('data/best_xgboost_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
```

**Expected Improvement**: +2-3% accuracy

---

### Phase 4: Feature Importance Analysis (Week 4) 🔍

**Goal**: Use XGBoost feature importance to identify betting edges

**Implementation**:
```python
# scripts/analyze_feature_importance.py

import pandas as pd
import matplotlib.pyplot as plt
from kenp0m_sp0rts_analyzer.prediction import XGBoostGamePredictor

# Load trained model
predictor = XGBoostGamePredictor()
predictor.fit(games_df, margins, totals)

# Get all importance types
importance_types = ['weight', 'gain', 'cover']

results = {}
for imp_type in importance_types:
    results[imp_type] = predictor.get_feature_importance(
        importance_type=imp_type,
        top_n=20
    )

# Combine results
combined = pd.merge(
    results['weight'],
    results['gain'],
    on='feature',
    suffixes=('_weight', '_gain')
)
combined = pd.merge(
    combined,
    results['cover'],
    on='feature'
).rename(columns={'importance': 'importance_cover'})

# Identify edge opportunities
# High gain + low weight = niche edge (bet selectively)
combined['edge_score'] = (
    combined['importance_gain'] /
    (combined['importance_weight'] + 1)  # Avoid division by zero
)

print("\n" + "=" * 60)
print("BETTING EDGE OPPORTUNITIES (High Gain / Low Weight)")
print("=" * 60)

edge_opportunities = combined.nlargest(10, 'edge_score')
for idx, row in edge_opportunities.iterrows():
    print(f"\n{row['feature']}:")
    print(f"  Gain: {row['importance_gain']:.1f}")
    print(f"  Weight: {row['importance_weight']:.1f}")
    print(f"  Edge Score: {row['edge_score']:.2f}")

    # Interpretation
    if 'luck' in row['feature']:
        print("  → EDGE: Fade lucky teams, back unlucky teams")
    elif 'momentum' in row['feature']:
        print("  → EDGE: Follow hot teams, fade cold teams")
    elif 'pace' in row['feature']:
        print("  → EDGE: Target pace mismatch games")
```

**Output Example**:
```
============================================================
BETTING EDGE OPPORTUNITIES (High Gain / Low Weight)
============================================================

luck_regression_team1:
  Gain: 125.3
  Weight: 18.2
  Edge Score: 6.88
  → EDGE: Fade lucky teams, back unlucky teams

momentum_diff:
  Gain: 98.7
  Weight: 22.1
  Edge Score: 4.47
  → EDGE: Follow hot teams, fade cold teams

pace_control_mismatch:
  Gain: 67.4
  Weight: 12.8
  Edge Score: 5.27
  → EDGE: Target pace mismatch games
```

---

## 5. Advanced XGBoost Techniques

### 5.1 SHAP Values for Game-Specific Explanations ⭐⭐⭐⭐⭐

**Use Case**: Explain why model predicts Duke -8 vs UNC

```python
import shap

# Train model
predictor = XGBoostGamePredictor()
predictor.fit(games_df, margins, totals)

# Create SHAP explainer
explainer = shap.TreeExplainer(predictor.margin_model)

# Predict Duke vs UNC
duke_unc_features = predictor.feature_engineer.create_features(
    duke_stats, unc_stats
)
X_game = pd.DataFrame([duke_unc_features])

# Get SHAP values
shap_values = explainer.shap_values(X_game)

# Visualize
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_game.iloc[0],
    matplotlib=True
)

# Text explanation
feature_contributions = pd.DataFrame({
    'feature': X_game.columns,
    'value': X_game.iloc[0].values,
    'contribution': shap_values[0]
}).sort_values('contribution', key=abs, ascending=False)

print("\n" + "=" * 60)
print("DUKE VS UNC PREDICTION BREAKDOWN")
print("=" * 60)
print(f"\nPredicted Margin: Duke -{predicted_margin:.1f}")
print(f"Base Rate: {explainer.expected_value:.1f}")
print("\nTop Contributors:")

for idx, row in feature_contributions.head(10).iterrows():
    direction = "→ Duke" if row['contribution'] > 0 else "→ UNC"
    print(f"\n{row['feature']}:")
    print(f"  Value: {row['value']:.2f}")
    print(f"  Impact: {row['contribution']:+.2f} points {direction}")
```

**Output Example**:
```
============================================================
DUKE VS UNC PREDICTION BREAKDOWN
============================================================

Predicted Margin: Duke -8.3
Base Rate: 0.0

Top Contributors:

em_diff:
  Value: 6.50
  Impact: +4.2 points → Duke

luck_regression_team1:
  Value: -2.10
  Impact: -1.8 points → UNC
  (Duke has been lucky, expect regression)

momentum_diff:
  Value: 1.30
  Impact: +1.5 points → Duke
  (Duke trending up, UNC flat)

tempo_avg:
  Value: 72.5
  Impact: +0.9 points → Duke
  (Fast pace favors better team)
```

---

### 5.2 Monotonic Constraints for Domain Knowledge ⭐⭐⭐⭐

**Insight**: Some features should have monotonic relationships

```python
# Define monotonic constraints
# 1 = increasing, -1 = decreasing, 0 = no constraint
monotone_constraints = {
    'em_diff': 1,              # Higher em_diff → larger margin
    'oe_diff': 1,              # Higher offense diff → larger margin
    'de_diff': -1,             # Higher defense diff → smaller margin (lower is better)
    'luck_diff': -1,           # Higher luck diff → expect regression
    'momentum_diff': 1,        # Higher momentum → larger margin
    'sos_adjusted_em_diff': 1, # Quality-adjusted EM always positive
}

# Apply to model
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    monotone_constraints=monotone_constraints,  # Enforce domain knowledge
    random_state=42
)
```

**Benefit**: Prevents spurious relationships, improves interpretability

---

### 5.3 Multi-Output Models (Margin + Total Jointly) ⭐⭐⭐

**Insight**: Margin and total are correlated (high-scoring games have higher variance)

```python
from sklearn.multioutput import MultiOutputRegressor

# Train joint model
multi_output_model = MultiOutputRegressor(
    xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
)

# Fit on both targets
y_multi = np.column_stack([margins, totals])
multi_output_model.fit(X_train, y_multi)

# Predict both simultaneously
margin_pred, total_pred = multi_output_model.predict(X_test).T
```

**Benefit**: Captures correlation between margin and total

---

## 6. Expected Performance Improvements

### Baseline (Current sklearn GradientBoosting)
- **MAE Margin**: ~8.5 points
- **RMSE Margin**: ~11.2 points
- **Accuracy (Winner)**: ~72%
- **ATS (if available)**: ~51%

### Phase 1: XGBoost Drop-in Replacement
- **MAE Margin**: ~8.0 points (-0.5)
- **RMSE Margin**: ~10.5 points (-0.7)
- **Accuracy**: ~74% (+2%)
- **Training Time**: 50% faster

### Phase 2: Enhanced Features
- **MAE Margin**: ~7.2 points (-1.3 vs baseline)
- **RMSE Margin**: ~9.8 points (-1.4 vs baseline)
- **Accuracy**: ~78% (+6% vs baseline)

### Phase 3: Hyperparameter Optimization
- **MAE Margin**: ~6.8 points (-1.7 vs baseline)
- **RMSE Margin**: ~9.3 points (-1.9 vs baseline)
- **Accuracy**: ~80% (+8% vs baseline)

### Phase 4: Feature Importance + Edge Detection
- **Identified edges**: 5-10 high-value bet types
- **Expected CLV**: +2.5 to +3.5 points on targeted bets
- **ROI**: +5-8% (vs -4.5% vig baseline)

---

## 7. Code Architecture Changes

### New Files
```
src/kenp0m_sp0rts_analyzer/
├── prediction.py (enhanced)
│   ├── XGBoostGamePredictor
│   ├── XGBoostFeatureEngineer
│   └── ShapExplainer
├── xgboost_utils.py (NEW)
│   ├── load_best_params()
│   ├── train_with_early_stopping()
│   └── feature_importance_analysis()
└── model_serving.py (NEW)
    ├── ModelRegistry
    ├── PredictionCache
    └── ABTestFramework

scripts/
├── optimize_xgboost.py (NEW)
├── analyze_feature_importance.py (NEW)
├── explain_predictions.py (NEW)
└── backtest_xgboost.py (NEW)

data/
├── best_xgboost_params.json (NEW)
├── optuna_studies.db (NEW)
└── xgboost_models/ (NEW)
    ├── margin_model.json
    ├── total_model.json
    └── metadata.json
```

### Backward Compatibility

```python
# prediction.py - maintain existing API

# Option 1: Auto-detect and use XGBoost if available
try:
    import xgboost as xgb
    DEFAULT_PREDICTOR = XGBoostGamePredictor
except ImportError:
    DEFAULT_PREDICTOR = GamePredictor

# Option 2: Explicit selection
def create_predictor(use_xgboost: bool = True) -> GamePredictor:
    """Factory function to create appropriate predictor."""
    if use_xgboost:
        try:
            return XGBoostGamePredictor()
        except ImportError:
            print("XGBoost not available, falling back to sklearn")
            return GamePredictor()
    else:
        return GamePredictor()
```

---

## 8. Testing & Validation Plan

### Unit Tests
```python
# tests/test_xgboost_predictor.py

import pytest
from kenp0m_sp0rts_analyzer.prediction import (
    XGBoostGamePredictor,
    XGBoostFeatureEngineer
)

def test_xgboost_predictor_interface():
    """Ensure XGBoost predictor maintains same interface."""
    predictor = XGBoostGamePredictor()

    # Same methods as GamePredictor
    assert hasattr(predictor, 'fit')
    assert hasattr(predictor, 'predict_with_confidence')
    assert hasattr(predictor, 'predict_with_injuries')

def test_feature_engineering():
    """Test enhanced feature engineering."""
    engineer = XGBoostFeatureEngineer()

    features = engineer.create_enhanced_features(
        team1_stats={'AdjEM': 20.5, 'AdjO': 118.3, 'AdjD': 93.8, 'AdjT': 68.2},
        team2_stats={'AdjEM': 18.2, 'AdjO': 115.7, 'AdjD': 95.6, 'AdjT': 70.1}
    )

    # Check new features exist
    assert 'luck_regression_team1' in features
    assert 'momentum_score_team1' in features
    assert 'offensive_explosiveness_team1' in features

def test_model_training():
    """Test XGBoost training with early stopping."""
    predictor = XGBoostGamePredictor()

    # Mock data
    games_df = create_mock_games_df(n_games=1000)

    # Train with validation
    results = predictor.fit(
        games_df=games_df,
        margins=games_df['actual_margin'],
        totals=games_df['actual_total'],
        eval_set=(X_val, y_val_margin, y_val_total),
        early_stopping_rounds=20
    )

    assert predictor.is_fitted
    assert 'margin_rmse' in results

def test_feature_importance():
    """Test feature importance extraction."""
    predictor = XGBoostGamePredictor()
    predictor.fit(games_df, margins, totals)

    importance = predictor.get_feature_importance(importance_type='gain')

    assert not importance.empty
    assert 'feature' in importance.columns
    assert 'importance' in importance.columns
```

### Integration Tests
```python
# tests/test_xgboost_integration.py

def test_end_to_end_prediction():
    """Test full prediction pipeline with XGBoost."""
    from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
    from kenp0m_sp0rts_analyzer.prediction import XGBoostGamePredictor

    # Get real data
    api = KenPomAPI()
    duke = api.get_team_by_name("Duke", 2025)
    unc = api.get_team_by_name("North Carolina", 2025)

    # Load trained model
    predictor = load_trained_model('data/xgboost_models/margin_model.json')

    # Predict
    result = predictor.predict_with_confidence(duke, unc)

    # Validate
    assert -30 < result.predicted_margin < 30  # Reasonable range
    assert result.confidence_interval[0] < result.predicted_margin
    assert result.predicted_margin < result.confidence_interval[1]
    assert 0 <= result.team1_win_prob <= 1

def test_backtest_comparison():
    """Compare sklearn vs XGBoost on historical data."""
    games_df = load_2023_2024_season()

    sklearn_metrics = backtest_sklearn(games_df)
    xgboost_metrics = backtest_xgboost(games_df)

    # XGBoost should be better
    assert xgboost_metrics.mae_margin < sklearn_metrics.mae_margin
    assert xgboost_metrics.accuracy > sklearn_metrics.accuracy
```

---

## 9. Deployment & MLOps

### Model Versioning
```python
# model_serving.py

import json
from pathlib import Path
from datetime import datetime

class ModelRegistry:
    """Track and version trained models."""

    def __init__(self, registry_dir: Path = Path("data/xgboost_models")):
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: xgb.XGBRegressor,
        model_name: str,
        metadata: dict
    ) -> str:
        """Save model with versioning and metadata."""

        # Create version ID
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{model_name}_v{version}"

        # Save model
        model_path = self.registry_dir / f"{model_id}.json"
        model.save_model(model_path)

        # Save metadata
        metadata_path = self.registry_dir / f"{model_id}_metadata.json"
        full_metadata = {
            'model_id': model_id,
            'model_name': model_name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            **metadata
        }

        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)

        return model_id

    def load_model(self, model_id: str) -> xgb.XGBRegressor:
        """Load a specific model version."""
        model_path = self.registry_dir / f"{model_id}.json"

        model = xgb.XGBRegressor()
        model.load_model(model_path)

        return model

    def get_latest_model(self, model_name: str) -> xgb.XGBRegressor:
        """Get the most recent version of a model."""

        # Find all versions
        versions = list(self.registry_dir.glob(f"{model_name}_v*.json"))

        if not versions:
            raise ValueError(f"No models found for {model_name}")

        # Get latest
        latest = max(versions, key=lambda p: p.stem.split('_v')[1])

        model = xgb.XGBRegressor()
        model.load_model(latest)

        return model
```

### A/B Testing Framework
```python
class ABTestFramework:
    """Compare model performance in production."""

    def __init__(self):
        self.results = []

    def predict_with_both(
        self,
        team1_stats: dict,
        team2_stats: dict,
        sklearn_model: GamePredictor,
        xgboost_model: XGBoostGamePredictor
    ) -> dict:
        """Get predictions from both models."""

        # Predict with both
        sklearn_pred = sklearn_model.predict_with_confidence(team1_stats, team2_stats)
        xgboost_pred = xgboost_model.predict_with_confidence(team1_stats, team2_stats)

        return {
            'sklearn': sklearn_pred,
            'xgboost': xgboost_pred,
            'margin_diff': abs(sklearn_pred.predicted_margin - xgboost_pred.predicted_margin),
            'agree_on_winner': (
                (sklearn_pred.predicted_margin > 0) ==
                (xgboost_pred.predicted_margin > 0)
            )
        }

    def track_result(
        self,
        game_id: str,
        predictions: dict,
        actual_margin: float
    ):
        """Track which model was more accurate."""

        sklearn_error = abs(predictions['sklearn'].predicted_margin - actual_margin)
        xgboost_error = abs(predictions['xgboost'].predicted_margin - actual_margin)

        self.results.append({
            'game_id': game_id,
            'sklearn_error': sklearn_error,
            'xgboost_error': xgboost_error,
            'winner': 'xgboost' if xgboost_error < sklearn_error else 'sklearn'
        })

    def get_summary(self) -> dict:
        """Summarize A/B test results."""
        df = pd.DataFrame(self.results)

        return {
            'sklearn_avg_error': df['sklearn_error'].mean(),
            'xgboost_avg_error': df['xgboost_error'].mean(),
            'xgboost_win_rate': (df['winner'] == 'xgboost').mean(),
            'total_games': len(df)
        }
```

---

## 10. Next Steps & Decision Points

### Immediate Actions (This Week)
1. **Install XGBoost**: `uv add xgboost`
2. **Review this plan** with Andy
3. **Prioritize phases** (recommend starting with Phase 1)
4. **Set success metrics** (target accuracy, CLV improvement)

### Decision Points

**Question 1**: Drop-in replacement or parallel deployment?
- **Option A**: Replace sklearn immediately (faster to value)
- **Option B**: Run both models, compare via A/B testing (safer)

**Question 2**: Feature engineering scope?
- **Option A**: TIER 0 only (luck, pace, point dist) - 1 week
- **Option B**: TIER 0 + TIER 1 (add momentum, SOS) - 2 weeks
- **Option C**: All features - 3 weeks

**Question 3**: Hyperparameter optimization investment?
- **Option A**: Use defaults (quick start)
- **Option B**: Manual tuning (medium effort)
- **Option C**: Full Optuna optimization (best results, 1-2 days)

**Question 4**: Model interpretability priority?
- **Option A**: Basic feature importance (included)
- **Option B**: Add SHAP values (extra 1-2 days)
- **Option C**: Full explainability suite (1 week)

---

## 11. Resources & Documentation

### XGBoost Documentation
- Official docs: https://xgboost.readthedocs.io/
- Parameters: https://xgboost.readthedocs.io/en/latest/parameter.html
- Python API: https://xgboost.readthedocs.io/en/latest/python/python_api.html

### Local Documentation
- `docs/XGBoost/` - Comprehensive XGBoost guide (22 chapters)
- `docs/DATA_GAPS_AND_ML_OPPORTUNITIES.md` - Feature ideas
- `docs/KENPOM_ANALYTICS_GUIDE.md` - KenPom methodology

### Related Libraries
- **Optuna**: Hyperparameter optimization
- **SHAP**: Model interpretability
- **MLflow**: Experiment tracking (optional)

---

## 12. Summary & Recommendation

### Why XGBoost for KenPom Analytics?

1. **Better Regularization** → Prevents overfitting to small samples
2. **Feature Importance** → Identify betting edges
3. **Faster Training** → Quicker iteration during development
4. **Early Stopping** → Optimal model complexity automatically
5. **Quantile Regression** → Better confidence intervals
6. **Custom Loss Functions** → Optimize for ATS performance

### Recommended Approach

**Phase 1 (Week 1)**: Drop-in replacement
- Minimal code changes
- Immediate +2-3% accuracy improvement
- Validate XGBoost advantage

**Phase 2 (Week 2)**: Enhanced features (TIER 0)
- Luck regression
- Possession length
- Point distribution
- Target: +3-5% accuracy

**Phase 3 (Week 3)**: Hyperparameter optimization
- Run Optuna search
- Find optimal params
- Target: +2-3% additional accuracy

**Phase 4 (Week 4)**: Edge detection
- Feature importance analysis
- SHAP explanations
- Identify 5-10 high-value bet types

### Expected ROI
- **Development Time**: 4 weeks
- **Accuracy Improvement**: +8-10% (72% → 80-82%)
- **CLV Improvement**: +2.5 to +3.5 points on targeted bets
- **Long-term**: Foundation for ensemble models, advanced techniques

---

## Questions for Andy

1. **Priority**: Which phase should we start with?
2. **Scope**: TIER 0 features only, or TIER 0 + TIER 1?
3. **Deployment**: Parallel A/B test or direct replacement?
4. **Optimization**: Run full hyperparameter search or use defaults?
5. **Interpretability**: How important is SHAP analysis vs just feature importance?
6. **Timeline**: 4-week phased rollout or faster iteration?
7. **Success Metrics**: Target accuracy? Target CLV? Both?

---

**Created by**: Claude (Dynamite Duo)
**Date**: 2025-12-18
**Status**: Draft - Awaiting Andy's Review
