# Chapter 20: Better Models with ICE, Partial Dependence, Monotonic Constraints, and Calibration

## Why this chapter exists
- ICE vs PDP
- SHAP PDP helper
- Monotonic constraints

## Core ideas (LLM-friendly)
- ICE plots show per-row response curves; PDPs average ICE to show general effect.
- Monotonic constraints encode domain knowledge (feature must increase/decrease prediction).
- Calibration checks whether predicted probabilities match observed frequencies; fix with calibration methods.

## Practical checklist
- [ ] Define the goal (metric + constraints).
- [ ] Set up train/validation/test splits (or CV) before tuning.
- [ ] Log everything: params, metric, seed, data version.
- [ ] Start simple → add complexity only if it buys verified lift.

## Minimal reference snippets (non-verbatim)
- PDP/ICE via scikit-learn:

```python
from sklearn.inspection import PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(model, X_train, features=[...], kind='both', centered=True)
```

## Study prompts
- What problem does this technique solve, and what failure mode does it introduce?
- Which metric would you *refuse* to use here, and why?
- What’s your ‘sanity check’ plot/table for this chapter’s main idea?
