# Chapter 10: Early Stopping

## Why this chapter exists
- early_stopping_rounds
- eval_set + eval_metric
- Plotting learning dynamics

## Core ideas (LLM-friendly)
- Early stopping watches a validation metric during training and stops when improvement stalls.
- You typically provide an `eval_set` and choose an `eval_metric` (logloss, auc, error, etc.).
- Use the best iteration (tree count) discovered by early stopping as your final n_estimators.

## Practical checklist
- [ ] Define the goal (metric + constraints).
- [ ] Use an explicit validation set for early stopping; never early-stop on the test set.
- [ ] Set up train/validation/test splits (or CV) before tuning.
- [ ] Log everything: params, metric, seed, data version.
- [ ] Start simple → add complexity only if it buys verified lift.

## Minimal reference snippets (non-verbatim)
- Train with validation monitoring:

```python
model = XGBClassifier(n_estimators=..., eval_metric='logloss')
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=...) 
best_iter = model.best_iteration
```
- Plot metric vs tree count (use evals_result()).

## Study prompts
- What problem does this technique solve, and what failure mode does it introduce?
- Which metric would you *refuse* to use here, and why?
- What’s your ‘sanity check’ plot/table for this chapter’s main idea?
