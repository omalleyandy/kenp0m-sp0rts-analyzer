# Chapter 12: Hyperopt

## Why this chapter exists
- Bayesian optimization
- Parameter distributions
- Trials exploration

## Core ideas (LLM-friendly)
- Hyperopt uses Bayesian optimization: it learns where good hyperparameters tend to live.
- You define search spaces (distributions) and an objective function tied to CV or validation score.
- Inspect `Trials` to understand sensitivity and tradeoffs, not just the ‘best’ run.

## Practical checklist
- [ ] Define the goal (metric + constraints).
- [ ] Set up train/validation/test splits (or CV) before tuning.
- [ ] Log everything: params, metric, seed, data version.
- [ ] Start simple → add complexity only if it buys verified lift.

## Minimal reference snippets (non-verbatim)
- Keep the code path: load → clean → split → fit → evaluate → interpret → deploy.

## Study prompts
- What problem does this technique solve, and what failure mode does it introduce?
- Which metric would you *refuse* to use here, and why?
- What’s your ‘sanity check’ plot/table for this chapter’s main idea?
