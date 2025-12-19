# Chapter 8: Random Forest

## Why this chapter exists
- Bagging intuition
- RF hyperparameters
- Training # trees

## Core ideas (LLM-friendly)
- Random forests reduce variance via bagging (bootstrap rows) and feature subsampling.
- More trees usually stabilize results; single-tree interpretability is traded for ensemble stability.
- Forest ‘votes’ approximate a wise crowd when members are decent and not too correlated.

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
