# Chapter 4: Tree Creation

## Why this chapter exists
- How splits work
- Gini impurity
- Stopping criteria / hyperparameters

## Core ideas (LLM-friendly)
- A tree is a greedy sequence of splits that increases label purity.
- Gini impurity is a common purity score for classification; lower is purer.
- Stopping rules (max_depth, min_samples_leaf, etc.) trade bias vs variance.

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
