# Chapter 3: Exploratory Data Analysis

## Why this chapter exists
- Summary stats
- Correlations
- Visual EDA to guide modeling

## Core ideas (LLM-friendly)
- EDA is about *shape* (distributions), *relationships* (correlations/interactions), and *leakage* (too-good-to-be-true signals).
- Use visuals to find non-linearities and subgroup behavior that trees can exploit.
- Turn EDA findings into feature engineering hypotheses.

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
