# Chapter 2: Datasets

## Why this chapter exists
- Download/load dataset
- Cleanup pipeline
- Train/test split & transform

## Core ideas (LLM-friendly)
- Treat data prep as a reproducible *pipeline* so train/test transforms are identical.
- Keep a clear separation: raw data → cleaned features → model-ready matrix.
- Stratify splits when class imbalance matters.

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
