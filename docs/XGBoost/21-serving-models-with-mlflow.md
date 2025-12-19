# Chapter 21: Serving Models with MLFlow

## Why this chapter exists
- Track experiments
- Inspect artifacts
- Serve predictions

## Core ideas (LLM-friendly)
- MLflow tracks runs, parameters, metrics, and artifacts so experiments are reproducible.
- Serving turns a trained model into an HTTP endpoint; Docker packaging makes deployment portable.
- Always validate inputs/feature schema at serving time—training/serving skew is the silent killer.

## Practical checklist
- [ ] Define the goal (metric + constraints).
- [ ] Set up train/validation/test splits (or CV) before tuning.
- [ ] Freeze the feature pipeline (same preprocessing code/config in training and serving).
- [ ] Log everything: params, metric, seed, data version.
- [ ] Start simple → add complexity only if it buys verified lift.

## Minimal reference snippets (non-verbatim)
- Serve an MLflow model locally, then POST JSON to `/invocations`.
- Build a Docker image from a model artifact directory.

## Study prompts
- What problem does this technique solve, and what failure mode does it introduce?
- Which metric would you *refuse* to use here, and why?
- What’s your ‘sanity check’ plot/table for this chapter’s main idea?
