# Chapter 15: Model Evaluation

## Why this chapter exists
- Accuracy vs PR metrics
- Confusion matrix
- ROC/threshold curves

## Core ideas (LLM-friendly)
- Accuracy can be misleading; choose metrics aligned with the cost of false positives/negatives.
- Confusion matrix → precision/recall → F1 provide a useful ladder of nuance.
- Threshold curves (ROC, gains/lift) help you pick operating points for real decisions.

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
