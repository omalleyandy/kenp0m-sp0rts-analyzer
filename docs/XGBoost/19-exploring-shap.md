# Chapter 19: Exploring SHAP

## Why this chapter exists
- Single prediction explanations
- Waterfall/force
- Dependence + beeswarm

## Core ideas (LLM-friendly)
- SHAP explains predictions by assigning each feature a contribution relative to a baseline.
- Single-prediction plots (waterfall/force) show local reasoning; beeswarm/dependence show global behavior.
- Dependence plots reveal non-linearities and interactions; jittering/heatmaps help interpret density and correlation.

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
