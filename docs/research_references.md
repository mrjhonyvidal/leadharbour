# Research notes and boundaries

These sources guide the lab. They do not turn the old UCI data into a current marketing policy.

## Propensity and uplift

The [UCI Bank Marketing source](https://archive.ics.uci.edu/dataset/222/bank+marketing) describes a dated, ordered campaign dataset with contact outcomes. It contains no randomised no-contact control. Its label can support a historical *response propensity* exercise, not the causal effect of contacting a person.

For causal uplift, compare potential outcomes under contact and no contact. [Gutierrez and Gérardy's causal inference and uplift review](https://arxiv.org/abs/1809.09820) and [uplift modelling in direct marketing](https://arxiv.org/abs/1910.00393) explain why treatment assignment, overlap and validation matter. Randomised assignment is a strong route to evidence. Observational estimates need justified adjustment and sensitivity checks. If no such data exists, label any policy comparison as hypothetical.

## Value and stopping

LeadHarbour's `expected_net_value` multiplies historical propensity by a hypothetical value and subtracts contact cost. Its `rank_for_review` enforces a capacity limit. Both are teaching arithmetic; neither is an optimal outreach policy. The correct incremental value of contact would use estimated uplift, not raw propensity. If a lead would convert anyway, contact has no incremental conversion benefit.

[Gittins' allocation index](https://academic.oup.com/jrsssb/article/41/2/148/7027626) is a result for sequential allocation under a specific stochastic model. The [secretary problem](https://www.jstor.org/stable/2237763) illustrates a different stopping rule under assumptions about ordered observations and irrevocable choice. Those assumptions rarely hold for a real campaign. A defensible sequential system would define remaining budget, horizon, delayed outcomes, treatment effects and exploration policy, then test off policy bias and guardrails. The current repository intentionally stops at a ranked human review list.

## Agents and evaluation

The [ReAct paper](https://arxiv.org/abs/2210.03629) joins reasoning with tool actions. In LeadHarbour, the agent receives two narrow tools instead of a general CRM or mail credential. [Google ADK evaluation](https://adk.dev/evaluate/) can inspect tool trajectories and final responses. The included JSON cases are valid examples, but a live run needs a model and credentials. Exact wording metrics may be brittle; review tool arguments, unsupported claims, refusal behaviour and human approval state as well.

For deployment details, use the maintained [ADK Cloud Run guide](https://adk.dev/deploy/cloud-run/), [Google Gen AI SDK documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview), [BigQuery ML logistic regression guide](https://cloud.google.com/bigquery/docs/create-machine-learning-model), [Vertex AI Pipelines documentation](https://cloud.google.com/vertex-ai/docs/pipelines/introduction), and the [Cloud Run pricing page](https://cloud.google.com/run/pricing). LeadHarbour's default cloud path deploys only the private deterministic scoring API. Any later agent service needs its own session, approval, evaluation, observability and access design.
