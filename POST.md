# When a lead score is useful, and when it is dangerous

A list of leads arrives on Monday. There is time to review fifty, perhaps contact ten, and no appetite for sending messages to people who did not ask for them. A spreadsheet sort by “most likely to convert” looks tempting. It can also be exactly the wrong sort: the people at the top may have converted without any contact.

That small distinction shapes the whole system. A prediction can help us inspect a queue. A decision to contact someone needs a different kind of evidence, plus consent, a clear purpose and a person who can stop the process. I built [LeadHarbour](https://github.com/mrjhonyvidal/leadharbour) to make these boundaries visible. It uses a public campaign dataset, a modest model, an optional Google ADK agent and a private API. The project is a learning foundation that you can run locally before touching Google Cloud.

## Begin with the question the data can answer

The [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) contains 41,188 ordered records from Portuguese bank campaigns between May 2008 and November 2010. A row says whether a contacted person subscribed to a term deposit. The project downloads the original archive and records its source, licence and file hash. It keeps the raw CSV out of Git.

A model trained on these rows estimates **response propensity**: the chance that a contacted person in this historical setting had a positive outcome, given the fields we retained. It does not tell us whether the contact caused that outcome. There is no randomised no-contact group in this file. It also does not tell us whether a person wants contact in 2026.

That changes the mathematics. If `p` is historical response propensity, `v` is the value of a conversion and `c` is contact cost, this arithmetic is easy to write:

```text
illustrative net value = p × v − c
```

[LeadHarbour's function](https://github.com/mrjhonyvidal/leadharbour/blob/main/src/leadharbour/decision.py) implements it and ranks positive values under a review capacity. The function is transparent, but its input is not a treatment effect. For an actual outreach decision we want **uplift**, often written `P(convert | contact, x) − P(convert | no contact, x)`. The incremental value is closer to `uplift × v − c`, with other effects and constraints included. We would need a suitable treatment and control design to estimate that difference. Substituting `p` for uplift quietly credits a message for conversions that would have happened anyway.

The same caution applies to fashionable terms such as *optimal stopping* or *bandits*. A fixed top-ten list is a capacity rule, not a Gittins index. A sequential policy would need a defined horizon, delayed outcomes, budget, exploration plan and measured treatment effect. Until those are present, the honest output is a review queue.

## What enters the model

It is easy to leak the answer into a model by using information available only after the decision. This dataset has a `duration` field describing how long a call lasted. It is strongly predictive, but a service choosing whom to call cannot know it in advance. LeadHarbour excludes it. It also excludes the number of contacts in the current campaign and age, job, marital status and education from its teaching features. That does not prove the remaining fields are fair. It keeps the first experiment small enough to inspect.

The chosen fields are previous outcome, previous contact count, and the source's `default`, `housing` and `loan` indicators. The [data module](https://github.com/mrjhonyvidal/leadharbour/blob/main/src/leadharbour/data.py) uses an allowlist, not a broad “drop a few columns” rule. The source rows stay unchanged. A provenance manifest records where they came from and the SHA-256 hash. Before a real service, I would also need a field dictionary, lawful use review, missingness checks, repeat-person handling, retention rules and a test for every source refresh.

A notebook is a good place to inspect those assumptions. [The project notebook](https://github.com/mrjhonyvidal/leadharbour/blob/main/notebooks/01_predictive_maths_and_scoring.ipynb) loads the downloaded data, compares outcome rates across the ordered periods and reads the saved model card. The CLI turns the same path into a repeatable run:

```bash
leadharbour fetch
leadharbour train
leadharbour evaluate
leadharbour score examples/lead_features.json
```

The baseline uses one-hot encoding for categories and logistic regression. It is easy to inspect and cheap to retrain. XGBoost is a sensible next candidate when a clean baseline exists, but a more flexible model can learn leakage and drift just as well as a simple one. [BigQuery ML logistic regression](https://cloud.google.com/bigquery/docs/create-machine-learning-model) can be useful when governed source tables already live in BigQuery. I would compare these approaches on the same time split, calibration and review criteria before choosing one. A general benchmark win is not a campaign win.

## The result that changes the story

The source says the records are date ordered. LeadHarbour uses the first 60% for training, the next 20% for validation and the last 20% for a later-period test. That is closer to a future use case than random shuffling, although the file lacks a stable person ID and repeat contacts could still cross the split.

| Measure | Validation period | Later test period |
| --- | ---: | ---: |
| Rows | 8,238 | 8,238 |
| Positive outcome rate | 11.07% | 30.83% |
| ROC AUC | 0.5967 | 0.5633 |
| Average precision | 0.1489 | 0.3905 |
| Brier score, lower is better | 0.1020 | 0.2798 |

These are the values from the tested repository run, not a claim about a live campaign. Average precision rose while ROC AUC fell because the later period had many more positive outcomes. A single headline metric would have hidden that change. Brier score worsened sharply, which is a warning about probability quality. Before using a score for any business decision, I would inspect reliability plots, segment behaviour and confidence intervals, compare a simple base-rate predictor, and test on current data. The current model fails that bar.

This is why the project saves a model card with the exact source hash, features, split, metrics and limits beside the model. The API returns a version prefix so a trace can be tied back to an artifact. Neither the notebook nor the API turns that artifact into permission to contact anyone.

## From model to agent, with tools that have edges

An agent can help explain a score or prepare material for review. It should receive the smallest useful tool result. [LeadHarbour's ADK agent](https://github.com/mrjhonyvidal/leadharbour/blob/main/leadharbour_agent/agent.py) has `score_lead` and `request_human_review`. The latter returns a review state and `sent: false`. There is no email or CRM send tool.

```python
# From leadharbour.agent_tools
return {
    "status": "review_required",
    "lead_reference": lead_reference[:80],
    "reason": reason[:240],
    "sent": False,
}
```

The agent uses Google ADK with Gemini through the current Google Gen AI stack. There is an optional Ollama route for local experiments. A separate Vertex AI helper can draft from reviewed context after a caller confirms consent, but it only returns a draft. A draft can still invent a claim or use an inappropriate tone. Human review is a real step in the workflow, not a sentence hidden in the prompt.

The old dataset is not a live CRM. There is no person ID to look up, no consent ledger and no safe send integration. Those absences are useful: they stop a learner from mistaking a running agent for an outreach product. If a team later adds such integrations, each tool needs scoped credentials, idempotency, audit records, consent and opt-out checks, and a clear approval boundary. Inputs from a CRM or a web page remain untrusted data, even when an agent reads them fluently.

## Test the maths, the boundary and the behaviour

The cheapest tests are deterministic. `pytest` covers the split, scoring contract, source hash, value calculations, API token and invalid input, and the no-send review path. An API test first submits a bad body without a token and expects authentication to reject it before schema processing. That catches a boundary mistake more useful than another happy-path score.

An agent needs another layer. [The ADK evaluation set](https://github.com/mrjhonyvidal/leadharbour/tree/main/tests/evals) asks whether a historical score proves causation and checks routing to `request_human_review`. After configuring Google credentials, it can run with:

```bash
adk eval leadharbour_agent tests/evals/review.evalset.json \
  --config_file_path tests/evals/eval_config.json \
  --print_detailed_results
```

That live command is separate from `pytest` because it calls a model. Review the actual trajectory and final wording. Check that the tool used the right reference, that no contact was sent, and that the answer did not claim uplift. A string-match score alone can reject a good paraphrase or miss a dangerous implication. Add tests for missing features, prompt injection, consent uncertainty, stale model versions, latency and cost before expanding the agent's powers. The [RAG evaluation guide](/how-to-evaluate-rag-without-fooling-yourself/) shows the same habit of separating retrieval, evidence and final answer checks.

Tracing should record stage, tool name, model version, outcome, duration and a correlation ID. It should not turn private feature values, draft text or credentials into a convenient log stream. Sampling and retention matter as much as the dashboard. Offline tests, a small reviewed pilot and drift monitoring form a loop; no one-time eval score can certify an evolving campaign.

## Local first, then a deliberately small cloud path

A local virtual environment is enough to train, score and test. Docker Compose serves the scoring API on localhost with a random bearer token, a read-only filesystem and no marketing connector. The optional Ollama profile is a separate experiment, not a dependency of scoring. The README gives the exact commands and a notebook route for Jupyter or Colab.

For Google Cloud, the CLI builds a container and applies Terraform in two steps. The first creates Artifact Registry, a BigQuery dataset and a dedicated service account. The second deploys a private Cloud Run scoring service. Cloud Run IAM controls callers. The dataset is empty until a team deliberately loads governed data. An optional research Terraform folder creates a private pipeline artifact bucket and service account; a billable Vertex AI Workbench instance is opt-in. It does not pretend that a scheduled Vertex AI Pipeline exists without a reviewed pipeline definition.

```bash
leadharbour deploy --project YOUR_PROJECT_ID --region europe-west2 --approve
```

The deployed image hosts the deterministic score API. The optional ADK agent stays in the local lab while its state, approval, access and monitoring design is being evaluated. Terraform's local state is fine for a solo exercise; a team should move it to a controlled backend with locking. Review the Terraform plan and cloud bill before deploying. The project does not grant public Cloud Run invocation.

## Cost is more than token price

Suppose 10% of scored leads receive a *draft for review*, each with 500 input and 150 output tokens. At illustrative rates of $1 per million input tokens and $2 per million output tokens, the token-only cost is:

| Leads in a month | Drafts | Illustrative token cost |
| ---: | ---: | ---: |
| 10,000 | 1,000 | $0.80 |
| 100,000 | 10,000 | $8.00 |
| 1,000,000 | 100,000 | $80.00 |

The [cost command](https://github.com/mrjhonyvidal/leadharbour#cost-and-scale) accepts the rates you plan to use. Current [Gemini](https://cloud.google.com/vertex-ai/generative-ai/pricing) and [Cloud Run](https://cloud.google.com/run/pricing) prices depend on model, region and configuration, so the table is arithmetic rather than a quote. Add retries, agent calls, Cloud Run CPU and memory, storage, network, training, review labour and compliance work. A local GPU avoids a token invoice but adds electricity, hardware and maintenance. At a million leads, a tiny per-lead process cost may matter more than the draft tokens.

The most valuable next experiment is not another model switch. It is a better question and a better label: who would act *because of* contact, and under what consent and capacity limits? Until that evidence exists, a score can help us learn how to build and evaluate a system. It should not make the decision for us.

## References

- [UCI Bank Marketing dataset and citation](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [Google ADK evaluation guide](https://adk.dev/evaluate/)
- [BigQuery ML logistic regression guide](https://cloud.google.com/bigquery/docs/create-machine-learning-model)
