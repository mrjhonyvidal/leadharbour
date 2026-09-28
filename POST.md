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

## A modelling loop that knows when to stop

The quickest way to lose trust in a model is to start with an algorithm and work backwards to a question it seems to answer. I start with the decision and the point in time when it will be made. Only then do I ask what label and features existed at that point.

```text
Question and decision boundary
          ↓
Source, label, licence and consent review
          ↓
Time-ordered split: training | validation | later test kept aside
          ↓
Train-only cleaning and feature rules → simple baseline → candidate models
          ↑                                         ↓
          └──── revise on validation ← inspect errors, drift and metrics
                                                    ↓ freeze the candidate
                                      Check the later test once
                                                    ↓
                                Private API → agent tool → human review
                                                    ↓
                         Monitor new outcomes and start a new iteration
```

The loop goes back through training and validation. If I use the later test repeatedly to pick features or parameters, it stops being a fair test. Once a system is running, newly labelled outcomes call for a fresh training window and a new future holdout. They do not make the old test period fresh again. In this project there is no live send step, so the final human review is a boundary rather than an automated campaign.

## What enters the model

It is easy to leak the answer into a model by using information available only after the decision. This dataset has a `duration` field describing how long a call lasted. It is strongly predictive, but a service choosing whom to call cannot know it in advance. LeadHarbour excludes it. It also excludes the number of contacts in the current campaign and age, job, marital status and education from its teaching features. That does not prove the remaining fields are fair. It keeps the first experiment small enough to inspect.

The chosen fields are previous outcome, previous contact count, and the source's `default`, `housing` and `loan` indicators. The [data module](https://github.com/mrjhonyvidal/leadharbour/blob/main/src/leadharbour/data.py) uses an allowlist, not a broad “drop a few columns” rule. The source rows stay unchanged. A provenance manifest records where they came from and the SHA-256 hash. Before a real service, I would also need a field dictionary, lawful use review, missingness checks, repeat-person handling, retention rules and a test for every source refresh.

I inspect values before inventing new features. The original source uses `unknown` as a category in several fields, which can mean something different from a missing cell. An imputer or encoder fitted on all rows would also let the later period influence the training process. The project fits those steps on training rows inside each model pipeline. It turns categories into indicator columns and uses the median for the one numerical field. A future change to this rule would require the same transformation in training and the API, a new artifact, and another evaluation.

Feature engineering is a hypothesis, not a prize for adding columns. The comparison derives `had_previous_contact = previous > 0`, which is knowable before contact. It is mostly redundant with `previous` and `poutcome` here, and the measured result barely moves. A calendar month might capture seasonality, but in this old campaign it could also act as a shortcut for one particular wave. Before adding it I would ask whether it is available for the real decision, whether its meaning remains stable, and how it behaves across periods and groups. The same scrutiny applies to fields that may proxy sensitive traits.

A notebook is a good place to inspect those assumptions. [The project notebook](https://github.com/mrjhonyvidal/leadharbour/blob/main/notebooks/01_predictive_maths_and_scoring.ipynb) loads the downloaded data, compares outcome rates across the ordered periods and reads the saved model card. The CLI turns the same path into a repeatable run:

```bash
leadharbour fetch
leadharbour train
leadharbour evaluate
leadharbour compare
leadharbour score examples/lead_features.json
```

The serving baseline uses one-hot encoding for categories and logistic regression. Despite its name, logistic regression is a **binary classifier**: it adds weighted features and passes the result through a sigmoid to estimate a probability. Its regularisation discourages very large coefficients. It is easy to inspect and cheap to retrain. [DataCamp's logistic regression walkthrough](https://www.datacamp.com/tutorial/understanding-logistic-regression-python) is a helpful companion if the odds and sigmoid are new to you.

The comparison adds a constant training-rate baseline and XGBoost, which builds shallow trees in sequence to correct earlier errors. More trees and deeper trees can fit more patterns, including accidental ones. The fixed exercise uses 120 trees, depth 2 and learning rate 0.05, with row sampling, minimum child weight and regularisation to limit complexity. These are declared teaching settings, not optimised parameters. [DataCamp's XGBoost guide](https://www.datacamp.com/tutorial/xgboost-in-python) explains how the main knobs interact. [BigQuery ML logistic regression](https://cloud.google.com/bigquery/docs/create-machine-learning-model) is another route when governed source tables already live in BigQuery. I would keep the same label definition and ordered split across all candidates.

## The result that changes the story

The source says the records are date ordered. LeadHarbour uses the first 60% for training, the next 20% for validation and the last 20% for a later-period test. That is closer to a future use case than random shuffling. The file lacks a stable person ID, though, so repeat contacts could cross the split. The exact decision date is not recorded for each row in the subset, which limits what this chronological test can prove.

Before comparing algorithms, I compare the periods themselves:

| Source measure | Training | Validation | Later test |
| --- | ---: | ---: | ---: |
| Positive outcome rate | 4.81% | 11.07% | 30.83% |
| Had any previous contact | 0.53% | 26.04% | 40.67% |
| Previous outcome was success | 0.02% | 1.97% | 14.63% |
| `default` marked unknown | 27.70% | 14.23% | 7.04% |

The model learns from a period where previous successes are almost absent, then faces a later period where they are common. That is a much stronger warning than the word *drift* alone. It gives me a reason to inspect errors by period and feature value before changing an algorithm. It also means I should not assume that a fresh campaign will look like any of these old periods.

### What each metric can and cannot tell us

A classifier can produce a probability and then apply a threshold to make a yes/no prediction. The threshold produces a **confusion matrix**: true positives, false positives, false negatives and true negatives. From it, precision is the share of selected cases that were positive, recall is the share of all positives that were selected, and F1 balances those two at one threshold. Accuracy counts all correct labels, but a model that predicts “no” for everyone would already be 95.19% accurate in the training period and would find nobody. There is no natural reason to use a 0.5 threshold when the actual task is to review a limited number of records.

The notebook makes that concrete on validation data. At a 0.5 threshold, the fitted logistic model selects nobody. At 0.05, it selects 4,476 records: 616 with a positive historical outcome and 3,860 without one. A lower threshold finds more positives while creating many more false positives. Neither number says which contact caused a conversion.

| Measure | What I use it to ask | Main trap |
| --- | --- | --- |
| ROC AUC | Does a random positive tend to rank above a random negative across thresholds? | It says little about the probability value or the first few places in a short queue. |
| Average precision | How does precision hold up as recall grows? | Its reference level changes with the positive rate, so compare it with that period's base rate. |
| Brier score | How far are predicted probabilities from 0 and 1 outcomes on average? | A lower score alone does not prove good calibration for every group. |
| Log loss | Are confident wrong probabilities being heavily penalised? | It can improve while the very top of the ranking worsens. |
| Precision at 10% | Among the top tenth by score, what share had a positive outcome? | It depends on capacity, ties and the period's base rate; it is still not uplift. |
| Recall at 10% and lift | How many positives fit in that queue, and how does its precision compare with the base rate? | A historical outcome is not proof that contact helped. |

A reliability plot adds another view: group predictions into probability bands and compare the mean prediction with the observed outcome rate. If a group of records scored near 0.2 converts at 0.4, calling its score a 20% chance is misleading. [DataCamp's ROC AUC explanation](https://www.datacamp.com/tutorial/auc) is a useful visual introduction, but I would never choose a service from ROC AUC alone.

The [comparison code](https://github.com/mrjhonyvidal/leadharbour/blob/main/src/leadharbour/comparison.py) holds the fields and split fixed. Its [metric function](https://github.com/mrjhonyvidal/leadharbour/blob/main/src/leadharbour/model.py) first finds the score at the top-ten cutoff:

```python
capacity = ceil(len(labels) * 0.1)
cutoff = np.partition(scores, len(scores) - capacity)[len(scores) - capacity]
above = scores > cutoff
tied = scores == cutoff
```

Only a few distinct combinations exist in the five fields, so many records share the same score. Picking the first rows in a tie would make a top-ten result depend on the old file's order, not on a model preference. The project reports the **expected precision if tied records are selected at random**, the number tied at the cutoff, and the lowest and highest precision any tie choice could produce on these historical labels. Those bounds are a warning, not a way to choose people with hindsight. A constant base-rate predictor has no meaningful top-ten ranking, so its cell is blank.

| Fixed model | Validation ROC AUC | Validation average precision | Validation Brier | Validation precision at 10% |
| --- | ---: | ---: | ---: | ---: |
| Training-rate baseline | 0.5000 | 0.1107 | 0.1024 | No ranking |
| Logistic regression | 0.5967 | 0.1489 | 0.1020 | 0.1643 |
| Logistic with history flag | 0.5955 | 0.1485 | 0.1020 | 0.1643 |
| XGBoost | 0.5355 | 0.1181 | 0.1022 | 0.1138 |

The validation positive rate is 0.1107. Logistic regression improves the broad ranking over the constant model. The extra history flag adds almost nothing. XGBoost's expected top-ten precision is near the base rate. But each of these top-ten values has a wide tie-related range, so I would not treat the apparent decimal precision as a real operational promise.

| Fixed model | Later ROC AUC | Later average precision | Later Brier | Later log loss | Later precision at 10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training-rate baseline | 0.5000 | 0.3083 | 0.2810 | 0.9699 | No ranking |
| Logistic regression | 0.5633 | 0.3905 | 0.2798 | 0.9659 | 0.5880 |
| Logistic with history flag | 0.5595 | 0.3888 | 0.2802 | 0.9665 | 0.5880 |
| XGBoost | 0.5291 | 0.3224 | 0.2802 | 0.9618 | 0.3268 |

Average precision is higher in the later period partly because positives are much more common: even the constant baseline moves from 0.1107 to 0.3083. XGBoost has slightly lower log loss than logistic regression there, yet its ROC AUC is worse and its top-ten expectation is close to the 0.3083 base rate. More strikingly, **2,772 XGBoost records tie for the highest score**, while the review queue has only 824 places. Depending on which tied records a reviewer picks, the observed top-ten precision in this file could span 0 to 1. Logistic regression also ties 1,629 records at its cutoff, although it orders more records above that tie. The code reports these limits because a queue that cannot distinguish most of its candidates is a weak decision aid. None of these figures measure the incremental effect of outreach.

What would I tune next? First I would look at source availability and the period shift, not launch a large hyperparameter search. For logistic regression I might test the regularisation strength `C` and inspect coefficient stability. The numeric count and one-hot categories have different scales, so I would not compare raw coefficient sizes as if they were direct measures of importance. For XGBoost I would vary tree depth, number of trees, learning rate, minimum child weight and regularisation within a small validation search chosen in advance. Class weighting can change recall but may also make the probabilities less useful as probabilities. If I used early stopping, it would watch validation data only. A calibration method needs its own held-out slice or careful cross-validation, never the final test. I would then freeze the whole choice before using a new future holdout. The later test above has already been inspected while writing this article, so a new claim of improvement would need new data.

A single score table also hides uncertainty and harm. I would inspect false positives and false negatives, calibration bands, slices with enough data to measure, changes in missing values and performance under the actual review capacity. I would check whether any allowed field stands in for a sensitive one, and whether the label reflects a fair outcome. With no stable person ID, I cannot honestly produce a person-grouped confidence interval from this source. The old dataset is a place to learn the method, not to authorise a 2026 campaign.

This is why the project saves a model card with the source hash, features, split, metrics and limits beside the deployed logistic model. The API returns a version prefix so a trace can be tied back to an artifact. Neither the notebook nor the API turns that artifact into permission to contact anyone.

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
