# When a Lead Score Is Useful, and When It Is Dangerous

A model says someone is likely to buy. That sounds useful if you have a long list and time to review only a few records. But a likely buyer might buy anyway. A prediction about an outcome is not the same as evidence that contacting someone will help.

I built [LeadHarbour](https://github.com/mrjhonyvidal/leadharbour) to work through that distinction with a small model, a public dataset and a result you can inspect. The first run needs no agent or cloud service.

## Run it before adding anything else

You need Python 3.11 or newer and internet for the packages and public dataset. Allow about ten minutes. No account is needed; estimated service cost is £0 on your own laptop.

```bash
git clone https://github.com/mrjhonyvidal/leadharbour.git
cd leadharbour
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
leadharbour fetch
leadharbour train
leadharbour evaluate
leadharbour score examples/lead_features.json
```

You should see metrics for two periods, a model card in `artifacts/model.json` and a score between zero and one. The [README](https://github.com/mrjhonyvidal/leadharbour#readme) includes Windows instructions and troubleshooting. If the dataset is already present, skip `fetch`.

The score estimates historical response probability. It is a teaching result, not permission to contact someone. No command sends a message.

## Ask what the rows can tell you

The [UCI Bank Marketing source](https://archive.ics.uci.edu/dataset/222/bank+marketing) includes an ordered file with 41,188 records from campaigns between May 2008 and November 2010. A row records whether a contacted person subscribed to a term deposit. The project preserves the CC BY 4.0 attribution and a file hash in its source manifest.

The model estimates **response propensity**, meaning how likely a positive outcome was in that historical setting given the retained fields. The file has no randomised no-contact group. It cannot tell us how much an extra contact caused the outcome to change.

That second question concerns **uplift**: the difference between outcomes with and without an intervention for comparable people. Estimating it needs a suitable treatment and control design. Replacing it with a response score credits contact for outcomes that might have happened anyway.

The value function in `src/leadharbour/decision.py` deliberately keeps the arithmetic visible:

```text
illustrative net value = response probability × conversion value − contact cost
```

You can learn from the calculation without treating it as a validated contact policy. A real incremental-value calculation would need a credible estimate of uplift, along with consent, capacity and other costs.

## Use only information available at the decision

The dataset includes call duration. It can help predict the outcome after a call, but it is unknown when deciding whom to call. Training a pre-call model with it would leak future information into the decision.

LeadHarbour excludes duration and starts with five fields: previous outcome, previous contact count and the source's `default`, `housing` and `loan` indicators. Age, job, marital status and education are excluded too. That keeps the experiment small; it does not establish that the remaining fields are fair or appropriate for a real service.

The code uses a feature allowlist so new source columns cannot silently enter training. It fits preprocessing on training rows only, then reuses the fitted transformation when scoring.

The model is logistic regression, which combines weighted inputs to estimate a probability for a binary outcome. Despite its name, it is used here as a classifier. It is a useful first model because the pipeline is small and the result is cheap to reproduce.

## Look at the periods before comparing models

The first 60% of rows train the model, the next 20% validate it and the last 20% form a later test. This preserves source order instead of shuffling future rows into training. There is no stable person ID, so repeat contacts may still cross the split.

The most useful warning is visible before choosing another algorithm:

| Logistic regression run | Validation | Later test |
| --- | ---: | ---: |
| Positive outcome rate | 11.07% | 30.83% |
| ROC AUC | 0.5967 | 0.5633 |
| Brier score | 0.1020 | 0.2798 |

These values come from the included source and fixed teaching pipeline, reproduced for this revision. Package versions are recorded in the model card. They are not a claim about a current campaign.

**ROC AUC** asks whether positive cases generally rank above negative ones. A value of 0.5 is the broad ranking baseline. **Brier score** measures average squared probability error; lower is better. Neither tells me whether sending a message changes an outcome.

The later positive rate is almost three times the validation rate. The model's probability error also worsens. That gives me a reason to inspect changed source conditions and prediction errors before adding complexity.

## More complex does not automatically mean better

The optional comparison fits a constant training-rate predictor, logistic regression, a regression variant with a history flag and a fixed XGBoost candidate. XGBoost combines decision trees. All candidates use the same ordered periods.

```bash
python -m pip install -e '.[comparison]'
leadharbour compare
```

On macOS, XGBoost may also require `brew install libomp`. This step is separate from the basic model.

| Fixed candidate | Validation ROC AUC | Later ROC AUC | Later precision in the top 10% |
| --- | ---: | ---: | ---: |
| Constant training rate | 0.5000 | 0.5000 | No ranking |
| Logistic regression | 0.5967 | 0.5633 | 0.5880 |
| Regression with history flag | 0.5955 | 0.5595 | 0.5880 |
| XGBoost | 0.5355 | 0.5291 | 0.3268 |

The later positive rate is 0.3083. Top-ten precision is the expected proportion of positives if score ties are resolved randomly. Many rows have identical scores: 1,629 tie at the later logistic cutoff and 2,772 at the XGBoost cutoff. The command reports tie counts and precision bounds. A neat four-decimal number should not hide an unresolved queue.

This fixed run gives no reason to prefer the more complex candidate. It also does not establish a general contest winner. I would use validation for changes and preserve a fresh later holdout for a final check. These particular test results are already known, so tuning against them would need new evidence before claiming improvement.

The [notebook](https://github.com/mrjhonyvidal/leadharbour/blob/main/notebooks/01_predictive_maths_and_scoring.ipynb) goes further into thresholds, probability errors and the model comparison.

## Keep scoring and action separate

The optional API scores an input. The optional agent can explain a score and request human review. It has no email or CRM send tool. This makes the boundary visible in code, rather than relying on a prompt that asks an otherwise powerful agent to be careful.

Before connecting a real workflow, I would need a lawful purpose, consent and opt-out handling, reviewed feature use, suitable labels, monitoring and an accountable owner. The public historical dataset cannot supply those for me.

Test the deterministic pieces first:

```bash
python -m pip install -e '.[test]'
pytest -q
```

Those tests cover the data and scoring contract, API authentication, value calculations and the review path. Live agent evaluation is separate because it calls a model and can vary. The [advanced guide](https://github.com/mrjhonyvidal/leadharbour/blob/main/docs/guide.md) explains the optional setup. You do not need it to learn from the model's failures.

## Keep the first experiment cheap

The local model makes no paid model calls. Optional hosting and hosted agents can charge a cloud account. For an arithmetic example, 1,000 reviewed drafts using 500 input and 150 output tokens each cost $0.80 at assumed rates of $1 per million input and $2 per million output tokens. Those are illustrative rates, not a provider quote.

Retries, hosting, storage and a person's review time add to the bill. The [FlowCheck cost experiment](https://trailblazergarage.com/what-does-one-correct-ai-result-cost/) shows why I count every attempt and divide by useful completed work.

For a next step, read the model card and inspect five mistakes. Ask whether the source, label and decision still match before changing a parameter. That habit travels much further than this particular dataset or algorithm.

## References

- [UCI Bank Marketing dataset and citation](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [LeadHarbour source and setup](https://github.com/mrjhonyvidal/leadharbour)
