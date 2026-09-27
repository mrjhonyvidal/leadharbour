# ◆ LeadHarbour

**Score, inspect, decide.** A small predictive outreach lab using a real public dataset, an explainable model, narrow agent tools and a private scoring API. The model is a teaching baseline. It is not approved to choose real recipients or send messages.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB)](pyproject.toml) [![Licence](https://img.shields.io/badge/licence-MIT-blue)](LICENSE)

## Five-minute local route

You need Python 3.11+, internet access for the one-time data download, and about 100 MB of disk space. From this directory:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
leadharbour fetch
leadharbour train
leadharbour evaluate
leadharbour score examples/lead_features.json
pytest -q
```

`fetch` downloads the original [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing), records a SHA-256 hash and source attribution, and keeps the raw CSV out of Git. The archive contains 41,188 ordered records from May 2008 to November 2010. Its CC BY 4.0 terms and original citation are in `data/source_manifest.json` after download. The `duration` field is excluded because a future contact duration is unknowable before contact. Age, job, marital status and education are also excluded from this teaching model. The retained five fields are only a starting point, not an approval of their fairness or predictive value.

The model trains on the first 60% of source rows, validates on the next 20%, and tests on the final 20%. The source says records are date ordered, but it has no stable person ID, so repeat contacts may cross the split. The included run found validation ROC AUC **0.5967** and later-period test ROC AUC **0.5633**. The test outcome rate moved from **0.1107** in validation to **0.3083** in test, and the Brier score worsened from **0.1020** to **0.2798**. Run it yourself before trusting those numbers on your machine. This is an example of why a working pipeline does not make a useful campaign model.

The [notebook](notebooks/01_predictive_maths_and_scoring.ipynb) explores the changing outcome rate, reads the model card and works through a small expected-value example. Run `leadharbour fetch` and `leadharbour train` before opening it. The notebook works in JupyterLab or Colab after installing the project and uploading or fetching the data there.

## What is here

| Part | Purpose |
| --- | --- |
| `src/leadharbour/data.py` | Source fetch, provenance and feature allowlist. |
| `src/leadharbour/model.py` | Ordered split, preprocessing, logistic regression, metrics and model card. |
| `src/leadharbour/decision.py` | Explicit value and capacity calculations for human review. |
| `src/leadharbour/api.py` | Small authenticated scoring service. |
| `src/leadharbour/agent_tools.py` | Narrow tools for scoring and requesting review. |
| `leadharbour_agent/agent.py` | Optional Google ADK agent. |
| `tests/unit` and `tests/evals` | Deterministic tests and a live ADK evaluation set. |
| `terraform/bootstrap`, `runtime`, `research` | Minimal cloud service plus optional research resources. |

The model output is **historical response propensity**: a probability estimated from people contacted in the old dataset. It is not the extra probability caused by contact. The data has no randomised no-contact group, so it cannot support an uplift estimate. The expected-value function is a transparent arithmetic exercise, not a validated intervention policy.

## Local API sandbox

Train first, then make a long random token:

```bash
export LEADHARBOUR_API_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
leadharbour serve
```

In another terminal, use `curl` with the same token:

```bash
curl -s -H "Authorization: Bearer $LEADHARBOUR_API_TOKEN" \
  -H 'Content-Type: application/json' \
  --data @examples/lead_features.json http://127.0.0.1:8080/score
```

Or run `docker compose up --build` with the token set. The container binds to localhost, drops Linux capabilities, uses a read-only filesystem and has no outbound marketing integration. The API returns the score, source hash prefix and a trace ID. It does not log input features or make contact decisions. Keep the token secret and rotate it if shared.

## Google ADK and Gemini

The agent is optional. `pip install -e '.[agent]'` adds Google ADK and the current Google Gen AI SDK. Set up [Application Default Credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc), a billed Google Cloud project and access to a supported Gemini model, then set `GOOGLE_GENAI_USE_VERTEXAI=TRUE`, `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` (for example `global` where supported). From the repository root:

```bash
adk run leadharbour_agent
adk web .
adk eval leadharbour_agent tests/evals/review.evalset.json \
  --config_file_path tests/evals/eval_config.json \
  --print_detailed_results
```

The ADK cases check an explanation about propensity versus uplift and route a review request to the right tool. They need live model credentials and outputs can vary. Review trajectories and final text, including cases that fail. Unit tests run without a model. The ADK agent can ask for human review, but cannot email or call a CRM. `drafts.py` has a separate Vertex AI draft helper that requires a caller to confirm consent; its output still needs review and is never sent.

For an **optional local model experiment**, install `.[local-agent]`, run Ollama locally, pull a model you have assessed, then set `LEADHARBOUR_OLLAMA_MODEL` and `OLLAMA_API_BASE=http://127.0.0.1:11434`. `docker compose --profile local-llm up` starts a local Ollama service; model download and ADK execution are separate. Local model quality, privacy and energy use need their own checks. No Gemini or Ollama credentials are needed for the baseline model and API.

## Private Google Cloud deployment

You need a billed GCP project you control, `gcloud`, Docker with `linux/amd64` build support, Terraform 1.7+, and permission to enable services and create the resources shown below. Authenticate with `gcloud auth login` and `gcloud auth application-default login`, check `gcloud config get-value project`, then train the model locally. Inspect the Terraform plans and the expected cost before applying.

```bash
leadharbour deploy --project YOUR_PROJECT_ID --region europe-west2 --approve
```

This CLI enables required APIs, creates Artifact Registry, a BigQuery dataset and a dedicated runtime service account, builds and pushes the scoring image, and deploys a private Cloud Run service. It leaves invocation restricted to IAM. Grant `roles/run.invoker` to a reviewed caller and use an identity token to call `/score`. **The cloud image hosts the deterministic API, not the optional ADK agent.** Keeping the agent local until its evaluation, durable state, approval workflow and access model are ready is deliberate.

Terraform uses local state for the learning route. Protect that state and move it to a controlled remote backend with locking before a team or production deployment. The bootstrap, runtime and optional research folders have separate state. The Cloud Run endpoint still requires standard network, IAM, logging, data retention and incident controls for a real service. A live campaign also needs consent, opt-out handling, regional privacy review, fairness checks, monitoring, and human approval. This repository has no CRM connector or send path.

`terraform/research` is optional: it creates a private Cloud Storage bucket for Vertex AI Pipeline artifacts and a least-privilege research account. Setting `create_workbench=true` also creates a billable Vertex AI Workbench instance. It does not create a pipeline job by itself. Use a reviewed pipeline definition and data contract before scheduling one. The BigQuery dataset is created but source rows are **not** uploaded automatically. [BigQuery ML logistic regression](https://cloud.google.com/bigquery/docs/create-machine-learning-model) is an alternative once a governed table and split are in place.

## Cost and scale

Pricing changes by model, region and service configuration. Use the [Gemini pricing page](https://cloud.google.com/vertex-ai/generative-ai/pricing) and [Cloud Run pricing page](https://cloud.google.com/run/pricing) for the rate you will actually pay. The CLI calculates draft token spend from your inputs:

```bash
leadharbour cost --input-price-per-million 1 --output-price-per-million 2
```

At the illustrative rates above, 10% of leads drafted with 500 input and 150 output tokens each costs **$0.80, $8 and $80** at 10,000, 100,000 and 1,000,000 leads. Add Cloud Run CPU, memory and requests, storage, egress, training, human review and any agent retries. Local hardware has purchase, maintenance and electricity costs; it is not free merely because no token invoice arrives. Send no full contact list to an LLM by default. Score cheaply first, sample and review, then estimate the cost of a restricted drafting stage.

## Learning path

1. Inspect the raw source and provenance manifest, then run the notebook.
2. Change one allowed feature or split rule and see how the held-out result moves.
3. Read the model card and add calibration and group checks before considering another model.
4. Keep the API private. Test invalid features, authentication and the ADK review path.
5. Only with suitable treatment and control data, estimate uplift and compare policies offline.

See [the research notes](docs/research_references.md) and [the article](POST.md). Contributions that improve tests, documentation or the learning route are welcome. Please do not add real personal data or deploy an autonomous send operation in a pull request.
