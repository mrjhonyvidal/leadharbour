# LeadHarbour

Train a small model on a public dataset, score an example and see where the model fails. It is a way to learn prediction, evaluation and human review without connecting a real contact list.

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

**You need:** Python 3.11 or newer and internet for the packages and public dataset. Allow about ten minutes on a laptop. **Account: none. Estimated service cost: £0.** This excludes your laptop and electricity. Google Cloud, Gemini and Docker are optional.

You should see evaluation metrics, a model card in `artifacts/model.json`, and a score between zero and one. That score estimates historical response probability. It does not measure the effect of contacting someone or give permission to contact them.

## Test it

```bash
python -m pip install -e '.[test]'
pytest -q
```

The test extra includes XGBoost for the optional model comparison. On macOS it may also need `brew install libomp`. You can run the quick start without it. Use `leadharbour compare` when you are ready to compare the fixed candidates on the same data periods.

## If setup gets stuck

- Check `python3 --version`. On Windows, use `py -3` to create the environment and `.venv\Scripts\Activate.ps1` in PowerShell to activate it.
- Run commands from the project folder so the CLI can find `data/`, `examples/` and `artifacts/`.
- If `fetch` says the data already exists, continue to `train`. Use `leadharbour fetch --force` only when you want a fresh download.
- Install the core package first. The agent and notebook extras are separate downloads, not requirements for scoring.

## One useful experiment

Open `artifacts/model.json`. Compare the validation and later test periods. Did the proportion of positive outcomes change? Did the probability error change too? A pipeline can run correctly while its predictions become less useful.

The source is the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing), used under CC BY 4.0. The source manifest records attribution and a file hash. The project excludes call duration because it would not be known before a call. It uses the first 60% of rows for training, the next 20% for validation and the final 20% for testing. Repeat contacts may cross those boundaries because the source has no stable person ID.

## Choose a next step

| I want to... | Start here |
| --- | --- |
| Understand the results and modelling choices | [Worked article](POST.md) |
| Work through the maths | [Notebook](notebooks/01_predictive_maths_and_scoring.ipynb) |
| Compare regression, a simple baseline and XGBoost | [Model comparison](docs/guide.md#local-route) |
| Call the score over HTTP | [Local API](docs/guide.md#local-api-sandbox) |
| Try an agent with narrow tools | [Optional agent setup](docs/guide.md#google-adk-and-gemini) |
| Try a private cloud service | [Google Cloud setup](docs/guide.md#private-google-cloud-deployment) |

## Cost before adding an agent

The local model and tests make no paid model calls. A local Ollama experiment uses your own hardware. Cloud deployment and hosted models can charge your account.

For a planning example, 1,000 drafts at 500 input and 150 output tokens each would cost **$0.80 in tokens** if the rates were $1 per million input tokens and $2 per million output tokens. Those are illustrative inputs, not current provider prices. [The cost guide](docs/guide.md#cost-and-scale) gives the calculation and pricing links. Storage, hosting, retries and review add to it.

The code never sends messages. `src/leadharbour/model.py` trains and scores, `decision.py` contains the value calculations, and `agent_tools.py` keeps review separate from scoring. Use public or synthetic data while learning.

[Blog walkthrough](https://trailblazergarage.com/when-a-lead-score-is-useful-and-when-it-is-dangerous/) · [Further setup](docs/guide.md) · [MIT licence](LICENSE)
