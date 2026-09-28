# Repository guidance

- Use British English, plain names, short explanations and no em dash in prose or comments.
- Keep the public UCI source attribution and licence with every derived example. Never commit downloaded rows, model artifacts, secrets or personal lead data.
- Treat the fitted model as an educational baseline. Never call its score uplift or contact approval.
- Keep the scoring and review tools separate from message sending. Any new external action needs explicit consent, scoped credentials, a review boundary and tests.
- Add meaningful unit tests for data contracts, calculations and API boundaries. Run `pytest -q`, `leadharbour evaluate`, notebook cells and Terraform validation before publishing infrastructure changes.
- Compare models on the same ordered periods. Report base rates and score ties with top-capacity metrics; keep the later test out of tuning.
- Cloud resources cost money. Keep Workbench and agent services optional and the default Cloud Run service private.
