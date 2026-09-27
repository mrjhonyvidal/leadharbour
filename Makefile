.PHONY: setup fetch train test eval serve sandbox deploy

REGION ?= europe-west2

setup:
	python3 -m venv .venv
	.venv/bin/pip install -e '.[test]'

fetch:
	.venv/bin/leadharbour fetch

train:
	.venv/bin/leadharbour train

test:
	.venv/bin/pytest -q

eval:
	.venv/bin/leadharbour evaluate

serve:
	.venv/bin/leadharbour serve

sandbox:
	docker compose up --build

deploy:
	.venv/bin/leadharbour deploy --project $(PROJECT) --region $(REGION) --approve
