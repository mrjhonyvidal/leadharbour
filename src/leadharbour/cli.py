"""One entry point for the local lab and reviewed Google Cloud deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from .api import app_from_environment
from .comparison import compare_models
from .data import fetch_dataset
from .decision import estimate_generation_cost
from .model import evaluate, score, train

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = Path("data/bank-additional-full.csv")
DEFAULT_ARTIFACT = Path("artifacts/model.joblib")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="leadharbour", description="◆ LEADHARBOUR | score, inspect, decide")
    actions = command.add_subparsers(dest="action", required=True)
    fetch = actions.add_parser("fetch", help="Download the attributed UCI dataset")
    fetch.add_argument("--force", action="store_true")
    actions.add_parser("train", help="Train the ordered logistic regression baseline")
    actions.add_parser("evaluate", help="Check the held-out source period")
    actions.add_parser("compare", help="Compare base rate, regression and XGBoost on one ordered split")
    scoring = actions.add_parser("score", help="Score one local JSON feature record")
    scoring.add_argument("features_file", type=Path)
    server = actions.add_parser("serve", help="Serve the local or Cloud Run scoring API")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8080)
    cost = actions.add_parser("cost", help="Estimate token cost at three monthly volumes")
    cost.add_argument("--draft-fraction", type=float, default=0.1)
    cost.add_argument("--input-tokens", type=int, default=500)
    cost.add_argument("--output-tokens", type=int, default=150)
    cost.add_argument("--input-price-per-million", type=float, required=True)
    cost.add_argument("--output-price-per-million", type=float, required=True)
    deploy = actions.add_parser("deploy", help="Build and deploy a private GCP scoring service")
    deploy.add_argument("--project", required=True)
    deploy.add_argument("--region", default="europe-west2")
    deploy.add_argument("--approve", action="store_true")
    return command


def run(command: list[str], *, cwd: Path = PROJECT_ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def execute(arguments: argparse.Namespace) -> dict | None:
    if arguments.action == "fetch":
        if DEFAULT_CSV.exists() and not arguments.force:
            raise ValueError("Dataset already exists. Pass --force to download it again")
        return fetch_dataset(DEFAULT_CSV.parent)
    if arguments.action == "train":
        return train(DEFAULT_CSV, DEFAULT_ARTIFACT)
    if arguments.action == "evaluate":
        return evaluate(DEFAULT_CSV, DEFAULT_ARTIFACT)
    if arguments.action == "compare":
        return compare_models(DEFAULT_CSV)
    if arguments.action == "score":
        features = json.loads(arguments.features_file.read_text())
        return {"propensity": score(features, DEFAULT_ARTIFACT),
                "meaning": "Historical response propensity, not uplift or contact approval"}
    if arguments.action == "serve":
        import uvicorn
        app = app_from_environment()
        uvicorn.run(app, host=arguments.host, port=arguments.port, access_log=False)
        return None
    if arguments.action == "cost":
        return {
            str(volume): estimate_generation_cost(
                volume, arguments.draft_fraction, arguments.input_tokens,
                arguments.output_tokens, arguments.input_price_per_million,
                arguments.output_price_per_million)
            for volume in (10_000, 100_000, 1_000_000)
        }
    if arguments.action == "deploy":
        if not arguments.approve:
            raise ValueError("Review cloud costs and pass --approve to create resources")
        if not DEFAULT_ARTIFACT.exists():
            raise ValueError("Train a model before deployment")
        if not arguments.project.replace("-", "").isalnum():
            raise ValueError("Project ID contains unsupported characters")
        bootstrap = PROJECT_ROOT / "terraform/bootstrap"
        runtime = PROJECT_ROOT / "terraform/runtime"
        common = ["-var", f"project_id={arguments.project}", "-var", f"region={arguments.region}"]
        run(["terraform", "init", "-input=false"], cwd=bootstrap)
        run(["terraform", "apply", "-input=false", "-auto-approve", *common], cwd=bootstrap)
        artifact_version = hashlib.sha256(DEFAULT_ARTIFACT.read_bytes()).hexdigest()[:12]
        image = (f"{arguments.region}-docker.pkg.dev/{arguments.project}/"
                 f"leadharbour/leadharbour:{artifact_version}")
        run(["gcloud", "auth", "configure-docker", f"{arguments.region}-docker.pkg.dev", "--quiet"])
        run(["docker", "build", "--platform", "linux/amd64", "-t", image, "."])
        run(["docker", "push", image])
        run(["terraform", "init", "-input=false"], cwd=runtime)
        run(["terraform", "apply", "-input=false", "-auto-approve", *common,
             "-var", f"image={image}"], cwd=runtime)
        return {"image": image, "access": "Cloud Run IAM only; grant invoker explicitly"}
    raise ValueError(f"Unknown action: {arguments.action}")


def main(argv: list[str] | None = None) -> None:
    print("◆ LEADHARBOUR | score, inspect, decide", file=sys.stderr)
    try:
        result = execute(parser().parse_args(argv))
        if result is not None:
            print(json.dumps(result, indent=2))
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
