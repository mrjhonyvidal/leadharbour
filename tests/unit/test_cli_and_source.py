import io
import json
import zipfile
from pathlib import Path

import pytest

from leadharbour import cli, data


def test_fetch_preserves_attribution_and_hash(monkeypatch, tmp_path, trained_lab):
    csv_path, _, _ = trained_lab
    inner_buffer = io.BytesIO()
    with zipfile.ZipFile(inner_buffer, "w") as archive:
        archive.writestr(data.SOURCE_FILE, csv_path.read_bytes())
    outer_buffer = io.BytesIO()
    with zipfile.ZipFile(outer_buffer, "w") as archive:
        archive.writestr(data.INNER_ARCHIVE, inner_buffer.getvalue())

    class SourceResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    monkeypatch.setattr(data.urllib.request, "urlopen",
                        lambda request, timeout: SourceResponse(outer_buffer.getvalue()))
    manifest = data.fetch_dataset(tmp_path / "source")
    assert manifest["rows"] == 60
    assert manifest["source"] == data.SOURCE_PAGE
    assert json.loads((tmp_path / "source/source_manifest.json").read_text()) == manifest


def test_cost_cli_and_deploy_requires_approval(monkeypatch, tmp_path, capsys):
    cli.main(["cost", "--input-price-per-million", "1",
              "--output-price-per-million", "2"])
    assert json.loads(capsys.readouterr().out) == {
        "10000": 0.8, "100000": 8.0, "1000000": 80.0}
    with pytest.raises(SystemExit) as error:
        cli.main(["deploy", "--project", "example-project"])
    assert error.value.code == 1
    assert "--approve" in capsys.readouterr().err


def test_deploy_orders_bootstrap_image_and_private_runtime(monkeypatch, trained_lab):
    _, artifact_path, _ = trained_lab
    commands = []
    monkeypatch.setattr(cli, "DEFAULT_ARTIFACT", artifact_path)
    monkeypatch.setattr(cli, "run", lambda command, cwd=cli.PROJECT_ROOT:
                        commands.append((command, Path(cwd))))
    result = cli.execute(cli.parser().parse_args([
        "deploy", "--project", "learning-project", "--approve"]))
    assert [command[0][0] for command in commands] == [
        "terraform", "terraform", "gcloud", "docker", "docker", "terraform", "terraform"]
    assert commands[3][0][1:4] == ["build", "--platform", "linux/amd64"]
    assert commands[4][0][1] == "push"
    assert commands[1][1].name == "bootstrap"
    assert commands[-1][1].name == "runtime"
    assert "Cloud Run IAM" in result["access"]
