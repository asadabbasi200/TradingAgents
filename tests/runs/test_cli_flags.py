from pathlib import Path

from tradingagents.runs.orchestrator import start_run


def test_list_runs_cli_shows_started_run(tmp_path: Path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    start_run("AAPL", "2026-04-10", "cheap", {}, results_root=tmp_path / "results")
    from typer.testing import CliRunner
    from cli.main import app
    runner = CliRunner()
    result = runner.invoke(app, ["list-runs"])
    assert result.exit_code == 0
    assert "AAPL" in result.output
    assert "cheap" in result.output
