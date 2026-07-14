import pytest
from typer.testing import CliRunner

from tja_ai_chartgen.cli import app

runner = CliRunner()


def _capture_web_start(monkeypatch):
    calls = {}

    def fake_create_app(**kwargs):
        calls["create_app"] = kwargs
        return object()

    def fake_run(application, **kwargs):
        calls["application"] = application
        calls["uvicorn"] = kwargs

    monkeypatch.setattr("tja_ai_chartgen.web.create_app", fake_create_app)
    monkeypatch.setattr("uvicorn.run", fake_run)
    return calls


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "localhost"])
def test_web_allows_loopback_hosts_without_remote_opt_in(monkeypatch, tmp_path, host):
    calls = _capture_web_start(monkeypatch)

    result = runner.invoke(
        app,
        ["web", "--host", host, "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert calls["create_app"] == {
        "output_dir": tmp_path,
        "remote_mode": False,
        "allow_instrument_analysis": False,
    }
    assert calls["uvicorn"]["host"] == host


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "example.test"])
def test_web_rejects_non_loopback_hosts_without_remote_opt_in(monkeypatch, tmp_path, host):
    calls = _capture_web_start(monkeypatch)

    result = runner.invoke(
        app,
        ["web", "--host", host, "--output-dir", str(tmp_path)],
    )

    assert result.exit_code != 0
    assert "--allow-remote" in result.output
    assert calls == {}


def test_web_allows_non_loopback_host_with_remote_opt_in(monkeypatch, tmp_path):
    calls = _capture_web_start(monkeypatch)

    result = runner.invoke(
        app,
        [
            "web",
            "--host",
            "0.0.0.0",
            "--allow-remote",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["create_app"] == {
        "output_dir": tmp_path,
        "remote_mode": True,
        "allow_instrument_analysis": False,
    }


def test_web_can_explicitly_allow_remote_instrument_analysis(monkeypatch, tmp_path):
    calls = _capture_web_start(monkeypatch)

    result = runner.invoke(
        app,
        [
            "web",
            "--host",
            "0.0.0.0",
            "--allow-remote",
            "--allow-instrument-analysis",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["create_app"] == {
        "output_dir": tmp_path,
        "remote_mode": True,
        "allow_instrument_analysis": True,
    }
