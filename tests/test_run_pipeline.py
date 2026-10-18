"""Pruebas del punto de entrada único ``run_pipeline.py``."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

import run_pipeline
from src.config import PipelineSettings
from src.data import fetch


@pytest.fixture()
def patched_settings(
    monkeypatch: pytest.MonkeyPatch, test_settings: PipelineSettings
) -> PipelineSettings:
    monkeypatch.setattr(run_pipeline, "get_settings", lambda: test_settings)
    return test_settings


@pytest.mark.usefixtures("mock_yfinance")
def test_full_cycle_runs_without_input_and_exits_zero(
    patched_settings: PipelineSettings, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")

    assert run_pipeline.main([]) == 0

    with sqlite3.connect(patched_settings.paths.database_path) as conn:
        signals = {
            row[0]
            for row in conn.execute("SELECT signal_type FROM predictions")
        }
    assert signals == {"anomaly", "garch_volatility_forecast"}
    assert "RESUMEN DE LA CORRIDA" in caplog.text
    assert "Fallos: ninguno." in caplog.text


def test_data_failure_is_reported_and_exit_code_is_one(
    patched_settings: PipelineSettings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        fetch, "_download_raw", lambda symbol, **kw: pd.DataFrame()
    )
    caplog.set_level("INFO")

    assert run_pipeline.main(["--signals", "garch"]) == 1
    assert "DataUnavailableError" in caplog.text
    assert "Fallos:" in caplog.text


@pytest.mark.usefixtures("mock_yfinance")
@pytest.mark.parametrize("mode", ["report", "backtest"])
def test_report_mode_writes_html_pdf_and_figures_without_touching_db(
    patched_settings: PipelineSettings, mode: str
) -> None:
    assert run_pipeline.main(["--mode", mode]) == 0

    [directory] = list(patched_settings.paths.reports_dir.glob("report_*"))
    html = (directory / "report.html").read_text(encoding="utf-8")
    assert "Test Index" in html
    assert "Resumen ejecutivo" in html
    # Autocontenido: ninguna referencia a recursos externos.
    assert "http://" not in html and "https://" not in html
    assert (directory / "report.pdf").read_bytes().startswith(b"%PDF")
    assert len(list((directory / "figures").glob("*.png"))) >= 5
    assert not patched_settings.paths.database_path.exists()
