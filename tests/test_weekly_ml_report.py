"""Reporte semanal de mejora del modelo, sobre una base SQLite de prueba."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from src.persistence import database
from src.reporting import weekly_ml

WEEK_END = datetime(2026, 11, 14, 3, 0, tzinfo=UTC)  # vie 21:00 CDMX
GARCH = "garch_volatility_forecast"


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "weekly.db"
    database.init_db(path)
    return path


def _run(
    db_path: Path,
    days_ago: float,
    model: float,
    baseline: float,
    signal: str = GARCH,
    sufficient: bool = True,
    report: dict[str, Any] | None = None,
) -> None:
    database.insert_evaluation_run(
        db_path,
        "^GSPC",
        signal,
        "v1",
        WEEK_END - timedelta(days=days_ago),
        60,
        "qlike" if signal == GARCH else "f1",
        model,
        baseline,
        sufficient,
        {"baseline": "ewma"} if report is None else report,
    )


def test_advantage_improves_week_over_week(db_path: Path) -> None:
    # QLIKE: menor es mejor. Ventaja = baseline - modelo.
    _run(db_path, 10, model=0.70, baseline=0.75)  # semana previa: +0.05
    _run(db_path, 2, model=0.60, baseline=0.75)  # esta semana: +0.15

    [row] = weekly_ml.build_rows(db_path, WEEK_END)

    assert row.weeks[0].advantage == pytest.approx(0.15)
    assert row.weeks[1].advantage == pytest.approx(0.05)
    assert row.delta == pytest.approx(0.10)
    assert row.verdict == "mejoró"


def test_f1_advantage_uses_higher_is_better(db_path: Path) -> None:
    _run(db_path, 3, model=0.10, baseline=0.08, signal="anomaly", report={})

    [row] = weekly_ml.build_rows(db_path, WEEK_END)

    assert row.weeks[0].advantage == pytest.approx(0.02)
    assert row.weeks[0].baseline_name == "aleatorio"


def test_insufficient_sample_is_flagged_not_scored(db_path: Path) -> None:
    _run(db_path, 2, model=0.60, baseline=0.75, sufficient=False)

    [row] = weekly_ml.build_rows(db_path, WEEK_END)

    assert row.weeks[0].advantage is None
    assert row.weeks[0].insufficient is True
    assert row.verdict == "muestra insuficiente"


def test_baseline_change_is_not_compared(db_path: Path) -> None:
    _run(db_path, 10, 0.70, 0.80, report={})  # SD-20d (registros viejos)
    _run(db_path, 2, 0.70, 0.72)  # EWMA

    [row] = weekly_ml.build_rows(db_path, WEEK_END)

    assert row.delta is None
    assert row.verdict == "cambio de baseline"


def test_markdown_lists_retrains_with_trigger(db_path: Path) -> None:
    _run(db_path, 2, model=0.60, baseline=0.75)
    database.insert_model_version(
        db_path,
        "^GSPC",
        GARCH,
        "v2",
        "candidate",
        {"family": "har"},
        {"triggered_by": "drift", "promoted": True},
        WEEK_END - timedelta(days=1),
        promote=True,
    )

    text = weekly_ml.render_markdown(db_path, WEEK_END)

    assert "2026-11-13" in text  # fecha local CDMX del corte
    assert "drift" in text
    assert "har" in text
    assert "^GSPC" in text


def test_markdown_flags_champion_that_does_not_beat_ewma(
    db_path: Path,
) -> None:
    database.insert_model_version(
        db_path,
        "EURUSD=X",
        GARCH,
        "v3",
        "candidate",
        {"family": "garch"},
        {"triggered_by": "fixed_cadence", "beats_ewma": False},
        WEEK_END - timedelta(days=1),
        promote=True,
    )

    text = weekly_ml.render_markdown(db_path, WEEK_END)

    assert "| EURUSD=X | garch_volatility_forecast | `v3` | garch | " in text
    assert "fixed_cadence | sí | **no** |" in text
    assert "sobre EWMA (sigue sirviendo; vigilar): EURUSD=X." in text


def test_write_report_creates_dated_file(
    db_path: Path, tmp_path: Path
) -> None:
    _run(db_path, 2, model=0.60, baseline=0.75)

    path = weekly_ml.write_report(db_path, tmp_path / "weekly", WEEK_END)

    assert path.name == "2026-11-13.md"
    assert path.read_text(encoding="utf-8").startswith("# Reporte semanal")


@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 11, 14, 3, 0, tzinfo=UTC),  # a la hora exacta
        datetime(2026, 11, 14, 9, 50, tzinfo=UTC),  # cron con 7 h de retraso
        datetime(2026, 11, 20, 23, 0, tzinfo=UTC),  # manual el viernes 17:00
    ],
)
def test_default_cutoff_is_last_friday_9pm_cdmx(now: datetime) -> None:
    assert weekly_ml.last_cutoff(now) == WEEK_END


def _prediction(db_path: Path, signal: str, run_ts: datetime) -> None:
    database.insert_prediction(
        db_path, "^GSPC", signal, "v1", run_ts, 1, {}, run_ts
    )


def test_markdown_reports_pipeline_runs_and_warns_about_missing_days(
    db_path: Path,
) -> None:
    # Volatilidad corrió los 5 días; anomalías solo 2 de 5.
    for days_ago in (1, 2, 3, 4, 5):
        _prediction(db_path, GARCH, WEEK_END - timedelta(days=days_ago))
    for days_ago in (1, 1.1, 4):
        _prediction(db_path, "anomaly", WEEK_END - timedelta(days=days_ago))

    text = weekly_ml.render_markdown(db_path, WEEK_END)

    assert "| Volatilidad (QLIKE) | 5 | 5 de 5 |" in text
    assert "| Anomalías (F1) | 3 | 2 de 5 |" in text
    assert "**Aviso:** el pipeline no corrió todos los días" in text
    assert "Anomalías (F1)" in text.split("**Aviso:**")[1].splitlines()[0]
