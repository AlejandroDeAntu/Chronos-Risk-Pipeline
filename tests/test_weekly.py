"""Resumen semanal: ventanas de la base, historial, git y alertas."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.persistence import database
from src.reporting.weekly import (
    Metric,
    build_weekly_summary,
    live_metrics,
    load_previous_snapshot,
    project_metrics,
    snapshot_line,
)
from src.reporting.weekly_render import render_markdown
from src.schemas import SignalType

NOW = datetime(2026, 10, 26, 1, 0, tzinfo=UTC)
ANOMALY = SignalType.ANOMALY.value
VOL = SignalType.GARCH_VOLATILITY_FORECAST.value


def _prediction(
    db: Path, i: int, signal: str, run_ts: datetime, **payload: bool
) -> int | None:
    return database.insert_prediction(
        db_path=db,
        asset_symbol="^GSPC",
        signal_type=signal,
        model_version="v1",
        reference_timestamp=datetime(2026, 1, 1, tzinfo=UTC)
        + timedelta(minutes=i),
        horizon_bars=1,
        payload=payload,
        run_ts=run_ts,
    )


def _outcome(db: Path, pid: int, when: datetime, **values: float) -> None:
    database.insert_outcome(
        db_path=db,
        prediction_id=pid,
        baseline_predicted_value=0.0,
        evaluated_at=when,
        **values,
    )


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    path = tmp_path / "pipeline.db"
    database.init_db(path)
    this_week = NOW - timedelta(days=2)
    last_week = NOW - timedelta(days=9)
    # Esta semana: 2 ciclos de anomalías, 1 de volatilidad.
    a1 = _prediction(path, 1, ANOMALY, this_week, is_anomaly=True)
    a2 = _prediction(path, 2, ANOMALY, this_week, is_anomaly=False)
    a3 = _prediction(
        path, 3, ANOMALY, this_week + timedelta(hours=1), is_anomaly=True
    )
    v1 = _prediction(path, 4, VOL, this_week)
    _outcome(path, a1, this_week, realized_value=1.0, error_metric=0.0,
             baseline_error_metric=1.0)  # fmt: skip
    _outcome(path, a2, this_week, realized_value=1.0, error_metric=1.0,
             baseline_error_metric=1.0)  # fmt: skip
    _outcome(path, a3, this_week, realized_value=0.0, error_metric=1.0,
             baseline_error_metric=0.0)  # fmt: skip
    _outcome(path, v1, this_week, realized_value=1.0, error_metric=0.5,
             baseline_error_metric=0.8)  # fmt: skip
    # Semana anterior: 1 ciclo de volatilidad peor que el baseline.
    v2 = _prediction(path, 5, VOL, last_week)
    _outcome(path, v2, last_week, realized_value=1.0, error_metric=0.9,
             baseline_error_metric=0.4)  # fmt: skip
    # Pendiente vieja (backlog) y un modelo promovido esta semana.
    _prediction(path, 6, VOL, NOW - timedelta(days=20))
    database.insert_model_version(
        path, "^GSPC", VOL, "v2", "active", {}, {}, this_week
    )
    database.insert_model_version(
        path, "^GSPC", ANOMALY, "v3", "candidate", {}, {}, this_week
    )
    return path


def test_weekly_activity_counts_cycles_and_live_losses(db: Path) -> None:
    week = database.fetch_weekly_activity(db, NOW - timedelta(days=7), NOW)

    assert week["cycles"] == {ANOMALY: 2, VOL: 1}
    assert week["predictions"] == {ANOMALY: 3, VOL: 1}
    assert week["outcomes"] == 4
    assert week["vol_model_loss"] == pytest.approx(0.5)
    assert week["vol_baseline_loss"] == pytest.approx(0.8)
    assert week["vol_win_share"] == pytest.approx(1.0)
    # 2 anomalías marcadas, 1 revirtió; prevalencia 2 de 3.
    assert week["anomaly_precision"] == pytest.approx(0.5)
    assert week["anomaly_prevalence"] == pytest.approx(2 / 3)
    assert week["models_registered"] == 2
    assert week["models_promoted"] == 1


def test_weekly_activity_empty_window_has_no_losses(db: Path) -> None:
    week = database.fetch_weekly_activity(
        db, NOW - timedelta(days=60), NOW - timedelta(days=30)
    )

    assert week["cycles"] == {}
    assert week["outcomes"] == 0
    assert week["vol_model_loss"] is None
    assert week["anomaly_precision"] is None


def test_live_metrics_compare_against_previous_week(db: Path) -> None:
    metrics = {m.label: m for m in live_metrics(db, NOW)}

    vol = metrics["Días en que el modelo de volatilidad superó al baseline"]
    assert (vol.current, vol.previous) == (1.0, 0.0)
    assert metrics["Predicciones pendientes con más de 7 días"].current == 1
    assert metrics["Tamaño de la base (MB)"].current > 0


def test_live_metrics_without_database_is_empty(tmp_path: Path) -> None:
    assert live_metrics(tmp_path / "missing.db", NOW) == ()


def test_metric_delta_and_direction() -> None:
    better = Metric("QLIKE", current=0.4, previous=0.5, higher_is_better=False)
    unknown = Metric("Nueva", current=3.0, previous=None)

    assert better.delta == pytest.approx(-0.1)
    assert better.improved is True
    assert unknown.delta is None
    assert unknown.improved is None


def test_previous_snapshot_skips_same_week_and_bad_lines(
    tmp_path: Path,
) -> None:
    history = tmp_path / "weekly.jsonl"
    old = {"generated_at": (NOW - timedelta(days=7)).isoformat(), "id": 1}
    same_week = {"generated_at": (NOW - timedelta(hours=5)).isoformat()}
    history.write_text(
        "\n".join([json.dumps(old), "{no es json", json.dumps(same_week)]),
        encoding="utf-8",
    )

    assert load_previous_snapshot(history, NOW) == old
    assert load_previous_snapshot(tmp_path / "missing.jsonl", NOW) is None


def _git(repo: Path, *args: str, when: datetime | None = None) -> None:
    env = None
    if when is not None:
        stamp = when.isoformat()
        env = {"GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env=None if env is None else {**_base_env(), **env},
    )


def _base_env() -> dict[str, str]:
    import os

    return dict(os.environ)


def _commit(repo: Path, author: str, when: datetime, files: dict) -> None:
    for name, text in files.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        f"user.name={author}",
        "-c",
        "user.email=a@b.c",
        "commit",
        "-q",
        "-m",
        "c",
        when=when,
    )


def test_project_metrics_from_git_history(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _commit(
        repo,
        "dev",
        NOW - timedelta(days=10),
        {
            "tests/test_a.py": "def test_a():\n    pass\n",
            "src/m.py": "x = 1\n",
        },
    )
    _commit(
        repo,
        "dev",
        NOW - timedelta(days=2),
        {"tests/test_b.py": "def test_b():\n    pass\n\ndef test_c(): ..."},
    )
    _commit(
        repo,
        "trading-pipeline-bot",
        NOW - timedelta(days=1),
        {"data/x.txt": "1"},
    )

    metrics = {m.label: m for m in project_metrics(repo, NOW)}

    assert metrics["Commits de código"].current == 1
    assert metrics["Commits de código"].previous == 1
    assert metrics["Commits automáticos del pipeline"].current == 1
    assert metrics["Funciones de prueba"].current == 3
    assert metrics["Funciones de prueba"].previous == 1
    assert metrics["Líneas agregadas"].current == 4


def test_project_metrics_outside_git_is_empty(tmp_path: Path) -> None:
    assert project_metrics(tmp_path, NOW) == ()


def _report(champion: str, kupiec_p: float) -> SimpleNamespace:
    def row(name: str, qlike: float, p: float | None) -> SimpleNamespace:
        vs = None if p is None else SimpleNamespace(p_value_one_sided=p)
        return SimpleNamespace(name=name, mean_qlike=qlike, vs_champion=vs)

    vol = SimpleNamespace(
        champion=champion,
        rows=(
            row(champion, 0.60, None),
            row("garch", 0.66, 0.01),
            row("ewma", 0.70, 0.001),
        ),
        var={0.01: SimpleNamespace(hit_rate=0.016, p_unconditional=kupiec_p)},
        forecast_vol_pct=1.1,
    )
    asset = SimpleNamespace(
        symbol="^GSPC",
        name="S&P 500",
        volatility=vol,
        anomaly=None,
        zero_return_share=0.01,
        errors=(),
    )
    return SimpleNamespace(generated_at=NOW, alpha=0.05, assets=(asset,))


def test_summary_compares_with_previous_snapshot(tmp_path: Path) -> None:
    history = tmp_path / "weekly.jsonl"
    previous = _report("garch", 0.5)
    first = build_weekly_summary(previous, history, tmp_path, tmp_path)
    old = json.loads(snapshot_line(first))
    old["generated_at"] = (NOW - timedelta(days=7)).isoformat()
    history.write_text(json.dumps(old) + "\n", encoding="utf-8")

    summary = build_weekly_summary(
        _report("har", 0.002), history, tmp_path, tmp_path
    )

    row = summary.assets[0]
    assert summary.has_previous_snapshot is True
    assert (row.previous_champion, row.champion) == ("garch", "har")
    assert row.gain_vs_legacy == pytest.approx(-0.06)
    assert any("cambió de campeón" in a for a in summary.alerts)
    assert any("Kupiec" in a for a in summary.alerts)
    text = render_markdown(summary)
    assert "S&P 500" in text and "Alertas" in text
    assert "\n" not in snapshot_line(summary)


def test_first_week_has_no_previous_snapshot(tmp_path: Path) -> None:
    summary = build_weekly_summary(
        _report("har", 0.5), tmp_path / "none.jsonl", tmp_path, tmp_path
    )

    assert summary.has_previous_snapshot is False
    assert summary.assets[0].qlike_previous is None
    assert "primera semana" in render_markdown(summary).lower()
