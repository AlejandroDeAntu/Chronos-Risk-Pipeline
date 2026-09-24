"""Pruebas de persistencia: idempotencia, cierre, métricas y promoción."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.exceptions import PersistenceError
from src.persistence import database

REF = datetime(2026, 9, 18, 15, 0, tzinfo=UTC)
RUN = datetime(2026, 9, 18, 15, 10, tzinfo=UTC)


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    database.init_db(path)
    return path


def _insert(db_path: Path, version: str = "v1", minutes: int = 0) -> int:
    prediction_id = database.insert_prediction(
        db_path,
        "TEST",
        "anomaly",
        version,
        REF + timedelta(minutes=minutes),
        10,
        {"is_anomaly": True},
        RUN,
    )
    assert prediction_id is not None
    return prediction_id


def test_insert_prediction_is_idempotent_per_bar(db_path: Path) -> None:
    _insert(db_path)
    duplicate = database.insert_prediction(
        db_path, "TEST", "anomaly", "v1", REF, 10, {"is_anomaly": True}, RUN
    )
    assert duplicate is None
    pending = database.fetch_pending_predictions(db_path, "TEST", "anomaly")
    assert len(pending) == 1


def test_outcome_closes_prediction_and_cannot_be_duplicated(
    db_path: Path,
) -> None:
    prediction_id = _insert(db_path)
    database.insert_outcome(db_path, prediction_id, 1.0, 0.0, 0.0, 1.0, RUN)

    assert database.fetch_pending_predictions(db_path, "TEST", "anomaly") == []
    with pytest.raises(PersistenceError):
        database.insert_outcome(
            db_path, prediction_id, 1.0, 0.0, 0.0, 1.0, RUN
        )


def test_recent_outcomes_filter_by_model_version(db_path: Path) -> None:
    old = _insert(db_path, "v1", minutes=0)
    new = _insert(db_path, "v2", minutes=5)
    for prediction_id in (old, new):
        database.insert_outcome(
            db_path, prediction_id, 1.0, 0.0, 0.0, 1.0, RUN
        )

    recent = database.fetch_recent_outcomes(
        db_path, "TEST", "anomaly", "v2", 10
    )

    assert [r["prediction_id"] for r in recent] == [new]


def test_evaluation_runs_return_only_sufficient_in_order(
    db_path: Path,
) -> None:
    for minutes, sufficient in ((0, True), (5, False), (10, True)):
        database.insert_evaluation_run(
            db_path,
            "TEST",
            "anomaly",
            "v1",
            RUN + timedelta(minutes=minutes),
            100,
            "f1",
            0.1 * minutes,
            0.2,
            sufficient,
            {},
        )
    runs = database.fetch_recent_evaluation_runs(
        db_path, "TEST", "anomaly", "v1", 5
    )
    assert [r["model_metric"] for r in runs] == pytest.approx([0.0, 1.0])


def test_promote_retires_previous_active(db_path: Path) -> None:
    database.insert_model_version(
        db_path, "TEST", "anomaly", "v1", "active", {}, {}, RUN
    )
    database.insert_model_version(
        db_path, "TEST", "anomaly", "v2", "candidate", {}, {}, RUN
    )
    database.promote_model(db_path, "TEST", "anomaly", "v2")

    active = database.get_active_model(db_path, "TEST", "anomaly")
    assert active is not None
    assert active["version"] == "v2"


def test_promote_unknown_version_fails_loudly(db_path: Path) -> None:
    with pytest.raises(PersistenceError, match="no se puede promover"):
        database.promote_model(db_path, "TEST", "anomaly", "nope")


def test_last_registration_time(db_path: Path) -> None:
    assert (
        database.get_last_registration_time(db_path, "TEST", "anomaly") is None
    )
    database.insert_model_version(
        db_path, "TEST", "anomaly", "v1", "candidate", {}, {}, RUN
    )
    assert (
        database.get_last_registration_time(db_path, "TEST", "anomaly") == RUN
    )
