"""Linaje de evidencia: qué versiones de volatilidad comparten historial."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from src.persistence import database, registry

SIGNAL = "garch_volatility_forecast"
T0 = datetime(2026, 10, 1, tzinfo=UTC)


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "lineage.db"
    database.init_db(path)
    return path


def _promote(
    db_path: Path,
    version: str,
    day: int,
    family: str | None = "garch",
    triggered_by: str | None = "fixed_cadence",
) -> None:
    metadata: dict[str, Any] = {} if family is None else {"family": family}
    metrics: dict[str, Any] = (
        {} if triggered_by is None else {"triggered_by": triggered_by}
    )
    database.insert_model_version(
        db_path,
        "^TEST",
        SIGNAL,
        version,
        "candidate",
        metadata,
        metrics,
        T0 + timedelta(days=day),
        promote=True,
    )


def test_cadence_refits_of_same_family_share_evidence(db_path: Path) -> None:
    _promote(db_path, "v1", 0, triggered_by="cold_start")
    _promote(db_path, "v2", 7)
    _promote(db_path, "v3", 14)

    assert registry.volatility_lineage(db_path, "^TEST") == (
        "v3",
        "v2",
        "v1",
    )


def test_family_change_starts_new_lineage(db_path: Path) -> None:
    _promote(db_path, "v1", 0)
    _promote(db_path, "v2", 7, family="har")
    _promote(db_path, "v3", 14, family="har")

    assert registry.volatility_lineage(db_path, "^TEST") == ("v3", "v2")


def test_degradation_retrain_starts_new_lineage(db_path: Path) -> None:
    _promote(db_path, "v1", 0)
    _promote(db_path, "v2", 3, triggered_by="degradation")
    _promote(db_path, "v3", 10)

    assert registry.volatility_lineage(db_path, "^TEST") == ("v3", "v2")


def test_legacy_versions_without_family_are_excluded(db_path: Path) -> None:
    _promote(db_path, "v1", 0, family=None, triggered_by=None)
    _promote(db_path, "v2", 7, triggered_by=None)

    assert registry.volatility_lineage(db_path, "^TEST") == ("v2",)


def test_no_active_model_has_empty_lineage(db_path: Path) -> None:
    assert registry.volatility_lineage(db_path, "^TEST") == ()
