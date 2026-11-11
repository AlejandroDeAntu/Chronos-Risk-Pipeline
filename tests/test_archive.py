"""Archivo de payloads viejos: nada se pierde, la base deja de crecer."""

from __future__ import annotations

import gzip
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.persistence import archive, database

NOW = datetime(2026, 11, 14, 3, 0, tzinfo=UTC)
CUTOFF = NOW - timedelta(days=180)


def _predict(db_path: Path, reference: datetime, closed: bool) -> int:
    prediction_id = database.insert_prediction(
        db_path, "^GSPC", "anomaly", "v1", reference, 10, {"z": 2.5}, NOW
    )
    assert prediction_id is not None
    if closed:
        database.insert_outcome(
            db_path, prediction_id, 1.0, 0.0, 0.0, 1.0, NOW
        )
    return prediction_id


def _payload(db_path: Path, prediction_id: int) -> dict[str, object]:
    with sqlite3.connect(db_path) as conn:
        return json.loads(
            conn.execute(
                "SELECT payload_json FROM predictions WHERE id = ?",
                (prediction_id,),
            ).fetchone()[0]
        )


def test_archives_only_old_closed_predictions(tmp_path: Path) -> None:
    db_path = tmp_path / "pipeline.db"
    database.init_db(db_path)
    old = _predict(db_path, CUTOFF - timedelta(days=3), closed=True)
    old_pending = _predict(db_path, CUTOFF - timedelta(days=2), closed=False)
    recent = _predict(db_path, CUTOFF + timedelta(days=1), closed=True)

    n = archive.archive_old_payloads(db_path, tmp_path / "archive", CUTOFF)
    again = archive.archive_old_payloads(db_path, tmp_path / "archive", CUTOFF)

    assert (n, again) == (1, 0)
    assert _payload(db_path, old) == {"archived": True}
    assert _payload(db_path, old_pending) == {"z": 2.5}
    assert _payload(db_path, recent) == {"z": 2.5}
    [path] = list((tmp_path / "archive").glob("predictions_*.jsonl.gz"))
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        [line] = [json.loads(row) for row in handle]
    assert line["id"] == old
    assert line["payload"] == {"z": 2.5}
    # El resultado y la fila siguen en la base: solo se movió el detalle.
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0] == 2
        assert (
            conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 3
        )
