"""Mueve a ``data/archive/`` el detalle de predicciones viejas ya cerradas.

``payload_json`` es más de la mitad de ``data/pipeline.db`` y solo se
consulta en la ventana reciente de evaluación. Pasada la retención, el
detalle se copia a ``predictions_AAAA-MM.jsonl.gz`` (un archivo por mes,
que ya no cambia y git guarda una sola vez) y en la base queda
``{"archived": true}``. La fila, su resultado y sus métricas no se tocan::

    python -m src.persistence.archive            # retención de 180 días
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from datetime import UTC, datetime, timedelta
from itertools import groupby
from pathlib import Path

from src.config import get_settings
from src.persistence.database import _connect

logger = logging.getLogger(__name__)

RETENTION_DAYS = 180
"""Muy por encima de cualquier ventana que lea payloads: la más larga son
las 500 barras de 5 min de la degradación de anomalías (~2 días)."""

_ARCHIVED = json.dumps({"archived": True})


def archive_old_payloads(
    db_path: Path, archive_dir: Path, cutoff: datetime
) -> int:
    """Archiva los payloads de predicciones cerradas anteriores a ``cutoff``.

    Primero escribe los archivos y solo después vacía la base, en una
    transacción: si algo falla a la mitad puede quedar una línea repetida
    en el archivo (se identifica por ``id``), nunca un payload perdido.

    Returns
    -------
    int
        Predicciones archivadas en esta llamada.
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, asset_symbol, signal_type, model_version,
                   reference_timestamp, payload_json
            FROM predictions
            WHERE evaluated != 0 AND reference_timestamp < ?
              AND payload_json != ?
            ORDER BY reference_timestamp
            """,
            (cutoff.isoformat(), _ARCHIVED),
        ).fetchall()
        if not rows:
            return 0
        archive_dir.mkdir(parents=True, exist_ok=True)
        for month, group in groupby(
            rows, key=lambda row: row["reference_timestamp"][:7]
        ):
            path = archive_dir / f"predictions_{month}.jsonl.gz"
            with gzip.open(path, "at", encoding="utf-8") as handle:
                for row in group:
                    record = dict(row)
                    record["payload"] = json.loads(record.pop("payload_json"))
                    handle.write(json.dumps(record) + "\n")
        conn.executemany(
            "UPDATE predictions SET payload_json = ? WHERE id = ?",
            [(_ARCHIVED, row["id"]) for row in rows],
        )
    # Fuera de la transacción: VACUUM devuelve el espacio al sistema.
    with _connect(db_path) as conn:
        conn.execute("VACUUM")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada: archiva con la retención indicada."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=RETENTION_DAYS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    db_path = get_settings().paths.database_path
    n = archive_old_payloads(
        db_path,
        db_path.parent / "archive",
        datetime.now(UTC) - timedelta(days=args.days),
    )
    logger.info("Payloads archivados: %d.", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
