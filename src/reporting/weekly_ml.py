"""Reporte semanal de cuánto mejoró el modelo frente a su baseline.

Lee ``evaluation_runs`` y ``model_registry`` de ``data/pipeline.db`` (solo
lectura) y escribe ``reports/weekly/AAAA-MM-DD.md``. Lo ejecuta cada viernes
a las 21:00 de Ciudad de México ``.github/workflows/weekly-ml-report.yml``::

    python -m src.reporting.weekly_ml            # corte = ahora
    python -m src.reporting.weekly_ml --week-end 2026-11-14T03:00:00+00:00

Métrica de mejora: la **ventaja** sobre el baseline en las corridas con
muestra suficiente de cada semana. Para QLIKE (menor es mejor) es
``baseline - modelo``; para F1 (mayor es mejor), ``modelo - baseline``. Que
la ventaja suba de una semana a otra es "mejoró". Las ventanas de
evaluación se solapan entre corridas, así que es una lectura descriptiva,
no una prueba estadística. Si el baseline cambió entre semanas (SD-20d a
EWMA en volatilidad), no se comparan.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean

from src.config import get_settings

CDMX = timezone(timedelta(hours=-6), "CDMX")
"""Ciudad de México no tiene horario de verano desde 2022 (UTC-6 fijo)."""

N_WEEKS = 4
_WEEK = timedelta(days=7)
_SIGNAL_LABELS = {
    "anomaly": "Anomalías (F1)",
    "garch_volatility_forecast": "Volatilidad (QLIKE)",
}


@dataclass(frozen=True)
class WeekStat:
    """Resumen de una semana para un activo y una señal."""

    model: float | None
    baseline: float | None
    advantage: float | None
    n_runs: int
    insufficient: bool
    baseline_name: str | None


@dataclass(frozen=True)
class AssetRow:
    """Fila del reporte: semanas de la más reciente a la más antigua."""

    signal_type: str
    asset_symbol: str
    weeks: tuple[WeekStat, ...]

    @property
    def delta(self) -> float | None:
        """Cambio de la ventaja respecto de la semana anterior."""
        current, previous = self.weeks[0], self.weeks[1]
        if (
            current.advantage is None
            or previous.advantage is None
            or current.baseline_name != previous.baseline_name
        ):
            return None
        return current.advantage - previous.advantage

    @property
    def verdict(self) -> str:
        """Lectura en una palabra de la semana."""
        current, previous = self.weeks[0], self.weeks[1]
        if current.advantage is None:
            return "muestra insuficiente"
        if previous.advantage is None:
            return "sin semana previa"
        if current.baseline_name != previous.baseline_name:
            return "cambio de baseline"
        delta = self.delta
        if delta is None or delta == 0.0:
            return "igual"
        return "mejoró" if delta > 0.0 else "empeoró"


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    """Conexión de solo lectura: el reporte nunca modifica la base."""
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _baseline_name(signal_type: str, report_json: str) -> str:
    if signal_type == "anomaly":
        return "aleatorio"
    return json.loads(report_json).get("baseline", "sd20")


def _week_stat(signal_type: str, runs: list[sqlite3.Row]) -> WeekStat:
    sufficient = [
        r
        for r in runs
        if r["sufficient_data"]
        and r["model_metric"] is not None
        and r["baseline_metric"] is not None
    ]
    if not sufficient:
        latest = runs[-1] if runs else None
        return WeekStat(
            model=None if latest is None else latest["model_metric"],
            baseline=None if latest is None else latest["baseline_metric"],
            advantage=None,
            n_runs=len(runs),
            insufficient=bool(runs),
            baseline_name=(
                None
                if latest is None
                else _baseline_name(signal_type, latest["report_json"])
            ),
        )
    model = fmean(r["model_metric"] for r in sufficient)
    baseline = fmean(r["baseline_metric"] for r in sufficient)
    advantage = (
        model - baseline if signal_type == "anomaly" else baseline - model
    )
    return WeekStat(
        model=model,
        baseline=baseline,
        advantage=advantage,
        n_runs=len(sufficient),
        insufficient=False,
        baseline_name=_baseline_name(
            signal_type, sufficient[-1]["report_json"]
        ),
    )


def build_rows(db_path: Path, week_end: datetime) -> list[AssetRow]:
    """Ventaja semanal por activo y señal en las últimas ``N_WEEKS``."""
    start = week_end - N_WEEKS * _WEEK
    with closing(_connect_ro(db_path)) as conn:
        runs = conn.execute(
            """
            SELECT asset_symbol, signal_type, evaluated_at, model_metric,
                   baseline_metric, sufficient_data, report_json
            FROM evaluation_runs
            WHERE evaluated_at >= ? AND evaluated_at < ?
            ORDER BY signal_type, asset_symbol, evaluated_at
            """,
            (start.isoformat(), week_end.isoformat()),
        ).fetchall()
    grouped: dict[tuple[str, str], list[list[sqlite3.Row]]] = {}
    for run in runs:
        key = (run["signal_type"], run["asset_symbol"])
        weeks_ago = int(
            (week_end - datetime.fromisoformat(run["evaluated_at"])) / _WEEK
        )
        grouped.setdefault(key, [[] for _ in range(N_WEEKS)])[
            weeks_ago
        ].append(run)
    return [
        AssetRow(
            signal_type=signal,
            asset_symbol=asset,
            weeks=tuple(_week_stat(signal, week) for week in weeks),
        )
        for (signal, asset), weeks in sorted(grouped.items())
    ]


def _registrations(db_path: Path, week_end: datetime) -> list[sqlite3.Row]:
    with closing(_connect_ro(db_path)) as conn:
        return conn.execute(
            """
            SELECT asset_symbol, signal_type, version, stage, created_at,
                   metadata_json, validation_metrics_json
            FROM model_registry
            WHERE created_at >= ? AND created_at < ?
            ORDER BY created_at
            """,
            ((week_end - _WEEK).isoformat(), week_end.isoformat()),
        ).fetchall()


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:+.{digits}f}"


def _plain(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _row_line(row: AssetRow) -> str:
    current = row.weeks[0]
    trend = " → ".join(_fmt(week.advantage, 3) for week in reversed(row.weeks))
    sample = (
        "insuficiente"
        if current.insufficient
        else f"{current.n_runs} corridas"
    )
    return (
        f"| {row.asset_symbol} | {_plain(current.model)} | "
        f"{_plain(current.baseline)} ({current.baseline_name or '—'}) | "
        f"{_fmt(current.advantage)} | {_fmt(row.delta)} | {trend} | "
        f"{sample} | **{row.verdict}** |"
    )


def _summary(rows: list[AssetRow]) -> list[str]:
    scored = [r for r in rows if r.delta is not None]
    better = sum(1 for r in scored if r.verdict == "mejoró")
    beats = sum(
        1
        for r in rows
        if r.weeks[0].advantage is not None and r.weeks[0].advantage > 0
    )
    with_data = sum(1 for r in rows if r.weeks[0].advantage is not None)
    return [
        f"- **Mejoró** frente a la semana anterior en **{better} de "
        f"{len(scored)}** pares activo/señal comparables.",
        f"- **Supera a su baseline** esta semana en **{beats} de "
        f"{with_data}** pares con muestra suficiente.",
        f"- Pares sin lectura (muestra insuficiente, sin semana previa o "
        f"cambio de baseline): {len(rows) - len(scored)}.",
    ]


def _registration_lines(registrations: list[sqlite3.Row]) -> list[str]:
    if not registrations:
        return ["Sin reentrenamientos esta semana."]
    lines = [
        "| Activo | Señal | Versión | Familia | Disparador | ¿Promovido? | "
        "¿Supera EWMA? |",
        "|---|---|---|---|---|---|---|",
    ]
    for reg in registrations:
        metrics = json.loads(reg["validation_metrics_json"])
        metadata = json.loads(reg["metadata_json"])
        promoted = reg["stage"] in ("active", "retired")
        lines.append(
            f"| {reg['asset_symbol']} | {reg['signal_type']} | "
            f"`{reg['version']}` | {metadata.get('family', 'detector')} | "
            f"{metrics.get('triggered_by', 'sin registro')} | "
            f"{'sí' if promoted else 'no'} | {_beats_ewma(metrics)} |"
        )
    return lines


def _beats_ewma(metrics: dict[str, object]) -> str:
    flag = metrics.get("beats_ewma")
    if flag is None:
        return "—"
    return "sí" if flag else "**no**"


def _ewma_warning(registrations: list[sqlite3.Row]) -> list[str]:
    laggards = sorted(
        {
            r["asset_symbol"]
            for r in registrations
            if r["stage"] in ("active", "retired")
            and json.loads(r["validation_metrics_json"]).get("beats_ewma")
            is False
        }
    )
    if not laggards:
        return []
    return [
        "- Campeón promovido sin ventaja significativa sobre EWMA (sigue "
        f"sirviendo; vigilar): {', '.join(laggards)}."
    ]


def render_markdown(db_path: Path, week_end: datetime) -> str:
    """Arma el reporte completo en Markdown."""
    rows = build_rows(db_path, week_end)
    registrations = _registrations(db_path, week_end)
    local_end = week_end.astimezone(CDMX)
    drift = [
        r
        for r in registrations
        if json.loads(r["validation_metrics_json"]).get("triggered_by")
        == "drift"
    ]
    lines = [
        f"# Reporte semanal del modelo — {local_end:%Y-%m-%d}",
        "",
        f"Corte: {local_end:%Y-%m-%d %H:%M} hora de Ciudad de México "
        f"({week_end.astimezone(UTC):%Y-%m-%d %H:%M} UTC). Semana = los 7 "
        "días previos al corte.",
        "",
        "## Resumen",
        "",
        *_summary(rows),
        f"- Reentrenamientos: {len(registrations)} "
        f"({sum(r['stage'] in ('active', 'retired') for r in registrations)}"
        f" promovidos); alertas de drift: {len(drift)}.",
        *_ewma_warning(registrations),
        "",
        "**Ventaja** = cuánto mejor que el baseline es el modelo (QLIKE: "
        "baseline − modelo; F1: modelo − baseline). Positivo es bueno. "
        "**Δ semana** = cambio de la ventaja frente a la semana previa. Las "
        "ventanas de evaluación se solapan: es una lectura descriptiva.",
    ]
    for signal, label in _SIGNAL_LABELS.items():
        signal_rows = [r for r in rows if r.signal_type == signal]
        if not signal_rows:
            continue
        lines += [
            "",
            f"## {label}",
            "",
            "| Activo | Modelo | Baseline | Ventaja | Δ semana | "
            f"Tendencia {N_WEEKS} sem. (antigua → reciente) | Muestra | "
            "Lectura |",
            "|---|---|---|---|---|---|---|---|",
            *(_row_line(r) for r in signal_rows),
        ]
    lines += [
        "",
        "## Reentrenamientos de la semana",
        "",
        *_registration_lines(registrations),
        "",
    ]
    return "\n".join(lines)


def write_report(
    db_path: Path, out_dir: Path, week_end: datetime | None = None
) -> Path:
    """Escribe el reporte y devuelve su ruta (``AAAA-MM-DD.md`` en CDMX)."""
    end = week_end or datetime.now(UTC)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{end.astimezone(CDMX):%Y-%m-%d}.md"
    path.write_text(render_markdown(db_path, end), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada: imprime la ruta del reporte generado."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db", type=Path, default=settings.paths.database_path
    )
    parser.add_argument("--out-dir", type=Path, default=Path("reports/weekly"))
    parser.add_argument(
        "--week-end",
        type=datetime.fromisoformat,
        default=None,
        help="Corte ISO 8601 con zona horaria (por defecto, ahora).",
    )
    args = parser.parse_args(argv)
    if args.week_end is not None and args.week_end.tzinfo is None:
        parser.error("--week-end debe incluir zona horaria.")
    print(write_report(args.db, args.out_dir, args.week_end))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
