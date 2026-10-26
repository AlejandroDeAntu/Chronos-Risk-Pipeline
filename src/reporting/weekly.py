"""Resumen semanal: qué cambió respecto a la semana anterior.

Tres fuentes, todas reproducibles:

- el reporte de esta corrida (métricas de los modelos por activo),
  comparado con la foto de la semana anterior guardada en
  ``data/weekly_snapshots.jsonl`` (una línea JSON por semana);
- la base auditable (operación en vivo: ciclos, resultados, pérdidas);
- el historial de git (avance del código y de las pruebas).
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.config import DataQualityConfig
from src.exceptions import PersistenceError
from src.persistence import database
from src.reporting.data import AssetSection, ReportData
from src.reporting.style import FAMILY_LABELS, format_p
from src.schemas import SignalType

logger = logging.getLogger(__name__)

WEEK = timedelta(days=7)
# Una foto con menos de 3 días es de la misma semana (p. ej. una corrida
# manual repetida) y no sirve como referencia.
_MIN_SNAPSHOT_AGE = timedelta(days=3)
# Ciclos programados por semana en .github/workflows/pipeline.yml.
ANOMALY_CYCLES_PER_WEEK = 36 * 5  # cada 15 min de 13 a 21 UTC, lun-vie
VOLATILITY_CYCLES_PER_WEEK = 5  # 00:30 UTC, mar-sáb
_MIN_CRON_COVERAGE = 0.5
# GitHub advierte desde 50 MB por archivo y rechaza desde 100 MB: la base
# versionada debe migrarse antes de llegar ahí.
DB_SIZE_WARN_MB = 50.0
BOT_AUTHOR = "trading-pipeline-bot"
_ANOMALY = SignalType.ANOMALY.value
_VOL = SignalType.GARCH_VOLATILITY_FORECAST.value


@dataclass(frozen=True)
class Metric:
    """Un indicador de esta semana frente a la anterior."""

    label: str
    current: float | None
    previous: float | None = None
    fmt: str = "{:,.0f}"
    higher_is_better: bool | None = True

    @property
    def delta(self) -> float | None:
        """Cambio respecto a la semana anterior (``None`` si falta uno)."""
        if self.current is None or self.previous is None:
            return None
        return self.current - self.previous

    @property
    def improved(self) -> bool | None:
        """``True``/``False`` si mejoró/empeoró; ``None`` si no aplica."""
        delta = self.delta
        if not delta or self.higher_is_better is None:
            return None
        return (delta > 0) == self.higher_is_better

    def show(self, value: float | None) -> str:
        """Formatea un valor del indicador (``—`` si no hay dato)."""
        return "—" if value is None else self.fmt.format(value)


@dataclass(frozen=True)
class AssetWeek:
    """Modelo de volatilidad de un activo: hoy, semana pasada y legado."""

    symbol: str
    name: str
    champion: str
    previous_champion: str | None
    qlike: float
    qlike_previous: float | None
    gain_vs_legacy: float | None  # QLIKE campeón − GARCH(1,1)-t; < 0 mejora
    p_vs_legacy: float | None
    gain_vs_ewma: float
    p_vs_ewma: float
    var99_hit_rate: float
    var99_hit_rate_previous: float | None
    kupiec_p: float
    forecast_vol_pct: float


@dataclass(frozen=True)
class WeeklySummary:
    """Contenido del resumen semanal."""

    week_start: datetime
    week_end: datetime
    has_previous_snapshot: bool
    highlights: tuple[str, ...]
    assets: tuple[AssetWeek, ...]
    live: tuple[Metric, ...]
    project: tuple[Metric, ...]
    alerts: tuple[str, ...]
    snapshot: dict[str, Any]


def load_previous_snapshot(path: Path, now: datetime) -> dict | None:
    """Última foto con al menos 3 días de antigüedad, o ``None``.

    Las líneas ilegibles se omiten con una advertencia: el historial es
    informativo y no debe tumbar el reporte.
    """
    if not path.is_file():
        return None
    best: dict | None = None
    for number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        try:
            entry = json.loads(line)
            when = datetime.fromisoformat(entry["generated_at"])
        except (ValueError, KeyError, TypeError):
            logger.warning("Línea %d ilegible en %s; se omite.", number, path)
            continue
        if now - when >= _MIN_SNAPSHOT_AGE:
            best = entry
    return best


def _asset_snapshot(asset: AssetSection) -> dict[str, Any] | None:
    section = asset.volatility
    if section is None:
        return None
    rows = {row.name: row for row in section.rows}
    legacy = rows.get("garch")
    legacy_p = (
        None
        if legacy is None or legacy.vs_champion is None
        else legacy.vs_champion.p_value_one_sided
    )
    var99 = section.var[0.01]
    return {
        "champion": section.champion,
        "qlike": rows[section.champion].mean_qlike,
        "qlike_garch": None if legacy is None else legacy.mean_qlike,
        "p_vs_garch": legacy_p,
        "qlike_ewma": rows["ewma"].mean_qlike,
        "p_vs_ewma": rows["ewma"].vs_champion.p_value_one_sided,
        "var99_hit_rate": var99.hit_rate,
        "kupiec_p": var99.p_unconditional,
        "christoffersen_p": getattr(var99, "p_independence", None),
        "forecast_vol_pct": section.forecast_vol_pct,
    }


def _asset_week(
    asset: AssetSection, now: dict[str, Any], before: dict[str, Any] | None
) -> AssetWeek:
    before = before or {}
    legacy = now["qlike_garch"]
    return AssetWeek(
        symbol=asset.symbol,
        name=asset.name,
        champion=now["champion"],
        previous_champion=before.get("champion"),
        qlike=now["qlike"],
        qlike_previous=before.get("qlike"),
        gain_vs_legacy=None if legacy is None else now["qlike"] - legacy,
        p_vs_legacy=now["p_vs_garch"],
        gain_vs_ewma=now["qlike"] - now["qlike_ewma"],
        p_vs_ewma=now["p_vs_ewma"],
        var99_hit_rate=now["var99_hit_rate"],
        var99_hit_rate_previous=before.get("var99_hit_rate"),
        kupiec_p=now["kupiec_p"],
        forecast_vol_pct=now["forecast_vol_pct"],
    )


def live_metrics(db_path: Path, now: datetime) -> tuple[Metric, ...]:
    """Operación en vivo: últimos 7 días frente a los 7 anteriores."""
    if not db_path.is_file():
        return ()
    try:
        cur = database.fetch_weekly_activity(db_path, now - WEEK, now)
        prev = database.fetch_weekly_activity(
            db_path, now - 2 * WEEK, now - WEEK
        )
        stale = database.count_stale_pending(db_path, now - WEEK)
    except PersistenceError as exc:
        logger.warning("Sin métricas en vivo: %s", exc)
        return ()

    def pair(key: str, signal: str | None = None) -> tuple[Any, Any]:
        if signal is None:
            return cur[key], prev[key]
        return cur[key].get(signal, 0), prev[key].get(signal, 0)

    anomaly_cycles = pair("cycles", _ANOMALY)
    pct = "{:.0%}"
    return (
        Metric("Ciclos de anomalías con registro", *anomaly_cycles),
        Metric(
            "Cobertura del cron de anomalías",
            *(n / ANOMALY_CYCLES_PER_WEEK for n in anomaly_cycles),
            fmt=pct,
        ),
        Metric(
            f"Ciclos de volatilidad (de {VOLATILITY_CYCLES_PER_WEEK})",
            *pair("cycles", _VOL),
        ),
        Metric(
            "Predicciones registradas",
            sum(cur["predictions"].values()),
            sum(prev["predictions"].values()),
        ),
        Metric("Resultados evaluados", *pair("outcomes")),
        Metric(
            "QLIKE en vivo del modelo de volatilidad",
            *pair("vol_model_loss"),
            fmt="{:.3f}",
            higher_is_better=False,
        ),
        Metric(
            "QLIKE en vivo del baseline (SD 20 días)",
            *pair("vol_baseline_loss"),
            fmt="{:.3f}",
            higher_is_better=None,
        ),
        Metric(
            "Días en que el modelo de volatilidad superó al baseline",
            *pair("vol_win_share"),
            fmt=pct,
        ),
        Metric(
            "Precisión en vivo de anomalías",
            *pair("anomaly_precision"),
            fmt=pct,
        ),
        Metric(
            "Prevalencia de reversión (azar)",
            *pair("anomaly_prevalence"),
            fmt=pct,
            higher_is_better=None,
        ),
        Metric(
            "Modelos entrenados",
            *pair("models_registered"),
            higher_is_better=None,
        ),
        Metric(
            "Modelos promovidos",
            *pair("models_promoted"),
            higher_is_better=None,
        ),
        Metric(
            "Predicciones pendientes con más de 7 días",
            stale,
            higher_is_better=False,
        ),
        Metric(
            "Tamaño de la base (MB)",
            db_path.stat().st_size / 1e6,
            fmt="{:.1f}",
            higher_is_better=None,
        ),
    )


def _git(repo: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout


def _commits(repo: Path, start: datetime, end: datetime) -> dict[str, int]:
    out = _git(
        repo,
        "log",
        f"--since={start.isoformat()}",
        f"--until={end.isoformat()}",
        "--format=@%an",
        "--numstat",
    )
    counts = {"code": 0, "bot": 0, "added": 0, "deleted": 0}
    author = None
    for line in (out or "").splitlines():
        if line.startswith("@"):
            author = line[1:]
            counts["bot" if author == BOT_AUTHOR else "code"] += 1
        elif line and author != BOT_AUTHOR:
            added, deleted, _ = line.split("\t", 2)
            if added.isdigit():  # "-" en binarios
                counts["added"] += int(added)
                counts["deleted"] += int(deleted)
    return counts


def _count_at(repo: Path, rev: str | None, pattern: str, path: str) -> int:
    if not rev:
        return 0
    out = _git(repo, "grep", "-c", "-E", pattern, rev, "--", path)
    return sum(
        int(line.rsplit(":", 1)[1]) for line in (out or "").splitlines()
    )


def _rev_before(repo: Path, when: datetime) -> str | None:
    out = _git(repo, "rev-list", "-1", f"--before={when.isoformat()}", "HEAD")
    return (out or "").strip() or None


def project_metrics(repo: Path, now: datetime) -> tuple[Metric, ...]:
    """Avance del código: commits, líneas y pruebas en la semana."""
    top = _git(repo, "rev-parse", "--show-toplevel")
    if top is None or Path(top.strip()).resolve() != repo.resolve():
        return ()
    cur = _commits(repo, now - WEEK, now)
    prev = _commits(repo, now - 2 * WEEK, now - WEEK)
    head, week_start = "HEAD", _rev_before(repo, now - WEEK)
    tests = r"^[[:space:]]*def test_"
    return (
        Metric(
            "Commits de código",
            cur["code"],
            prev["code"],
            higher_is_better=None,
        ),
        Metric(
            "Commits automáticos del pipeline",
            cur["bot"],
            prev["bot"],
            higher_is_better=None,
        ),
        Metric(
            "Líneas agregadas",
            cur["added"],
            prev["added"],
            higher_is_better=None,
        ),
        Metric(
            "Líneas eliminadas",
            cur["deleted"],
            prev["deleted"],
            higher_is_better=None,
        ),
        Metric(
            "Funciones de prueba",
            _count_at(repo, head, tests, "tests"),
            _count_at(repo, week_start, tests, "tests"),
        ),
        Metric(
            "Líneas de código en src/",
            _count_at(repo, head, ".?", "src"),
            _count_at(repo, week_start, ".?", "src"),
            higher_is_better=None,
        ),
    )


def since_start(repo: Path) -> str | None:
    """Avance acumulado desde el primer commit, en una frase."""
    roots = (_git(repo, "rev-list", "--max-parents=0", "HEAD") or "").split()
    authors = (_git(repo, "log", "--format=%an") or "").splitlines()
    if not roots:
        return None
    first = roots[-1]
    date = (_git(repo, "show", "-s", "--format=%cs", first) or "").strip()
    tests = r"^[[:space:]]*def test_"
    code = sum(1 for a in authors if a != BOT_AUTHOR)
    return (
        f"Desde el primer commit ({date}): {code} commits de código, "
        f"{len(authors) - code} ciclos automáticos versionados y "
        f"funciones de prueba de {_count_at(repo, first, tests, 'tests')} a "
        f"{_count_at(repo, 'HEAD', tests, 'tests')}."
    )


def _alerts(
    data: ReportData, assets: tuple[AssetWeek, ...], live: dict[str, Metric]
) -> tuple[str, ...]:
    alerts = []
    for a in assets:
        if a.previous_champion and a.previous_champion != a.champion:
            alerts.append(
                f"{a.name} cambió de campeón: "
                f"{FAMILY_LABELS[a.previous_champion]} → "
                f"{FAMILY_LABELS[a.champion]}."
            )
        if a.kupiec_p < data.alpha:
            alerts.append(
                f"{a.name}: VaR 99% con cobertura rechazada (Kupiec p "
                f"{format_p(a.kupiec_p)}; {a.var99_hit_rate:.2%} de "
                "excepciones contra 1% nominal)."
            )
    stale_limit = DataQualityConfig().max_zero_return_share
    for asset in data.assets:
        var99 = getattr(asset.volatility, "var", {}).get(0.01)
        p_ind = getattr(var99, "p_independence", None)
        if p_ind is not None and p_ind < data.alpha:
            alerts.append(
                f"{asset.name}: excepciones del VaR 99% agrupadas "
                f"(Christoffersen p {format_p(p_ind)})."
            )
        share = asset.zero_return_share
        if share is not None and share > stale_limit:
            alerts.append(
                f"{asset.name}: {share:.0%} de retornos de 5 min en cero "
                "(cotizaciones rancias)."
            )
        alerts.extend(f"{asset.name}: error — {e}" for e in asset.errors)
    alerts.extend(_live_alerts(live))
    return tuple(alerts)


def _live_alerts(live: dict[str, Metric]) -> list[str]:
    alerts = []
    coverage = live.get("Cobertura del cron de anomalías")
    if (
        coverage
        and coverage.current is not None
        and coverage.current < _MIN_CRON_COVERAGE
    ):
        alerts.append(
            "El cron de anomalías solo registró el "
            f"{coverage.current:.0%} de los ciclos programados: GitHub "
            "Actions no garantiza los cron frecuentes."
        )
    model = live.get("QLIKE en vivo del modelo de volatilidad")
    base = live.get("QLIKE en vivo del baseline (SD 20 días)")
    if (
        model
        and base
        and None not in (model.current, base.current)
        and model.current > base.current
    ):
        alerts.append(
            "En vivo, el modelo de volatilidad tuvo más pérdida que el "
            f"baseline esta semana ({model.current:.3f} vs "
            f"{base.current:.3f})."
        )
    stale = live.get("Predicciones pendientes con más de 7 días")
    if stale and stale.current:
        alerts.append(
            f"{stale.current:,.0f} predicciones llevan más de 7 días sin "
            "evaluarse."
        )
    size = live.get("Tamaño de la base (MB)")
    if size and size.current and size.current > DB_SIZE_WARN_MB:
        alerts.append(
            f"La base versionada pesa {size.current:.0f} MB: GitHub rechaza "
            "archivos de más de 100 MB; conviene migrarla."
        )
    return alerts


def _highlights(
    data: ReportData,
    assets: tuple[AssetWeek, ...],
    live: dict[str, Metric],
    project: dict[str, Metric],
    history: str | None,
    has_previous: bool,
) -> tuple[str, ...]:
    out = []
    if assets:
        families: dict[str, list[str]] = {}
        for a in assets:
            families.setdefault(a.champion, []).append(a.name)
        champions = "; ".join(
            f"{FAMILY_LABELS[f]} en {len(n)} ({', '.join(n)})"
            for f, n in sorted(families.items(), key=lambda kv: -len(kv[1]))
        )
        changes = sum(
            1
            for a in assets
            if a.previous_champion and a.previous_champion != a.champion
        )
        tail = (
            f"{changes} cambio(s) de campeón respecto a la semana anterior."
            if has_previous
            else "Primera semana con foto guardada: la comparación de "
            "modelos semana a semana empieza el próximo lunes."
        )
        out.append(f"Campeón de volatilidad: {champions}. {tail}")
        better = [
            a
            for a in assets
            if a.gain_vs_legacy is not None
            and a.gain_vs_legacy < 0
            and a.p_vs_legacy is not None
            and a.p_vs_legacy < data.alpha
        ]
        best = min(better, key=lambda a: a.gain_vs_legacy, default=None)
        out.append(
            "Frente al modelo original (GARCH(1,1)-t), el campeón reduce la "
            f"pérdida QLIKE con significancia en {len(better)} de "
            f"{len(assets)} activos"
            + (
                f"; la mayor mejora es {best.name} "
                f"(ΔQLIKE {best.gain_vs_legacy:+.3f})."
                if best
                else "."
            )
        )
    outcomes = live.get("Resultados evaluados")
    wins = live.get("Días en que el modelo de volatilidad superó al baseline")
    if outcomes and wins:
        out.append(
            f"En vivo: {outcomes.show(outcomes.current)} resultados evaluados "
            f"(semana anterior {outcomes.show(outcomes.previous)}); el modelo "
            f"de volatilidad superó al baseline el {wins.show(wins.current)} "
            f"de los días (semana anterior {wins.show(wins.previous)})."
        )
    commits, tests = (
        project.get("Commits de código"),
        project.get("Funciones de prueba"),
    )
    if commits and tests:
        added = project["Líneas agregadas"].current
        deleted = project["Líneas eliminadas"].current
        out.append(
            f"Código: {commits.show(commits.current)} commits esta semana "
            f"(+{added:,.0f} / −{deleted:,.0f} líneas); "
            f"{tests.show(tests.current)} funciones de prueba "
            f"({tests.delta or 0:+,.0f} en la semana)."
        )
    if history:
        out.append(history)
    return tuple(out)


def build_weekly_summary(
    data: ReportData, history_path: Path, db_path: Path, repo_root: Path
) -> WeeklySummary:
    """Arma el resumen semanal a partir del reporte, la base y git.

    Parameters
    ----------
    data:
        ``ReportData`` de esta corrida.
    history_path:
        JSONL con las fotos de semanas anteriores.
    db_path:
        Base auditable del pipeline.
    repo_root:
        Raíz del repositorio git (sin git, se omite el avance del código).
    """
    now = data.generated_at
    previous = load_previous_snapshot(history_path, now)
    before = (previous or {}).get("assets", {})
    snaps = {a.symbol: _asset_snapshot(a) for a in data.assets}
    assets = tuple(
        _asset_week(a, snaps[a.symbol], before.get(a.symbol))
        for a in data.assets
        if snaps[a.symbol] is not None
    )
    live = live_metrics(db_path, now)
    project = project_metrics(repo_root, now)
    live_map = {m.label: m for m in live}
    project_map = {m.label: m for m in project}
    snapshot = {
        "generated_at": now.isoformat(),
        "assets": {s: v for s, v in snaps.items() if v is not None},
        "live": {m.label: m.current for m in live},
        "project": {m.label: m.current for m in project},
    }
    return WeeklySummary(
        week_start=now - WEEK,
        week_end=now,
        has_previous_snapshot=previous is not None,
        highlights=_highlights(
            data,
            assets,
            live_map,
            project_map,
            since_start(repo_root) if project else None,
            previous is not None,
        ),
        assets=assets,
        live=live,
        project=project,
        alerts=_alerts(data, assets, live_map),
        snapshot=snapshot,
    )


def snapshot_line(summary: WeeklySummary) -> str:
    """Foto de esta semana como una sola línea JSON (para el historial)."""
    return json.dumps(
        summary.snapshot, ensure_ascii=False, separators=(",", ":")
    )
