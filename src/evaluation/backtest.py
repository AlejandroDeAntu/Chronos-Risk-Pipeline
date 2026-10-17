"""Backtest walk-forward sobre datos históricos, sin escribir en la base.

Reutiliza exactamente la misma validación que decide las promociones en
producción (``walk_forward_anomaly`` / ``walk_forward_volatility``) y
la convierte en un reporte Markdown con las tablas modelo vs baseline y
un QQ-plot de los residuos estandarizados del campeón por activo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from scipy import stats

from src.config import AssetConfig, PipelineSettings
from src.evaluation.metrics import evaluate_volatility_forecast
from src.evaluation.volatility_selection import (
    ChampionDecision,
    VolatilityWalkForward,
    select_champion,
    walk_forward_volatility,
)
from src.exceptions import TradingPipelineError
from src.models.volatility import volatility_frame
from src.pipeline.predict import load_daily_ohlc, load_intraday_returns
from src.pipeline.retrain import (
    AnomalyWalkForwardResult,
    walk_forward_anomaly,
)
from src.schemas import SignalType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssetBacktest:
    """Resultado del backtest de un activo (``None`` si no se evaluó)."""

    asset: AssetConfig
    anomaly: AnomalyWalkForwardResult | None
    volatility: VolatilityWalkForward | None
    champion: ChampionDecision | None
    errors: tuple[str, ...]


def run_backtest(
    settings: PipelineSettings,
    signals: tuple[SignalType, ...],
    now: datetime,
) -> list[AssetBacktest]:
    """Ejecuta la validación walk-forward de cada señal sobre cada activo.

    Un fallo en un activo o señal se registra en ``errors`` y no detiene el
    resto del backtest.
    """
    results: list[AssetBacktest] = []
    for asset in settings.assets:
        anomaly: AnomalyWalkForwardResult | None = None
        volatility: VolatilityWalkForward | None = None
        champion: ChampionDecision | None = None
        errors: list[str] = []
        if SignalType.ANOMALY in signals:
            try:
                log_return = load_intraday_returns(
                    asset, settings, settings.anomaly.history_period, now
                )
                anomaly = walk_forward_anomaly(log_return, settings)
            except TradingPipelineError as exc:
                errors.append(f"anomaly: {type(exc).__name__}: {exc}")
        if SignalType.GARCH_VOLATILITY_FORECAST in signals:
            try:
                frame = volatility_frame(load_daily_ohlc(asset, settings, now))
                volatility = walk_forward_volatility(
                    frame, settings, settings.garch.candidate_families
                )
                champion = select_champion(
                    volatility,
                    settings.garch.default_family,
                    settings.retrain.promotion_alpha,
                )
            except TradingPipelineError as exc:
                errors.append(f"garch: {type(exc).__name__}: {exc}")
        for error in errors:
            logger.error("[backtest] %s: %s", asset.symbol, error)
        results.append(
            AssetBacktest(asset, anomaly, volatility, champion, tuple(errors))
        )
    return results


def _fmt_ci(interval: tuple[float, float]) -> str:
    low, high = interval
    if not (np.isfinite(low) and np.isfinite(high)):
        return "n/d"
    return f"[{low:.3f}, {high:.3f}]"


def _anomaly_table(results: list[AssetBacktest]) -> list[str]:
    lines = [
        "## Detector de anomalías (walk-forward purgado, velas de 5 min)",
        "",
        "Positivo = hubo reversión en las siguientes barras. Baseline = "
        "marcar anomalías al azar con la misma tasa que el modelo.",
        "",
        "| Activo | n | Anomalías | TN / FP / FN / TP | Precisión [IC95] "
        "| Recall | F1 | Precisión base | F1 base | ¿Supera? |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    f1_values: list[float] = []
    base_values: list[float] = []
    for item in results:
        if item.anomaly is None:
            continue
        rep = item.anomaly.pooled
        (tn, fp), (fn, tp) = rep.confusion_matrix
        f1_values.append(rep.f1)
        base_values.append(rep.baseline_f1)
        lines.append(
            f"| {item.asset.display_name} | {rep.n_samples} "
            f"| {rep.n_predicted_positive} | {tn} / {fp} / {fn} / {tp} "
            f"| {rep.precision:.3f} {_fmt_ci(rep.precision_ci_95)} "
            f"| {rep.recall:.3f} | {rep.f1:.3f} "
            f"| {rep.baseline_precision:.3f} | {rep.baseline_f1:.3f} "
            f"| {'sí' if rep.beats_baseline_f1 else 'no'} |"
        )
    if f1_values:
        lines.append(
            f"| **Promedio (macro)** | | | | | | **{np.mean(f1_values):.3f}** "
            f"| | **{np.mean(base_values):.3f}** | |"
        )
    return lines


def _volatility_table(results: list[AssetBacktest]) -> list[str]:
    families = sorted(
        {f for r in results if r.volatility for f in r.volatility.forecasts}
    )
    header = " | ".join(f"QLIKE {f}" for f in families)
    lines = [
        "## Volatilidad diaria (walk-forward, pronóstico a 1 día)",
        "",
        "QLIKE contra r^2: menor es mejor. El campeón parte de GARCH y solo "
        "cambia si un retador gana con Diebold-Mariano (Holm). E[z^2] cerca "
        "de 1 indica varianza bien calibrada.",
        "",
        f"| Activo | n | Campeón | {header} | QLIKE SD-20d | QLIKE EWMA "
        "| E[z^2] |",
        "|---|---|---|" + "---|" * len(families) + "---|---|---|",
    ]
    for item in results:
        if item.volatility is None or item.champion is None:
            continue
        wf = item.volatility
        cells = " | ".join(
            f"{wf.mean_qlike(f):.4f}" if f in wf.forecasts else "n/d"
            for f in families
        )
        report = evaluate_volatility_forecast(
            wf.volatility(item.champion.champion),
            wf.realized_pct,
            wf.baselines["sd20"],
        )
        lines.append(
            f"| {item.asset.display_name} | {len(wf.dates)} "
            f"| {item.champion.champion} | {cells} "
            f"| {wf.mean_qlike('sd20'):.4f} | {wf.mean_qlike('ewma'):.4f} "
            f"| {report.mean_squared_standardized_residual:.3f} |"
        )
    return lines


def _save_qq_plot(
    residuals: tuple[float, ...], title: str, path: Path
) -> None:
    """Guarda un QQ-plot contra la normal (Figure sin pyplot: headless)."""
    figure = Figure(figsize=(5, 5))
    axes = figure.subplots()
    stats.probplot(np.asarray(residuals), dist="norm", plot=axes)
    axes.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=110)


def write_backtest_report(
    results: list[AssetBacktest],
    settings: PipelineSettings,
    generated_at: datetime,
) -> Path:
    """Escribe el reporte Markdown (y los QQ-plots) en ``reports/``.

    Returns
    -------
    pathlib.Path
        Ruta del archivo Markdown generado.
    """
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    out_dir = settings.paths.reports_dir / f"backtest_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Backtest walk-forward — {generated_at:%Y-%m-%d %H:%M} UTC",
        "",
        f"Bloques TimeSeriesSplit: {settings.retrain.walk_forward_splits}. "
        f"Historia intradía: {settings.anomaly.history_period}. "
        f"Historia diaria desde: {settings.garch.history_start_date}.",
        "",
        *_anomaly_table(results),
        "",
        *_volatility_table(results),
        "",
        "## QQ-plots de residuos estandarizados (campeón)",
        "",
    ]
    for item in results:
        if item.volatility is None or item.champion is None:
            continue
        sigma = item.volatility.volatility(item.champion.champion)
        residuals = tuple(item.volatility.realized_pct / sigma)
        file_name = f"qq_{item.asset.symbol.strip('^').replace('=', '_')}.png"
        _save_qq_plot(
            residuals,
            f"QQ residuos z — {item.asset.display_name}",
            out_dir / file_name,
        )
        lines.append(f"![{item.asset.display_name}]({file_name})")

    failures = [e for item in results for e in item.errors]
    lines += ["", "## Errores", ""]
    lines += [f"- {e}" for e in failures] if failures else ["- Ninguno."]

    report_path = out_dir / "backtest.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
