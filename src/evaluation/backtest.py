"""Backtest walk-forward sobre datos históricos, sin escribir en la base.

Reutiliza exactamente la misma validación que decide las promociones en
producción (``walk_forward_anomaly`` / ``walk_forward_garch``) y la
convierte en un reporte Markdown con las tablas modelo vs baseline y un
QQ-plot de los residuos estandarizados del GARCH por activo.
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
from src.exceptions import TradingPipelineError
from src.pipeline.predict import load_daily_returns, load_intraday_returns
from src.pipeline.retrain import (
    AnomalyWalkForwardResult,
    GarchWalkForwardResult,
    walk_forward_anomaly,
    walk_forward_garch,
)
from src.schemas import SignalType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssetBacktest:
    """Resultado del backtest de un activo (``None`` si no se evaluó)."""

    asset: AssetConfig
    anomaly: AnomalyWalkForwardResult | None
    garch: GarchWalkForwardResult | None
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
        garch: GarchWalkForwardResult | None = None
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
                _, daily_return = load_daily_returns(asset, settings)
                garch = walk_forward_garch(daily_return, settings)
            except TradingPipelineError as exc:
                errors.append(f"garch: {type(exc).__name__}: {exc}")
        for error in errors:
            logger.error("[backtest] %s: %s", asset.symbol, error)
        results.append(AssetBacktest(asset, anomaly, garch, tuple(errors)))
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


def _garch_table(results: list[AssetBacktest]) -> list[str]:
    lines = [
        "## Volatilidad GARCH(1,1)-t (walk-forward, pronóstico a 1 día)",
        "",
        "QLIKE: menor es mejor. Baseline = desviación estándar de los "
        "últimos días (ventana móvil, sin incluir el día pronosticado). "
        "E[z²] cercano a 1 indica varianza bien calibrada; p(LB z²) > 0.05 "
        "indica que no quedan efectos ARCH sin modelar.",
        "",
        "| Activo | n | QLIKE | QLIKE base | RMSE | RMSE base | E[z²] "
        "| p(JB) | p(LB z²) | ¿Supera? |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in results:
        if item.garch is None:
            continue
        rep = item.garch.pooled
        lb = (
            f"{rep.residual_sq_ljung_box_pvalue:.3f}"
            if rep.residual_sq_ljung_box_pvalue is not None
            else "n/d"
        )
        lines.append(
            f"| {item.asset.display_name} | {rep.n_samples} "
            f"| {rep.qlike:.4f} | {rep.baseline_qlike:.4f} "
            f"| {rep.rmse:.4f} | {rep.baseline_rmse:.4f} "
            f"| {rep.mean_squared_standardized_residual:.3f} "
            f"| {rep.residual_jarque_bera_pvalue:.3f} | {lb} "
            f"| {'sí' if rep.beats_baseline_qlike else 'no'} |"
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
        *_garch_table(results),
        "",
        "## QQ-plots de residuos estandarizados (GARCH)",
        "",
    ]
    for item in results:
        if item.garch is None:
            continue
        file_name = f"qq_{item.asset.symbol.strip('^').replace('=', '_')}.png"
        _save_qq_plot(
            item.garch.standardized_residuals,
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
