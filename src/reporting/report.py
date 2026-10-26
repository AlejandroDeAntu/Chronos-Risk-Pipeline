"""Orquesta el reporte: datos -> PNG -> HTML autocontenido -> PDF."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

from matplotlib.figure import Figure

from src.config import PipelineSettings
from src.evaluation.backtest import AssetBacktest
from src.reporting.data import build_report_data
from src.reporting.figures import (
    PNG_DPI,
    draw_anomaly_lift,
    draw_cumulative_qlike,
    draw_fan_chart,
    draw_hourly_profiles,
    draw_qq,
    draw_var_exceptions,
    draw_volatility_tracking,
    save_png,
)
from src.reporting.html import render_html
from src.reporting.pdf import render_pdf
from src.reporting.weekly import build_weekly_summary, snapshot_line
from src.reporting.weekly_render import render_markdown

_WIDE, _SQUARE = (9.0, 3.6), (5.4, 4.2)


@dataclass(frozen=True)
class ReportPaths:
    """Archivos generados por ``write_report``."""

    directory: Path
    html: Path
    pdf: Path
    weekly: Path
    snapshot: Path
    figures: tuple[Path, ...]


def _slug(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", symbol).strip("_").lower()


def write_report(
    results: list[AssetBacktest],
    settings: PipelineSettings,
    generated_at: datetime,
) -> ReportPaths:
    """Genera ``reports/report_<UTC>/`` con HTML, PDF y PNG a 200 dpi.

    Incluye el resumen semanal (``weekly.md``) y su foto en una línea
    JSON (``snapshot.json``), que el workflow semanal agrega al historial.

    Parameters
    ----------
    results:
        Salida de ``run_backtest``.
    settings:
        Configuración usada (rutas, parámetros de la metodología).
    generated_at:
        Instante UTC del reporte.

    Returns
    -------
    ReportPaths
        Rutas del directorio, HTML, PDF y figuras.
    """
    data = build_report_data(results, settings, generated_at)
    out = settings.paths.reports_dir / f"report_{generated_at:%Y%m%dT%H%M%SZ}"
    figures_dir = out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    asset_figures: dict[str, dict[str, Path]] = {}
    for asset in data.assets:
        section = asset.volatility
        if section is None:
            continue
        slug = _slug(asset.symbol)
        draws = {
            "fan": (
                partial(draw_fan_chart, section=section, name=asset.name),
                _WIDE,
            ),
            "tracking": (
                partial(draw_volatility_tracking, section=section),
                _WIDE,
            ),
            "qq": (partial(draw_qq, section=section), _SQUARE),
            "qlike": (
                partial(draw_cumulative_qlike, section=section),
                _SQUARE,
            ),
            "var": (partial(draw_var_exceptions, section=section), _WIDE),
        }
        asset_figures[asset.symbol] = {
            kind: save_png(draw, figures_dir / f"{slug}_{kind}.png", size)
            for kind, (draw, size) in draws.items()
        }

    summary: dict[str, Path] = {}
    anomaly_assets = [a for a in data.assets if a.anomaly]
    if anomaly_assets:
        summary["lift"] = save_png(
            partial(draw_anomaly_lift, assets=data.assets, alpha=data.alpha),
            figures_dir / "anomaly_lift.png",
            (9.0, 0.45 * len(anomaly_assets) + 1.4),
        )
        figure = Figure(figsize=(9.0, 4.6), layout="constrained")
        axes = figure.subplots(2, 4, sharey=True).ravel()
        draw_hourly_profiles(list(axes), data.assets)
        summary["profiles"] = figures_dir / "hourly_profiles.png"
        figure.savefig(summary["profiles"], dpi=PNG_DPI, facecolor="white")

    weekly = build_weekly_summary(
        data,
        settings.paths.weekly_history_path,
        settings.paths.database_path,
        settings.paths.project_root,
    )
    weekly_md = out / "weekly.md"
    weekly_md.write_text(render_markdown(weekly), encoding="utf-8")
    snapshot = out / "snapshot.json"
    snapshot.write_text(snapshot_line(weekly) + "\n", encoding="utf-8")
    html = render_html(
        data, asset_figures, summary, out / "report.html", weekly
    )
    pdf = render_pdf(data, out / "report.pdf", weekly)
    figures = tuple(sorted(figures_dir.glob("*.png")))
    return ReportPaths(
        directory=out,
        html=html,
        pdf=pdf,
        weekly=weekly_md,
        snapshot=snapshot,
        figures=figures,
    )
