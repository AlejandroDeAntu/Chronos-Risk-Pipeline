"""PDF ejecutivo (A4 horizontal) con ``matplotlib.backends.backend_pdf``.

Reutiliza las mismas funciones ``draw_*`` que los PNG, compuestas por
página. Sin dependencias nuevas (WeasyPrint exigiría GTK en Windows).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from src.reporting.data import AssetSection, ReportData
from src.reporting.figures import (
    draw_anomaly_lift,
    draw_cumulative_qlike,
    draw_fan_chart,
    draw_hourly_profiles,
    draw_qq,
    draw_var_exceptions,
    draw_volatility_tracking,
)
from src.reporting.html import LIMITATIONS
from src.reporting.style import (
    FAMILY_LABELS,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    format_p,
)

A4_LANDSCAPE = (11.69, 8.27)
_WRAP = 150


def _page(title: str, data: ReportData, number: int) -> Figure:
    figure = Figure(figsize=A4_LANDSCAPE)
    figure.text(0.04, 0.955, title, fontsize=16, weight="bold", color=INK)
    figure.text(
        0.04,
        0.02,
        f"Fuente: Yahoo Finance vía yfinance · Generado "
        f"{data.generated_at:%Y-%m-%d %H:%M} UTC",
        fontsize=7.5,
        color=INK_MUTED,
    )
    figure.text(
        0.96, 0.02, str(number), fontsize=7.5, color=INK_MUTED, ha="right"
    )
    return figure


def _kpi_rows(assets: tuple[AssetSection, ...]) -> list[list[str]]:
    rows = []
    for a in assets:
        v = a.volatility
        if v is None:
            continue
        by_name = {r.name: r for r in v.rows}
        dm = by_name["ewma"].vs_champion
        rows.append(
            [
                a.name,
                FAMILY_LABELS[v.champion],
                f"{v.n_common:,}",
                f"{by_name[v.champion].mean_qlike:.3f}",
                f"{by_name['ewma'].mean_qlike:.3f}",
                f"{dm.t_stat:.2f} ({format_p(dm.p_value_one_sided)})",
                f"{v.forecast_vol_pct:.2f}%",
                f"{v.var_99:.2%}",
                format_p(v.var[0.01].p_unconditional),
                format_p(v.var[0.01].p_independence),
            ]
        )
    return rows


def _cover(data: ReportData) -> Figure:
    figure = _page("Reporte de volatilidad y riesgo", data, 1)
    y = 0.885
    figure.text(0.04, y, "Resumen ejecutivo", fontsize=12, weight="bold")
    for i, finding in enumerate(data.findings, start=1):
        wrapped = textwrap.fill(f"{i}. {finding}", _WRAP)
        y -= 0.035 + 0.022 * wrapped.count("\n")
        figure.text(
            0.05, y, wrapped, fontsize=9, color=INK_SECONDARY, va="bottom"
        )
    rows = _kpi_rows(data.assets)
    if rows:
        ax = figure.add_axes((0.04, 0.06, 0.92, y - 0.12))
        ax.axis("off")
        table = ax.table(
            cellText=rows,
            colLabels=[
                "Activo",
                "Campeón",
                "n",
                "QLIKE campeón",
                "QLIKE EWMA",
                "DM t (p)",
                "Vol. 1d",
                "VaR 99%",
                "Kupiec p",
                "Christoffersen p",
            ],
            loc="upper center",
            cellLoc="right",
            colLoc="right",
            colWidths=[
                0.11,
                0.11,
                0.08,
                0.11,
                0.09,
                0.13,
                0.07,
                0.08,
                0.08,
                0.12,
            ],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.5)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(GRID)
            cell.visible_edges = "B"
            if col == 0:
                cell.set_text_props(ha="left")
            if row == 0:
                cell.set_text_props(weight="bold", color=INK_SECONDARY)
    return figure


def _asset_page(asset: AssetSection, data: ReportData, number: int) -> Figure:
    section = asset.volatility
    figure = _page(asset.name, data, number)
    narrative = textwrap.fill(
        f"{section.narrative} Validación {section.start:%Y-%m-%d} a "
        f"{section.end:%Y-%m-%d}, n = {section.n_common:,} días.",
        _WRAP,
    )
    figure.text(
        0.04, 0.925, narrative, fontsize=8.5, color=INK_SECONDARY, va="top"
    )
    # Cada línea extra de la narrativa baja la rejilla para no pisarla.
    top = 0.87 - 0.022 * narrative.count("\n")
    grid = figure.add_gridspec(
        3,
        2,
        left=0.06,
        right=0.97,
        top=top,
        bottom=0.1,
        hspace=0.8,
        wspace=0.18,
    )
    draw_fan_chart(
        figure.add_subplot(grid[0, 0]),
        section,
        asset.name,
        compact=True,
    )
    draw_volatility_tracking(figure.add_subplot(grid[0, 1]), section)
    draw_qq(figure.add_subplot(grid[1, 0]), section)
    draw_cumulative_qlike(figure.add_subplot(grid[1, 1]), section)
    draw_var_exceptions(figure.add_subplot(grid[2, :]), section)
    return figure


def _anomaly_page(data: ReportData, number: int) -> Figure:
    figure = _page("Detector de anomalías intradía", data, number)
    grid = figure.add_gridspec(
        3,
        4,
        left=0.12,
        right=0.97,
        top=0.9,
        bottom=0.07,
        hspace=0.75,
        wspace=0.12,
        height_ratios=(1.5, 1.0, 1.0),
    )
    draw_anomaly_lift(figure.add_subplot(grid[0, :]), data.assets, data.alpha)
    first = figure.add_subplot(grid[1, 0])
    profiles = [first] + [
        figure.add_subplot(grid[1 + i // 4, i % 4], sharey=first)
        for i in range(1, 8)
    ]
    draw_hourly_profiles(profiles, data.assets)
    return figure


def _methodology_page(data: ReportData, number: int) -> Figure:
    figure = _page("Metodología y limitaciones", data, number)
    y = 0.89
    for key, value in data.methodology.items():
        wrapped = textwrap.fill(value, 120)
        figure.text(0.04, y, key, fontsize=9, weight="bold", va="top")
        figure.text(
            0.24, y, wrapped, fontsize=9, color=INK_SECONDARY, va="top"
        )
        y -= 0.04 + 0.025 * wrapped.count("\n")
    y -= 0.02
    figure.text(0.04, y, "Limitaciones", fontsize=11, weight="bold", va="top")
    for limitation in LIMITATIONS:
        y -= 0.04
        wrapped = textwrap.fill(f"• {limitation}", _WRAP)
        figure.text(
            0.05, y, wrapped, fontsize=9, color=INK_SECONDARY, va="top"
        )
        y -= 0.022 * wrapped.count("\n")
    errors = [f"{a.name}: {e}" for a in data.assets for e in a.errors]
    if errors:
        y -= 0.05
        figure.text(0.04, y, "Errores", fontsize=11, weight="bold", va="top")
        for error in errors:
            y -= 0.035
            figure.text(
                0.05,
                y,
                textwrap.fill(f"• {error}", _WRAP),
                fontsize=8,
                color=INK_SECONDARY,
                va="top",
            )
    return figure


def render_pdf(data: ReportData, path: Path) -> Path:
    """Escribe el PDF ejecutivo; devuelve la ruta."""
    pages = [_cover(data)]
    pages += [
        _asset_page(a, data, len(pages) + i + 1)
        for i, a in enumerate(a for a in data.assets if a.volatility)
    ]
    if any(a.anomaly for a in data.assets):
        pages.append(_anomaly_page(data, len(pages) + 1))
    pages.append(_methodology_page(data, len(pages) + 1))
    with PdfPages(path) as pdf:
        for page in pages:
            pdf.savefig(page, facecolor="white")
        info = pdf.infodict()
        info["Title"] = "Reporte de volatilidad y riesgo"
    return path
