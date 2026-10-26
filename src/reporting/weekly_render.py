"""Presentación del resumen semanal: Markdown, sección HTML y página PDF.

El Markdown se publica en el resumen del job de GitHub Actions y se
versiona en ``docs/semanal/``, así se lee sin descargar el artefacto.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from html import escape

from matplotlib.figure import Figure

from src.reporting.style import FAMILY_LABELS, GRID, INK_SECONDARY, format_p
from src.reporting.weekly import AssetWeek, Metric, WeeklySummary

_UP, _DOWN = "▲", "▼"


def _change(metric: Metric) -> str:
    delta = metric.delta
    if delta is None:
        return "—"
    if metric.fmt.format(abs(delta)) == metric.fmt.format(0):
        return "sin cambio"
    arrow = _UP if delta > 0 else _DOWN
    text = f"{arrow} {metric.fmt.format(abs(delta))}"
    verdict = {True: " (mejora)", False: " (empeora)", None: ""}
    return text + verdict[metric.improved]


def _signed(value: float | None, p_value: float | None = None) -> str:
    if value is None:
        return "—"
    text = f"{value:+.3f}"
    return text if p_value is None else f"{text} (p {format_p(p_value)})"


def _asset_cells(a: AssetWeek) -> list[str]:
    previous = (
        FAMILY_LABELS[a.previous_champion] if a.previous_champion else "—"
    )
    weekly = None if a.qlike_previous is None else a.qlike - a.qlike_previous
    return [
        a.name,
        FAMILY_LABELS[a.champion],
        previous,
        f"{a.qlike:.3f}",
        _signed(weekly),
        "mismo modelo"
        if a.champion == "garch"
        else _signed(a.gain_vs_legacy, a.p_vs_legacy),
        _signed(a.gain_vs_ewma, a.p_vs_ewma),
        f"{a.var99_hit_rate:.2%}",
        format_p(a.kupiec_p),
        f"{a.forecast_vol_pct:.2f}%",
    ]


ASSET_HEADERS = [
    "Activo",
    "Campeón",
    "Campeón semana anterior",
    "QLIKE",
    "Δ QLIKE semana",
    "Δ vs GARCH original (p)",
    "Δ vs EWMA (p)",
    "Excepciones VaR 99%",
    "Kupiec p",
    "Vol. 1d",
]
_METRIC_HEADERS = ["Indicador", "Esta semana", "Semana anterior", "Cambio"]
# En el PDF las columnas son angostas: encabezados cortos.
_PDF_ASSET_HEADERS = [
    "Activo",
    "Campeón",
    "Antes",
    "QLIKE",
    "Δ sem.",
    "Δ vs GARCH orig. (p)",
    "Δ vs EWMA (p)",
    "Exc. VaR 99%",
    "Kupiec p",
    "Vol. 1d",
]
_PDF_METRIC_HEADERS = ["Indicador", "Esta sem.", "Sem. ant.", "Cambio"]
_NOTE = (
    "ΔQLIKE < 0 significa menor pérdida (mejor). «GARCH original» es el "
    "modelo con el que arrancó el proyecto; p de una cola de "
    "Diebold-Mariano. Operación en vivo: últimos 7 días contra los 7 "
    "anteriores, según la base auditable."
)


def _metric_cells(m: Metric) -> list[str]:
    return [m.label, m.show(m.current), m.show(m.previous), _change(m)]


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def render_markdown(summary: WeeklySummary) -> str:
    """Resumen semanal en Markdown (GitHub)."""
    parts = [
        f"# Resumen semanal · {summary.week_start:%Y-%m-%d} a "
        f"{summary.week_end:%Y-%m-%d}",
        "## Lo más destacado",
        "\n".join(f"- {h}" for h in summary.highlights),
    ]
    if summary.assets:
        parts += [
            "## Modelos de volatilidad por activo",
            _md_table(
                ASSET_HEADERS, [_asset_cells(a) for a in summary.assets]
            ),
        ]
    for title, metrics in (
        ("Operación en vivo", summary.live),
        ("Avance del proyecto", summary.project),
    ):
        if metrics:
            parts += [
                f"## {title}",
                _md_table(
                    _METRIC_HEADERS, [_metric_cells(m) for m in metrics]
                ),
            ]
    parts += [
        "## Alertas",
        "\n".join(f"- {a}" for a in summary.alerts) or "- Ninguna.",
        f"_{_NOTE}_",
    ]
    return "\n\n".join(parts) + "\n"


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(c)}</td>" for c in r) + "</tr>"
        for r in rows
    )
    return (
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


def _html_metrics(metrics: tuple[Metric, ...]) -> str:
    rows = []
    for m in metrics:
        css = {True: "pos", False: "neg", None: ""}[m.improved]
        cells = "".join(f"<td>{escape(c)}</td>" for c in _metric_cells(m)[:3])
        rows.append(
            f'<tr>{cells}<td class="{css}">{escape(_change(m))}</td></tr>'
        )
    head = "".join(f"<th>{h}</th>" for h in _METRIC_HEADERS)
    body = "".join(rows)
    return (
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


def render_html_section(summary: WeeklySummary) -> str:
    """Sección «Resumen semanal» para el reporte HTML."""
    parts = [
        "<section><h2>Resumen semanal</h2>",
        f'<p class="note">Semana del {summary.week_start:%Y-%m-%d} al '
        f"{summary.week_end:%Y-%m-%d}.</p>",
        '<ul class="findings">'
        + "".join(f"<li>{escape(h)}</li>" for h in summary.highlights)
        + "</ul>",
    ]
    if summary.assets:
        parts += [
            "<h3>Modelos de volatilidad por activo</h3>",
            _html_table(
                ASSET_HEADERS, [_asset_cells(a) for a in summary.assets]
            ),
        ]
    if summary.live:
        parts += ["<h3>Operación en vivo</h3>", _html_metrics(summary.live)]
    if summary.project:
        parts += [
            "<h3>Avance del proyecto</h3>",
            _html_metrics(summary.project),
        ]
    alerts = summary.alerts or ("Ninguna.",)
    parts += [
        "<h3>Alertas</h3><ul>",
        "".join(f"<li>{escape(a)}</li>" for a in alerts),
        f'</ul><p class="note">{escape(_NOTE)}</p></section>',
    ]
    return "".join(parts)


def _draw_table(
    figure: Figure,
    rect: tuple[float, float, float, float],
    headers: list[str],
    rows: list[list[str]],
    widths: list[float] | None = None,
) -> None:
    ax = figure.add_axes(rect)
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="upper center",
        cellLoc="right",
        colLoc="right",
        colWidths=widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.visible_edges = "B"
        if col == 0:
            cell.set_text_props(ha="left")
        if row == 0:
            cell.set_text_props(weight="bold", color=INK_SECONDARY)


def draw_weekly_pages(
    summary: WeeklySummary, page: Callable[[str, int], Figure]
) -> list[Figure]:
    """Dos páginas: destacados + modelos, y operación + avance + alertas.

    ``page(title, offset)`` crea la página ``offset`` (0, 1) con el
    encabezado y pie del PDF.
    """
    first = page("Resumen semanal", 0)
    y = 0.9
    for text in summary.highlights:
        wrapped = textwrap.fill(f"• {text}", 150)
        first.text(
            0.04, y, wrapped, fontsize=8.5, color=INK_SECONDARY, va="top"
        )
        y -= 0.03 + 0.021 * wrapped.count("\n")
    if summary.assets:
        _draw_table(
            first,
            (0.04, 0.08, 0.92, y - 0.1),
            _PDF_ASSET_HEADERS,
            [_asset_cells(a) for a in summary.assets],
            [0.11, 0.11, 0.1, 0.06, 0.07, 0.16, 0.15, 0.09, 0.07, 0.07],
        )
        first.text(
            0.04,
            0.1,
            textwrap.fill(_NOTE, 170),
            fontsize=7.5,
            color=INK_SECONDARY,
            va="bottom",
        )
    second = page("Operación en vivo y avance del proyecto", 1)
    if summary.live:
        _draw_table(
            second,
            (0.04, 0.3, 0.55, 0.6),
            _PDF_METRIC_HEADERS,
            [_metric_cells(m) for m in summary.live],
            [0.5, 0.15, 0.17, 0.18],
        )
    if summary.project:
        _draw_table(
            second,
            (0.62, 0.55, 0.35, 0.35),
            _PDF_METRIC_HEADERS,
            [_metric_cells(m) for m in summary.project],
            [0.46, 0.15, 0.18, 0.21],
        )
    y = 0.44  # justo debajo de la tabla de operación (14 filas)
    second.text(0.04, y, "Alertas", fontsize=11, weight="bold", va="top")
    for alert in summary.alerts or ("Ninguna.",):
        y -= 0.03
        wrapped = textwrap.fill(f"• {alert}", 160)
        second.text(
            0.05, y, wrapped, fontsize=8, color=INK_SECONDARY, va="top"
        )
        y -= 0.018 * wrapped.count("\n")
    return [first, second]
