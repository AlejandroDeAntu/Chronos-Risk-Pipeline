"""Reporte HTML autocontenido: CSS embebido e imágenes en base64.

Abre sin internet y se imprime limpio (``@media print``): cada activo
empieza en página nueva y las tablas no se cortan entre páginas.
"""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path
from string import Template

from src.reporting.data import AssetSection, ReportData
from src.reporting.style import FAMILY_LABELS, format_p, format_price

_CSS = """
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9;
  --surface:#fcfcfb; --page:#f9f9f7; --accent:#2a78d6; --critical:#d03b3b;
  --good:#006300; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
  font:14px/1.55 "Segoe UI", system-ui, -apple-system, sans-serif; }
main { max-width:1080px; margin:0 auto; padding:40px 24px 64px; }
header { border-bottom:2px solid var(--ink); padding-bottom:18px;
  margin-bottom:28px; }
h1 { font-size:28px; margin:0 0 6px; letter-spacing:-0.01em; }
h2 { font-size:20px; margin:40px 0 4px; }
h3 { font-size:15px; margin:24px 0 8px; color:var(--ink2); }
.meta { color:var(--ink2); margin:0; }
section { background:var(--surface); border:1px solid rgba(11,11,11,.08);
  border-radius:8px; padding:20px 24px; margin:20px 0; }
ol.findings li { margin:6px 0; }
table { border-collapse:collapse; width:100%; font-size:12.5px;
  font-variant-numeric:tabular-nums; margin:8px 0; }
th, td { padding:6px 8px; border-bottom:1px solid var(--grid);
  text-align:right; }
th:first-child, td:first-child { text-align:left; }
th { color:var(--ink2); font-weight:600;
  border-bottom:1px solid var(--muted); }
tr.champion td { font-weight:600; }
.pos { color:var(--good); } .neg { color:var(--critical); }
figure { margin:18px 0; }
figure img { width:100%; height:auto; border:1px solid var(--grid);
  border-radius:4px; background:#fff; }
figcaption { color:var(--muted); font-size:12px; margin-top:4px; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
dl { display:grid; grid-template-columns:220px 1fr; gap:6px 16px; }
dt { color:var(--ink2); font-weight:600; } dd { margin:0; }
.note { color:var(--ink2); font-size:13px; }
.error { color:var(--critical); }
@media (max-width:760px) { .grid2, dl { grid-template-columns:1fr; } }
@media print {
  body { background:#fff; font-size:11px; } main { padding:0; }
  section { border:none; padding:0; }
  section.asset { break-before:page; }
  table, figure { break-inside:avoid; }
}
"""

_PAGE = Template(
    """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reporte de volatilidad y riesgo</title>
<style>$css</style>
</head>
<body><main>
<header>
<h1>Reporte de volatilidad y riesgo</h1>
<p class="meta">Generado $generated · Fuente: $source · $window</p>
</header>
<section>
<h2>Resumen ejecutivo</h2>
<ol class="findings">$findings</ol>
<h3>Indicadores clave por activo</h3>
$kpis
<p class="note">QLIKE: pérdida de pronóstico de varianza, menor es mejor.
DM: Diebold-Mariano del campeón contra EWMA (t &lt; 0 favorece al
campeón; p de una cola). VaR y CVaR: log-retorno del día siguiente con el
Monte Carlo del campeón. Kupiec y Christoffersen: p-valores del backtest
del VaR 99%.</p>
</section>
$assets
$anomalies
$live
<section>
<h2>Metodología y limitaciones</h2>
<dl>$methodology</dl>
<h3>Limitaciones</h3>
<ul>$limitations</ul>
</section>
$errors
</main></body></html>
"""
)

LIMITATIONS = (
    "Yahoo Finance es una fuente gratuita sin licencia comercial; las "
    "cotizaciones FX son indicativas, no precios ejecutables.",
    "En FX, Yahoo fecha el cierre diario al inicio del día; el sistema lo "
    "re-alinea al día que describe y por eso el último día FX disponible "
    "va un día detrás de los índices.",
    "Los estimadores de varianza por rango subestiman con muestreo discreto; "
    "la selección de modelos usa r² (insesgado) y el rango solo se usa "
    "re-escalado en entrenamiento.",
    "La historia intradía de Yahoo se limita a 60 días: las métricas del "
    "detector cubren un solo régimen de mercado.",
    "Las cifras son estimaciones fuera de muestra sobre datos históricos; "
    "no garantizan desempeño futuro ni constituyen una recomendación de "
    "inversión.",
)


def _img(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _figure(path: Path, caption: str) -> str:
    return (
        f'<figure><img src="{_img(path)}" alt="{escape(caption)}">'
        f"<figcaption>{escape(caption)}</figcaption></figure>"
    )


def _table(
    headers: list[str], rows: list[list[str]], classes: list[str] | None = None
) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = []
    for i, row in enumerate(rows):
        css = f' class="{classes[i]}"' if classes and classes[i] else ""
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        body.append(f"<tr{css}>{cells}</tr>")
    rows_html = "".join(body)
    return (
        f"<table><thead><tr>{head}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )


def _p(value: float) -> str:
    return escape(format_p(value))


def _kpi_table(assets: tuple[AssetSection, ...]) -> str:
    rows = []
    for a in assets:
        v = a.volatility
        if v is None:
            rows.append([escape(a.name), "n/d"] + [""] * 9)
            continue
        by_name = {r.name: r for r in v.rows}
        dm = by_name["ewma"].vs_champion
        rows.append(
            [
                escape(a.name),
                escape(FAMILY_LABELS[v.champion]),
                f"{v.n_common:,}",
                f"{by_name[v.champion].mean_qlike:.3f}",
                f"{by_name['ewma'].mean_qlike:.3f}",
                f"{dm.t_stat:.2f} ({_p(dm.p_value_one_sided)})",
                f"{v.forecast_vol_pct:.2f}%",
                f"{v.var_95:.2%} / {v.cvar_95:.2%}",
                f"{v.var_99:.2%} / {v.cvar_99:.2%}",
                _p(v.var[0.01].p_unconditional),
                _p(v.var[0.01].p_independence),
            ]
        )
    return _table(
        [
            "Activo",
            "Campeón",
            "n",
            "QLIKE campeón",
            "QLIKE EWMA",
            "DM t (p)",
            "Vol. 1d",
            "VaR / CVaR 95%",
            "VaR / CVaR 99%",
            "Kupiec p",
            "Christoffersen p",
        ],
        rows,
    )


def _asset_section(a: AssetSection, figures: dict[str, Path]) -> str:
    v = a.volatility
    if v is None:
        return ""
    period = f"{v.start:%Y-%m-%d} a {v.end:%Y-%m-%d}"
    rows, classes = [], []
    for r in sorted(v.rows, key=lambda r: r.mean_qlike):
        dm = r.vs_champion
        rows.append(
            [
                escape(FAMILY_LABELS[r.name]),
                f"{r.mean_qlike:.4f}",
                "campeón" if dm is None else f"{dm.t_stat:.2f}",
                "" if dm is None else _p(dm.p_value_one_sided),
            ]
        )
        classes.append("champion" if dm is None else "")
    comparison = _table(
        ["Modelo", "QLIKE medio", "DM t (campeón vs modelo)", "p (una cola)"],
        rows,
        classes,
    )
    source = "Fuente: Yahoo Finance"
    tracking_days = min(500, v.n_common)
    var_days = min(750, v.var[0.01].n_obs)
    captions = {
        "fan": (
            f"{a.name} · escenarios a {len(v.scenarios_mc)} días desde "
            f"{v.origin:%Y-%m-%d} con {FAMILY_LABELS[v.champion]} · {source}"
        ),
        "tracking": (
            f"{a.name} · últimos {tracking_days:,} de {v.n_common:,} días "
            f"fuera de muestra ({period}) · {source}"
        ),
        "qq": (
            f"{a.name} · {v.calibration.n_samples:,} residuos del campeón "
            f"({period}) · {source}"
        ),
        "qlike": (
            f"{a.name} · {v.n_common:,} días comunes ({period}) · {source}"
        ),
        "var": (
            f"{a.name} · gráfica: últimos {var_days:,} días; estadísticas "
            f"sobre {v.var[0.01].n_obs:,} días ({period}) · {source}"
        ),
    }
    forecast = (
        f"Origen {v.origin:%Y-%m-%d}, último cierre "
        f"{format_price(v.last_close)}; "
        f"volatilidad pronosticada a 1 día {v.forecast_vol_pct:.2f}%."
    )
    return (
        f'<section class="asset"><h2>{escape(a.name)}</h2>'
        f'<p class="note">{escape(v.narrative)} {escape(forecast)}</p>'
        f"{_figure(figures['fan'], captions['fan'])}"
        f"<h3>Comparación de modelos (muestra común)</h3>{comparison}"
        '<p class="note">DM t &lt; 0: el campeón tuvo menor pérdida que el '
        "modelo de la fila. Un retador con menor QLIKE medio solo reemplaza "
        "al campeón si la diferencia es significativa tras el ajuste de "
        "Holm.</p>"
        f"{_figure(figures['tracking'], captions['tracking'])}"
        f'<div class="grid2">{_figure(figures["qq"], captions["qq"])}'
        f"{_figure(figures['qlike'], captions['qlike'])}</div>"
        f"{_figure(figures['var'], captions['var'])}</section>"
    )


def _anomaly_section(data: ReportData, figures: dict[str, Path]) -> str:
    rows = []
    for a in data.assets:
        s = a.anomaly
        if s is None:
            continue
        low, high = s.lift_ci
        rows.append(
            [
                escape(a.name),
                f"{s.n_bars:,}",
                f"{s.lift.n_predicted_positive:,}",
                f"{s.lift.precision:.3f}",
                f"{s.lift.prevalence:.3f}",
                f"{s.lift.lift:.3f} [{low:.2f}, {high:.2f}]",
                _p(s.lift.p_value),
                _p(s.p_bonferroni),
                f"{s.auc:.3f}",
                ""
                if a.zero_return_share is None
                else f"{a.zero_return_share:.1%}",
            ]
        )
    if not rows:
        return ""
    table = _table(
        [
            "Activo",
            "Velas",
            "Anomalías",
            "Precisión",
            "Prevalencia",
            "Lift [IC 95%]",
            "p",
            "p Bonferroni",
            "AUC",
            "Retornos en cero",
        ],
        rows,
    )
    caption = (
        "Velas de 5 min fuera de muestra (60 días) · Fuente: Yahoo Finance"
    )
    return (
        "<section><h2>Detector de anomalías intradía</h2>"
        '<p class="note">Positivo = el precio revierte en las 10 velas '
        "siguientes. Lift = precisión / prevalencia: 1 equivale a marcar "
        "velas al azar.</p>"
        f"{table}{_figure(figures['lift'], caption)}"
        f"{_figure(figures['profiles'], caption)}</section>"
    )


def _live_section(data: ReportData) -> str:
    if not data.live_runs:
        return ""
    rows = [
        [
            escape(r["asset_symbol"]),
            escape(r["signal_type"]),
            escape(str(r["evaluated_at"])[:16]),
            f"{r['n_outcomes']:,}",
            escape(r["metric_name"]),
            "" if r["model_metric"] is None else f"{r['model_metric']:.4f}",
            ""
            if r["baseline_metric"] is None
            else f"{r['baseline_metric']:.4f}",
            "sí" if r["sufficient_data"] else "no",
        ]
        for r in data.live_runs
    ]
    return (
        "<section><h2>Monitoreo en producción</h2>"
        '<p class="note">Última evaluación registrada por activo y señal en '
        "la base auditable (predicciones comparadas contra el resultado "
        "real).</p>"
        + _table(
            [
                "Activo",
                "Señal",
                "Evaluado (UTC)",
                "Resultados",
                "Métrica",
                "Modelo",
                "Baseline",
                "Muestra suficiente",
            ],
            rows,
        )
        + "</section>"
    )


def render_html(
    data: ReportData,
    asset_figures: dict[str, dict[str, Path]],
    summary_figures: dict[str, Path],
    path: Path,
) -> Path:
    """Escribe el reporte HTML autocontenido.

    Parameters
    ----------
    data:
        Contenido del reporte.
    asset_figures:
        PNG por símbolo y tipo (``fan``, ``tracking``, ``qq``, ``qlike``,
        ``var``).
    summary_figures:
        PNG de resumen (``lift``, ``profiles``).
    path:
        Ruta del HTML a escribir.

    Returns
    -------
    pathlib.Path
        La ruta escrita.
    """
    vol = [a.volatility for a in data.assets if a.volatility]
    window = (
        f"Validación {min(v.start for v in vol):%Y-%m-%d} a "
        f"{max(v.end for v in vol):%Y-%m-%d}"
        if vol
        else "sin validación de volatilidad"
    )
    errors = [f"{a.name}: {e}" for a in data.assets for e in a.errors]
    page = _PAGE.substitute(
        css=_CSS,
        generated=escape(f"{data.generated_at:%Y-%m-%d %H:%M} UTC"),
        source="Yahoo Finance vía yfinance",
        window=escape(window),
        findings="".join(f"<li>{escape(f)}</li>" for f in data.findings),
        kpis=_kpi_table(data.assets),
        assets="".join(
            _asset_section(a, asset_figures[a.symbol])
            for a in data.assets
            if a.symbol in asset_figures
        ),
        anomalies=_anomaly_section(data, summary_figures),
        live=_live_section(data),
        methodology="".join(
            f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>"
            for k, v in data.methodology.items()
        ),
        limitations="".join(f"<li>{escape(t)}</li>" for t in LIMITATIONS),
        errors=(
            '<section><h2 class="error">Errores</h2><ul>'
            + "".join(f"<li>{escape(e)}</li>" for e in errors)
            + "</ul></section>"
            if errors
            else ""
        ),
    )
    path.write_text(page, encoding="utf-8")
    return path
