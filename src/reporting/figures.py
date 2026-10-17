"""Gráficas del reporte (matplotlib sin pyplot: headless y sin estado global).

Cada función ``draw_*`` dibuja sobre un ``Axes`` recibido, así la misma
gráfica sirve para el PNG individual y para la página compuesta del PDF.
Los títulos enuncian el hallazgo; las unidades van en los ejes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats

from src.reporting.data import AssetSection, VolatilitySection
from src.reporting.style import (
    BAND_DARK,
    BAND_LIGHT,
    BASELINE,
    CHAMPION,
    CRITICAL,
    EWMA,
    FAMILY_LABELS,
    INK_MUTED,
    INK_SECONDARY,
    LABEL_SIZE,
    format_p,
    format_price,
    style_axes,
    style_legend,
)

PNG_DPI = 200
_TRACKING_DAYS = 500
_VAR_DAYS = 750
_QQ_MAX_POINTS = 2500
_STALE_SHARE = 0.20


def _date_axis(ax: Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _end_label(ax: Axes, x: object, y: float, text: str, color: str) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        fontsize=LABEL_SIZE,
        color=color,
    )


def draw_fan_chart(
    ax: Axes,
    section: VolatilitySection,
    name: str,
    compact: bool = False,
) -> None:
    """Precio reciente y abanico de escenarios a ``h`` días.

    La leyenda va debajo del área de datos para no tapar la serie;
    ``compact`` reduce su tipografía en páginas compuestas del PDF.
    """
    history = section.close_tail
    days = [q.day for q in section.scenarios_mc]
    future = pd.bdate_range(section.origin, periods=len(days) + 1)[1:]
    x = [history.index[-1], *future]
    last = section.last_close

    def band(rows: Sequence, attr: str) -> list[float]:
        return [last, *(getattr(r, attr) for r in rows)]

    mc, fhs = section.scenarios_mc, section.scenarios_fhs
    ax.fill_between(
        x,
        band(mc, "p05"),
        band(mc, "p95"),
        color=BAND_LIGHT,
        lw=0,
        label="Rango 90% (Monte Carlo)",
    )
    ax.fill_between(
        x,
        band(mc, "p25"),
        band(mc, "p75"),
        color=BAND_DARK,
        lw=0,
        label="Rango 50% (Monte Carlo)",
    )
    ax.plot(
        x,
        band(fhs, "p05"),
        color=INK_MUTED,
        lw=0.8,
        label="Rango 90% (simulación histórica)",
    )
    ax.plot(x, band(fhs, "p95"), color=INK_MUTED, lw=0.8)
    ax.plot(
        history.index,
        history.to_numpy(),
        color=INK_SECONDARY,
        lw=1.2,
        label="Cierre observado",
    )
    ax.plot(x, band(mc, "p50"), color=CHAMPION, lw=1.8, label="Mediana")

    low, high = mc[-1].p05, mc[-1].p95
    style_axes(
        ax,
        f"{name}: rango 90% a {days[-1]} días de {format_price(low)} a "
        f"{format_price(high)} "
        f"({low / last - 1:+.1%} / {high / last - 1:+.1%})",
        ylabel="Precio",
    )
    _date_axis(ax)
    # El año ya va en el pie de figura; la etiqueta de desfase chocaba
    # con la leyenda inferior.
    ax.xaxis.get_offset_text().set_visible(False)
    ax.legend(
        frameon=False,
        fontsize=LABEL_SIZE - 1.5 if compact else LABEL_SIZE,
        labelcolor=INK_SECONDARY,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.14),
    )


def draw_volatility_tracking(ax: Axes, section: VolatilitySection) -> None:
    """Volatilidad pronosticada (campeón y EWMA) contra la realizada."""
    data = section.series[section.series["common"]].iloc[-_TRACKING_DAYS:]
    ax.plot(
        data.index,
        data["realized_vol"],
        color=BASELINE,
        lw=0.8,
        label="Realizada (rango OHLC)",
    )
    ax.plot(
        data.index,
        data["ewma"],
        color=EWMA,
        lw=1.1,
        label=FAMILY_LABELS["ewma"],
    )
    ax.plot(
        data.index,
        data["champion"],
        color=CHAMPION,
        lw=1.5,
        label=f"{FAMILY_LABELS[section.champion]} (campeón)",
    )
    rows = {r.name: r for r in section.rows}
    style_axes(
        ax,
        "Pronóstico vs realizada: QLIKE "
        f"{rows[section.champion].mean_qlike:.3f} campeón vs "
        f"{rows['ewma'].mean_qlike:.3f} EWMA",
        ylabel="Volatilidad diaria (%)",
    )
    _date_axis(ax)
    style_legend(ax)


def draw_qq(ax: Axes, section: VolatilitySection) -> None:
    """Residuos estandarizados contra la t de Student ajustada."""
    data = section.series
    own = np.isfinite(data["champion"]) & (data["champion"] > 0)
    z = np.sort((data["realized"][own] / data["champion"][own]).to_numpy())
    nu = max(section.residual_nu, 2.05)
    probs = (np.arange(1, len(z) + 1) - 0.5) / len(z)
    theory = stats.t.ppf(probs, nu) * np.sqrt((nu - 2.0) / nu)
    if len(z) > _QQ_MAX_POINTS:
        # Submuestreo uniforme en el centro; las colas se conservan.
        keep = np.unique(
            np.r_[
                np.arange(50),
                np.linspace(50, len(z) - 51, _QQ_MAX_POINTS).astype(int),
                np.arange(len(z) - 50, len(z)),
            ]
        )
        z, theory = z[keep], theory[keep]
    limit = float(np.nanmax(np.abs(np.r_[z, theory])))
    ax.plot([-limit, limit], [-limit, limit], color=BASELINE, lw=1.0)
    ax.scatter(theory, z, s=8, color=CHAMPION, linewidths=0, alpha=0.8)
    style_axes(
        ax,
        f"Residuos vs t({nu:.1f}): E[z²] = "
        f"{section.calibration.mean_squared_standardized_residual:.3f}",
        xlabel="Cuantil teórico t estandarizada",
        ylabel="Residuo estandarizado",
    )


def draw_cumulative_qlike(ax: Axes, section: VolatilitySection) -> None:
    """Ventaja acumulada del campeón frente a cada baseline."""
    series = section.series
    data = series[series["common"]]
    for name, color in (("ewma", EWMA), ("sd20", INK_MUTED)):
        gap = (data["qlike_champion"] - data[f"qlike_{name}"]).cumsum()
        # Reindexar deja NaN fuera de la muestra común: la línea se corta
        # en vez de dibujar un tramo plano que no existe.
        ax.plot(
            series.index,
            gap.reindex(series.index),
            color=color,
            lw=1.4,
            label=f"vs {FAMILY_LABELS[name]}",
        )
        _end_label(
            ax, data.index[-1], gap.iloc[-1], f"{gap.iloc[-1]:,.0f}", color
        )
    ax.axhline(0.0, color=BASELINE, lw=1.0)
    style_axes(
        ax,
        "QLIKE acumulada: campeón − baseline (< 0 = mejor)",
        ylabel="QLIKE acumulada",
    )
    _date_axis(ax)
    style_legend(ax)


def draw_var_exceptions(ax: Axes, section: VolatilitySection) -> None:
    """Retornos, VaR 99% filtrado y excepciones de los últimos días."""
    data = section.series.dropna(subset=["var_01"]).iloc[-_VAR_DAYS:]
    hits = data["realized"] < data["var_01"]
    result = section.var[0.01]
    ax.plot(
        data.index,
        data["realized"],
        color=BASELINE,
        lw=0.7,
        label="Retorno diario",
    )
    ax.plot(
        data.index,
        data["var_01"],
        color=CHAMPION,
        lw=1.3,
        label="VaR 99% filtrado",
    )
    ax.scatter(
        data.index[hits],
        data["realized"][hits],
        s=16,
        color=CRITICAL,
        zorder=4,
        label="Excepción (pérdida > VaR)",
    )
    style_axes(
        ax,
        f"VaR 99%: {result.n_hits} excepciones en {result.n_obs:,} días "
        f"({result.hit_rate:.2%}; nominal 1%, Kupiec p = "
        f"{result.p_unconditional:.2f})",
        xlabel=f"Últimos {len(data):,} días de la muestra",
        ylabel="Retorno (%)",
    )
    _date_axis(ax)
    ax.legend(
        frameon=False,
        fontsize=LABEL_SIZE,
        labelcolor=INK_SECONDARY,
        ncol=3,
        loc="upper left",
    )


def draw_anomaly_lift(ax: Axes, assets: Sequence[AssetSection]) -> None:
    """Lift de precisión con IC 95% por activo (1 = azar)."""
    rows = [a for a in assets if a.anomaly]
    for i, asset in enumerate(rows):
        section = asset.anomaly
        color = CHAMPION if section.p_bonferroni < 0.05 else INK_MUTED
        low, high = section.lift_ci
        ax.plot(
            [low, high], [i, i], color=color, lw=2.0, solid_capstyle="round"
        )
        ax.scatter([section.lift.lift], [i], s=40, color=color, zorder=3)
        ax.annotate(
            f"{section.lift.lift:.2f} "
            f"(p Bonf. {_p_text(section.p_bonferroni)})",
            (high, i),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=LABEL_SIZE,
            color=INK_SECONDARY,
        )
    ax.axvline(1.0, color=BASELINE, lw=1.0)
    # Margen derecho para la etiqueta del intervalo más alto.
    lows = [a.anomaly.lift_ci[0] for a in rows]
    highs = [a.anomaly.lift_ci[1] for a in rows]
    span = max(highs) - min(lows)
    ax.set_xlim(min(lows) - 0.03 * span, max(highs) + 0.45 * span)
    ax.set_yticks(range(len(rows)), [a.name for a in rows])
    ax.invert_yaxis()
    significant = sum(a.anomaly.p_bonferroni < 0.05 for a in rows)
    style_axes(
        ax,
        f"Precisión / prevalencia con IC 95%: {significant} de {len(rows)} "
        "activos superan al azar tras Bonferroni",
        xlabel="Lift de precisión (1 = azar)",
    )
    ax.grid(axis="y", visible=False)


def draw_hourly_profiles(
    axes: Sequence[Axes], assets: Sequence[AssetSection]
) -> None:
    """Perfil de volatilidad por hora UTC (small multiples, misma escala).

    Solo se dibujan las horas con velas en la muestra: un índice no cotiza
    de madrugada y mostrar ahí un factor 1.0 sugeriría actividad.
    """
    rows = [a for a in assets if a.anomaly]
    for ax, asset in zip(axes, rows, strict=False):
        section = asset.anomaly
        hours = list(section.active_hours)
        profile = np.asarray(section.profile)[hours]
        ax.bar(hours, profile, width=0.8, color=CHAMPION)
        ax.axhline(1.0, color=INK_MUTED, lw=0.8)
        peak = hours[int(np.argmax(profile))]
        note = f"pico {peak:02d}h UTC ({profile.max():.1f}×)"
        stale = asset.zero_return_share
        if stale is not None and stale > _STALE_SHARE:
            note = f"plano: {stale:.0%} retornos = 0"
        style_axes(ax, f"{asset.name}\n{note}")
        ax.title.set_fontsize(LABEL_SIZE + 0.5)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlim(-0.7, 23.7)
    for ax in axes[len(rows) :]:
        ax.set_visible(False)


def _p_text(value: float) -> str:
    text = format_p(value)
    return text if text.startswith("<") else f"= {text}"


def save_png(
    draw: Callable[[Axes], None], path: Path, size: tuple[float, float]
) -> Path:
    """Dibuja una gráfica individual y la guarda a ``PNG_DPI``."""
    figure = Figure(figsize=size, layout="constrained")
    draw(figure.subplots())
    figure.savefig(path, dpi=PNG_DPI, facecolor="white")
    return path
