"""Identidad visual del reporte: paleta validada, tipografía y ejes.

La paleta categórica (azul, naranja, aqua) pasa el validador de daltonismo
en pares adyacentes sobre la superficie clara (ΔE CVD >= 9.2). El aqua
queda bajo 3:1 de contraste, así que solo se usa con etiqueta directa. El
rojo es de estado (excepción de VaR) y nunca identifica una serie.
"""

from __future__ import annotations

from matplotlib.axes import Axes

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

CHAMPION = "#2a78d6"  # slot 1: el modelo campeón
EWMA = "#eb6834"  # slot 2: baseline de referencia
ACCENT = "#1baf7a"  # slot 3: solo con etiqueta directa
CRITICAL = "#d03b3b"  # estado: excepción de VaR
BAND_DARK = "#86b6ef"  # rampa azul, paso 250
BAND_LIGHT = "#cde2fb"  # rampa azul, paso 100

FONT_FAMILY = ["Segoe UI", "DejaVu Sans", "sans-serif"]
TITLE_SIZE = 10.5
LABEL_SIZE = 8.5
TICK_SIZE = 8.0

FAMILY_LABELS = {
    "garch": "GARCH(1,1)-t",
    "gjr": "GJR-GARCH-t",
    "egarch": "EGARCH-t",
    "har": "HAR-RV",
    "sd20": "SD 20 días",
    "ewma": "EWMA 0.94",
}


def style_axes(
    ax: Axes, title: str, xlabel: str = "", ylabel: str = ""
) -> None:
    """Aplica el estilo común: rejilla de trazo fino, ejes discretos."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=TITLE_SIZE, color=INK, pad=8)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, color=INK_SECONDARY)
    ax.tick_params(colors=INK_MUTED, labelsize=TICK_SIZE, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(0.8)


def style_legend(ax: Axes) -> None:
    """Leyenda sin marco, en tinta secundaria."""
    legend = ax.legend(
        frameon=False, fontsize=LABEL_SIZE, labelcolor=INK_SECONDARY
    )
    if legend is not None:
        legend.set_zorder(5)


def format_price(value: float) -> str:
    """Precio legible según su magnitud (sin notación científica)."""
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:,.2f}"
    return f"{value:.4f}"


def format_p(value: float) -> str:
    """P-valor con piso de presentación en 0.001."""
    return "< 0.001" if value < 0.001 else f"{value:.3f}"
