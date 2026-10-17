"""Resumen de escenarios simulados: cuantiles de precio y cola de pérdida.

Los escenarios vienen de ``VolatilityModel.simulate_returns`` (retornos log
en %, forma ``(n_paths, horizon)``), sea por Monte Carlo paramétrico o por
simulación histórica filtrada. Aquí solo se resumen; no se simula nada.
"""

from __future__ import annotations

import numpy as np

from src.schemas import PriceQuantiles

PERCENTILES: tuple[int, ...] = (5, 25, 50, 75, 95)


def price_quantiles(
    last_close: float, sim_returns_pct: np.ndarray
) -> list[PriceQuantiles]:
    """Cuantiles del precio al cierre de cada día del horizonte.

    Parameters
    ----------
    last_close:
        Último cierre observado (origen del pronóstico).
    sim_returns_pct:
        Retornos log simulados en %, ``(n_paths, horizon)``.

    Returns
    -------
    list[PriceQuantiles]
        Una fila por día con los percentiles 5, 25, 50, 75 y 95.
    """
    cumulative = np.cumsum(sim_returns_pct / 100.0, axis=1)
    prices = last_close * np.exp(cumulative)
    levels = np.percentile(prices, PERCENTILES, axis=0)
    return [
        PriceQuantiles(
            day=day + 1,
            **{
                f"p{p:02d}": float(levels[i, day])
                for i, p in enumerate(PERCENTILES)
            },
        )
        for day in range(prices.shape[1])
    ]


def tail_risk(
    sim_returns_pct: np.ndarray, level: float
) -> tuple[float, float]:
    """VaR y CVaR del primer día como log-retorno (proporción, negativo).

    El CVaR (expected shortfall) es la pérdida media condicionada a superar
    el VaR; a diferencia del VaR, es una medida de riesgo coherente.

    Parameters
    ----------
    sim_returns_pct:
        Retornos log simulados en %, ``(n_paths, horizon)``.
    level:
        Probabilidad de cola, p. ej. 0.05.

    Returns
    -------
    tuple[float, float]
        ``(VaR, CVaR)`` del día 1.
    """
    day_one = sim_returns_pct[:, 0] / 100.0
    var = float(np.quantile(day_one, level))
    return var, float(day_one[day_one <= var].mean())
