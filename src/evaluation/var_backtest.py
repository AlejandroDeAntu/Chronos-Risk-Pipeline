"""Backtest de Value-at-Risk: cobertura, independencia y conjunta.

* Kupiec (1995), cobertura incondicional: ¿la tasa de excepciones coincide
  con el nivel nominal? ``LR_uc ~ chi2(1)``.
* Christoffersen (1998), independencia: ¿una excepción hace más probable
  otra al día siguiente? (cadena de Markov de primer orden,
  ``LR_ind ~ chi2(1)``). Un VaR con cobertura correcta pero excepciones
  agrupadas subestima el riesgo justo cuando importa.
* Cobertura condicional: ``LR_cc = LR_uc + LR_ind ~ chi2(2)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class VarBacktestResult:
    """Excepciones del VaR y pruebas de verosimilitud."""

    level: float
    n_obs: int
    n_hits: int
    hit_rate: float
    lr_unconditional: float
    p_unconditional: float
    lr_independence: float
    p_independence: float
    lr_conditional: float
    p_conditional: float


def _bernoulli_loglik(successes: int, trials: int, prob: float) -> float:
    """Log-verosimilitud de Bernoulli con la convención ``0 ln 0 = 0``."""
    failures = trials - successes
    loglik = 0.0
    if successes:
        loglik += successes * np.log(prob)
    if failures:
        loglik += failures * np.log1p(-prob)
    return float(loglik)


def _independence_lr(hits: np.ndarray) -> float:
    """Estadístico de Christoffersen sobre la cadena de excepciones."""
    previous, current = hits[:-1], hits[1:]
    n00 = int(np.sum(~previous & ~current))
    n01 = int(np.sum(~previous & current))
    n10 = int(np.sum(previous & ~current))
    n11 = int(np.sum(previous & current))
    pi_all = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    pi_0 = n01 / max(n00 + n01, 1)
    pi_1 = n11 / max(n10 + n11, 1)
    restricted = _bernoulli_loglik(n01 + n11, n00 + n01 + n10 + n11, pi_all)
    unrestricted = _bernoulli_loglik(n01, n00 + n01, pi_0)
    unrestricted += _bernoulli_loglik(n11, n10 + n11, pi_1)
    return max(-2.0 * (restricted - unrestricted), 0.0)


def var_backtest(
    returns: np.ndarray, var: np.ndarray, level: float
) -> VarBacktestResult:
    """Cuenta excepciones (``r_t < VaR_t``) y aplica las tres pruebas.

    Parameters
    ----------
    returns:
        Retornos realizados.
    var:
        VaR pronosticado para cada día, como cuantil de retorno (negativo
        para una pérdida), en las mismas unidades que ``returns``.
    level:
        Nivel nominal de cola (p. ej. 0.01 para VaR al 99%).

    Returns
    -------
    VarBacktestResult
        Excepciones, tasa y estadísticos con sus p-valores.

    Raises
    ------
    ValueError
        Si las entradas difieren en longitud o están vacías.
    """
    returns = np.asarray(returns, dtype=float)
    var = np.asarray(var, dtype=float)
    if len(returns) != len(var):
        raise ValueError("returns y var deben tener la misma longitud.")
    if len(returns) == 0:
        raise ValueError("No se puede evaluar un VaR sin observaciones.")

    hits = returns < var
    n_obs, n_hits = len(hits), int(hits.sum())
    hit_rate = n_hits / n_obs
    lr_uc = -2.0 * (
        _bernoulli_loglik(n_hits, n_obs, level)
        - _bernoulli_loglik(n_hits, n_obs, hit_rate)
    )
    lr_uc = max(lr_uc, 0.0)
    lr_ind = _independence_lr(hits)
    lr_cc = lr_uc + lr_ind
    return VarBacktestResult(
        level=level,
        n_obs=n_obs,
        n_hits=n_hits,
        hit_rate=hit_rate,
        lr_unconditional=lr_uc,
        p_unconditional=float(stats.chi2.sf(lr_uc, 1)),
        lr_independence=lr_ind,
        p_independence=float(stats.chi2.sf(lr_ind, 1)),
        lr_conditional=lr_cc,
        p_conditional=float(stats.chi2.sf(lr_cc, 2)),
    )
