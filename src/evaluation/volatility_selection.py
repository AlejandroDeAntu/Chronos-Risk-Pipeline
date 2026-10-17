"""Walk-forward de todas las familias de volatilidad y champion-challenger.

Todas las familias se validan sobre exactamente los mismos días de prueba
(``TimeSeriesSplit``, ventana expansiva): los parámetros se estiman solo con
el bloque de entrenamiento y el pronóstico del día ``t`` usa información
hasta ``t-1``. Ninguna función de este módulo escribe en la base, así que
lo comparten el reentrenamiento en producción, el backtest y los reportes.

Pérdida de selección: QLIKE contra ``r^2``. ``r^2`` es ruidoso pero
insesgado, y Patton (2011) muestra que con un proxy insesgado QLIKE ordena
los modelos como si se conociera la varianza verdadera. La varianza por
rango es menos ruidosa pero subestima (muestreo discreto), así que se
re-escala con un factor estimado solo en entrenamiento y se usa como
métrica complementaria y para graficar, no para decidir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.config import PipelineSettings
from src.evaluation.metrics import (
    DieboldMarianoResult,
    diebold_mariano,
    qlike_loss,
)
from src.exceptions import InsufficientHistoryError, TradingPipelineError
from src.models.baseline import (
    ewma_volatility_baseline,
    trailing_volatility_baseline,
)
from src.models.volatility import fit_volatility_model

logger = logging.getLogger(__name__)

BASELINES: tuple[str, ...] = ("sd20", "ewma")
VAR_LEVELS: tuple[float, ...] = (0.01, 0.05)


@dataclass(frozen=True)
class VolatilityWalkForward:
    """Pronósticos fuera de muestra alineados de modelos y baselines.

    Attributes
    ----------
    dates:
        Días de prueba, en orden cronológico.
    realized_pct:
        Retorno realizado de cada día (%).
    rv_proxy_pct2:
        Varianza realizada por rango re-escalada a ``r^2`` (%^2).
    baselines:
        Volatilidad pronosticada (%) por ``sd20`` y ``ewma``.
    forecasts:
        Volatilidad pronosticada (%) por familia; ``NaN`` en bloques donde
        el ajuste falló.
    var_z:
        Por familia, cuantiles ``VAR_LEVELS`` de los residuos estandarizados
        del bloque de entrenamiento que produjo cada pronóstico, forma
        ``(n, len(VAR_LEVELS))``. VaR filtrado: ``VaR_t = z_q sigma_t``.
    n_folds:
        Bloques ajustados con éxito por familia.
    failed:
        Familias que no produjeron ningún pronóstico, con la causa.
    """

    dates: tuple[pd.Timestamp, ...]
    realized_pct: np.ndarray
    rv_proxy_pct2: np.ndarray
    baselines: dict[str, np.ndarray]
    forecasts: dict[str, np.ndarray]
    var_z: dict[str, np.ndarray]
    n_folds: dict[str, int]
    failed: dict[str, str]

    def volatility(self, name: str) -> np.ndarray:
        """Pronóstico de una familia o de un baseline (%)."""
        if name in self.forecasts:
            return self.forecasts[name]
        return self.baselines[name]

    def qlike(self, name: str) -> np.ndarray:
        """QLIKE punto a punto contra ``r^2`` (``NaN`` sin pronóstico)."""
        sigma = self.volatility(name)
        loss = np.full(len(sigma), np.nan)
        valid = np.isfinite(sigma) & (sigma > 0.0)
        loss[valid] = qlike_loss(sigma[valid], self.realized_pct[valid])
        return loss

    def value_at_risk(self, family: str, level: float) -> np.ndarray:
        """VaR filtrado del día (cuantil de retorno en %, negativo).

        Usa el cuantil empírico de los residuos estandarizados de
        entrenamiento en lugar de suponer normalidad o una t.

        Raises
        ------
        ValueError
            Si ``level`` no está en ``VAR_LEVELS``.
        """
        if level not in VAR_LEVELS:
            raise ValueError(f"level debe ser uno de {VAR_LEVELS}.")
        column = VAR_LEVELS.index(level)
        return self.var_z[family][:, column] * self.forecasts[family]

    @property
    def common_mask(self) -> np.ndarray:
        """Días en que todos los modelos y baselines tienen pronóstico.

        Si una familia falla en un bloque, ese bloque sale de todas las
        comparaciones: promediar pérdidas sobre muestras distintas haría
        incomparables las medias (y contradictorias con Diebold-Mariano).
        """
        series = [*self.forecasts.values(), *self.baselines.values()]
        return np.logical_and.reduce(
            [np.isfinite(s) & (s > 0.0) for s in series]
        )

    def mean_qlike(self, name: str) -> float:
        """QLIKE medio sobre la muestra común."""
        return float(np.mean(self.qlike(name)[self.common_mask]))


@dataclass(frozen=True)
class ChallengerComparison:
    """Diebold-Mariano de un retador contra el campeón vigente."""

    challenger: str
    dm: DieboldMarianoResult
    p_value_holm: float


@dataclass(frozen=True)
class ChampionDecision:
    """Resultado de la selección champion-challenger."""

    champion: str
    previous: str
    switched: bool
    reason: str
    comparisons: tuple[ChallengerComparison, ...]


def walk_forward_volatility(
    frame: pd.DataFrame,
    settings: PipelineSettings,
    families: tuple[str, ...],
) -> VolatilityWalkForward:
    """Valida cada familia en los mismos bloques de prueba.

    Parameters
    ----------
    frame:
        Salida de ``volatility_frame`` (retornos y varianzas realizadas).
    settings:
        Configuración (bloques, historia mínima y ventana del baseline).
    families:
        Familias a validar.

    Returns
    -------
    VolatilityWalkForward
        Pronósticos alineados; una familia que falla en todos los bloques
        queda en ``failed`` con su causa, nunca se omite en silencio.

    Raises
    ------
    InsufficientHistoryError
        Si no hay ningún día de prueba con baselines definidos.
    """
    returns = frame["return_pct"]
    baselines_full = {
        "sd20": trailing_volatility_baseline(
            returns, settings.garch.baseline_trailing_window
        ).to_numpy(),
        "ewma": ewma_volatility_baseline(returns).to_numpy(),
    }
    splitter = TimeSeriesSplit(n_splits=settings.retrain.walk_forward_splits)
    folds = list(splitter.split(frame))
    test_rows = np.concatenate([test for _, test in folds])
    position = {row: i for i, row in enumerate(test_rows)}

    forecasts = {
        family: np.full(len(test_rows), np.nan) for family in families
    }
    var_z = {
        family: np.full((len(test_rows), len(VAR_LEVELS)), np.nan)
        for family in families
    }
    n_folds = dict.fromkeys(families, 0)
    errors: dict[str, str] = {}
    rv_proxy = np.full(len(test_rows), np.nan)

    for fold, (train, test) in enumerate(folds):
        slots = [position[row] for row in test]
        rv_proxy[slots] = _scaled_rv(frame, train, test)
        history = frame.iloc[: test[-1] + 1]
        for family in families:
            try:
                model = fit_volatility_model(
                    frame.iloc[train],
                    family,
                    settings.garch.min_history_days,
                    version=f"walk_forward_{family}_{fold}",
                )
            except TradingPipelineError as exc:
                errors[family] = f"{type(exc).__name__}: {exc}"
                logger.info("Bloque %d de %s omitido: %s", fold, family, exc)
                continue
            variance = model.variance_path(history).to_numpy()[test]
            forecasts[family][slots] = np.sqrt(variance)
            var_z[family][slots] = _training_z_quantiles(
                model.variance_path(frame.iloc[train]),
                frame["return_pct"].iloc[train],
            )
            n_folds[family] += 1

    baselines = {
        name: values[test_rows] for name, values in baselines_full.items()
    }
    keep = np.isfinite(baselines["sd20"]) & np.isfinite(baselines["ewma"])
    if not keep.any():
        raise InsufficientHistoryError(
            "Ningún día de prueba tiene baselines de volatilidad definidos."
        )
    failed = {f: errors[f] for f in families if n_folds[f] == 0}
    return VolatilityWalkForward(
        dates=tuple(frame.index[test_rows][keep]),
        realized_pct=returns.to_numpy()[test_rows][keep],
        rv_proxy_pct2=rv_proxy[keep],
        baselines={k: v[keep] for k, v in baselines.items()},
        forecasts={
            f: v[keep] for f, v in forecasts.items() if f not in failed
        },
        var_z={f: v[keep] for f, v in var_z.items() if f not in failed},
        n_folds={f: n for f, n in n_folds.items() if f not in failed},
        failed=failed,
    )


def _training_z_quantiles(
    variance: pd.Series, returns_pct: pd.Series
) -> np.ndarray:
    """Cuantiles ``VAR_LEVELS`` de ``r / sigma`` en entrenamiento."""
    standardized = (returns_pct / np.sqrt(variance)).to_numpy()
    return np.quantile(standardized[np.isfinite(standardized)], VAR_LEVELS)


def _scaled_rv(
    frame: pd.DataFrame, train: np.ndarray, test: np.ndarray
) -> np.ndarray:
    """Varianza por rango del bloque de prueba, escalada en entrenamiento."""
    train_rows = frame.iloc[train]
    scale = (train_rows["return_pct"] ** 2).mean() / train_rows[
        "rv_day"
    ].mean()
    return (scale * frame["rv_day"].iloc[test]).to_numpy()


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """P-valores ajustados por Holm-Bonferroni (step-down).

    Controla la tasa de error familiar con menos pérdida de potencia que
    Bonferroni.
    """
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    m = len(p_values)
    stepped = np.minimum(1.0, (m - np.arange(m)) * p_values[order])
    adjusted = np.empty(m)
    adjusted[order] = np.maximum.accumulate(stepped)
    return adjusted


def select_champion(
    walk_forward: VolatilityWalkForward,
    current: str,
    alpha: float,
) -> ChampionDecision:
    """Elige el campeón: el vigente se queda salvo derrota significativa.

    Cada retador se compara contra el campeón con Diebold-Mariano (QLIKE,
    una cola, sobre la muestra común a todos los modelos). Los p-valores se
    ajustan por Holm. Entre los retadores significativos gana el de menor
    QLIKE medio. Si el campeón vigente no produjo pronósticos (falló el
    ajuste), se elige la familia disponible de menor QLIKE.

    Raises
    ------
    InsufficientHistoryError
        Si ninguna familia produjo pronósticos.
    """
    available = list(walk_forward.forecasts)
    if not available:
        raise InsufficientHistoryError(
            f"Ninguna familia produjo pronósticos: {walk_forward.failed}."
        )
    if current not in available:
        best = min(available, key=walk_forward.mean_qlike)
        return ChampionDecision(
            champion=best,
            previous=current,
            switched=True,
            reason=(
                f"El campeón '{current}' no produjo pronósticos; se elige "
                f"'{best}' por menor QLIKE."
            ),
            comparisons=(),
        )

    common = walk_forward.common_mask
    champion_loss = walk_forward.qlike(current)
    challengers = [f for f in available if f != current]
    results: list[DieboldMarianoResult] = []
    for family in challengers:
        loss = walk_forward.qlike(family)
        results.append(diebold_mariano(loss[common], champion_loss[common]))
    adjusted = holm_adjust(np.array([r.p_value_one_sided for r in results]))
    comparisons = tuple(
        ChallengerComparison(f, r, float(p))
        for f, r, p in zip(challengers, results, adjusted, strict=True)
    )
    winners = [
        c
        for c in comparisons
        if c.p_value_holm < alpha and c.dm.mean_loss_diff < 0.0
    ]
    if not winners:
        return ChampionDecision(
            champion=current,
            previous=current,
            switched=False,
            reason=(
                f"Ningún retador supera a '{current}' con p_Holm < {alpha}."
            ),
            comparisons=comparisons,
        )
    best = min(winners, key=lambda c: walk_forward.mean_qlike(c.challenger))
    return ChampionDecision(
        champion=best.challenger,
        previous=current,
        switched=True,
        reason=(
            f"'{best.challenger}' supera a '{current}': DM "
            f"t={best.dm.t_stat:.2f}, p_Holm={best.p_value_holm:.4f}."
        ),
        comparisons=comparisons,
    )
