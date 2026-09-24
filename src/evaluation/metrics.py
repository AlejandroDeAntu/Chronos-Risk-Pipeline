"""Métricas completas, siempre contra baseline y con tamaño de muestra.

* Clasificación (detector de anomalías): matriz de confusión, precisión,
  recall y F1 del modelo y de un baseline aleatorio con la misma tasa de
  positivos, más intervalos de confianza de Wilson al 95% para precisión y
  recall.
* Regresión (volatilidad GARCH): QLIKE como pérdida principal, más MAE,
  RMSE y diagnóstico de residuos estandarizados (Jarque-Bera, Ljung-Box
  sobre residuos al cuadrado y calibración de la varianza).

Por qué QLIKE y no RMSE como criterio principal: la volatilidad real no se
observa; se usa el retorno al cuadrado como aproximación ruidosa. Patton
(2011, *Journal of Econometrics*) muestra que QLIKE y el MSE sobre
varianzas ordenan los modelos igual que si se conociera la volatilidad
verdadera, mientras que el RMSE sobre ``|r|`` no tiene esa garantía.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.proportion import proportion_confint

_MIN_RELIABLE_SAMPLE = 30
_LJUNG_BOX_LAG = 10


@dataclass(frozen=True)
class ClassificationReport:
    """Evaluación del detector de anomalías contra su baseline."""

    n_samples: int
    n_predicted_positive: int
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    precision: float
    precision_ci_95: tuple[float, float]
    recall: float
    recall_ci_95: tuple[float, float]
    f1: float
    baseline_precision: float
    baseline_recall: float
    baseline_f1: float
    beats_baseline_f1: bool
    sample_size_warning: str | None


@dataclass(frozen=True)
class RegressionReport:
    """Evaluación del forecast de volatilidad contra su baseline."""

    n_samples: int
    qlike: float
    baseline_qlike: float
    beats_baseline_qlike: bool
    mae: float
    baseline_mae: float
    rmse: float
    baseline_rmse: float
    mean_squared_standardized_residual: float
    residual_jarque_bera_pvalue: float
    residual_sq_ljung_box_pvalue: float | None
    residual_skew: float
    residual_excess_kurtosis: float
    sample_size_warning: str | None


def _sample_size_warning(n_samples: int) -> str | None:
    """Advierte cuando la muestra es demasiado pequeña para concluir."""
    if n_samples < _MIN_RELIABLE_SAMPLE:
        return (
            f"n={n_samples} < {_MIN_RELIABLE_SAMPLE}: resultado orientativo, "
            "no concluyente; la variabilidad esperada es alta."
        )
    return None


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Intervalo de Wilson al 95% (``(nan, nan)`` si no hay ensayos)."""
    if trials == 0:
        return (float("nan"), float("nan"))
    low, high = proportion_confint(
        successes, trials, alpha=0.05, method="wilson"
    )
    return (float(low), float(high))


def evaluate_anomaly_classifier(
    y_true_reverted: np.ndarray,
    y_pred_anomaly: np.ndarray,
    y_baseline_anomaly: np.ndarray,
) -> ClassificationReport:
    """Matriz de confusión, precisión, recall y F1 del modelo y del baseline.

    Parameters
    ----------
    y_true_reverted:
        Ground truth booleano: ¿hubo reversión tras la barra?
    y_pred_anomaly:
        Predicción booleana del detector.
    y_baseline_anomaly:
        Predicción del baseline aleatorio con la misma tasa de positivos.

    Returns
    -------
    ClassificationReport
        Métricas completas del modelo y del baseline.

    Raises
    ------
    ValueError
        Si los arreglos difieren en longitud o están vacíos.
    """
    lengths = {
        len(y_true_reverted),
        len(y_pred_anomaly),
        len(y_baseline_anomaly),
    }
    if len(lengths) != 1:
        raise ValueError("Los tres arreglos deben tener la misma longitud.")
    if len(y_true_reverted) == 0:
        raise ValueError("No se puede evaluar un conjunto vacío.")

    y_true = np.asarray(y_true_reverted, dtype=bool)
    y_pred = np.asarray(y_pred_anomaly, dtype=bool)
    y_base = np.asarray(y_baseline_anomaly, dtype=bool)

    matrix = confusion_matrix(y_true, y_pred, labels=[False, True])
    n_predicted_positive = int(y_pred.sum())
    n_true_positive = int((y_pred & y_true).sum())
    n_actual_positive = int(y_true.sum())

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    baseline_f1 = float(f1_score(y_true, y_base, zero_division=0))

    return ClassificationReport(
        n_samples=len(y_true),
        n_predicted_positive=n_predicted_positive,
        confusion_matrix=(
            (int(matrix[0, 0]), int(matrix[0, 1])),
            (int(matrix[1, 0]), int(matrix[1, 1])),
        ),
        precision=precision,
        precision_ci_95=_wilson_interval(
            n_true_positive, n_predicted_positive
        ),
        recall=recall,
        recall_ci_95=_wilson_interval(n_true_positive, n_actual_positive),
        f1=f1,
        baseline_precision=float(
            precision_score(y_true, y_base, zero_division=0)
        ),
        baseline_recall=float(recall_score(y_true, y_base, zero_division=0)),
        baseline_f1=baseline_f1,
        beats_baseline_f1=f1 > baseline_f1,
        sample_size_warning=_sample_size_warning(n_predicted_positive),
    )


def qlike_loss(
    predicted_volatility_pct: np.ndarray, realized_returns_pct: np.ndarray
) -> np.ndarray:
    """Pérdida QLIKE punto a punto: ``log(sigma^2) + r^2 / sigma^2``.

    Raises
    ------
    ValueError
        Si alguna volatilidad pronosticada no es estrictamente positiva.
    """
    predicted = np.asarray(predicted_volatility_pct, dtype=float)
    if np.any(predicted <= 0.0) or not np.all(np.isfinite(predicted)):
        raise ValueError("La volatilidad pronosticada debe ser finita y > 0.")
    variance = predicted**2
    realized = np.asarray(realized_returns_pct, dtype=float)
    return np.log(variance) + realized**2 / variance


def evaluate_volatility_forecast(
    predicted_volatility_pct: np.ndarray,
    realized_returns_pct: np.ndarray,
    baseline_volatility_pct: np.ndarray,
) -> RegressionReport:
    """QLIKE, MAE, RMSE y diagnóstico de residuos del modelo vs baseline.

    Parameters
    ----------
    predicted_volatility_pct:
        Volatilidad pronosticada a un paso por el GARCH (en %).
    realized_returns_pct:
        Retorno realizado del día pronosticado (en %, con signo).
    baseline_volatility_pct:
        Pronóstico del baseline (desviación estándar móvil previa).

    Returns
    -------
    RegressionReport
        Métricas y diagnóstico de residuos estandarizados ``r / sigma``.

    Raises
    ------
    ValueError
        Si los arreglos difieren en longitud, están vacíos o hay
        volatilidades no positivas.
    """
    predicted = np.asarray(predicted_volatility_pct, dtype=float)
    realized = np.asarray(realized_returns_pct, dtype=float)
    baseline = np.asarray(baseline_volatility_pct, dtype=float)
    if not len(predicted) == len(realized) == len(baseline):
        raise ValueError("Los tres arreglos deben tener la misma longitud.")
    if len(predicted) == 0:
        raise ValueError("No se puede evaluar un conjunto vacío.")

    model_loss = qlike_loss(predicted, realized)
    baseline_loss = qlike_loss(baseline, realized)
    abs_realized = np.abs(realized)

    standardized = realized / predicted
    ljung_box_pvalue: float | None = None
    if len(standardized) > 2 * _LJUNG_BOX_LAG:
        ljung_box = acorr_ljungbox(
            standardized**2, lags=[_LJUNG_BOX_LAG], return_df=True
        )
        ljung_box_pvalue = float(ljung_box["lb_pvalue"].iloc[0])

    qlike = float(model_loss.mean())
    baseline_qlike = float(baseline_loss.mean())
    return RegressionReport(
        n_samples=len(predicted),
        qlike=qlike,
        baseline_qlike=baseline_qlike,
        beats_baseline_qlike=qlike < baseline_qlike,
        mae=float(np.mean(np.abs(predicted - abs_realized))),
        baseline_mae=float(np.mean(np.abs(baseline - abs_realized))),
        rmse=float(np.sqrt(np.mean((predicted - abs_realized) ** 2))),
        baseline_rmse=float(np.sqrt(np.mean((baseline - abs_realized) ** 2))),
        mean_squared_standardized_residual=float(np.mean(standardized**2)),
        residual_jarque_bera_pvalue=float(
            stats.jarque_bera(standardized).pvalue
        ),
        residual_sq_ljung_box_pvalue=ljung_box_pvalue,
        residual_skew=float(stats.skew(standardized)),
        residual_excess_kurtosis=float(
            stats.kurtosis(standardized, fisher=True)
        ),
        sample_size_warning=_sample_size_warning(len(predicted)),
    )
