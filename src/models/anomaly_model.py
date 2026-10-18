"""Detector de anomalías de retorno logarítmico, con umbral auto-calibrado.

El proyecto original fijaba el umbral en ``threshold = 2`` (desviaciones
estándar) de forma arbitraria, asumiendo implícitamente normalidad. Aquí el
umbral se calibra empíricamente a partir de la distribución histórica de
z-scores causales para alcanzar una tasa objetivo de anomalías
(``target_anomaly_rate``), lo cual es más robusto a la curtosis (colas
pesadas) típica de los retornos financieros. La recalibración periódica es,
precisamente, el mecanismo de "reentrenamiento" de este modelo.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.exceptions import InsufficientHistoryError, ModelFitError
from src.features.returns import compute_causal_zscore_bands
from src.features.seasonality import HOURS, deseasonalize, hourly_profile


@dataclass(frozen=True)
class AnomalyDetector:
    """Detector de anomalías con umbral de z-score calibrado empíricamente.

    Attributes
    ----------
    version:
        Identificador de versión (timestamp ISO 8601 de entrenamiento).
    rolling_window:
        Barras usadas para la media/std móviles causales.
    zscore_threshold:
        Umbral absoluto de z-score calibrado en ``fit``.
    target_anomaly_rate:
        Tasa objetivo de anomalías usada para calibrar el umbral.
    reversion_horizon_bars:
        Barras hacia adelante usadas para etiquetar reversión (ground truth).
    trained_at:
        Marca de tiempo UTC del entrenamiento.
    train_rows:
        Barras usadas para calibrar (tamaño de muestra).
    seasonal_profile:
        Factor de volatilidad por hora UTC estimado en entrenamiento; el
        z-score se calcula sobre el retorno dividido por el factor de su
        hora. Todos 1.0 equivale al detector sin ajuste estacional.
    """

    version: str
    rolling_window: int
    zscore_threshold: float
    target_anomaly_rate: float
    reversion_horizon_bars: int
    trained_at: datetime
    train_rows: int
    seasonal_profile: tuple[float, ...] = (1.0,) * len(HOURS)

    def _bands(
        self, log_return: pd.Series, min_bars_required: int
    ) -> pd.DataFrame:
        """Bandas causales sobre el retorno desestacionalizado.

        ``log_return`` conserva el retorno crudo (la etiqueta de reversión
        se define sobre él); ``rolling_mean``, ``rolling_std`` y ``zscore``
        están en unidades desestacionalizadas.
        """
        profile = pd.Series(self.seasonal_profile, index=HOURS)
        bands = compute_causal_zscore_bands(
            deseasonalize(log_return, profile),
            self.rolling_window,
            min_bars_required,
        )
        bands["seasonal_factor"] = profile.reindex(bands.index.hour).to_numpy()
        bands["log_return"] = log_return.reindex(bands.index)
        return bands

    @classmethod
    def fit(
        cls,
        log_return: pd.Series,
        rolling_window: int,
        target_anomaly_rate: float,
        reversion_horizon_bars: int,
        min_bars_required: int,
        version: str | None = None,
        seasonal: bool = True,
    ) -> AnomalyDetector:
        """Calibra el umbral de z-score sobre un historial de retornos.

        Parameters
        ----------
        log_return:
            Retornos logarítmicos históricos (barra a barra).
        rolling_window:
            Tamaño de la ventana móvil causal.
        target_anomaly_rate:
            Proporción objetivo de barras marcadas como anómalas (ej. 0.025
            para un ~2.5% en cada cola, análogo a un umbral de ~2 sigma bajo
            normalidad, pero derivado empíricamente en vez de asumido).
        reversion_horizon_bars:
            Barras hacia adelante usadas para el ground truth de reversión.
        min_bars_required:
            Mínimo de observaciones causales requeridas para calibrar.
        version:
            Identificador de versión; si es ``None`` se usa el timestamp
            actual en UTC.
        seasonal:
            Si es ``True`` (por defecto) estima el perfil horario con estos
            mismos datos de entrenamiento y lo aplica antes del z-score.

        Returns
        -------
        AnomalyDetector
            Detector calibrado, listo para ``predict``.

        Raises
        ------
        InsufficientHistoryError
            Si no hay suficiente historia para calibrar de forma confiable.
        ModelFitError
            Si ``target_anomaly_rate`` no produce un umbral finito válido.
        """
        if not 0.0 < target_anomaly_rate < 0.5:
            raise ModelFitError(
                "target_anomaly_rate debe estar en (0, 0.5), se recibió "
                f"{target_anomaly_rate}."
            )

        profile = (
            tuple(float(f) for f in hourly_profile(log_return))
            if seasonal
            else (1.0,) * len(HOURS)
        )
        trained_at = datetime.now(UTC)
        unfitted = cls(
            version=version or trained_at.strftime("%Y%m%dT%H%M%S%fZ"),
            rolling_window=rolling_window,
            zscore_threshold=float("inf"),
            target_anomaly_rate=target_anomaly_rate,
            reversion_horizon_bars=reversion_horizon_bars,
            trained_at=trained_at,
            train_rows=0,
            seasonal_profile=profile,
        )
        bands = unfitted._bands(log_return, min_bars_required)
        abs_z = bands["zscore"].abs()

        threshold = float(
            np.quantile(abs_z.to_numpy(), 1.0 - target_anomaly_rate)
        )
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ModelFitError(
                f"El umbral calibrado no es válido: {threshold}."
            )

        return dataclasses.replace(
            unfitted, zscore_threshold=threshold, train_rows=len(bands)
        )

    def predict(
        self, log_return: pd.Series, min_bars_required: int
    ) -> pd.DataFrame:
        """Aplica el umbral calibrado a una serie de retornos recientes.

        Parameters
        ----------
        log_return:
            Retornos logarítmicos recientes (misma frecuencia que en ``fit``).
        min_bars_required:
            Mínimo de observaciones causales requeridas para predecir.

        Returns
        -------
        pandas.DataFrame
            Columnas ``log_return`` (crudo), ``rolling_mean``,
            ``rolling_std``, ``zscore`` (desestacionalizados),
            ``seasonal_factor`` e ``is_anomaly``.
        """
        bands = self._bands(log_return, min_bars_required)
        bands["is_anomaly"] = bands["zscore"].abs() > self.zscore_threshold
        return bands

    def to_metadata(self) -> dict[str, Any]:
        """Serializa el detector a un diccionario JSON-compatible."""
        return {
            "version": self.version,
            "rolling_window": self.rolling_window,
            "zscore_threshold": self.zscore_threshold,
            "target_anomaly_rate": self.target_anomaly_rate,
            "reversion_horizon_bars": self.reversion_horizon_bars,
            "trained_at": self.trained_at.isoformat(),
            "train_rows": self.train_rows,
            "seasonal_profile": list(self.seasonal_profile),
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> AnomalyDetector:
        """Reconstruye un detector a partir de metadata serializada."""
        return cls(
            version=metadata["version"],
            rolling_window=int(metadata["rolling_window"]),
            zscore_threshold=float(metadata["zscore_threshold"]),
            target_anomaly_rate=float(metadata["target_anomaly_rate"]),
            reversion_horizon_bars=int(metadata["reversion_horizon_bars"]),
            trained_at=datetime.fromisoformat(metadata["trained_at"]),
            train_rows=int(metadata["train_rows"]),
            # Los detectores previos al ajuste estacional no lo tienen.
            seasonal_profile=tuple(
                float(f)
                for f in metadata.get("seasonal_profile", [1.0] * len(HOURS))
            ),
        )


def label_reversion_outcomes(
    bands: pd.DataFrame, horizon_bars: int
) -> pd.DataFrame:
    """Etiqueta si cada barra fue seguida de una reversión de signo.

    El ground truth es deliberadamente simple y directamente ligado a la
    hipótesis que el propio proyecto original imprimía en pantalla
    ("Buscar posible corrección"): una anomalía "acierta" si el retorno
    acumulado de las siguientes ``horizon_bars`` barras tiene signo opuesto
    al retorno anómalo. No se inventa una regla de trading; se opera lo que
    el indicador ya afirmaba implícitamente.

    Parameters
    ----------
    bands:
        DataFrame devuelto por ``AnomalyDetector.predict`` (debe incluir
        ``log_return`` e ``is_anomaly``).
    horizon_bars:
        Número de barras hacia adelante usadas para el resultado observado.

    Returns
    -------
    pandas.DataFrame
        Copia de ``bands`` con una columna adicional ``reverted`` (``bool``).
        Se eliminan las filas sin suficientes barras futuras y los empates
        (retorno actual o futuro acumulado exactamente 0).

    Raises
    ------
    InsufficientHistoryError
        Si no queda ninguna fila con horizonte futuro completo.
    """
    # Suma vectorizada de los retornos de las siguientes `horizon_bars` barras
    # (t+1 .. t+horizon_bars). El bucle recorre un número fijo y pequeño de
    # desplazamientos (no filas de un DataFrame) para construir columnas
    # desplazadas que luego se suman de forma vectorizada con `sum(axis=1)`.
    shifted_future_returns = pd.concat(
        [bands["log_return"].shift(-k) for k in range(1, horizon_bars + 1)],
        axis=1,
    )
    future_cum_return = shifted_future_returns.sum(axis=1, skipna=False)

    labeled = bands.copy()
    labeled["future_cum_return"] = future_cum_return
    labeled = labeled.dropna(subset=["future_cum_return"])
    # Empates: con retorno 0 (o retorno futuro 0) el signo no está definido.
    # Contarlos como "reversión" inflaba la prevalencia (EUR/USD: 34% de
    # las barras de 5 min tienen retorno 0) y con ella al baseline aleatorio.
    is_tie = (labeled["log_return"] == 0.0) | (
        labeled["future_cum_return"] == 0.0
    )
    labeled = labeled[~is_tie]

    if labeled.empty:
        raise InsufficientHistoryError(
            f"No quedan barras con {horizon_bars} barras futuras completas "
            "para etiquetar reversión."
        )

    labeled["reverted"] = np.sign(labeled["future_cum_return"]) != np.sign(
        labeled["log_return"]
    )
    return labeled
