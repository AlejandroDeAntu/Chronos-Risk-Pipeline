"""Modelo GARCH(1,1) con innovaciones t-Student y simulación Monte Carlo.

Generaliza el script original (hardcodeado a ``ticker = "SPY"``) a cualquier
activo y corrige un defecto de la primera reconstrucción: el forecast se
calculaba con el estado (último retorno y varianza) del final del
entrenamiento, de modo que entre dos reentrenamientos el sistema repetía el
mismo pronóstico aunque llegaran datos nuevos. Ahora los parámetros quedan
fijos tras el ajuste, pero la varianza condicional se filtra sobre los
retornos observados hasta el día del pronóstico.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model

from src.exceptions import InsufficientHistoryError, ModelFitError


@dataclass(frozen=True)
class GarchVolatilityModel:
    """Parámetros de un GARCH(1,1)-t ajustado (en unidades de %).

    Attributes
    ----------
    version:
        Identificador de versión (timestamp UTC de entrenamiento).
    omega, alpha, beta, mu, nu:
        Parámetros del GARCH(1,1) con distribución t-Student.
    aic, bic:
        Criterios de información del ajuste.
    trained_at:
        Marca de tiempo UTC del entrenamiento.
    train_rows:
        Observaciones usadas en el ajuste.
    train_end:
        Fecha de la última observación de entrenamiento (auditoría).
    """

    version: str
    omega: float
    alpha: float
    beta: float
    mu: float
    nu: float
    aic: float
    bic: float
    trained_at: datetime
    train_rows: int
    train_end: str

    @classmethod
    def fit(
        cls,
        log_returns: pd.Series,
        min_history_days: int,
        version: str | None = None,
    ) -> GarchVolatilityModel:
        """Ajusta un GARCH(1,1)-t sobre retornos logarítmicos diarios.

        Parameters
        ----------
        log_returns:
            Retornos logarítmicos diarios (proporciones).
        min_history_days:
            Mínimo de observaciones requerido.
        version:
            Identificador de versión; por defecto, el timestamp UTC actual.

        Returns
        -------
        GarchVolatilityModel
            Modelo ajustado.

        Raises
        ------
        InsufficientHistoryError
            Si no hay historia suficiente.
        ModelFitError
            Si el optimizador falla, no converge, o los parámetros no son
            válidos (``nu <= 2`` o proceso no estacionario).
        """
        if len(log_returns) < min_history_days:
            raise InsufficientHistoryError(
                f"Se requieren {min_history_days} observaciones diarias, "
                f"se recibieron {len(log_returns)}."
            )

        spec = arch_model(
            log_returns * 100.0,
            mean="Constant",
            vol="Garch",
            p=1,
            q=1,
            dist="t",
        )
        try:
            result = spec.fit(disp="off", show_warning=False)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ModelFitError(f"El ajuste GARCH falló: {exc}") from exc

        if result.convergence_flag != 0:
            raise ModelFitError(
                "El optimizador GARCH no convergió "
                f"(convergence_flag={result.convergence_flag})."
            )

        params = result.params
        required = ("omega", "alpha[1]", "beta[1]", "nu")
        missing = [name for name in required if name not in params.index]
        if missing:
            raise ModelFitError(f"Faltan parámetros en el ajuste: {missing}")

        alpha = float(params["alpha[1]"])
        beta = float(params["beta[1]"])
        nu = float(params["nu"])
        if nu <= 2.0:
            raise ModelFitError(f"nu={nu:.3f} debe ser > 2 (varianza finita).")
        if alpha + beta >= 1.0:
            raise ModelFitError(
                f"alpha + beta = {alpha + beta:.4f} >= 1: proceso no "
                "estacionario en varianza."
            )

        trained_at = datetime.now(UTC)
        return cls(
            version=version or trained_at.strftime("%Y%m%dT%H%M%S%fZ"),
            omega=float(params["omega"]),
            alpha=alpha,
            beta=beta,
            mu=float(params.get("mu", 0.0)),
            nu=nu,
            aic=float(result.aic),
            bic=float(result.bic),
            trained_at=trained_at,
            train_rows=len(log_returns),
            train_end=str(log_returns.index[-1]),
        )

    def conditional_variance_path(self, returns_pct: np.ndarray) -> np.ndarray:
        """Filtra la varianza condicional con parámetros fijos.

        ``h[t]`` es la varianza pronosticada para el retorno ``t`` usando solo
        ``returns_pct[:t]``: nunca usa el propio retorno ``t`` ni posteriores,
        por lo que es un pronóstico fuera de muestra a un paso. La recursión
        es secuencial en el tiempo por definición del GARCH, así que el
        bucle recorre posiciones de un arreglo NumPy, no filas de un
        DataFrame. ``h[0]`` se inicializa con la varianza muestral.

        Parameters
        ----------
        returns_pct:
            Retornos en porcentaje, en orden cronológico.

        Returns
        -------
        numpy.ndarray
            Varianza condicional ``h`` (en %^2), misma longitud que la
            entrada.
        """
        n_obs = len(returns_pct)
        if n_obs < 2:
            raise InsufficientHistoryError(
                "Se requieren al menos 2 retornos para filtrar la varianza."
            )
        variance = np.empty(n_obs)
        variance[0] = float(np.var(returns_pct, ddof=1))
        shocks_sq = (returns_pct - self.mu) ** 2
        for t in range(1, n_obs):
            variance[t] = (
                self.omega
                + self.alpha * shocks_sq[t - 1]
                + self.beta * variance[t - 1]
            )
        return variance

    def forecast_next_volatility_pct(self, returns_pct: np.ndarray) -> float:
        """Volatilidad condicional pronosticada para la siguiente barra (%).

        Parameters
        ----------
        returns_pct:
            Todos los retornos observados hasta hoy, en porcentaje.

        Returns
        -------
        float
            ``sqrt(omega + alpha * (r_T - mu)^2 + beta * h_T)``.
        """
        variance = self.conditional_variance_path(returns_pct)
        next_variance = (
            self.omega
            + self.alpha * (returns_pct[-1] - self.mu) ** 2
            + self.beta * variance[-1]
        )
        return float(np.sqrt(next_variance))

    def simulate_paths(
        self,
        returns_pct: np.ndarray,
        horizon_days: int,
        n_paths: int,
        rng_seed: int,
    ) -> np.ndarray:
        """Simula trayectorias Monte Carlo de retornos desde el estado actual.

        Parameters
        ----------
        returns_pct:
            Retornos observados hasta hoy (en %), para fijar el estado inicial.
        horizon_days:
            Barras futuras a simular.
        n_paths:
            Número de trayectorias.
        rng_seed:
            Semilla del generador, para reproducibilidad.

        Returns
        -------
        numpy.ndarray
            Retornos simulados en %, de forma ``(n_paths, horizon_days)``.
        """
        rng = np.random.default_rng(rng_seed)
        scale = np.sqrt(self.nu / (self.nu - 2.0))

        variance_path = self.conditional_variance_path(returns_pct)
        h_t = np.full(n_paths, variance_path[-1])
        r_t = np.full(n_paths, returns_pct[-1])
        sim_returns = np.empty((n_paths, horizon_days))

        # Bucle sobre el horizonte (pocos pasos, recursión secuencial);
        # las n_paths trayectorias se calculan vectorizadas en cada paso.
        for step in range(horizon_days):
            z = rng.standard_t(df=self.nu, size=n_paths) / scale
            h_t = (
                self.omega
                + self.alpha * (r_t - self.mu) ** 2
                + self.beta * h_t
            )
            r_t = self.mu + np.sqrt(h_t) * z
            sim_returns[:, step] = r_t
        return sim_returns

    def to_metadata(self) -> dict[str, Any]:
        """Serializa el modelo a un diccionario JSON-compatible."""
        return {
            "version": self.version,
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "mu": self.mu,
            "nu": self.nu,
            "aic": self.aic,
            "bic": self.bic,
            "trained_at": self.trained_at.isoformat(),
            "train_rows": self.train_rows,
            "train_end": self.train_end,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> GarchVolatilityModel:
        """Reconstruye un modelo a partir de metadata serializada."""
        return cls(
            version=str(metadata["version"]),
            omega=float(metadata["omega"]),
            alpha=float(metadata["alpha"]),
            beta=float(metadata["beta"]),
            mu=float(metadata["mu"]),
            nu=float(metadata["nu"]),
            aic=float(metadata["aic"]),
            bic=float(metadata["bic"]),
            trained_at=datetime.fromisoformat(metadata["trained_at"]),
            train_rows=int(metadata["train_rows"]),
            train_end=str(metadata["train_end"]),
        )
