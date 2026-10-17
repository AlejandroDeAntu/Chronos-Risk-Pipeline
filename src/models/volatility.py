"""Modelos de volatilidad diaria con una interfaz común.

Familias:

* ``garch``, ``gjr`` y ``egarch``: varianza condicional con innovaciones t
  de Student, estimadas con ``arch`` (``ConditionalVolModel``). GJR y
  EGARCH capturan la asimetría (los choques negativos suben más la
  volatilidad), documentada en índices bursátiles.
* ``har``: HAR-RV de Corsi (2009) sobre varianza realizada por rango
  (``HarModel``): componentes diaria, semanal y mensual en logaritmos.

Todas las varianzas están en %^2 (retornos log x 100). La interfaz es:

* ``variance_path(frame)``: varianza pronosticada para cada día ``t`` con
  información hasta ``t-1`` (pronóstico fuera de muestra a un paso).
* ``forecast_variance(frame, horizon)``: varianza de los próximos
  ``horizon`` días tras la última fila de ``frame``.
* ``simulate_returns(frame, horizon, n_paths, seed, method)``: escenarios
  de retornos en %, por Monte Carlo paramétrico o simulación histórica
  filtrada (``"fhs"``: remuestreo de residuos estandarizados).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from scipy import stats

from src.config import VOLATILITY_FAMILIES
from src.exceptions import InsufficientHistoryError, ModelFitError
from src.features.range_volatility import realized_variance_proxy, yang_zhang

ArchFamily = Literal["garch", "gjr", "egarch"]
SimulationMethod = Literal["monte_carlo", "fhs"]

_PCT2 = 1e4  # log-retorno^2 -> %^2
_HAR_WEEK, _HAR_MONTH = 5, 22
_FHS_LOOKBACK = 1000
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_ARCH_SPECS: dict[str, dict[str, Any]] = {
    "garch": {"vol": "GARCH", "p": 1, "o": 0, "q": 1},
    "gjr": {"vol": "GARCH", "p": 1, "o": 1, "q": 1},
    "egarch": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1},
}


def volatility_frame(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Prepara las entradas comunes a todos los modelos.

    Parameters
    ----------
    ohlc:
        Velas diarias validadas (``open``, ``high``, ``low``, ``close``).

    Returns
    -------
    pandas.DataFrame
        ``return_pct`` (log-retorno cierre a cierre x 100) y las varianzas
        realizadas en %^2: ``rv_day`` (overnight^2 + Garman-Klass),
        ``rv_week`` y ``rv_month`` (Yang-Zhang de 5 y 22 días). Empieza en
        la segunda vela, la primera con retorno definido.
    """
    frame = pd.DataFrame(
        {
            "return_pct": 100.0 * np.log(ohlc["close"]).diff(),
            "rv_day": _PCT2 * realized_variance_proxy(ohlc),
            "rv_week": _PCT2 * yang_zhang(ohlc, _HAR_WEEK),
            "rv_month": _PCT2 * yang_zhang(ohlc, _HAR_MONTH),
        }
    )
    return frame.iloc[1:]


class VolatilityModel(Protocol):
    """Contrato que cumplen todas las familias de volatilidad.

    Los atributos son de solo lectura: las implementaciones son dataclasses
    congeladas.
    """

    @property
    def family(self) -> str:
        """Familia del modelo."""
        ...

    @property
    def version(self) -> str:
        """Versión registrada."""
        ...

    @property
    def trained_at(self) -> datetime:
        """Instante UTC del ajuste."""
        ...

    @property
    def train_rows(self) -> int:
        """Filas usadas en el ajuste."""
        ...

    def variance_path(self, frame: pd.DataFrame) -> pd.Series:
        """Varianza a un paso para cada fila, con información hasta t-1."""
        ...

    def forecast_variance(
        self, frame: pd.DataFrame, horizon: int
    ) -> np.ndarray:
        """Varianza de los próximos ``horizon`` días."""
        ...

    def simulate_returns(
        self,
        frame: pd.DataFrame,
        horizon: int,
        n_paths: int,
        seed: int,
        method: SimulationMethod = "monte_carlo",
    ) -> np.ndarray:
        """Escenarios de retornos en %, forma ``(n_paths, horizon)``."""
        ...

    def to_metadata(self) -> dict[str, Any]:
        """Serializa el modelo a JSON."""
        ...


def _require_history(frame: pd.DataFrame, min_history_days: int) -> None:
    if len(frame) < min_history_days:
        raise InsufficientHistoryError(
            f"Se requieren {min_history_days} observaciones diarias, se "
            f"recibieron {len(frame)}."
        )


def _version_now() -> tuple[str, datetime]:
    trained_at = datetime.now(UTC)
    return trained_at.strftime("%Y%m%dT%H%M%S%fZ"), trained_at


def _fhs_draws(
    standardized: np.ndarray, n_paths: int, horizon: int, seed: int
) -> np.ndarray:
    """Remuestrea residuos estandarizados históricos (con reemplazo)."""
    pool = standardized[np.isfinite(standardized)][-_FHS_LOOKBACK:]
    if pool.size < 30:
        raise InsufficientHistoryError(
            "FHS requiere al menos 30 residuos estandarizados válidos."
        )
    rng = np.random.default_rng(seed)
    return rng.choice(pool, size=(n_paths, horizon), replace=True)


@dataclass(frozen=True)
class ConditionalVolModel:
    """GARCH, GJR-GARCH o EGARCH (1,1) con innovaciones t de Student.

    ``arch`` solo estima los parámetros. El filtrado y la simulación usan
    las recursiones explícitas de abajo porque ``arch.fix()`` inicializa la
    varianza y la acota con estadísticas de toda la muestra (fuga hacia
    atrás desde datos futuros) y su simulación ignora ``random_state``.
    Aquí la varianza inicial es la de entrenamiento (``backcast``), así que
    el valor en ``t`` depende solo de retornos hasta ``t-1``.

    Recursiones, con ``eps = r - mu`` y ``e = eps / sigma``:

    * GARCH: ``h_t = omega + alpha eps^2 + beta h_{t-1}``
    * GJR: ``h_t = omega + (alpha + gamma 1[eps < 0]) eps^2 + beta h_{t-1}``
    * EGARCH: ``ln h_t = omega + alpha (|e| - sqrt(2/pi)) + gamma e
      + beta ln h_{t-1}``

    Attributes
    ----------
    family:
        ``"garch"``, ``"gjr"`` o ``"egarch"``.
    params:
        Parámetros con los nombres de ``arch`` (``mu``, ``omega``,
        ``alpha[1]``, ``gamma[1]``, ``beta[1]``, ``nu``).
    backcast:
        Varianza inicial de la recursión (%^2): la de entrenamiento.
    version, trained_at, train_rows, train_end:
        Auditoría del ajuste.
    aic, bic:
        Criterios de información del ajuste.
    """

    family: ArchFamily
    params: dict[str, float]
    backcast: float
    version: str
    trained_at: datetime
    train_rows: int
    train_end: str
    aic: float = float("nan")
    bic: float = float("nan")

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        family: ArchFamily,
        min_history_days: int,
        version: str | None = None,
    ) -> ConditionalVolModel:
        """Estima el modelo por máxima verosimilitud sobre ``frame``.

        Raises
        ------
        InsufficientHistoryError
            Si hay menos de ``min_history_days`` filas.
        ModelFitError
            Si el optimizador falla o no converge, ``nu <= 2`` o el proceso
            no es estacionario.
        """
        _require_history(frame, min_history_days)
        returns = frame["return_pct"].to_numpy()
        spec = arch_model(
            returns,
            mean="Constant",
            dist="t",
            rescale=False,
            **_ARCH_SPECS[family],
        )
        try:
            result = spec.fit(disp="off", show_warning=False)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ModelFitError(f"El ajuste {family} falló: {exc}") from exc
        if result.convergence_flag != 0:
            raise ModelFitError(
                f"El optimizador {family} no convergió "
                f"(convergence_flag={result.convergence_flag})."
            )
        params = {name: float(value) for name, value in result.params.items()}
        _check_arch_params(family, params)
        default_version, trained_at = _version_now()
        return cls(
            family=family,
            params=params,
            backcast=float(np.var(returns, ddof=1)),
            version=version or default_version,
            trained_at=trained_at,
            train_rows=len(frame),
            train_end=str(frame.index[-1]),
            aic=float(result.aic),
            bic=float(result.bic),
        )

    def _next_variance(
        self, variance: np.ndarray, shock: np.ndarray
    ) -> np.ndarray:
        """Un paso de la recursión, vectorizado sobre trayectorias."""
        p = self.params
        if self.family == "egarch":
            standardized = shock / np.sqrt(variance)
            log_next = (
                p["omega"]
                + p["alpha[1]"] * (np.abs(standardized) - _SQRT_2_OVER_PI)
                + p["gamma[1]"] * standardized
                + p["beta[1]"] * np.log(variance)
            )
            return np.exp(log_next)
        asymmetry = p.get("gamma[1]", 0.0) * (shock < 0.0)
        return (
            p["omega"]
            + (p["alpha[1]"] + asymmetry) * shock**2
            + p["beta[1]"] * variance
        )

    def _filter(self, returns_pct: np.ndarray) -> np.ndarray:
        """Varianzas ``h_0..h_n``: ``h_t`` usa retornos hasta ``t-1``.

        La recursión es secuencial por definición; el bucle recorre un
        arreglo NumPy, no filas de un DataFrame.
        """
        shocks = returns_pct - self.params["mu"]
        variance = np.empty(len(returns_pct) + 1)
        variance[0] = self.backcast
        for t in range(len(returns_pct)):
            variance[t + 1] = self._next_variance(
                variance[t : t + 1], shocks[t : t + 1]
            )[0]
        return variance

    def variance_path(self, frame: pd.DataFrame) -> pd.Series:
        """Varianza condicional filtrada con parámetros fijos (%^2)."""
        variance = self._filter(frame["return_pct"].to_numpy())
        return pd.Series(variance[:-1], index=frame.index, name="variance")

    def _innovations(
        self,
        frame: pd.DataFrame,
        n_paths: int,
        horizon: int,
        seed: int,
        method: SimulationMethod,
    ) -> np.ndarray:
        """Innovaciones estandarizadas (media 0, varianza 1)."""
        if method == "fhs":
            returns = frame["return_pct"].to_numpy()
            sigma = np.sqrt(self._filter(returns)[:-1])
            standardized = (returns - self.params["mu"]) / sigma
            return _fhs_draws(standardized, n_paths, horizon, seed)
        nu = self.params["nu"]
        rng = np.random.default_rng(seed)
        draws = rng.standard_t(nu, size=(n_paths, horizon))
        return draws / math.sqrt(nu / (nu - 2.0))

    def _simulate_shocks(
        self, frame: pd.DataFrame, innovations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propaga la recursión sobre ``innovations``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Choques ``eps`` y varianzas ``h``, ambos ``(n_paths, horizon)``.
        """
        n_paths, horizon = innovations.shape
        variance = np.full(
            n_paths, self._filter(frame["return_pct"].to_numpy())[-1]
        )
        shocks = np.empty_like(innovations)
        variances = np.empty_like(innovations)
        # Bucle sobre el horizonte (pocos pasos); trayectorias vectorizadas.
        for step in range(horizon):
            variances[:, step] = variance
            shocks[:, step] = np.sqrt(variance) * innovations[:, step]
            variance = self._next_variance(variance, shocks[:, step])
        return shocks, variances

    def forecast_variance(
        self, frame: pd.DataFrame, horizon: int
    ) -> np.ndarray:
        """Varianza esperada de los próximos días.

        GARCH/GJR: fórmula cerrada ``E[h_{k+1}] = omega + phi E[h_k]`` con
        ``phi = alpha + gamma / 2 + beta`` (innovación simétrica). EGARCH:
        media de 5,000 trayectorias simuladas con semilla fija.
        """
        next_variance = self._filter(frame["return_pct"].to_numpy())[-1]
        if self.family == "egarch" and horizon > 1:
            innovations = self._innovations(
                frame, 5000, horizon, 0, "monte_carlo"
            )
            _, variances = self._simulate_shocks(frame, innovations)
            return np.r_[next_variance, variances.mean(axis=0)[1:]]
        p = self.params
        phi = p["alpha[1]"] + 0.5 * p.get("gamma[1]", 0.0) + p["beta[1]"]
        variances = [next_variance]
        for _ in range(horizon - 1):
            variances.append(p["omega"] + phi * variances[-1])
        return np.asarray(variances)

    def simulate_returns(
        self,
        frame: pd.DataFrame,
        horizon: int,
        n_paths: int,
        seed: int,
        method: SimulationMethod = "monte_carlo",
    ) -> np.ndarray:
        """Escenarios de retornos en % (t de Student o FHS)."""
        innovations = self._innovations(frame, n_paths, horizon, seed, method)
        shocks, _ = self._simulate_shocks(frame, innovations)
        return self.params["mu"] + shocks

    def to_metadata(self) -> dict[str, Any]:
        """Serializa el modelo a un diccionario JSON-compatible."""
        return {
            "family": self.family,
            "params": self.params,
            "backcast": self.backcast,
            "version": self.version,
            "trained_at": self.trained_at.isoformat(),
            "train_rows": self.train_rows,
            "train_end": self.train_end,
            "aic": self.aic,
            "bic": self.bic,
        }


def _check_arch_params(family: str, params: dict[str, float]) -> None:
    """Valida varianza finita y estacionariedad del proceso estimado."""
    if params["nu"] <= 2.0:
        raise ModelFitError(f"nu={params['nu']:.3f} debe ser > 2.")
    if family == "egarch":
        persistence = abs(params["beta[1]"])
    else:
        persistence = (
            params["alpha[1]"]
            + 0.5 * params.get("gamma[1]", 0.0)
            + params["beta[1]"]
        )
    if persistence >= 1.0:
        raise ModelFitError(
            f"Persistencia {persistence:.4f} >= 1 en {family}: proceso no "
            "estacionario en varianza."
        )


@dataclass(frozen=True)
class HarModel:
    """HAR-RV en logaritmos sobre varianza realizada por rango.

    ``log RV_{t+1} = b0 + b1 log RV_t + b2 log RV^w_t + b3 log RV^m_t + e``
    con ``RV`` = overnight^2 + Garman-Klass y ``RV^w``, ``RV^m`` =
    Yang-Zhang de 5 y 22 días. El pronóstico de varianza del retorno es
    ``scale * exp(Xb + s2 / 2)``: la corrección ``s2 / 2`` deshace el sesgo
    de Jensen del logaritmo y ``scale`` (estimado en entrenamiento) mapea
    la varianza por rango, que subestima por muestreo discreto y no ve
    toda la sesión, a la varianza del retorno cierre a cierre.

    Attributes
    ----------
    coef:
        ``(b0, b1, b2, b3)``.
    resid_var:
        Varianza de los residuos de la regresión en logaritmos.
    scale:
        Media de ``r^2`` entre media del pronóstico, en entrenamiento.
    floor:
        Piso de RV en %^2 (percentil 1 de los valores positivos de
        entrenamiento) para velas planas con RV = 0.
    nu:
        Grados de libertad de la t ajustada a los residuos estandarizados,
        para el Monte Carlo paramétrico.
    """

    coef: tuple[float, float, float, float]
    resid_var: float
    scale: float
    floor: float
    nu: float
    version: str
    trained_at: datetime
    train_rows: int
    train_end: str
    family: str = field(default="har")

    @staticmethod
    def _design(frame: pd.DataFrame, floor: float) -> pd.DataFrame:
        logs = np.log(
            frame[["rv_day", "rv_week", "rv_month"]].clip(lower=floor)
        )
        return sm.add_constant(logs, has_constant="add")

    def _raw_path(self, frame: pd.DataFrame) -> pd.Series:
        design = self._design(frame, self.floor)
        log_forecast = design.to_numpy() @ np.asarray(self.coef)
        raw = np.exp(log_forecast + 0.5 * self.resid_var)
        # Fila t pronostica t+1; se desplaza para alinear con el objetivo.
        return pd.Series(raw, index=frame.index).shift(1)

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        min_history_days: int,
        version: str | None = None,
    ) -> HarModel:
        """Estima el HAR por MCO y calibra la escala en el mismo bloque.

        Raises
        ------
        InsufficientHistoryError
            Si hay menos de ``min_history_days`` filas.
        ModelFitError
            Si no hay suficientes filas completas o la regresión no es
            finita.
        """
        _require_history(frame, min_history_days)
        positive = frame["rv_day"][frame["rv_day"] > 0.0]
        if positive.size < min_history_days // 2:
            raise ModelFitError("Muy pocas velas con varianza realizada > 0.")
        floor = float(positive.quantile(0.01))

        design = cls._design(frame, floor)
        target = np.log(frame["rv_day"].clip(lower=floor)).shift(-1)
        sample = pd.concat([design, target.rename("y")], axis=1).dropna()
        if len(sample) < min_history_days // 2:
            raise ModelFitError("Muy pocas filas completas para el HAR.")
        ols = sm.OLS(sample["y"], sample.drop(columns="y")).fit()
        coef = tuple(float(c) for c in ols.params)
        if not np.all(np.isfinite(coef)):
            raise ModelFitError(f"Coeficientes HAR no finitos: {coef}.")

        default_version, trained_at = _version_now()
        unscaled = cls(
            coef=coef,  # type: ignore[arg-type]
            resid_var=float(ols.mse_resid),
            scale=1.0,
            floor=floor,
            nu=30.0,
            version=version or default_version,
            trained_at=trained_at,
            train_rows=len(frame),
            train_end=str(frame.index[-1]),
        )
        raw = unscaled._raw_path(frame).dropna()
        squared = frame["return_pct"].loc[raw.index] ** 2
        scale = float(squared.mean() / raw.mean()) if len(raw) else np.nan
        if not np.isfinite(scale) or scale <= 0.0:
            raise ModelFitError(f"Escala HAR inválida: {scale}.")
        standardized = frame["return_pct"].loc[raw.index] / np.sqrt(
            scale * raw
        )
        try:
            nu, _, _ = stats.t.fit(standardized.to_numpy(), floc=0.0)
        except (ValueError, RuntimeError) as exc:
            raise ModelFitError(f"Ajuste t de residuos HAR: {exc}") from exc
        return cls(
            coef=coef,  # type: ignore[arg-type]
            resid_var=unscaled.resid_var,
            scale=scale,
            floor=floor,
            nu=float(max(nu, 2.5)),
            version=unscaled.version,
            trained_at=trained_at,
            train_rows=len(frame),
            train_end=unscaled.train_end,
        )

    def variance_path(self, frame: pd.DataFrame) -> pd.Series:
        """Varianza pronosticada (%^2); ``NaN`` sin componentes completas."""
        return (self.scale * self._raw_path(frame)).rename("variance")

    def forecast_variance(
        self, frame: pd.DataFrame, horizon: int
    ) -> np.ndarray:
        """Pronóstico iterado de la componente diaria.

        Las componentes semanal y mensual se mantienen en su último valor
        durante el horizonte.
        """
        # ponytail: semanal/mensual congeladas en h<=5; recalcularlas solo
        # si se piden horizontes largos.
        b0, b1, b2, b3 = self.coef
        last = frame.iloc[-1]
        # np.fmax ignora NaN (max() de Python lo propagaría).
        log_day = math.log(np.fmax(last["rv_day"], self.floor))
        fixed = b0 + b2 * math.log(np.fmax(last["rv_week"], self.floor))
        fixed += b3 * math.log(np.fmax(last["rv_month"], self.floor))
        variances = []
        for _ in range(horizon):
            log_day = fixed + b1 * log_day
            variances.append(
                self.scale * math.exp(log_day + 0.5 * self.resid_var)
            )
        return np.asarray(variances)

    def simulate_returns(
        self,
        frame: pd.DataFrame,
        horizon: int,
        n_paths: int,
        seed: int,
        method: SimulationMethod = "monte_carlo",
    ) -> np.ndarray:
        """Escenarios: t estandarizada (MC) o residuos históricos (FHS)."""
        sigma = np.sqrt(self.forecast_variance(frame, horizon))
        if method == "fhs":
            path = self.variance_path(frame)
            standardized = (frame["return_pct"] / np.sqrt(path)).to_numpy()
            draws = _fhs_draws(standardized, n_paths, horizon, seed)
        else:
            rng = np.random.default_rng(seed)
            draws = rng.standard_t(self.nu, size=(n_paths, horizon))
            draws /= math.sqrt(self.nu / (self.nu - 2.0))
        return draws * sigma

    def to_metadata(self) -> dict[str, Any]:
        """Serializa el modelo a un diccionario JSON-compatible."""
        return {
            "family": self.family,
            "coef": list(self.coef),
            "resid_var": self.resid_var,
            "scale": self.scale,
            "floor": self.floor,
            "nu": self.nu,
            "version": self.version,
            "trained_at": self.trained_at.isoformat(),
            "train_rows": self.train_rows,
            "train_end": self.train_end,
        }


def fit_volatility_model(
    frame: pd.DataFrame,
    family: str,
    min_history_days: int,
    version: str | None = None,
) -> VolatilityModel:
    """Ajusta la familia indicada.

    Raises
    ------
    ModelFitError
        Si la familia no existe o el ajuste falla.
    InsufficientHistoryError
        Si no hay historia suficiente.
    """
    if family == "har":
        return HarModel.fit(frame, min_history_days, version)
    if family in _ARCH_SPECS:
        return ConditionalVolModel.fit(
            frame,
            family,  # type: ignore[arg-type]
            min_history_days,
            version,
        )
    raise ModelFitError(
        f"Familia de volatilidad desconocida: '{family}'. Opciones: "
        f"{', '.join(VOLATILITY_FAMILIES)}."
    )


def model_from_metadata(metadata: dict[str, Any]) -> VolatilityModel:
    """Reconstruye cualquier familia desde ``model_registry``.

    Los registros anteriores a esta versión (sin ``family``) son GARCH(1,1)
    con claves ``omega``, ``alpha``, ``beta``, ``mu`` y ``nu``; se
    convierten a la nomenclatura de ``arch``.

    Raises
    ------
    KeyError, ValueError, TypeError
        Si la metadata está incompleta o corrupta.
    """
    family = metadata.get("family", "garch")
    if family not in VOLATILITY_FAMILIES:
        raise ValueError(f"Familia desconocida en metadata: {family!r}.")
    common = {
        "version": str(metadata["version"]),
        "trained_at": datetime.fromisoformat(metadata["trained_at"]),
        "train_rows": int(metadata["train_rows"]),
        "train_end": str(metadata["train_end"]),
    }
    if family == "har":
        return HarModel(
            coef=tuple(float(c) for c in metadata["coef"]),  # type: ignore[arg-type]
            resid_var=float(metadata["resid_var"]),
            scale=float(metadata["scale"]),
            floor=float(metadata["floor"]),
            nu=float(metadata["nu"]),
            **common,
        )
    if "params" in metadata:
        params = {k: float(v) for k, v in metadata["params"].items()}
    else:
        params = {
            "mu": float(metadata["mu"]),
            "omega": float(metadata["omega"]),
            "alpha[1]": float(metadata["alpha"]),
            "beta[1]": float(metadata["beta"]),
            "nu": float(metadata["nu"]),
        }
    backcast = metadata.get("backcast")
    if backcast is None:
        persistence = params["alpha[1]"] + params["beta[1]"]
        if persistence >= 1.0:
            raise ValueError(f"Persistencia {persistence} >= 1 en metadata.")
        backcast = params["omega"] / (1.0 - persistence)
    return ConditionalVolModel(
        family=family,
        params=params,
        backcast=float(backcast),
        aic=float(metadata.get("aic", float("nan"))),
        bic=float(metadata.get("bic", float("nan"))),
        **common,
    )
