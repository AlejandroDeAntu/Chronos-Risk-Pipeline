"""Configuración central del pipeline, validada con Pydantic.

Todo parámetro operativo (activos, ventanas, umbrales, disparadores de
reentrenamiento, rutas) vive aquí como un contrato tipado en vez de estar
disperso como literales mágicos en el código, como ocurría en el proyecto
original (``ticker = "SPY"`` hardcodeado, ``window = 20`` suelto en la
función).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AssetClass(StrEnum):
    """Clase de activo del universo monitoreado."""

    EQUITY_INDEX = "equity_index"
    FX = "fx"


class AssetConfig(BaseModel):
    """Contrato de un activo monitoreado por el pipeline.

    Attributes
    ----------
    symbol:
        Ticker tal como lo espera ``yfinance`` (ej. ``"^GSPC"``).
    display_name:
        Nombre legible para reportes y notificaciones.
    asset_class:
        Índice bursátil o par de forex.
    """

    model_config = {"frozen": True}

    symbol: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    asset_class: AssetClass

    @field_validator("symbol")
    @classmethod
    def _symbol_no_whitespace(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError(f"El símbolo '{value}' tiene espacios sobrantes.")
        return value


class AnomalyDetectorConfig(BaseModel):
    """Hiperparámetros del detector de anomalías (z-score causal)."""

    model_config = {"frozen": True}

    rolling_window: int = Field(
        default=20, ge=5, description="Barras para media/std móviles."
    )
    target_anomaly_rate: float = Field(
        default=0.0455,
        gt=0.0,
        lt=0.5,
        description=(
            "Tasa objetivo de anomalías para calibrar el umbral "
            "(equivalente a +-2 sigma bajo normalidad)."
        ),
    )
    reversion_horizon_bars: int = Field(
        default=10,
        ge=1,
        description="Barras hacia adelante del ground truth de reversión.",
    )
    bar_minutes: int = Field(default=5, ge=1)
    predict_period: str = Field(default="5d")
    history_period: str = Field(
        default="60d",
        description="Yahoo Finance limita las velas de 5m a 60 días.",
    )
    min_bars_required: int = Field(default=60, ge=20)

    @property
    def interval(self) -> str:
        """Intervalo de vela en el formato que espera ``yfinance``."""
        return f"{self.bar_minutes}m"


class GarchModelConfig(BaseModel):
    """Hiperparámetros del modelo GARCH(1,1)-t de volatilidad diaria."""

    model_config = {"frozen": True}

    history_start_date: str = Field(default="2010-01-01")
    forecast_horizon_days: int = Field(default=5, ge=1)
    min_history_days: int = Field(default=300, ge=100)
    n_monte_carlo_paths: int = Field(default=10_000, ge=500)
    baseline_trailing_window: int = Field(
        default=20,
        ge=5,
        description="Ventana del baseline de volatilidad histórica móvil.",
    )


class RetrainTriggerConfig(BaseModel):
    """Disparadores de reentrenamiento (ver ``feedback_loop.py``)."""

    model_config = {"frozen": True}

    fixed_cadence_days_anomaly: int = Field(default=30, ge=1)
    fixed_cadence_days_garch: int = Field(default=7, ge=1)
    degradation_window_anomaly: int = Field(
        default=500,
        ge=50,
        description=(
            "Resultados recientes (del modelo activo) sobre los que se "
            "calcula F1 vs baseline en cada corrida de evaluación."
        ),
    )
    degradation_window_garch: int = Field(
        default=60,
        ge=20,
        description="Resultados recientes para QLIKE vs baseline.",
    )
    min_predicted_positives: int = Field(
        default=10,
        ge=1,
        description=(
            "Mínimo de anomalías predichas en la ventana para que F1 sea "
            "interpretable; con menos, la corrida se marca insuficiente."
        ),
    )
    min_outcomes_garch: int = Field(
        default=20,
        ge=10,
        description="Mínimo de resultados para que QLIKE sea interpretable.",
    )
    degradation_consecutive_windows: int = Field(
        default=3,
        ge=1,
        description="Corridas consecutivas peor que baseline para disparar.",
    )
    retrain_cooldown_hours: int = Field(
        default=24,
        ge=0,
        description=(
            "Horas mínimas entre intentos de reentrenamiento del mismo "
            "activo/señal, para evitar bucles si el candidato no se promueve."
        ),
    )
    walk_forward_splits: int = Field(default=4, ge=2)
    drift_detection_enabled: bool = Field(default=True)


class PathsConfig(BaseModel):
    """Rutas de artefactos persistentes del pipeline.

    El registro de modelos vive dentro de la base SQLite (tabla
    ``model_registry``, como JSON): para 7 activos y dos familias de modelo
    no hace falta un almacén de artefactos separado.
    """

    model_config = {"frozen": True}

    database_path: Path = Field(default=_PROJECT_ROOT / "data" / "pipeline.db")
    reports_dir: Path = Field(default=_PROJECT_ROOT / "reports")

    @field_validator("database_path")
    @classmethod
    def _ensure_parent_exists(cls, value: Path) -> Path:
        value.parent.mkdir(parents=True, exist_ok=True)
        return value


class PipelineSettings(BaseModel):
    """Configuración raíz inyectada a todos los procesos del pipeline."""

    model_config = {"frozen": True}

    assets: tuple[AssetConfig, ...]
    anomaly: AnomalyDetectorConfig = Field(
        default_factory=AnomalyDetectorConfig
    )
    garch: GarchModelConfig = Field(default_factory=GarchModelConfig)
    retrain: RetrainTriggerConfig = Field(default_factory=RetrainTriggerConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @field_validator("assets")
    @classmethod
    def _unique_symbols(
        cls, value: tuple[AssetConfig, ...]
    ) -> tuple[AssetConfig, ...]:
        symbols = [asset.symbol for asset in value]
        duplicates = {s for s in symbols if symbols.count(s) > 1}
        if duplicates:
            raise ValueError(
                f"Símbolos duplicados: {sorted(duplicates)}."
                " El menú original tenía '^NDX' repetido (opciones 2 y 4); "
                "aquí se rechaza explícitamente."
            )
        if not value:
            raise ValueError("Se requiere al menos un activo configurado.")
        return value


# Los 7 activos únicos del menú original (se eliminó la entrada duplicada
# "US100 (Nasdaq 100)", que apuntaba también a "^NDX").
DEFAULT_ASSETS: tuple[AssetConfig, ...] = (
    AssetConfig(
        symbol="^GSPC",
        display_name="S&P 500",
        asset_class=AssetClass.EQUITY_INDEX,
    ),
    AssetConfig(
        symbol="^NDX",
        display_name="NASDAQ 100",
        asset_class=AssetClass.EQUITY_INDEX,
    ),
    AssetConfig(
        symbol="^DJI",
        display_name="Dow Jones 30",
        asset_class=AssetClass.EQUITY_INDEX,
    ),
    AssetConfig(
        symbol="EURUSD=X", display_name="EUR/USD", asset_class=AssetClass.FX
    ),
    AssetConfig(
        symbol="GBPUSD=X", display_name="GBP/USD", asset_class=AssetClass.FX
    ),
    AssetConfig(
        symbol="JPY=X", display_name="USD/JPY", asset_class=AssetClass.FX
    ),
    AssetConfig(
        symbol="CAD=X", display_name="USD/CAD", asset_class=AssetClass.FX
    ),
)


def get_settings() -> PipelineSettings:
    """Construye la configuración activa del pipeline.

    Returns
    -------
    PipelineSettings
        Configuración validada con el universo de activos por defecto.
    """
    return PipelineSettings(assets=DEFAULT_ASSETS)
