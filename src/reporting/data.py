"""Contenido del reporte: todo número sale del backtest o de la base.

Los hallazgos del resumen ejecutivo se redactan a partir de las cifras
calculadas aquí (conteos, p-valores, rangos); no hay texto escrito a mano
que pueda quedar desactualizado respecto a los datos.
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.config import PipelineSettings
from src.evaluation.backtest import AssetBacktest
from src.evaluation.metrics import (
    DieboldMarianoResult,
    PrecisionLiftResult,
    RegressionReport,
    diebold_mariano,
    evaluate_volatility_forecast,
    precision_lift_test,
)
from src.evaluation.var_backtest import VarBacktestResult, var_backtest
from src.evaluation.volatility_selection import (
    VAR_LEVELS,
    ChampionDecision,
    VolatilityWalkForward,
)
from src.exceptions import PersistenceError, TradingPipelineError
from src.models.scenarios import price_quantiles, tail_risk
from src.models.volatility import fit_volatility_model, volatility_frame
from src.persistence import database
from src.reporting.style import FAMILY_LABELS, format_p
from src.schemas import PriceQuantiles

logger = logging.getLogger(__name__)

SOURCE = "Yahoo Finance vía yfinance"
_FAN_HISTORY_DAYS = 60


@dataclass(frozen=True)
class ModelRow:
    """Un modelo o baseline frente al campeón, sobre la muestra común."""

    name: str
    mean_qlike: float
    vs_champion: DieboldMarianoResult | None  # None para el propio campeón


@dataclass(frozen=True)
class VolatilitySection:
    """Validación y pronóstico de volatilidad de un activo."""

    champion: str
    narrative: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_common: int
    rows: tuple[ModelRow, ...]
    calibration: RegressionReport
    residual_nu: float
    var: dict[float, VarBacktestResult]
    origin: pd.Timestamp
    last_close: float
    forecast_vol_pct: float
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    scenarios_mc: tuple[PriceQuantiles, ...]
    scenarios_fhs: tuple[PriceQuantiles, ...]
    series: pd.DataFrame  # walk-forward del campeón, para gráficas
    close_tail: pd.Series


@dataclass(frozen=True)
class AnomalySection:
    """Desempeño del detector de anomalías de un activo."""

    lift: PrecisionLiftResult
    lift_ci: tuple[float, float]
    p_bonferroni: float
    auc: float
    n_bars: int
    profile: tuple[float, ...]
    active_hours: tuple[int, ...]  # horas UTC con velas en la muestra


@dataclass(frozen=True)
class AssetSection:
    """Todo lo que el reporte muestra de un activo."""

    symbol: str
    name: str
    volatility: VolatilitySection | None
    anomaly: AnomalySection | None
    n_repaired_bars: int
    n_daily_bars: int
    zero_return_share: float | None
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ReportData:
    """Contenido completo del reporte."""

    generated_at: datetime
    assets: tuple[AssetSection, ...]
    findings: tuple[str, ...]
    live_runs: tuple[dict[str, Any], ...]
    methodology: dict[str, str]


def _residual_nu(standardized: np.ndarray) -> float:
    """Grados de libertad de la t ajustada a los residuos estandarizados."""
    nu, _, _ = stats.t.fit(standardized, floc=0.0)
    return float(nu)


def _narrative(wf: VolatilityWalkForward, decision: ChampionDecision) -> str:
    """Explica en lenguaje claro la decisión y la muestra comparada."""
    champion = FAMILY_LABELS[decision.champion]
    previous = FAMILY_LABELS[decision.previous]
    if decision.switched and decision.comparisons:
        best = next(
            c
            for c in decision.comparisons
            if c.challenger == decision.champion
        )
        text = (
            f"{champion} reemplaza a {previous}: le gana con "
            f"Diebold-Mariano t = {best.dm.t_stat:.2f}, p ajustado por "
            f"Holm {format_p(best.p_value_holm)}."
        )
    elif decision.switched:
        text = f"{previous} no produjo pronósticos; campeón: {champion}."
    else:
        text = (
            f"{champion} se mantiene como campeón: ningún retador le gana "
            "con p ajustado por Holm < 0.05."
        )
    max_folds = max(wf.n_folds.values())
    partial = [
        f"{FAMILY_LABELS[f]} ajustó {n} de {max_folds} bloques"
        for f, n in wf.n_folds.items()
        if n < max_folds
    ]
    if partial:
        text += (
            f" {'; '.join(partial)}: las comparaciones usan los "
            f"{int(wf.common_mask.sum()):,} días comunes a todos los modelos."
        )
    return text


def _volatility_section(
    item: AssetBacktest, settings: PipelineSettings
) -> VolatilitySection:
    """Comparación de modelos, calibración, VaR y pronóstico vigente."""
    wf, decision = item.volatility, item.champion
    champion = decision.champion
    common = wf.common_mask
    champion_loss = wf.qlike(champion)
    names = (*wf.forecasts, "ewma", "sd20")
    rows = tuple(
        ModelRow(
            name=name,
            mean_qlike=wf.mean_qlike(name),
            vs_champion=None
            if name == champion
            else diebold_mariano(
                champion_loss[common], wf.qlike(name)[common]
            ),
        )
        for name in names
    )

    sigma = wf.volatility(champion)
    own = np.isfinite(sigma) & (sigma > 0.0)
    realized = wf.realized_pct[own]
    calibration = evaluate_volatility_forecast(
        sigma[own], realized, wf.baselines["sd20"][own]
    )
    var = {
        level: var_backtest(
            realized, wf.value_at_risk(champion, level)[own], level
        )
        for level in VAR_LEVELS
    }

    frame = volatility_frame(item.ohlc)
    model = fit_volatility_model(
        frame, champion, settings.garch.min_history_days
    )
    origin = frame.index[-1]
    seed = zlib.crc32(f"{item.asset.symbol}|{origin}".encode())
    horizon = settings.garch.forecast_horizon_days
    n_paths = settings.garch.n_monte_carlo_paths
    monte_carlo = model.simulate_returns(frame, horizon, n_paths, seed)
    fhs = model.simulate_returns(frame, horizon, n_paths, seed, method="fhs")
    last_close = float(item.ohlc["close"].iloc[-1])
    var_95, cvar_95 = tail_risk(monte_carlo, 0.05)
    var_99, cvar_99 = tail_risk(monte_carlo, 0.01)

    dates = pd.DatetimeIndex(wf.dates)
    series = pd.DataFrame(
        {
            "realized": wf.realized_pct,
            "realized_vol": np.sqrt(wf.rv_proxy_pct2),
            "champion": sigma,
            "ewma": wf.baselines["ewma"],
            "sd20": wf.baselines["sd20"],
            "var_01": wf.value_at_risk(champion, 0.01),
            "var_05": wf.value_at_risk(champion, 0.05),
            "qlike_champion": champion_loss,
            "qlike_ewma": wf.qlike("ewma"),
            "qlike_sd20": wf.qlike("sd20"),
            "common": common,
        },
        index=dates,
    )
    return VolatilitySection(
        champion=champion,
        narrative=_narrative(wf, decision),
        start=dates[0],
        end=dates[-1],
        n_common=int(common.sum()),
        rows=rows,
        calibration=calibration,
        residual_nu=_residual_nu(realized / sigma[own]),
        var=var,
        origin=origin,
        last_close=last_close,
        forecast_vol_pct=float(np.sqrt(model.forecast_variance(frame, 1)[0])),
        var_95=var_95,
        cvar_95=cvar_95,
        var_99=var_99,
        cvar_99=cvar_99,
        scenarios_mc=tuple(price_quantiles(last_close, monte_carlo)),
        scenarios_fhs=tuple(price_quantiles(last_close, fhs)),
        series=series,
        close_tail=item.ohlc["close"].iloc[-_FAN_HISTORY_DAYS:],
    )


def _anomaly_section(item: AssetBacktest, n_tests: int) -> AnomalySection:
    """Lift de precisión con IC, Bonferroni y AUC del detector."""
    result = item.anomaly
    lift = precision_lift_test(result.pooled)
    low, high = result.pooled.precision_ci_95
    labeled = result.labeled
    return AnomalySection(
        lift=lift,
        lift_ci=(low / lift.prevalence, high / lift.prevalence),
        p_bonferroni=min(1.0, lift.p_value * n_tests),
        auc=float(roc_auc_score(labeled["reverted"], labeled["zscore"].abs())),
        n_bars=len(labeled),
        profile=result.seasonal_profile,
        active_hours=tuple(sorted(set(labeled.index.hour))),
    )


def _asset_section(
    item: AssetBacktest, settings: PipelineSettings, n_tests: int
) -> AssetSection:
    errors = list(item.errors)
    volatility = anomaly = None
    if item.volatility is not None and item.champion is not None:
        try:
            volatility = _volatility_section(item, settings)
        except (TradingPipelineError, ValueError) as exc:
            errors.append(f"reporte de volatilidad: {exc}")
    if item.anomaly is not None:
        try:
            anomaly = _anomaly_section(item, n_tests)
        except ValueError as exc:
            errors.append(f"reporte de anomalías: {exc}")
    return AssetSection(
        symbol=item.asset.symbol,
        name=item.asset.display_name,
        volatility=volatility,
        anomaly=anomaly,
        n_repaired_bars=item.n_repaired_bars,
        n_daily_bars=0 if item.ohlc is None else len(item.ohlc),
        zero_return_share=item.zero_return_share,
        errors=tuple(errors),
    )


def _names(sections: list[AssetSection]) -> str:
    return ", ".join(s.name for s in sections) if sections else "ninguno"


def build_findings(
    assets: tuple[AssetSection, ...], alpha: float
) -> tuple[str, ...]:
    """Cinco hallazgos redactados a partir de las cifras del reporte."""
    vol = [a for a in assets if a.volatility]
    anom = [a for a in assets if a.anomaly]
    findings: list[str] = []

    by_champion: dict[str, list[AssetSection]] = {}
    for a in vol:
        by_champion.setdefault(a.volatility.champion, []).append(a)
    parts = [
        f"{FAMILY_LABELS[c]} en {len(v)} de {len(vol)} ({_names(v)})"
        for c, v in sorted(by_champion.items(), key=lambda kv: -len(kv[1]))
    ]
    findings.append(
        "Modelo campeón de volatilidad por activo: " + "; ".join(parts) + "."
    )

    beats_ewma = [
        a
        for a in vol
        for row in a.volatility.rows
        if row.name == "ewma"
        and row.vs_champion is not None
        and row.vs_champion.p_value_one_sided < alpha
    ]
    findings.append(
        f"El campeón supera al baseline EWMA con significancia "
        f"(Diebold-Mariano, p < {alpha}) en {len(beats_ewma)} de "
        f"{len(vol)} activos: {_names(beats_ewma)}."
    )

    covered = [
        a for a in vol if a.volatility.var[0.01].p_unconditional >= 0.05
    ]
    clustered = [
        a for a in vol if a.volatility.var[0.01].p_independence < 0.05
    ]
    findings.append(
        f"VaR 99% filtrado: la cobertura no se rechaza (Kupiec p ≥ 0.05) "
        f"en {len(covered)} de {len(vol)} activos; hay agrupamiento de "
        f"excepciones (Christoffersen p < 0.05) en: {_names(clustered)}."
    )

    significant = [a for a in anom if a.anomaly.p_bonferroni < alpha]
    if anom:
        aucs = [a.anomaly.auc for a in anom]
        findings.append(
            f"Detector de anomalías: precisión significativamente mayor a "
            f"la prevalencia tras Bonferroni en {len(significant)} de "
            f"{len(anom)} activos ({_names(significant)}); AUC de reversión "
            f"entre {min(aucs):.3f} y {max(aucs):.3f} (0.5 = azar)."
        )

    stale = max(
        (a for a in assets if a.zero_return_share is not None),
        key=lambda a: a.zero_return_share,
        default=None,
    )
    repaired = sum(a.n_repaired_bars for a in assets)
    total = sum(a.n_daily_bars for a in assets)
    quality = (
        "Calidad de datos: en FX el cierre diario de Yahoo corresponde al "
        "inicio del día y se re-alineó al día que describe (sin esto el "
        "rango del día filtraba el retorno a pronosticar); además, "
        f"{repaired:,} de {total:,} velas diarias requirieron reparación "
        f"OHLC ({repaired / max(total, 1):.2%})"
    )
    if stale is not None:
        quality += (
            f"; la serie de 5 min más rancia es {stale.name}, con "
            f"{stale.zero_return_share:.1%} de retornos exactamente cero"
        )
    findings.append(quality + ".")
    return tuple(findings)


def _live_runs(db_path: Path) -> tuple[dict[str, Any], ...]:
    """Última corrida de evaluación en producción por activo y señal."""
    if not db_path.exists():
        return ()
    try:
        return tuple(database.fetch_latest_evaluation_runs(db_path))
    except PersistenceError as exc:
        logger.warning("No se pudo leer el monitoreo en producción: %s", exc)
        return ()


def build_report_data(
    results: list[AssetBacktest],
    settings: PipelineSettings,
    generated_at: datetime,
) -> ReportData:
    """Arma el contenido completo del reporte.

    Parameters
    ----------
    results:
        Salida de ``run_backtest``.
    settings:
        Configuración usada en el backtest.
    generated_at:
        Instante UTC del reporte.

    Returns
    -------
    ReportData
        Secciones por activo, hallazgos, monitoreo y metodología.
    """
    n_tests = sum(r.anomaly is not None for r in results) or 1
    assets = tuple(_asset_section(r, settings, n_tests) for r in results)
    alpha = settings.retrain.promotion_alpha
    methodology = {
        "Fuente": SOURCE,
        "Historia diaria desde": settings.garch.history_start_date,
        "Historia intradía": (
            settings.anomaly.history_period.replace("d", " días")
            + f" de velas de {settings.anomaly.bar_minutes} min"
        ),
        "Validación": (
            f"walk-forward TimeSeriesSplit de "
            f"{settings.retrain.walk_forward_splits} bloques, ventana "
            "expansiva; parámetros estimados solo con el bloque de "
            "entrenamiento"
        ),
        "Modelos candidatos": ", ".join(
            FAMILY_LABELS[f] for f in settings.garch.candidate_families
        ),
        "Baselines": "EWMA λ=0.94 (RiskMetrics) y desviación estándar móvil "
        "de 20 días",
        "Pérdida de selección": (
            "QLIKE contra r² (robusta a proxies ruidosos, Patton 2011)"
        ),
        "Regla de campeón": (
            "el vigente se mantiene salvo que un retador gane con "
            f"Diebold-Mariano (HAC) y p ajustado por Holm < {alpha}"
        ),
        "Comparaciones": "sobre la muestra común a todos los modelos",
        "VaR": (
            "filtrado: cuantil empírico de residuos estandarizados de "
            "entrenamiento por σ pronosticada; Kupiec y Christoffersen"
        ),
        "Escenarios": (
            f"{settings.garch.n_monte_carlo_paths:,} trayectorias a "
            f"{settings.garch.forecast_horizon_days} días, Monte Carlo "
            "paramétrico y simulación histórica filtrada"
        ),
        "Detector": (
            "z-score causal desestacionalizado por hora UTC; positivo = "
            f"reversión en {settings.anomaly.reversion_horizon_bars} velas; "
            "lift = precisión / prevalencia, Bonferroni por activos"
        ),
    }
    return ReportData(
        generated_at=generated_at,
        assets=assets,
        findings=build_findings(assets, alpha),
        live_runs=_live_runs(settings.paths.database_path),
        methodology=methodology,
    )
