"""Punto de entrada único del sistema de señales de anomalías y volatilidad.

Orquesta, sin ningún input manual y sobre todos los activos configurados en
``src/config.py``, el ciclo completo:

1. Ingesta y validación de datos (Yahoo Finance -> Pandera).
2. Predicción y registro de la señal (``predictions``).
3. Cierre de predicciones pendientes contra el resultado real
   (``outcomes``) y métricas modelo vs baseline (``evaluation_runs``).
4. Disparador de reentrenamiento y, si corresponde, reentrenamiento con
   validación walk-forward y promoción condicionada (``model_registry``).

Este archivo solo orquesta: toda la lógica vive en ``src/``.

Uso::

    python run_pipeline.py                      # ciclo completo, ambas señales
    python run_pipeline.py --signals anomaly    # solo anomalías (intradía)
    python run_pipeline.py --signals garch      # solo GARCH (diario)
    python run_pipeline.py --mode backtest      # walk-forward histórico,
                                                # no escribe en la base

Código de salida: 0 si todas las etapas de todos los activos terminaron
bien; 1 si alguna falló (los fallos ya se notificaron). Un error de
programación no se captura: termina con traceback y código distinto de 0.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime

from src.config import PipelineSettings, get_settings
from src.evaluation.backtest import run_backtest, write_backtest_report
from src.exceptions import PersistenceError
from src.notifications.notifier import notify_failure
from src.persistence import database
from src.pipeline.evaluate_and_retrain import run_evaluations
from src.pipeline.predict import run_predictions
from src.pipeline.results import AssetStageResult, Stage
from src.schemas import SignalType

logger = logging.getLogger("run_pipeline")

_SIGNAL_ALIASES: dict[str, SignalType] = {
    "anomaly": SignalType.ANOMALY,
    "garch": SignalType.GARCH_VOLATILITY_FORECAST,
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Interpreta los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Ciclo automatizado de señales sobre todos los activos."
    )
    parser.add_argument(
        "--mode",
        choices=["cycle", "backtest"],
        default="cycle",
        help="cycle: producción (por defecto). backtest: walk-forward "
        "histórico con reporte en reports/, sin escribir en la base.",
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        choices=sorted(_SIGNAL_ALIASES),
        default=sorted(_SIGNAL_ALIASES),
        help="Señales a procesar (por defecto, ambas).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args(argv)


def _configure_logging(level: str) -> None:
    """Configura el logging una sola vez, en el punto de entrada."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_cycle(
    settings: PipelineSettings,
    signals: tuple[SignalType, ...],
    now: datetime,
) -> list[AssetStageResult]:
    """Ejecuta predicción y evaluación para cada señal solicitada.

    Parameters
    ----------
    settings:
        Configuración validada.
    signals:
        Señales a procesar, en orden.
    now:
        Instante UTC de la corrida (común a todas las etapas).

    Returns
    -------
    list[AssetStageResult]
        Resultados de todas las etapas y activos.
    """
    results: list[AssetStageResult] = []
    for signal in signals:
        logger.info(
            "=== [%s] Etapa 1/2: ingesta + predicción ===", signal.value
        )
        results.extend(run_predictions(settings, signal, now))
        logger.info(
            "=== [%s] Etapa 2/2: resultado real + métricas + disparador ===",
            signal.value,
        )
        results.extend(run_evaluations(settings, signal, now))
    return results


def summarize(
    results: list[AssetStageResult], n_assets: int
) -> tuple[str, bool]:
    """Construye el resumen final y dice si hubo algún fallo.

    Returns
    -------
    tuple[str, bool]
        Texto del resumen y ``True`` si alguna etapa falló.
    """
    lines = ["", "=" * 70, "RESUMEN DE LA CORRIDA", "=" * 70]
    lines.append(f"Activos configurados: {n_assets}")
    for signal in dict.fromkeys(r.signal_type for r in results):
        predicted = [
            r
            for r in results
            if r.signal_type is signal and r.stage is Stage.PREDICT
        ]
        evaluated = [
            r
            for r in results
            if r.signal_type is signal and r.stage is Stage.EVALUATE
        ]
        retrained = [r for r in evaluated if r.retrain_promoted is not None]
        lines += [
            f"[{signal.value}]",
            f"  Predicción: {sum(r.succeeded for r in predicted)}/"
            f"{len(predicted)} activos OK; "
            f"{sum(r.prediction_id is not None for r in predicted)} "
            f"predicciones nuevas; "
            f"{sum(r.cold_start for r in predicted)} arranques en frío.",
            f"  Evaluación: {sum(r.succeeded for r in evaluated)}/"
            f"{len(evaluated)} activos OK; "
            f"{sum(r.outcomes_closed for r in evaluated)} resultados "
            "comparados contra el valor real.",
            f"  Reentrenamiento: {len(retrained)} ejecutados "
            f"({sum(bool(r.retrain_promoted) for r in retrained)} "
            f"promovidos, "
            f"{sum(not r.retrain_promoted for r in retrained)} rechazados "
            "por no superar al baseline).",
        ]
        lines += [
            f"    - {r.asset_symbol}: {r.retrain_decision.triggered_by} -> "
            f"{'promovido' if r.retrain_promoted else 'no promovido'}"
            for r in retrained
            if r.retrain_decision is not None
        ]

    failures = [r for r in results if not r.succeeded]
    lines.append("Fallos:" if failures else "Fallos: ninguno.")
    lines += [
        f"  - {r.stage.value}:{r.signal_type.value}:{r.asset_symbol} -> "
        f"{r.detail}"
        for r in failures
    ]
    lines.append("=" * 70)
    return "\n".join(lines), bool(failures)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de línea de comandos.

    Parameters
    ----------
    argv:
        Argumentos; si es ``None`` se usa ``sys.argv``.

    Returns
    -------
    int
        0 si todo terminó bien, 1 si alguna etapa o activo falló.
    """
    args = _parse_args(argv)
    _configure_logging(args.log_level)
    settings = get_settings()
    signals = tuple(_SIGNAL_ALIASES[name] for name in args.signals)
    now = datetime.now(UTC)
    logger.info(
        "Inicio | modo=%s | señales=%s | activos=%s",
        args.mode,
        ", ".join(s.value for s in signals),
        ", ".join(a.symbol for a in settings.assets),
    )

    if args.mode == "backtest":
        results = run_backtest(settings, signals, now)
        report_path = write_backtest_report(results, settings, now)
        logger.info("Reporte de backtest escrito en %s", report_path)
        return 1 if any(item.errors for item in results) else 0

    try:
        database.init_db(settings.paths.database_path)
    except PersistenceError as exc:
        notify_failure("init_db", exc)
        return 1

    results = run_cycle(settings, signals, now)
    summary, has_failures = summarize(results, len(settings.assets))
    logger.info(summary)
    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
