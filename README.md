# Señales automatizadas de anomalías y volatilidad

Sistema que vigila 7 activos (S&P 500, NASDAQ 100, Dow Jones 30, EUR/USD,
GBP/USD, USD/JPY, USD/CAD) sin intervención humana: detecta movimientos de
precio estadísticamente inusuales, pronostica la volatilidad del día
siguiente, **registra cada predicción, la compara después contra lo que
realmente pasó, mide su propio acierto frente a un baseline ingenuo y se
reentrena solo** cuando su desempeño lo justifica.

> No es una estrategia de trading ni una recomendación de inversión: es un
> indicador estadístico de condiciones de mercado.

## Qué hace, en una frase por señal

| Señal | Datos | Pregunta que responde | Se compara contra |
|---|---|---|---|
| Anomalía | Velas de 5 min (ya cerradas) | ¿El último movimiento es inusual y suele revertirse? | Marcar anomalías al azar con la misma frecuencia |
| Volatilidad GARCH(1,1)-t | Cierres diarios desde 2010 | ¿Cuánto se moverá el activo mañana? | Desviación estándar de los últimos 20 días |

## Arquitectura

```
run_pipeline.py                    punto de entrada único (solo orquesta)
  └─ por cada señal y cada activo:
     1. src/data            ingesta yfinance (reintentos) + contratos Pandera
     2. src/features        retornos log causales (sin fuga, sin saltos de sesión)
     3. src/models          detector / GARCH  ->  src/persistence (predictions)
     4. src/pipeline/evaluate_and_retrain.py
          resultado real -> pérdida modelo y baseline (outcomes)
          métrica agregada F1 / QLIKE vs baseline      (evaluation_runs)
          src/evaluation/feedback_loop.py  -> ¿reentrenar?
          src/pipeline/retrain.py  walk-forward (TimeSeriesSplit) -> promover
                                   solo si supera al baseline (model_registry)
```

Todo el historial vive en `data/pipeline.db` (SQLite), que GitHub Actions
actualiza y versiona en cada corrida: cualquier decisión del sistema se
puede auditar consultando qué modelo predijo qué, qué pasó después y por
qué se reentrenó (o no).

### Disparador de reentrenamiento

1. **Enfriamiento de 24 h** entre intentos (evita bucles si un candidato no
   se promueve).
2. **Cadencia fija**: 30 días (anomalías) / 7 días (GARCH).
3. **Degradación sostenida**: 3 corridas seguidas con el modelo activo peor
   que su baseline (F1 sobre los últimos 500 resultados / QLIKE sobre los
   últimos 60), con muestra mínima exigida.
4. **Drift**: ADWIN sobre la pérdida punto a punto del modelo activo.

Un candidato solo reemplaza al modelo activo si supera al baseline en una
validación walk-forward (`TimeSeriesSplit`, nunca split aleatorio; con
`gap` igual al horizonte de la etiqueta en anomalías).

## Cómo ejecutarlo

Requisitos: Python 3.11+ (probado localmente en 3.14; CI usa 3.12). No hace falta ninguna
API key: `yfinance` es pública.

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt
pytest -q                              # 69 pruebas, sin red
python run_pipeline.py                 # ciclo completo, todos los activos
python run_pipeline.py --mode backtest # walk-forward histórico -> reports/
```

| Comando | Qué hace | Escribe en la base |
|---|---|---|
| `python run_pipeline.py` | Predice, evalúa y decide reentrenar (ambas señales) | Sí |
| `python run_pipeline.py --signals anomaly` | Solo anomalías (lo que corre cada 15 min) | Sí |
| `python run_pipeline.py --signals garch` | Solo GARCH (lo que corre al cierre) | Sí |
| `python run_pipeline.py --mode backtest` | Valida ambos modelos contra baseline con datos históricos y genera `reports/backtest_<fecha>/backtest.md` + QQ-plots | No |

Código de salida: `0` si todo terminó bien, `1` si algún activo o etapa
falló (el fallo se registra y, si configuras
`PIPELINE_ALERT_WEBHOOK_URL`, se notifica; ver `.env.example`).

## Resultados (backtest walk-forward del 2026-09-24)

Reporte completo con matrices de confusión y QQ-plots en
`reports/backtest_20260924T155019Z/backtest.md`.

| Señal | Métrica | Modelo | Baseline | Lectura |
|---|---|---|---|---|
| Volatilidad GARCH(1,1)-t | QLIKE (menor es mejor) | Menor en **7/7** activos | Desviación móvil de 20 días | El GARCH pronostica mejor la volatilidad que el promedio móvil en todos los activos |
| Detector de anomalías | F1 macro (7 activos) | 0.085 | 0.084 (azar, misma tasa) | **Sin ventaja estadística**: los intervalos de confianza de la precisión se traslapan con el azar |

Hallazgos adicionales: los residuos del GARCH tienen colas pesadas
(Jarque-Bera p < 0.001, esperado y cubierto por la distribución t) y en
GBP/USD y USD/CAD quedan efectos ARCH sin modelar (Ljung-Box sobre z²,
p < 0.05). La conclusión honesta es que el componente con valor predictivo
demostrado es el de volatilidad; el detector de anomalías funciona como
filtro de atención, no como predictor de reversiones.

## Automatización

- `.github/workflows/pipeline.yml`: corre `run_pipeline.py` cada 15 min en
  horario de mercado (anomalías) y a las 22:00 UTC (GARCH), y versiona
  `data/pipeline.db`. GitHub envía un correo si el job falla.
- `.github/workflows/ci.yml`: en cada push, Ruff (PEP8 a 79 columnas, type
  hints, docstrings, sin `except` genérico) y pytest.

## Consultar el historial

```sql
-- Últimas métricas del modelo activo vs baseline, por activo
SELECT asset_symbol, signal_type, evaluated_at, n_outcomes,
       metric_name, model_metric, baseline_metric
FROM evaluation_runs
ORDER BY evaluated_at DESC
LIMIT 20;

-- Historial de modelos y por qué se promovieron o no
SELECT asset_symbol, signal_type, version, stage, created_at,
       validation_metrics_json
FROM model_registry
ORDER BY created_at DESC;
```

## Limitaciones conocidas

- Yahoo Finance (vía `yfinance`) es una fuente no oficial y gratuita: puede
  limitar peticiones o cambiar su formato. El sistema falla de forma
  explícita en ese caso, pero no lo evita.
- Las velas de 5 minutos solo están disponibles para los últimos 60 días,
  así que la validación del detector usa esa ventana.
- El horario del cron sigue el mercado de EE. UU.; en forex (24/5) no se
  evalúan las horas fuera de ese horario.
- SQLite versionado en el repo alcanza para un solo escritor secuencial;
  con escritores concurrentes habría que migrar `src/persistence/database.py`
  a Postgres.

## Problemas comunes al ejecutarlo en Windows

| Síntoma | Causa | Solución |
|---|---|---|
| `WinError 206` o error al cargar una DLL (Pillow, matplotlib, scikit-learn) | La ruta de la carpeta supera el límite de 260 caracteres de Windows | Mueve el proyecto a una ruta corta, por ejemplo `C:\dev\trading_signals` |
| `curl: (60) SSL certificate ... unable to get local issuer certificate`, o `yfinance` devuelve datos vacíos | Un antivirus con escaneo HTTPS (por ejemplo Avast Web Shield) firma los certificados con su propia raíz | Agrega `yahoo.com` como excepción del escaneo HTTPS del antivirus |

Ninguno de los dos afecta la ejecución en GitHub Actions (Linux, sin
antivirus ni límite de ruta).

## Notebook de exploración

`notebooks/01_eda_original.ipynb` es el EDA con el que empezó el proyecto
(GARCH sobre SPY con pruebas ADF, Ljung-Box y ARCH-LM, y el menú
interactivo original). Se conserva como referencia; no forma parte de
producción y no se valida con Ruff.

## Estructura

```
run_pipeline.py              punto de entrada único
src/                         código de producción (config, data, features,
                             models, persistence, evaluation, pipeline,
                             notifications)
tests/                       69 pruebas pytest (contratos, fuga, métricas,
                             disparador, integración, entry point)
data/pipeline.db             historial auditable (lo crea el pipeline)
notebooks/                   EDA original (referencia)
reports/                     reportes de backtest
.github/workflows/           pipeline programado + CI
.vscode/                     configuración de ejecución y depuración
```
