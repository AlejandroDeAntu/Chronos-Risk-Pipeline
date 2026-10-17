# Auditoría técnica: Chronos-Risk-Pipeline

> **Estado:** los quick wins QW1–QW5 de la sección 7, el cambio de workflow
> (`pull --rebase` y cron GARCH a 00:30 UTC mar–sáb) y los lockfiles
> (`requirements.lock`, `requirements-dev.lock`) ya están aplicados en este
> repositorio. El texto de abajo describe el estado auditado (`fa1d251`).

**Repositorio:** `AlejandroDeAntu/Chronos-Risk-Pipeline` (rama `main`, HEAD `fa1d251`, 61 commits del 2026-09-24 al 2026-10-17)
**Fecha de auditoría:** 2026-10-17 · **Entorno de ejecución:** Windows 11, Python 3.14.4, pandas 3.0.6, numpy 2.5.4, arch 8.0.0, scikit-learn 1.9.1, statsmodels 0.15.0, yfinance 1.7.0
**Fuentes de cada cifra:** (R) = existe en el repo (README, `docs/`, `reports/`, `data/pipeline.db`); (E) = ejecución real mía en esta auditoría; (S) = supuesto o estimación, siempre señalado.

---

## 1. Exploración

### 1.1 Inventario

| Componente | Contenido | Estado |
|---|---|---|
| `src/` | 5,487 LOC entre `src/`, `tests/` y `run_pipeline.py`; capas `data → features → models → persistence → evaluation → pipeline → notifications` | Bien estratificado, ningún archivo > 450 LOC |
| `tests/` | 69 tests pytest, fixtures + `monkeypatch` de `_download_raw` (sin red) | **69/69 pasan (E)**, cobertura de línea **92% (E)** |
| CI | `ci.yml`: `ruff check`, `ruff format --check`, `pytest` en py3.12 | Ruff limpio (E) |
| Orquestación | `pipeline.yml`: cron `*/15 13-21 * * 1-5` (anomalías) y `0 22 * * 1-5` (GARCH), commitea `data/pipeline.db` | Ver hallazgo C-1 |
| Datos | `data/pipeline.db` (SQLite, 479 KB): 337 predicciones, 322 outcomes, 293 `evaluation_runs`, 35 versiones en `model_registry` (R) | |
| Notebook | `01_eda_original.ipynb`: EDA original (GARCH sobre SPY + detector con menú). **Salidas limpiadas: no contiene métricas** (R) | |
| Reportes | `reports/backtest_20260924T155019Z/` con tablas y 7 QQ-plots (R) | |
| Dependencias | `requirements.txt` con rangos `>=`, sin lockfile | Ver hallazgo M-3 |

### 1.2 Historial de commits

- **2026-09-24 (7 commits, `99343e0`→`abe0c4e`)**: todo el código, tests, CI, README, notebook, backtest y reporte final entran el mismo día. No hay historia incremental de modelado en git: la evolución v0→v1 solo está narrada en `docs/reporte_final.md`.
- **2026-09-24 → 2026-10-17 (54 commits `data: ciclo …`)**: commits automáticos del bot con la base actualizada. Son la única "historia de modelo" posterior, vía `model_registry`.

---

## 2. Historia del modelo

| Fecha / commit | Versión | Cambio | Métricas (fuente) |
|---|---|---|---|
| Antes de 2026-09-24 (notebook) | v0 | GARCH(1,1)-t sobre SPY desde 2015; detector z-score con umbral fijo 2σ, ventana 20 **sin `shift(1)`** (fuga), menú manual, sin baseline | **No hay métricas en el repo** (salidas del notebook limpiadas) |
| 2026-09-24 `cc1e363` | v1 código | Umbral por cuantil empírico (tasa 4.55%), z-score causal, descarte de saltos de sesión y velas abiertas, GARCH generalizado a 7 activos, walk-forward `TimeSeriesSplit(4)` con `gap=10` en anomalías, baselines | — |
| 2026-09-24 15:50 `58edd4d` | Backtest v1 | Primer walk-forward | F1 macro **0.085 vs 0.084** baseline; GARCH QLIKE < baseline en **7/7** (R) |
| 2026-09-24 21:51 `34a93e5` | Detector `2026092…Z` ×7 | Arranque en frío, `active` sin validación | `{"note": "arranque en frío"}` (R). **Nunca se ha reentrenado** (cadencia 30 d vence el 2026-10-24) |
| 2026-09-25 00:18 `7456f10` | GARCH ×7 | Arranque en frío | Sin validación (R) |
| 2026-10-02 01:19 `8228153` | GARCH ×7 | Reentreno por cadencia fija (7 d), 7/7 promovidos | S&P 500 QLIKE 0.6857 vs 0.8341; EUR/USD solo **3 folds** (n=2,616) (R) |
| 2026-10-09 01:46 `73aeca3` | GARCH ×7 | Cadencia fija, 7/7 promovidos | S&P 500 0.6849 vs 0.8330; EUR/USD 4 folds (n=3,492) (R) |
| 2026-10-17 00:11 `fa1d251` | GARCH ×7 (activos) | Cadencia fija, 7/7 promovidos | S&P 500 0.6839 vs 0.8323 (R) |
| 2026-10-17 19:30 | Backtest de auditoría | Reproducción con datos actuales | Cifras de la sección 3 (E) |

Notas:
- **Hiperparámetros, features, algoritmo y esquema de validación no cambiaron entre versiones**; los reentrenos solo agregan datos. La métrica mejora marginalmente (QLIKE S&P 500: 0.6857 → 0.6849 → 0.6839) por la ventana expansiva, no por modelado.
- Ningún reentreno fue disparado por degradación o drift: **los 293 `evaluation_runs` tienen `sufficient_data = 0` (R)**, así que el disparador por degradación nunca ha estado activo (ver A-4).

---

## 3. Desempeño actual

### 3.1 Volatilidad GARCH(1,1)-t: walk-forward, pronóstico a 1 día (E)

n = 3,376 (índices) / 3,496 (FX) días fuera de muestra, 2010-01 → 2026-10. QLIKE: menor es mejor. DM = Diebold-Mariano con HAC Newey-West; t < 0 favorece al GARCH.

| Activo | QLIKE GARCH | Baseline SD-20d (repo) | **EWMA λ=0.94** (mío) | DM vs SD-20d (t, p) | **DM vs EWMA (t, p)** | GJR-GARCH-t (mío) | MZ β | Kupiec VaR 1% (hits, p) | VaR 5% (hits, p) |
|---|---|---|---|---|---|---|---|---|---|
| S&P 500 | 0.6839 | 0.8323 | 0.7522 | −4.80, <0.001 | −3.30, 0.001 | 0.6652 | 0.887 | 1.21%, 0.23 | 5.54%, 0.16 |
| NASDAQ 100 | 1.2449 | 1.3659 | 1.2997 | −5.13, <0.001 | −3.80, <0.001 | 1.2418 | 0.880 | 1.30%, 0.09 | **6.07%, 0.006** |
| Dow Jones 30 | 0.5996 | 0.7316 | 0.6627 | −5.27, <0.001 | −3.89, <0.001 | 0.5825 | 0.910 | 1.24%, 0.17 | 5.51%, 0.18 |
| EUR/USD | −0.5658 | −0.5032 | −0.5564 | −3.85, <0.001 | **−0.81, 0.42** | −0.5600 | 0.856 | 0.94%, 0.74 | 4.43%, 0.12 |
| GBP/USD | −0.3159 | −0.2667 | −0.3119 | −2.72, 0.007 | **−0.33, 0.74** | −0.3080 | 0.741 | 1.17%, 0.32 | 5.23%, 0.53 |
| USD/JPY | −0.2742 | −0.2004 | −0.2601 | −3.54, <0.001 | **−1.27, 0.21** | −0.2742 | 0.823 | 0.97%, 0.87 | 5.01%, 0.99 |
| USD/CAD | −0.7481 | −0.6917 | −0.7375 | −5.52, <0.001 | **−1.79, 0.07** | −0.7458 | 0.780 | 0.92%, 0.61 | 4.66%, 0.35 |

Lectura:
- **Contra el baseline del repo (SD-20d) el GARCH gana con significancia en 7/7.** Esto confirma lo que dice el README.
- **Contra un baseline ingenuo más fuerte (EWMA RiskMetrics) gana con significancia solo en los 3 índices.** En los 4 pares FX la ventaja no es significativa (p = 0.07 a 0.74). El README lee "GARCH supera al baseline en 7/7" como valor demostrado, pero en FX solo supera a un baseline débil.
- GJR (asimetría) baja el QLIKE en índices (−0.019 S&P, −0.017 DJI), pero **ningún DM GJR vs GARCH es significativo** (p ≥ 0.33).
- Calibración: E[z²] de 0.940 a 1.013 (E). En la regresión Mincer-Zarnowitz β queda entre 0.74 y 0.91 (< 1): los pronósticos son algo sobre-reactivos. R²_MZ es de 0.20 a 0.29 en índices y de 0.02 a 0.05 en FX; esto es esperable con r² como proxy ruidoso.
- Residuos: Jarque-Bera p < 0.001 en 7/7. Quedan efectos ARCH en GBP/USD (Ljung-Box z² p = 0.035) y USD/CAD (p < 0.001) (E). Los QQ-plots del repo son contra la normal y no contra la t ajustada, así que las colas "fallan" por construcción.
- Omisión en el reporte del repo: **en MAE contra |r| el GARCH pierde contra el baseline en 7/7** (`model_registry` del 2026-10-17, p. ej. S&P 0.5514 vs 0.5448) (R). El MAE sobre |r| no es una pérdida consistente para varianza (Patton 2011), así que no invalida al modelo, pero conviene reportarlo.

### 3.2 Detector de anomalías: walk-forward, velas de 5 min, 60 días (E)

Positivo = reversión de signo en las 10 velas siguientes. La precisión esperada de un baseline aleatorio con cualquier tasa es **la prevalencia**, así que el comparador correcto es el *lift* (precisión / prevalencia). "Prev. limpia" excluye los empates de signo (hallazgo A-2).

| Activo | n | Anomalías (k) | TN / FP / FN / TP | Precisión [IC95 Wilson] | Recall | F1 | F1 baseline (media de 1,000 semillas) | Prev. (repo) | **Prev. limpia** | **Lift** | p binomial (1 cola, tras QW1) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S&P 500 | 3,652 | 191 | 1697/91/1764/100 | 0.524 [0.453, 0.593] | 0.054 | 0.097 | 0.095 | 0.510 | 0.510 | 1.03 | 0.377 |
| NASDAQ 100 | 3,656 | 190 | 1733/79/1733/111 | 0.584 [0.513, 0.652] | 0.060 | 0.109 | 0.094 | 0.504 | 0.504 | 1.16 | 0.017 |
| Dow Jones 30 | 3,656 | 185 | 1702/96/1769/89 | 0.481 [0.410, 0.553] | 0.048 | 0.087 | 0.092 | 0.508 | 0.508 | 0.95 | 0.792 |
| EUR/USD | 13,368 | 650 | 3892/294/8826/356 | 0.548 [0.509, 0.586] | 0.039 | 0.072 | 0.091 | **0.687** | **0.524** | 1.05 | 0.123 |
| GBP/USD | 13,368 | 615 | 5883/277/6870/338 | 0.550 [0.510, 0.588] | 0.047 | 0.086 | 0.085 | 0.539 | 0.517 | 1.06 | 0.057 |
| USD/JPY | 13,312 | 600 | 6126/288/6586/312 | 0.520 [0.480, 0.560] | 0.045 | 0.083 | 0.083 | 0.518 | 0.509 | 1.02 | 0.309 |
| USD/CAD | 13,320 | 605 | 5740/231/6975/374 | 0.618 [0.579, 0.656] | 0.051 | 0.094 | 0.084 | 0.552 | 0.537 | 1.15 | **<0.001** |
| **Macro** | | | | | | **0.090** | **0.085** (repo, 1 semilla) | | | | |

Lectura:
- **AUC de |z| para predecir reversión: 0.496 a 0.534 (E).** No hay poder discriminante.
- Con corrección de Bonferroni (α = 0.05/7 = 0.0071), **solo USD/CAD tiene lift significativo** (1.15). NASDAQ (p = 0.017) no sobrevive a la corrección.
- El "EUR/USD pierde contra el baseline" del reporte original es un **artefacto de la etiqueta**: el 34.3% de las barras de EUR/USD tienen retorno 0 y se cuentan como "reversión". Al limpiar los empates, la prevalencia baja de 0.687 a 0.524 y el lift pasa de 0.80 a 1.05.
- F1 es una mala métrica de decisión aquí: con tasa de marcado fija (4.55%) y prevalencia ~0.5, el recall queda acotado en ~0.09 y el F1 casi no depende de la calidad de la señal.

### 3.3 Desempeño en producción (live, fuera de muestra real) (R)

| Señal | n | Resultado | Lectura |
|---|---|---|---|
| Anomalías (7 activos agregados) | 229 outcomes | TN 99 / FP 6 / FN 120 / TP 4 → precisión 0.40 [0.17, 0.69], recall 0.032, F1 0.060; baseline: TP 2 / FP 4 (precisión 0.33) | **k = 10 anomalías en 3 semanas: estadísticamente vacío** |
| GARCH (7 activos agregados) | 93 outcomes | QLIKE modelo −0.4091 vs baseline −0.3438; diferencia −0.065 ± 0.071 (EE ingenuo) | No significativo, y **contaminado en FX** (hallazgo A-1) |

Variabilidad esperada: con k ≈ 600 anomalías por par FX, el IC95 de la precisión mide ±0.04. En índices (k ≈ 190) mide ±0.07. Un lift de 1.05 no se distingue del azar con k < ~1,500 (S: cálculo de potencia aproximado con p = 0.5 y error estándar √(p(1−p)/k)).

---

## 4. KPIs ejecutivos

| KPI | Actual | Baseline | Mejora potencial (cons. / optim.) | Fuente |
|---|---|---|---|---|
| QLIKE GARCH, índices (S&P 500) | 0.6839 | SD-20d 0.8323 · EWMA 0.7522 | GJR: −0.000 / −0.019 · HAR-RV: −0.00 / −0.05 (S) | E / S |
| QLIKE GARCH, FX (EUR/USD) | −0.5658 | EWMA −0.5564 (DM p = 0.42) | ≈0 / −0.01 con GJR/EGARCH (S) | E / S |
| Lift de precisión del detector (mejor / peor activo) | 1.15 (CAD) / 0.95 (DJI) | 1.00 (azar) | 1.00 / 1.05–1.10 con la etiqueta actual (S) | E |
| AUC "movimiento grande" (reformulación propuesta) | 0.56–0.68 con un solo feature (`rolling_std`) | 0.50 | 0.60–0.68 / 0.70–0.75 con modelo multi-feature (S) | E / S |
| Outcomes de anomalía en producción por activo y día | ~1.8 | ~78 (índices) / ~276 (FX) velas posibles | ×40 a ×150 con QW5 (backfill) | R / E |
| Días para `sufficient_data` (≥10 anomalías) | ~119 (0.084 anomalías por activo y día) | — | ~3 días índices / ~1 día FX con QW5 | R / cálculo |
| Corridas programadas ejecutadas | 54 en 23 días (~9% de las ~630 previstas) | 100% | ≥99% con un scheduler externo | API de GitHub (E) |
| Outcomes GARCH-FX con `realized` erróneo | 30/45 (67%), error medio 0.043 pp, máx 0.237 pp | 0% | 0% con QW2 | E |
| Tests / cobertura | 69 / 92% | — | 78 tests tras quick wins (E) | E |

---

## 5. Auditoría técnica

### 5.1 Hallazgos

| ID | Severidad | Hallazgo | Ubicación | Evidencia |
|---|---|---|---|---|
| C-1 | **Crítico** | GitHub Actions descarta la mayoría de los cron `*/15`: 54 corridas programadas exitosas en 23 días frente a ~630 previstas. El detector solo predice **la última vela** por corrida, así que la base contiene ~2 de ~78–276 barras por activo y día. El feedback loop, el disparador por degradación y la evaluación live quedan sin muestra | `.github/workflows/pipeline.yml:15-18`; `src/pipeline/predict.py:155` | API de Actions (E); `run_ts` por día en la base = 1–4 (R) |
| A-1 | Alto | **Velas diarias no finalizadas.** Yahoo publica la vela del día en curso (en FX, incluso con fecha de sábado, 2026-10-17) y `fetch_daily_closes` no la filtra. Contamina el origen del forecast y el `realized` de la evaluación: en FX 67% de los outcomes difieren del dato final (máx 0.237 pp). En índices coinciden porque las corridas ocurren después del cierre | `src/data/fetch.py:120-158` | Comparación base vs re-descarga (E) |
| A-2 | Alto | **Empates en la etiqueta de reversión.** `np.sign(0) != np.sign(r)` cuenta como reversión: el 34% de las velas de EUR/USD tienen retorno 0 (cotización rancia). Infla la prevalencia y el baseline, y produjo la conclusión errónea "EUR/USD peor que el azar" | `src/models/anomaly_model.py:228-230` | tie_share 0.343 (E) |
| A-3 | Alto | **La promoción del detector es casi una moneda al aire**: `f1 > baseline_f1` contra *una* tirada aleatoria (`rng_seed=fold`). Con el modelo ≈ azar, el resultado depende de la semilla. El p95 del F1 aleatorio con 1,000 semillas supera al F1 del modelo en 5/7 activos | `src/pipeline/retrain.py:145,281`; `src/models/baseline.py:13` | f1_perm_p 0.008–1.0 (E) |
| A-4 | Alto | El disparador por degradación nunca se activa: `min_predicted_positives=10` y `min_outcomes_garch=20` no se alcanzan (consecuencia de C-1). `docs/reporte_final.md` estima "6 días hábiles" asumiendo 36 corridas diarias; con la cadencia real son ~119 días | `src/config.py:126,134`; `src/pipeline/evaluate_and_retrain.py:126,214` | 293/293 runs con `sufficient_data=0` (R) |
| A-5 | Alto (uso comercial) | `yfinance`/Yahoo: datos no licenciados, cotizaciones FX indicativas (no ejecutables) y velas rancias. Inservible como fuente para operar o vender | `src/data/fetch.py` | El propio `docs/reporte_final.md` lo reconoce |
| M-1 | Medio | Baseline débil (SD-20d). Contra EWMA, el GARCH-FX no es significativamente mejor. La afirmación "valor demostrado 7/7" queda sobredimensionada | `src/models/baseline.py:34-43` | DM vs EWMA (E) |
| M-2 | Medio | Sin champion–challenger ni test de significancia: el GARCH candidato se promueve si `qlike < baseline_qlike`, sin compararlo contra el modelo activo ni usar DM. Con cadencia de 7 días se promueve siempre (7/7 en 3 reentrenos) | `src/pipeline/retrain.py:346` | `model_registry` (R) |
| M-3 | Medio | Reproducibilidad: dependencias con `>=` y sin lockfile. Yo corrí con pandas 3.0.6 y numpy 2.5 y pasó, pero el CI instala versiones distintas en cada build. Tampoco hay semilla ni versión de datos fijada (Yahoo revisa el histórico) | `requirements.txt` | (E) |
| M-4 | Medio | El paso de push del workflow hace `git push` sin `pull --rebase`. Un commit manual entre el checkout y el push hace fallar el job y **se pierde el ciclo** (la base no se versiona) | `.github/workflows/pipeline.yml` (último step) | Revisión de código |
| M-5 | Medio | La ventana móvil del z-score (20 velas) y la etiqueta de 10 velas cruzan cierres de sesión: los retornos de salto se descartan, pero `rolling`/`shift(-k)` concatenan la sesión anterior con la siguiente. La etiqueta de una vela de las 15:55 ET mira el día siguiente | `src/features/returns.py:104`; `src/models/anomaly_model.py:212-216` | Revisión de código |
| M-6 | Medio | Pruebas múltiples: 7 activos × 2 señales sin corrección. "Supera en 5/7" en el backtest original es compatible con azar | `src/evaluation/backtest.py` | (E) |
| M-7 | Medio | Calidad de datos FX 5 min: 34% de retornos 0 en EUR/USD (4.6% en GBP, 3.1% en CAD) indican cotizaciones rancias de Yahoo. Ningún contrato Pandera lo detecta | `src/data/validation.py` | (E) |
| B-1 | Bajo | `h[0]` del filtro GARCH usa `np.var` de toda la serie de entrada, prueba incluida: look-ahead mínimo (decae como β^t, despreciable tras unas 100 observaciones) | `src/models/garch_model.py:172` | Revisión |
| B-2 | Bajo | Los residuos estandarizados no restan μ (`r/σ` en vez de `(r−μ)/σ`) | `src/evaluation/metrics.py:227`; `src/pipeline/retrain.py:253` | Revisión |
| B-3 | Bajo | QQ-plot contra la normal y no contra la t(ν) ajustada | `src/evaluation/backtest.py:161` | Revisión |
| B-4 | Bajo | `trained_at` usa el reloj de pared y no `run_ts`: rompe la reproducibilidad del backfill y de los tests temporales | `src/models/anomaly_model.py:116`, `garch_model.py:130` | Revisión |
| B-5 | Bajo | SQLite binaria versionada en git: el repo crece con cada commit (hoy `.git` pesa 627 KB) y el historial de predicciones queda público | `data/pipeline.db` | (E) |
| B-6 | Bajo | No se reporta el MAE, donde el GARCH pierde en 7/7 | `reports/…/backtest.md` | (R) |

### 5.2 Checklist de estándares

| Criterio | Estado | Comentario |
|---|---|---|
| Fuga: fit solo en train / transform en prueba | ✅ | Umbral y parámetros GARCH se estiman solo en train; bandas causales con `shift(1)` |
| Sin split aleatorio en series temporales | ✅ | `TimeSeriesSplit` con `gap = horizonte` en anomalías |
| Nulos, tipos, duplicados, encoding | ✅ / ⚠️ | Nulos se descartan y se registran; duplicados fallan; dtypes validados; encoding no aplica (API). **Falta**: detección de cotizaciones rancias y de velas diarias parciales |
| Supuestos estadísticos, QQ, residuos | ⚠️ | JB, Ljung-Box y E[z²] ✅; falta QQ contra t, MZ, Kupiec/Christoffersen, DM |
| PEP8, type hints, docstrings NumPy | ✅ | Ruff con `E,W,F,I,N,UP,B,BLE,ANN,D,PD,NPY,SIM`, 79 columnas |
| try/except específicos | ✅ | `BLE` activo; jerarquía propia de excepciones |
| Sin bucles sobre DataFrames ni `.append` | ✅ | Los únicos bucles son recursiones GARCH sobre arrays y sobre activos |
| SQL > Polars > Pandas | ✅ | Filtrado por SQL; agregación en Python sobre ≤500 filas (razonable); Polars no se justifica (<1 GB) |
| Pydantic / Pandera / pytest + mocking | ✅ | Contratos en config, payloads y decisiones; Pandera en OHLC y retornos; `monkeypatch` de la descarga |
| Secretos | ✅ | Solo `PIPELINE_ALERT_WEBHOOK_URL` vía `secrets`; `.env` ignorado; sin credenciales en el historial |
| README | ✅ / ⚠️ | Excelente. Corregir el "6 días hábiles", la lectura de EUR/USD y la afirmación de valor 7/7 |

Veredicto de ingeniería: **código por encima del promedio de un portafolio senior**. Los problemas no son de estilo: son de **validez estadística de la evaluación y de operación del scheduler**.

---

## 6. Cuánto se puede mejorar

### 6.1 Impacto por hallazgo

| ID | Fix | Impacto esperado (cons. / optim.) | Razonamiento | Esfuerzo |
|---|---|---|---|---|
| C-1 | QW5 backfill + scheduler externo | Muestra live ×40 / ×150; `sufficient_data` en 1–3 días | 78 velas/día (índices) o ~276 (FX) frente a ~1.8 observadas; `predict_period=5d` cubre huecos de hasta 5 días | 4 h (QW5) + 2 h (scheduler) |
| A-1 | QW2 vela diaria finalizada | 0 cambio en el backtest; elimina el sesgo en 67% de los outcomes FX live | El backtest usa datos finales; el error solo vive en producción | 2 h |
| A-2 | QW1 empates | Prevalencia EUR/USD 0.687→0.524; lift 0.80→1.05 (medido); **0 cambio en la calidad real de la señal** | Es un fix de medición | 1 h |
| A-3 | QW3 binomial + Bonferroni | Promociones válidas: 2/7 a α=0.05 (NDX, CAD), 1/7 con Bonferroni (medido) | Elimina las promociones por ruido | 2 h |
| M-1/M-2 | QW4 DM + EWMA + champion–challenger | GARCH-FX deja de "ganar" frente a EWMA (honestidad); índices se mantienen | DM vs EWMA medido | 3 h (+4 h champion–challenger) |
| M-3/M-4 | Lockfile + `pull --rebase` | Builds reproducibles; sin ciclos perdidos | — | 1 h |
| M-5 | Ventanas por sesión (`groupby(session)`) | Lift ±0.02 (S) | Elimina contaminación overnight en ~10/78 velas del borde por sesión | 3 h |
| GARCH | GJR/EGARCH elegidos por DM por activo | ΔQLIKE 0 / −0.019 (índices, medido, no significativo) | Asimetría documentada en índices | 4 h |
| GARCH | HAR-RV con varianza realizada intradía | ΔQLIKE 0 / −0.05 (S, basado en la literatura HAR: Corsi 2009; Andersen et al.) | Requiere ≥2 años de intradía: Yahoo solo da 60 días | 2–3 semanas + datos de pago |
| Detector | Reformular objetivo: magnitud o "burst" de volatilidad en lugar de signo | AUC 0.60–0.68 / 0.70–0.75 (S); hoy AUC de reversión 0.50–0.53 (E) | `rolling_std` solo ya da AUC 0.56–0.68 (E): el clustering de volatilidad es una regularidad robusta; la reversión de signo no lo es | 1–2 semanas |
| Detector | Desbalance y umbral | Lift en tasa 1–2%: 0.97–1.29 (E, exploratorio: umbral con cuantil del bloque de prueba, *look-ahead*) | k de 37 a 130: IC muy anchos | 1 día |
| Detector | Calibración (isotónica / Platt) + umbral por utilidad con costos | Precisión operable con costos explícitos (S) | Solo tiene sentido con una señal con AUC > 0.55 | 2 días |
| Monitoreo | PSI/KS de features, tablero sobre `evaluation_runs`, alerta de staleness (% de retornos 0) | Detección temprana de datos rancios (S) | — | 2–3 días |

### 6.2 Roadmap

**Quick wins (≤1 semana, ~15 h):** QW1–QW5 (implementados abajo y probados), lockfile, `git pull --rebase`, corrección de Bonferroni en el reporte, añadir EWMA y MAE al backtest, actualizar el README (EUR/USD, "6 días", 7/7 vs EWMA).

**Mediano plazo (2–6 semanas):**
1. Scheduler confiable: VPS o Raspberry con cron (~USD 5/mes), o un trigger externo a `workflow_dispatch`. GitHub no garantiza los cron de 15 min.
2. Champion–challenger: candidato vs activo con DM sobre el mismo walk-forward; GJR/EGARCH por activo.
3. Reformular el detector como clasificador de *volatility burst* (features: `rolling_std`, |r|, |z|, hora del día, día de la semana, régimen GARCH, volumen en índices). LightGBM con *purged K-fold* + embargo, calibración isotónica y umbral por utilidad.
4. Ventanas por sesión; contrato Pandera de staleness.

**Largo plazo (1–3 meses):** proveedor licenciado (Databento, Polygon, o la API de un broker como OANDA o IBKR) con ≥2 años de intradía; HAR-RV; Postgres; modelo de costos y paper trading; reporte con *Deflated Sharpe Ratio* y *White's Reality Check* / SPA de Hansen.

### 6.3 Proyección realista por métrica clave

| Métrica | Hoy | Tras quick wins | Mediano plazo | Largo plazo |
|---|---|---|---|---|
| QLIKE S&P 500 (↓) | 0.6839 | 0.6839 (fixes de medición) | 0.665–0.684 (GJR) | 0.63–0.68 (HAR-RV, S) |
| Ventaja GARCH-FX vs EWMA | No significativa | No significativa (reportada honestamente) | Marginal (S) | Depende de RV intradía (S) |
| Lift de reversión (mejor activo, con Bonferroni) | 1.15 (CAD) | 1.15 (CAD) | No es objetivo prioritario | — |
| AUC del objetivo reformulado | — | 0.56–0.68 (un feature, E) | 0.60–0.72 (S) | 0.65–0.75 (S) |
| Muestra live (outcomes/activo/día) | ~1.8 | 78–276 | 78–276 | 78–276 |

---

## 6.5 ¿Es viable para trading profesional?

**No. Hoy no es viable ni como sistema de trading ni como base de uno.** Esto es una evaluación técnica, no asesoría de inversión.

1. **No hay edge direccional.** AUC de reversión 0.50–0.53. Solo USD/CAD tiene lift significativo tras Bonferroni, en una sola ventana de 57 días y un solo régimen.
2. **El P&L bruto no cubre costos.** Una estrategia *fade* (operar contra la anomalía y salir a las 10 velas) da en promedio **+0.21 a +3.40 bps brutos** por operación, con DJI en −1.53 (E). Solo USD/CAD (+1.25 bps, t = 4.9) y GBP/USD (+0.60, t = 2.3) son significativos, y los t están **inflados** porque las operaciones se solapan dentro de 10 velas. Un spread retail en majors FX suele costar del orden de 0.5–1.5 bps por ida y vuelta, más slippage (S, rango típico de mercado, no medido aquí). El edge neto esperado es ≈ 0 o negativo.
3. **Los datos no son ejecutables.** Yahoo FX es indicativo, tiene un 34% de velas rancias en EUR/USD y no tiene licencia comercial.
4. **Falta toda la capa de trading:** sizing, stops, gestión de riesgo de cartera, modelo de costos y ejecución, paper trading, monitoreo de latencia, y la reconciliación con un broker.
5. **Validación insuficiente para capital real:** 60 días de intradía, sin out-of-time de otro régimen y sin corrección por *data snooping*.
6. **Lo que sí sirve:** el GARCH en índices es un **insumo de riesgo** legítimo (VaR 1% pasa Kupiec en 7/7; supera a EWMA en índices). Como componente de *volatility targeting* o de sizing tiene respaldo en la literatura (Moreira & Muir 2017, supuesto no testeado aquí), pero no genera alpha por sí mismo.
7. **Nota regulatoria (México):** operar capital propio no requiere registro. Gestionar o asesorar capital de terceros implica figuras reguladas por la CNBV (p. ej. asesor en inversiones). Valídalo con un especialista antes de ofrecerlo a clientes.

**Camino mínimo hacia "viable":** datos licenciados con ≥2 años de intradía → objetivo de volatilidad (no de signo) → backtest con costos, purged CV y DSR → 3 meses de paper trading con métricas live que repliquen el backtest → recién entonces capital pequeño.

---

## 7. Quick wins implementados

Implementados y probados en una **copia** del repo (el original no se modificó y no hay push). El patch completo, `chronos_quick_wins.patch` (14 archivos, +506/−25), está junto a este reporte. `git apply --check` aplica limpio sobre `fa1d251`. Resultado: **78/78 tests (69 originales + 9 nuevos), `ruff check` y `ruff format --check` limpios (E).**

```bash
git apply chronos_quick_wins.patch
```

### QW1: empates en la etiqueta (`src/models/anomaly_model.py`)

```python
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
```

```python
def test_label_reversion_outcomes_drops_zero_return_ties() -> None:
    index = pd.date_range("2026-01-01", periods=5, freq="D")
    bands = pd.DataFrame(
        {
            # Barra 1: retorno 0 (empate). Barra 2: futuro acumulado 0.
            "log_return": [0.01, 0.0, 0.02, 0.01, -0.01],
            "rolling_mean": [0.0] * 5,
            "rolling_std": [0.01] * 5,
            "zscore": [0.0] * 5,
            "is_anomaly": [False] * 5,
        },
        index=index,
    )
    labeled = label_reversion_outcomes(bands, horizon_bars=2)
    assert index[1] not in labeled.index
    assert index[2] not in labeled.index
    assert index[0] in labeled.index
```

Efecto medido (E): EUR/USD n 13,368 → 8,779; prevalencia 0.687 → 0.524 (precisión del baseline 0.688 → 0.519); el F1 base del backtest deja de "ganarle" al modelo.

Pendiente conocido: las predicciones live sobre velas con retorno 0 quedan `evaluated = 0` indefinidamente. Si molesta, márcalas como no evaluables.

### QW2: solo velas diarias finalizadas (`src/config.py`, `src/data/fetch.py`)

```python
# Desde la medianoche UTC de la fecha de la vela diaria hasta que su cierre
# es definitivo. Índices US: cierre 20:00/21:00 UTC (EDT/EST) + margen.
# FX (24/5): conservador, se espera al día UTC siguiente.
DAILY_BAR_FINAL_AFTER: dict[AssetClass, pd.Timedelta] = {
    AssetClass.EQUITY_INDEX: pd.Timedelta(hours=21, minutes=15),
    AssetClass.FX: pd.Timedelta(days=1),
}
```

```python
def fetch_daily_closes(
    symbol: str,
    start_date: str,
    min_rows: int,
    final_after: pd.Timedelta,
    now: datetime | None = None,
) -> pd.Series:
    """Descarga cierres diarios finalizados y los devuelve validados.
    ...
    """
    # ... descarga y _drop_nulls_and_check_duplicates sin cambios ...
    reference = pd.Timestamp(now or datetime.now(UTC)).tz_convert(None)
    bar_date = close.index.tz_localize(None).normalize()
    is_final = bar_date + final_after <= reference
    n_open = int((~is_final).sum())
    if n_open > 0:
        logger.info(
            "Se descartaron %d velas diarias no finalizadas de '%s'.",
            n_open,
            symbol,
        )
    close = close[is_final]
```

`load_daily_returns(asset, settings, now)` pasa `DAILY_BAR_FINAL_AFTER[asset.asset_class]`; los tres call sites (`predict`, `evaluate_and_retrain`, `backtest`) propagan `now`. Tests nuevos: `test_fetch_daily_closes_drops_bars_not_yet_final` (15:00 UTC → descarta la vela del día) y `test_fetch_daily_closes_keeps_bar_after_session_close` (22:00 UTC → la conserva).

Supuesto (S): el desfase de FX de 1 día es conservador y no está validado contra el timestamp crudo de Yahoo. Con él, la corrida de las 22:00 UTC pronostica FX con origen D−1. Mueve el cron de FX a ~00:30 UTC.

### QW3/QW4: promoción con significancia + baseline EWMA (`src/evaluation/metrics.py`, `src/models/baseline.py`, `src/pipeline/retrain.py`)

```python
def precision_lift_test(report: ClassificationReport) -> PrecisionLiftResult:
    """Test binomial exacto de una cola (H1: precisión > prevalencia).
    ...
    """
    (_, _), (fn, tp) = report.confusion_matrix
    k = report.n_predicted_positive
    prevalence = (fn + tp) / report.n_samples
    if k == 0 or not 0.0 < prevalence < 1.0:
        raise ValueError(
            f"Test indefinido: k={k}, prevalencia={prevalence:.3f}."
        )
    result = stats.binomtest(tp, k, prevalence, alternative="greater")
    precision = tp / k
    return PrecisionLiftResult(
        precision=precision,
        prevalence=prevalence,
        lift=precision / prevalence,
        p_value=float(result.pvalue),
        n_predicted_positive=k,
    )


def diebold_mariano(
    model_loss: np.ndarray, reference_loss: np.ndarray
) -> DieboldMarianoResult:
    """Test de Diebold-Mariano con error estándar HAC (Newey-West).
    ...
    """
    model = np.asarray(model_loss, dtype=float)
    reference = np.asarray(reference_loss, dtype=float)
    if len(model) != len(reference):
        raise ValueError("Las pérdidas deben tener la misma longitud.")
    if len(model) < _MIN_RELIABLE_SAMPLE:
        raise ValueError(
            f"Diebold-Mariano requiere >= {_MIN_RELIABLE_SAMPLE} puntos, "
            f"se recibieron {len(model)}."
        )
    diff = model - reference
    max_lag = int(np.ceil(len(diff) ** (1.0 / 3.0)))
    fit = sm.OLS(diff, np.ones_like(diff)).fit(
        cov_type="HAC", cov_kwds={"maxlags": max_lag}
    )
    t_stat = float(fit.tvalues[0])
    return DieboldMarianoResult(
        mean_loss_diff=float(diff.mean()),
        t_stat=t_stat,
        p_value_one_sided=float(stats.norm.cdf(t_stat)),
        n_samples=len(diff),
    )


def ewma_volatility_baseline(
    returns_pct: pd.Series, decay: float = 0.94
) -> pd.Series:
    """Baseline RiskMetrics: ``s2_t = d * s2_{t-1} + (1 - d) * r2_{t-1}``.
    ...
    """
    if not 0.0 < decay < 1.0:
        raise ValueError(f"decay debe estar en (0, 1), se recibió {decay}.")
    variance = (returns_pct**2).ewm(alpha=1.0 - decay, adjust=False).mean()
    return np.sqrt(variance).shift(1)
```

Reglas de promoción nuevas (`retrain.py`; `promotion_alpha=0.05` en `RetrainTriggerConfig`):

```python
    # Detector
    lift = precision_lift_test(pooled) if enough_positives else None
    promote = (
        lift is not None
        and lift.lift > 1.0
        and lift.p_value < settings.retrain.promotion_alpha
    )
    # GARCH
    dm = diebold_mariano(
        np.asarray(result.pointwise_qlike),
        np.asarray(result.pointwise_baseline_qlike),
    )
    promote = dm.p_value_one_sided < settings.retrain.promotion_alpha
```

Tests nuevos: `test_precision_lift_detects_signal_above_prevalence` (lift 1.6, p < 1e−6), `test_precision_lift_is_not_significant_at_chance` (p > 0.4), `test_diebold_mariano_prefers_lower_loss_model`, `test_diebold_mariano_rejects_small_samples`, `test_ewma_baseline_is_causal_and_matches_recursion`.

Efecto medido (E): el detector se promovería en NDX (p = 0.017) y CAD (p < 0.001); con α/7 solo en CAD. El GARCH se promueve 7/7 (DM t de −2.71 a −5.47).

Siguiente paso: cambiar `trailing_volatility_baseline` por `ewma_volatility_baseline` en `walk_forward_garch` y en `_close_garch_outcomes`. Hoy la función queda disponible pero no conectada, para no mezclar baselines dentro del historial de `evaluation_runs`.

### QW5: backfill de velas que el cron no alcanzó (`src/pipeline/predict.py`, `src/persistence/database.py`)

```python
def _backfill_anomaly_predictions(
    asset: AssetConfig,
    settings: PipelineSettings,
    detector: AnomalyDetector,
    bands: pd.DataFrame,
    run_ts: datetime,
) -> int:
    """Registra las velas cerradas que ninguna corrida previa alcanzó a ver.

    ... Solo se recuperan velas posteriores al entrenamiento del detector:
    anteriores serían in-sample.
    """
    out_of_sample = bands[bands.index > pd.Timestamp(detector.trained_at)]
    if out_of_sample.empty:
        return 0
    records = (
        out_of_sample.rename_axis("observed_at")
        .reset_index()
        .assign(
            asset_symbol=asset.symbol,
            model_version=detector.version,
            reversion_horizon_bars=detector.reversion_horizon_bars,
        )
        .to_dict("records")
    )
    try:
        payloads = _PAYLOAD_LIST.validate_python(records)  # TypeAdapter
    except ValidationError as exc:
        raise DataContractError(
            f"Backfill de anomalías inválido para '{asset.symbol}': {exc}"
        ) from exc
    return database.insert_predictions_bulk(
        db_path=settings.paths.database_path,
        signal_type=SignalType.ANOMALY.value,
        horizon_bars=detector.reversion_horizon_bars,
        payloads=[p.model_dump(mode="json") for p in payloads],
        run_ts=run_ts,
    )
```

`insert_predictions_bulk` usa `executemany` + `INSERT OR IGNORE` en una sola transacción (idempotente por `UNIQUE(asset, signal, reference_timestamp)`) y devuelve las filas insertadas vía `total_changes`. Test: `test_anomaly_backfills_out_of_sample_bars_missed_by_cron` (11 velas recuperadas; una segunda corrida recupera 0).

### Cambios en workflow y dependencias (no ejecutados: van fuera del patch)

```yaml
      - name: Versionar la base de datos actualizada
        if: ${{ !cancelled() }}
        run: |
          git config user.name "trading-pipeline-bot"
          git config user.email "actions@users.noreply.github.com"
          git add data/pipeline.db
          git diff --cached --quiet && exit 0
          git commit -m "data: ciclo ${{ steps.select.outputs.signals }} $(date -u +%Y-%m-%dT%H:%MZ)"
          git pull --rebase --autostash
          git push
```

```bash
pip install pip-tools
```

```bash
pip-compile requirements.txt -o requirements.lock
```

Instala en CI y en el pipeline con `pip install -r requirements.lock`.

---

## Supuestos y límites de esta auditoría

- Las métricas del backtest usan datos de Yahoo descargados el 2026-10-17. Yahoo revisa el histórico, así que una re-ejecución puede variar en la tercera cifra decimal.
- Los costos de transacción (0.5–1.5 bps en FX retail) son un rango típico de mercado y no se midieron.
- Las proyecciones HAR-RV, la del AUC multi-feature y la de volatility targeting son estimaciones (S) basadas en la literatura y en el AUC de un solo feature medido. No están probadas en este repo.
- El barrido de lift por tasa (1%, 2%, 10%) usa el cuantil del propio bloque de prueba: es exploratorio y tiene *look-ahead*.
- El P&L *fade* ignora el solapamiento de operaciones (t inflados) y los costos.

---

## Skills y herramientas usadas

| Skill / herramienta | Uso |
|---|---|
| `data:validate-data` | Checklist de QA previo a la entrega: pruebas múltiples, periodos incompletos (velas parciales), look-ahead, muestras pequeñas, t inflados |
| Bash + Python (venv aislado) | Clon, `git log`, consultas SQL a `pipeline.db`, pytest + cobertura, Ruff, backtest `--mode backtest`, análisis propio (DM, EWMA, GJR, Mincer-Zarnowitz, Kupiec, binomial, permutación de F1, AUC, P&L fade, verificación de velas parciales) |
| API REST de GitHub (pública) | Conteo y duración de corridas de Actions (hallazgo C-1) |
| Edición + pytest | Implementación y verificación de QW1–QW5 en una copia del repo |

No invoqué otras skills (revisión de código, testing strategy, tech-debt, arquitectura, SQL, visualización): hice esos análisis directamente con las herramientas de arriba, y lo declaro aquí para no inflar la lista.

---

## Resumen ejecutivo

1. Ingeniería sólida: 69/69 tests, 92% de cobertura, Ruff estricto, sin fuga en el walk-forward, contratos Pydantic y Pandera.
2. El GARCH supera con significancia al baseline del repo en 7/7, pero frente a EWMA solo en los 3 índices; en FX la ventaja no es significativa.
3. El detector de anomalías no tiene poder predictivo (AUC 0.50–0.53); solo USD/CAD muestra lift significativo tras Bonferroni, y el "EUR/USD peor que el azar" era un bug de empates en la etiqueta.
4. En producción, GitHub ejecutó solo ~9% de los cron, el disparador por degradación nunca se ha activado y el 67% de las evaluaciones GARCH-FX usan velas no finalizadas.
5. No es viable para trading profesional hoy; los 5 quick wins implementados (78/78 tests) corrigen la medición, y el camino de valor es reformular el detector hacia volatilidad, con datos licenciados.

---

## Versión para compartir con cliente (sin tecnicismos)

> **¿Qué es el sistema?** Un vigilante automático de 7 mercados (tres índices de bolsa de EE. UU. y cuatro pares de divisas). Hace dos cosas: avisa cuando un precio se mueve de forma inusual y estima cuánto podría moverse cada mercado al día siguiente.
>
> **¿Qué funciona bien?** El pronóstico de "cuánto se moverá mañana" es confiable en los índices de bolsa: acierta mejor que los métodos simples de referencia y sus estimaciones de riesgo extremo salen como deberían. Sirve para dimensionar riesgo, no para adivinar si el precio va a subir o bajar.
>
> **¿Qué no funciona?** Las alertas de "movimiento inusual que se va a corregir" no anticipan correcciones mejor que el azar. Además, el servicio gratuito que agenda las corridas solo ejecutó cerca de 1 de cada 10, así que el sistema casi no ha acumulado historia real para autoevaluarse. Y en divisas algunas evaluaciones se hicieron con precios del día aún no cerrados.
>
> **¿Qué ya corregimos?** Cinco ajustes, todos probados: medir el acierto de forma justa, usar solo precios definitivos, exigir evidencia estadística antes de cambiar un modelo, comparar contra una referencia más exigente y recuperar los datos que se saltaban las corridas.
>
> **¿Sirve para operar dinero?** Todavía no. Para eso hacen falta datos de un proveedor con licencia, al menos dos años de historia detallada, considerar los costos reales de operar y un periodo de prueba sin dinero real. Lo que hoy tiene valor es la parte de medición de riesgo.
