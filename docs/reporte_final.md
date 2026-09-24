# Reporte final — Chronos Risk Pipeline

## 1. Información general

| Campo | Detalle |
|---|---|
| Nombre del proyecto | Chronos Risk Pipeline |
| Especialidad | Data Science |
| Fuente de datos | Yahoo Finance, a través de la librería de código abierto `yfinance` |
| Link de la fuente | https://finance.yahoo.com · https://github.com/ranaroussi/yfinance |
| Repositorio | https://github.com/AlejandroDeAntu/Chronos-Risk-Pipeline |
| Activos | S&P 500, NASDAQ 100, Dow Jones 30, EUR/USD, GBP/USD, USD/JPY, USD/CAD |

## 2. Objetivo

Dar a un trader discrecional o a un área de riesgo una alerta automática y
auditable sobre 7 índices y pares de divisas: qué movimientos recientes son
estadísticamente inusuales y cuánta volatilidad esperar el día siguiente.
El sistema corre solo, mide su propio acierto contra un baseline ingenuo y
se reentrena cuando su desempeño lo justifica.

## 3. Plan de trabajo

1. **Exploración de datos.** Partí de un notebook con dos scripts
   desconectados: un GARCH(1,1)-t sobre SPY (pruebas ADF, Ljung-Box y
   ARCH-LM) y un detector de anomalías que dependía de un menú manual.
   Diagnostiqué sus fallas estadísticas (validación con un solo punto
   fuera de muestra, sin baseline, fuga en la ventana móvil).
2. **Limpieza y preparación.** Contratos de datos con Pandera y Pydantic,
   tratamiento explícito de nulos y duplicados, descarte de velas aún
   abiertas y de retornos que cruzan el cierre de sesión, estadísticas
   móviles causales.
3. **Construcción de los modelos y del pipeline.** Detector con umbral
   calibrado por cuantil, GARCH(1,1)-t con filtrado de varianza,
   persistencia en SQLite, registro de modelos, disparador de
   reentrenamiento y orquestación en GitHub Actions con un único punto de
   entrada (`run_pipeline.py`).
4. **Evaluación.** Validación walk-forward contra baselines, con matriz de
   confusión, precisión, recall y F1 (clasificación) y QLIKE más
   diagnóstico de residuos (volatilidad). 69 pruebas automatizadas y CI.
5. **Conclusiones.** Qué componente aporta valor demostrado, cuál no, y qué
   haría falta para llevarlo a un uso comercial.

## 4. Preguntas clave

**¿La fuente de datos sirve para producción y para vender el sistema?**
Para un proyecto personal o académico, sí. Para un producto comercial, no:
el propio repositorio de `yfinance` aclara que no está afiliado a Yahoo,
que está pensado para investigación y educación, y que la API de Yahoo
Finance es para uso personal. Un servicio de pago requeriría un proveedor
con licencia. Si Yahoo falla o cambia su formato, el sistema falla de forma
explícita (el activo queda marcado como fallido, el job termina con código
1 y GitHub envía un correo), pero no puede evitar la caída.

**¿El detector de anomalías aporta valor si su F1 está al nivel del azar?**
Hoy no hay evidencia de que prediga reversiones mejor que marcar barras al
azar con la misma frecuencia (sección 6). Por eso el sistema nunca promueve
un detector reentrenado que no supere al baseline. Además, su disparador de
degradación exige al menos 10 anomalías predichas, lo que en operación
real toma unos 6 días hábiles; antes de eso solo actúan la cadencia fija y
el detector de drift.

**¿Qué pasa si el pipeline deja de correr o dos corridas chocan?**
Las corridas están serializadas (un solo grupo de concurrencia en GitHub
Actions) porque todas escriben en la misma base SQLite. GitHub desactiva
los workflows programados de repositorios públicos tras 60 días sin
actividad, así que hay que revisar la pestaña Actions periódicamente. Con
varios escritores concurrentes habría que migrar la persistencia a
Postgres; el código lo aísla en un solo módulo para que ese cambio no
afecte al resto.

## 5. Qué se hizo y cómo

**Manejo de nulos, tipos y duplicados.** Los cierres nulos se descartan y
se registra cuántos. Los timestamps duplicados hacen fallar la ingesta en
vez de deduplicarse en silencio. Pandera valida tipo numérico, precios
positivos, índice único y zona horaria UTC antes de cualquier cálculo, y
Pydantic valida cada predicción antes de guardarla. La fuente es una API,
no un archivo de texto, así que la validación de encoding se reemplaza por
la de tipos y esquema.

**Transformaciones.**
- Retornos logarítmicos.
- Media y desviación móviles desplazadas un periodo (`shift(1)`), para que
  una barra nunca participe en su propio umbral.
- Se descartan los retornos intradía que abarcan un cierre de sesión o un
  fin de semana.
- Se descarta la vela de 5 minutos que todavía no cierra.

**Splits.** `TimeSeriesSplit` con 4 bloques de ventana expansiva, nunca un
split aleatorio. En el detector se deja un hueco (`gap`) de 10 barras,
igual al horizonte de la etiqueta de reversión, para que ninguna barra de
entrenamiento comparta ventana de resultado con la prueba. En el GARCH, los
parámetros se estiman solo con el bloque de entrenamiento y la varianza se
filtra hacia adelante, así que cada pronóstico usa solo información hasta
el día anterior.

**Modelos y baselines comparados.**

| Señal | Modelo | Baseline | Métrica de decisión |
|---|---|---|---|
| Anomalía (5 min) | Z-score causal con umbral calibrado por cuantil empírico (tasa objetivo 4.55%) | Marcar anomalías al azar con la misma tasa | F1 |
| Volatilidad (diaria) | GARCH(1,1) con innovaciones t de Student | Desviación estándar móvil de 20 días | QLIKE |

**Alternativas descartadas y por qué.**

| Alternativa | Por qué se descartó |
|---|---|
| Umbral fijo de 2σ (el original) | Supone normalidad; los retornos tienen colas pesadas. Se calibra por cuantil empírico |
| Validar con un solo día fuera de muestra (el original) | Con n = 1 no hay varianza del error ni significancia |
| RMSE contra el retorno absoluto para elegir el GARCH | La volatilidad real no se observa; QLIKE ordena los modelos de forma robusta con retornos al cuadrado como aproximación (Patton, 2011) |
| Acierto punto a punto como disparador | Con pocos puntos es ruido; se usa F1 o QLIKE agregado por corrida, con muestra mínima |
| Un "embargo" que excluía datos recientes del entrenamiento | Lo propuse al inicio y lo retiré: ninguno de los dos modelos usa etiquetas al entrenar, así que no evitaba ninguna fuga. La protección correcta es el `gap` en la validación |
| Airflow como orquestador | Sobredimensionado para un solo pipeline operado por una persona; GitHub Actions es gratuito y versionado |

## 6. Resultados

Backtest walk-forward del 24 de septiembre de 2026 (4 bloques; 60 días de
velas de 5 minutos para anomalías, diarios desde 2010 para GARCH).

**Métrica principal**

| Señal | Métrica | Modelo | Baseline | ¿Supera? |
|---|---|---|---|---|
| Volatilidad GARCH | QLIKE (menor es mejor) | Menor en 7 de 7 activos | Desviación móvil de 20 días | Sí, en todos |
| Detector de anomalías | F1 macro | 0.085 | 0.084 | Sin diferencia práctica |

**Volatilidad GARCH(1,1)-t**

| Activo | n | QLIKE modelo | QLIKE baseline | Diferencia | RMSE modelo | RMSE baseline | E[z²] | p Ljung-Box z² |
|---|---|---|---|---|---|---|---|---|
| S&P 500 | 3,364 | 0.6868 | 0.8360 | −0.1492 | 0.7486 | 0.7831 | 0.945 | 0.288 |
| NASDAQ 100 | 3,364 | 1.2442 | 1.3657 | −0.1215 | 0.9277 | 0.9605 | 0.994 | 0.239 |
| Dow Jones 30 | 3,364 | 0.6023 | 0.7347 | −0.1324 | 0.7257 | 0.7686 | 0.964 | 0.131 |
| EUR/USD | 2,610 | −0.4967 | −0.4305 | −0.0662 | 0.3582 | 0.3606 | 0.964 | 0.064 |
| GBP/USD | 3,480 | −0.3096 | −0.2588 | −0.0508 | 0.4047 | 0.4122 | 1.018 | 0.041 |
| USD/JPY | 3,480 | −0.2677 | −0.1944 | −0.0733 | 0.4084 | 0.4092 | 0.970 | 0.665 |
| USD/CAD | 3,480 | −0.7430 | −0.6872 | −0.0558 | 0.3108 | 0.3121 | 0.965 | 0.000 |

La varianza está bien calibrada: E[z²] va de 0.945 a 1.018, cerca de 1 en
todos los activos. Jarque-Bera rechaza normalidad en todos (p < 0.001),
algo esperado y cubierto por la distribución t. En GBP/USD y USD/CAD quedan
efectos ARCH sin modelar (p < 0.05).

**Detector de anomalías** (positivo = el precio revirtió en las siguientes
10 velas)

| Activo | n | Anomalías | TN / FP / FN / TP | Precisión [IC 95%] | Recall | F1 | Precisión baseline | F1 baseline |
|---|---|---|---|---|---|---|---|---|
| S&P 500 | 3,612 | 168 | 1705 / 82 / 1739 / 86 | 0.512 [0.437, 0.586] | 0.047 | 0.086 | 0.573 | 0.091 |
| NASDAQ 100 | 3,616 | 181 | 1732 / 81 / 1703 / 100 | 0.552 [0.480, 0.623] | 0.055 | 0.101 | 0.488 | 0.085 |
| Dow Jones 30 | 3,616 | 154 | 1695 / 78 / 1767 / 76 | 0.494 [0.416, 0.572] | 0.041 | 0.076 | 0.525 | 0.074 |
| EUR/USD | 13,532 | 607 | 3763 / 269 / 9162 / 338 | 0.557 [0.517, 0.596] | 0.036 | 0.067 | 0.697 | 0.085 |
| GBP/USD | 13,532 | 617 | 5947 / 276 / 6968 / 341 | 0.553 [0.513, 0.591] | 0.047 | 0.086 | 0.560 | 0.087 |
| USD/JPY | 13,476 | 601 | 6221 / 290 / 6654 / 311 | 0.517 [0.478, 0.557] | 0.045 | 0.082 | 0.515 | 0.081 |
| USD/CAD | 13,484 | 628 | 5742 / 225 / 7114 / 403 | 0.642 [0.603, 0.678] | 0.054 | 0.099 | 0.540 | 0.083 |

Si se compara la precisión del baseline contra el intervalo de confianza
del modelo:
- **USD/CAD** queda por encima del baseline.
- **EUR/USD** queda por debajo.
- En los otros 5 activos no se distinguen del azar.

La comparación es orientativa, porque la precisión del baseline también es
una estimación aleatoria. El recall es bajo por construcción: el detector
marca cerca del 4.5% de las barras, mientras que la reversión ocurre en
cerca de la mitad.

## 7. Conclusiones

**Qué aprendí.**
- Sin baseline no hay conclusión. El detector original se veía convincente
  en una gráfica, pero comparado contra el azar con la misma frecuencia no
  muestra ventaja.
- La fuga de información puede ser de un solo renglón (una ventana móvil
  sin `shift(1)`), y las decisiones de validación hay que justificarlas con
  el mecanismo real del modelo. Por eso corregí mi propio "embargo".
- El GARCH, un modelo clásico, sí demuestra valor en los 7 activos.

**Qué mejoraría con más tiempo o recursos.**
- Redefinir el objetivo del detector: predecir la magnitud del movimiento
  o de la volatilidad siguiente, en vez del signo de la reversión, y
  agregar variables como volumen, hora del día o régimen de volatilidad.
- Probar EGARCH o GJR-GARCH (asimetría) para GBP/USD y USD/CAD, donde
  quedan efectos ARCH.
- Cambiar a un proveedor de datos con licencia y a Postgres si se vuelve
  un servicio.
- Ajustar el horario de forex a 24/5 y agregar un tablero de monitoreo
  sobre `evaluation_runs`.

**Qué mencionaría en una entrevista técnica.** El ciclo de
retroalimentación con promoción condicionada: el sistema registra cada
predicción, la compara contra el resultado real, guarda su métrica contra
el baseline y solo reemplaza un modelo si el candidato gana en una
validación walk-forward, con enfriamiento para evitar bucles. También que
reporté con honestidad que uno de los dos modelos no supera al azar.

## 8. Checklist antes de publicar

| Punto | Estado | Cómo se verificó |
|---|---|---|
| README autoexplicativo | ✅ | Se muestra en GitHub con qué hace, arquitectura, comandos, resultados y limitaciones |
| Archivos organizados, sin versiones sueltas | ✅ | La raíz del repositorio solo tiene `.github`, `.vscode`, `data`, `docs`, `notebooks`, `reports`, `src`, `tests` y archivos de configuración |
| Sin credenciales ni datos sensibles | ✅ | Escaneo con `git grep` antes del primer commit (solo 2 coincidencias esperadas, ninguna con valores); `.env` está en `.gitignore` y en GitHub solo aparece `.env.example` |
| Link a la fuente original funcionando | ✅ | Se abrieron finance.yahoo.com y github.com/ranaroussi/yfinance el 24 de septiembre de 2026 |
| Proyecto publicado y accesible | ✅ | Repositorio público |
| Calidad verificada en GitHub | ✅ | CI #1 terminó con éxito (Ruff y pytest) |
| Pipeline de producción corriendo | ⏳ | Pendiente: todavía no hay ninguna corrida del workflow Pipeline |
| Link listo para compartir | ✅ | https://github.com/AlejandroDeAntu/Chronos-Risk-Pipeline |
