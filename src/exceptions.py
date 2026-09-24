"""Excepciones específicas del pipeline.

Se define una jerarquía propia para que cada capa (ingesta, validación,
modelado, persistencia) pueda fallar de forma explícita y distinguible.
Nunca se usa ``except Exception`` genérico en el resto del proyecto: siempre
se captura uno de estos tipos (u otro tipo específico de terceros) y se
relanza o se registra antes de propagar.
"""

from __future__ import annotations


class TradingPipelineError(Exception):
    """Clase base para todas las excepciones propias del pipeline."""


class DataUnavailableError(TradingPipelineError):
    """La fuente de datos no devolvió datos utilizables.

    Se lanza cuando ``yfinance`` responde con un DataFrame vacío, con menos
    filas que el mínimo requerido, o cuando la petición falla después de
    agotar los reintentos configurados.
    """


class DataContractError(TradingPipelineError):
    """Un DataFrame no cumple el contrato de esquema esperado (Pandera).

    Se lanza cuando el esquema Pandera detecta columnas faltantes, tipos
    incorrectos, nulos no permitidos, duplicados en el índice temporal o
    violaciones de rango. El pipeline debe detenerse aquí, nunca continuar
    con datos que no pasaron la validación.
    """


class ModelFitError(TradingPipelineError):
    """El ajuste de un modelo (GARCH o detector de anomalías) falló."""


class InsufficientHistoryError(TradingPipelineError):
    """No hay suficiente historial para entrenar o evaluar con confianza."""


class ModelRegistryError(TradingPipelineError):
    """Error al leer, escribir o promover una versión de modelo."""


class PersistenceError(TradingPipelineError):
    """Error al leer o escribir en la base de datos SQLite del pipeline."""
