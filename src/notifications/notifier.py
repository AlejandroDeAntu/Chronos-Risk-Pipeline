"""Notificación de fallos: nunca silenciosa.

GitHub Actions ya envía un correo automático cuando un job termina con
código de salida distinto de cero (ver ``.github/workflows/``); esta capa
añade, opcionalmente, un webhook (Slack/Discord/Telegram) para notificación
en tiempo real, sin depender de infraestructura adicional de pago.

Regla dura de este módulo: ``notify_failure`` nunca traga la excepción
original. Su contrato es notificar y luego dejar que la excepción se siga
propagando (o ser llamada desde un ``except`` que ya va a re-lanzar).
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)

_WEBHOOK_ENV_VAR = "PIPELINE_ALERT_WEBHOOK_URL"
_WEBHOOK_TIMEOUT_SECONDS = 10
_MAX_ERROR_CHARS = 300


def _safe_error_text(error: BaseException) -> str:
    """Mensaje acotado: una excepción de red puede traer URLs o rutas."""
    text = " ".join(str(error).split())
    return text[:_MAX_ERROR_CHARS]


def _is_allowed_webhook(url: str) -> bool:
    """Solo HTTPS: el payload no debe viajar en claro.

    No se fija una lista de proveedores para no silenciar un webhook ya
    configurado; la URL viene de un secret, no de entrada externa.
    """
    parts = urlsplit(url)
    return parts.scheme == "https" and bool(parts.hostname)


def notify_failure(context: str, error: BaseException) -> None:
    """Registra un fallo y, si está configurado, lo envía por webhook.

    Parameters
    ----------
    context:
        Descripción corta de dónde ocurrió el fallo (ej. ``"predict:^GSPC"``).
    error:
        Excepción capturada que motivó la notificación.

    Notes
    -----
    Un fallo al enviar el webhook se registra como advertencia pero NO
    reemplaza ni oculta el error original: la función no relanza nada, es
    responsabilidad del llamador seguir propagando ``error`` después de
    llamar a esta función.
    """
    logger.error(
        "Fallo en %s: %s: %s",
        context,
        type(error).__name__,
        _safe_error_text(error),
    )

    webhook_url = os.environ.get(_WEBHOOK_ENV_VAR)
    if not webhook_url:
        logger.info(
            "'%s' no configurada; no se envía notificación por webhook.",
            _WEBHOOK_ENV_VAR,
        )
        return
    if not _is_allowed_webhook(webhook_url):
        logger.warning(
            "'%s' debe ser una URL HTTPS; no se envía.", _WEBHOOK_ENV_VAR
        )
        return

    payload = {
        "text": (
            f"Pipeline de trading falló en `{context}`: "
            f"{type(error).__name__}: {_safe_error_text(error)}"
        )
    }
    try:
        response = requests.post(
            webhook_url, json=payload, timeout=_WEBHOOK_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as webhook_error:
        # Solo el tipo: el mensaje de requests incluye la URL con su token.
        logger.warning(
            "No se pudo enviar la notificación por webhook: %s",
            type(webhook_error).__name__,
        )
