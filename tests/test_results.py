"""Aislamiento de fallos por activo en las etapas del pipeline."""

from __future__ import annotations

import pytest

from src.exceptions import DataContractError
from src.pipeline.results import AssetStageResult, Stage, run_isolated
from src.schemas import SignalType


def _boom(error: Exception) -> AssetStageResult:
    raise error


@pytest.mark.parametrize(
    "error", [DataContractError("contrato"), KeyError("columna")]
)
def test_failure_of_one_asset_is_contained_and_reported(
    error: Exception,
) -> None:
    result = run_isolated(
        lambda: _boom(error), "^TEST", SignalType.ANOMALY, Stage.PREDICT
    )

    assert result.succeeded is False
    assert type(error).__name__ in result.detail


def test_success_passes_result_through() -> None:
    ok = AssetStageResult(
        asset_symbol="^TEST",
        signal_type=SignalType.ANOMALY,
        stage=Stage.PREDICT,
        succeeded=True,
        detail="ok",
    )

    assert (
        run_isolated(lambda: ok, "^TEST", SignalType.ANOMALY, Stage.PREDICT)
        is ok
    )
