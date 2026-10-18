"""Reporte: escapado del HTML y robustez ante activos sin resultados."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.reporting.data import AssetSection, ReportData, build_findings
from src.reporting.html import render_html
from src.reporting.pdf import render_pdf

HOSTILE = "<script>alert(1)</script>"


def _failed_report() -> ReportData:
    asset = AssetSection(
        symbol="^BAD",
        name=HOSTILE,
        volatility=None,
        anomaly=None,
        n_repaired_bars=0,
        n_daily_bars=0,
        zero_return_share=None,
        errors=('<img src=x onerror="alert(2)">',),
    )
    assets = (asset,)
    return ReportData(
        generated_at=datetime(2026, 10, 17, tzinfo=UTC),
        alpha=0.05,
        assets=assets,
        findings=build_findings(assets, 0.05),
        live_runs=(),
        methodology={"Fuente": "<b>Yahoo</b>"},
    )


def test_html_escapes_names_errors_and_methodology(tmp_path: Path) -> None:
    html = render_html(_failed_report(), {}, {}, tmp_path / "r.html")
    text = html.read_text(encoding="utf-8")

    assert "<script>alert" not in text
    assert "<img src=x" not in text
    assert "&lt;script&gt;" in text
    assert "&lt;b&gt;Yahoo&lt;/b&gt;" in text


def test_report_without_volatility_still_renders_with_errors(
    tmp_path: Path,
) -> None:
    data = _failed_report()

    assert data.findings[0].startswith("No hay resultados de volatilidad")
    render_html(data, {}, {}, tmp_path / "r.html")
    pdf = render_pdf(data, tmp_path / "r.pdf")
    assert pdf.read_bytes().startswith(b"%PDF")
