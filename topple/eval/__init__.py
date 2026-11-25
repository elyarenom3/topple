"""Evaluation: calibration metrics, scoring rules, Sobol indices and result tables (Section 11)."""

from .metrics import (  # noqa: F401
    brier_score,
    murphy_decomposition,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_diagram,
    interval_coverage,
    pit_values,
    margin_errors,
    spearman,
)
