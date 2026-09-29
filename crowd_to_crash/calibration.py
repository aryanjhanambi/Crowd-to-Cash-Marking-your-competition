"""Calibration utilities for the endogenous price-impact curve.

The module is deliberately data-source agnostic.  A future empirical adapter only
needs to produce aggregate flow and observed short-horizon price drift.  The same
least-squares routine can then be used for simulated or real observations.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .grid_solver import FloatArray


@dataclass(frozen=True)
class CalibrationResult:
    price_feedback: float
    liquidity_sensitivity: float
    mean_squared_error: float
    observations: int


def impact_curve(
    flow: FloatArray,
    price_feedback: float,
    liquidity_sensitivity: float,
    base_liquidity: float = 1.0,
    minimum_liquidity: float = 0.25,
) -> FloatArray:
    liquidity = minimum_liquidity + (base_liquidity - minimum_liquidity) / (
        1.0 + liquidity_sensitivity * flow**2
    )
    return price_feedback * flow / liquidity


def calibrate_impact_curve(
    flow: FloatArray,
    observed_drift: FloatArray,
    feedback_grid: FloatArray | None = None,
    sensitivity_grid: FloatArray | None = None,
    base_liquidity: float = 1.0,
    minimum_liquidity: float = 0.25,
) -> CalibrationResult:
    """Estimate beta and chi by deterministic grid-search least squares."""

    flow = np.asarray(flow, dtype=float)
    observed_drift = np.asarray(observed_drift, dtype=float)
    if flow.shape != observed_drift.shape or flow.ndim != 1:
        raise ValueError("flow and observed_drift must be equal-length vectors")
    if len(flow) < 3 or not np.all(np.isfinite(flow)) or not np.all(np.isfinite(observed_drift)):
        raise ValueError("calibration requires at least three finite observations")
    feedback_grid = (
        np.linspace(0.0, 2.0, 201) if feedback_grid is None else np.asarray(feedback_grid)
    )
    sensitivity_grid = (
        np.linspace(0.0, 5.0, 201)
        if sensitivity_grid is None
        else np.asarray(sensitivity_grid)
    )

    best = CalibrationResult(0.0, 0.0, float("inf"), len(flow))
    for sensitivity in sensitivity_grid:
        liquidity = minimum_liquidity + (base_liquidity - minimum_liquidity) / (
            1.0 + sensitivity * flow**2
        )
        regressor = flow / liquidity
        predictions = feedback_grid[:, None] * regressor[None, :]
        errors = np.mean((predictions - observed_drift[None, :]) ** 2, axis=1)
        index = int(np.argmin(errors))
        if errors[index] < best.mean_squared_error:
            best = CalibrationResult(
                float(feedback_grid[index]),
                float(sensitivity),
                float(errors[index]),
                len(flow),
            )
    return best


def read_flow_drift_csv(
    path: Path,
    flow_column: str = "aggregate_flow",
    drift_column: str = "observed_price_drift",
) -> tuple[FloatArray, FloatArray]:
    flow: list[float] = []
    drift: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            flow.append(float(row[flow_column]))
            drift.append(float(row[drift_column]))
    return np.asarray(flow), np.asarray(drift)

