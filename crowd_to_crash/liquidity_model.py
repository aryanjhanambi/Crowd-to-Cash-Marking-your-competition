"""Mean-field market model with endogenous liquidity and price feedback.

Aggregate order flow moves the asset price.  The same order flow reduces market
liquidity, which amplifies the price response.  Traders holding inventory account
for that endogenous price drift when choosing their optimal trading rate.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .grid_solver import (
    FloatArray,
    GridParameters,
    _solve_fokker_planck,
    _space_derivatives,
)


@dataclass(frozen=True)
class LiquidityParameters(GridParameters):
    """Parameters for the endogenous-liquidity extension."""

    price_feedback: float = 0.35
    base_liquidity: float = 1.0
    minimum_liquidity: float = 0.25
    liquidity_sensitivity: float = 2.0
    flow_relaxation: float = 0.15
    tolerance: float = 5e-7
    max_iterations: int = 1_000

    def validate(self) -> None:
        super().validate()
        if self.price_feedback < 0:
            raise ValueError("price_feedback must be non-negative")
        if self.base_liquidity <= 0:
            raise ValueError("base_liquidity must be positive")
        if not 0 < self.minimum_liquidity <= self.base_liquidity:
            raise ValueError(
                "minimum_liquidity must be positive and no larger than base_liquidity"
            )
        if self.liquidity_sensitivity < 0:
            raise ValueError("liquidity_sensitivity must be non-negative")
        if not 0 < self.flow_relaxation <= 1:
            raise ValueError("flow_relaxation must be in (0, 1]")


@dataclass(frozen=True)
class LiquiditySolution:
    time: FloatArray
    inventory: FloatArray
    value: FloatArray
    control: FloatArray
    density: FloatArray
    mean_inventory: FloatArray
    variance: FloatArray
    aggregate_flow: FloatArray
    liquidity: FloatArray
    price_drift: FloatArray
    cumulative_price_impact: FloatArray
    iterations: int
    mean_residual: float
    flow_residual: float
    max_mass_error: float
    min_density: float


def liquidity_from_flow(
    flow: FloatArray, params: LiquidityParameters
) -> FloatArray:
    """Smooth positive liquidity curve that falls as absolute flow grows."""

    return params.minimum_liquidity + (
        params.base_liquidity - params.minimum_liquidity
    ) / (1.0 + params.liquidity_sensitivity * flow**2)


def price_drift_from_flow(
    flow: FloatArray, params: LiquidityParameters
) -> FloatArray:
    """Permanent price drift caused by aggregate order flow."""

    liquidity = liquidity_from_flow(flow, params)
    return params.price_feedback * flow / liquidity


def _solve_liquidity_hjb(
    params: LiquidityParameters,
    inventory: FloatArray,
    mean_path: FloatArray,
    flow_path: FloatArray,
    dt: float,
    dq: float,
) -> tuple[FloatArray, FloatArray]:
    """Solve the HJB for proposed population mean and flow paths."""

    nt = params.time_steps
    value = np.empty((nt + 1, params.inventory_points), dtype=float)
    control = np.empty_like(value)
    value[-1] = 0.5 * params.terminal_penalty * inventory**2
    diffusion = 0.5 * params.noise**2
    price_drift = price_drift_from_flow(flow_path, params)

    for n in range(nt - 1, -1, -1):
        gradient, laplacian = _space_derivatives(value[n + 1], dq)
        running_state_cost = (
            0.5 * params.inventory_penalty * inventory**2
            + 0.5
            * params.crowding_strength
            * (inventory - mean_path[n]) ** 2
            - inventory * price_drift[n]
        )
        minimized_hamiltonian = -(gradient**2) / (2.0 * params.trading_cost)
        value[n] = value[n + 1] + dt * (
            running_state_cost + minimized_hamiltonian + diffusion * laplacian
        )

    for n in range(nt + 1):
        gradient, _ = _space_derivatives(value[n], dq)
        control[n] = -gradient / params.trading_cost
    return value, control


def _population_flow(
    control: FloatArray, density: FloatArray, dq: float
) -> FloatArray:
    """Average trading rate A_t = integral u(t,q)m(t,q)dq."""

    return np.sum(control * density, axis=1) * dq


def solve_liquidity_game(
    params: LiquidityParameters | None = None,
    initial_mean_guess: FloatArray | None = None,
    initial_flow_guess: FloatArray | None = None,
) -> LiquiditySolution:
    """Solve the joint mean-inventory and aggregate-flow fixed point."""

    params = params or LiquidityParameters()
    params.validate()
    time = np.linspace(0.0, params.horizon, params.time_steps + 1)
    inventory = np.linspace(
        params.inventory_min, params.inventory_max, params.inventory_points
    )
    dt = params.horizon / params.time_steps
    dq = (params.inventory_max - params.inventory_min) / (
        params.inventory_points - 1
    )

    if initial_mean_guess is None:
        mean_guess = params.initial_mean * (1.0 - 0.5 * time / params.horizon)
    else:
        mean_guess = np.asarray(initial_mean_guess, dtype=float).copy()
        if mean_guess.shape != time.shape:
            raise ValueError("initial_mean_guess must have one value per time point")
        mean_guess[0] = params.initial_mean

    if initial_flow_guess is None:
        flow_guess = np.full_like(time, -0.5 * params.initial_mean / params.horizon)
    else:
        flow_guess = np.asarray(initial_flow_guess, dtype=float).copy()
        if flow_guess.shape != time.shape:
            raise ValueError("initial_flow_guess must have one value per time point")
    mean_residual = math.inf
    flow_residual = math.inf

    for iteration in range(1, params.max_iterations + 1):
        value, control = _solve_liquidity_hjb(
            params, inventory, mean_guess, flow_guess, dt, dq
        )
        density, generated_mean, variance, mass_error, min_density = (
            _solve_fokker_planck(params, inventory, control, dt, dq)
        )
        generated_flow = _population_flow(control, density, dq)
        mean_residual = float(np.max(np.abs(generated_mean - mean_guess)))
        flow_residual = float(np.max(np.abs(generated_flow - flow_guess)))

        mean_guess = (
            (1.0 - params.relaxation) * mean_guess
            + params.relaxation * generated_mean
        )
        flow_guess = (
            (1.0 - params.flow_relaxation) * flow_guess
            + params.flow_relaxation * generated_flow
        )

        if max(mean_residual, flow_residual) < params.tolerance:
            value, control = _solve_liquidity_hjb(
                params, inventory, mean_guess, flow_guess, dt, dq
            )
            density, mean, variance, mass_error, min_density = (
                _solve_fokker_planck(params, inventory, control, dt, dq)
            )
            flow = _population_flow(control, density, dq)
            liquidity = liquidity_from_flow(flow, params)
            price_drift = price_drift_from_flow(flow, params)
            cumulative_price_impact = np.zeros_like(time)
            cumulative_price_impact[1:] = np.cumsum(
                0.5 * (price_drift[:-1] + price_drift[1:]) * dt
            )
            return LiquiditySolution(
                time=time,
                inventory=inventory,
                value=value,
                control=control,
                density=density,
                mean_inventory=mean,
                variance=variance,
                aggregate_flow=flow,
                liquidity=liquidity,
                price_drift=price_drift,
                cumulative_price_impact=cumulative_price_impact,
                iterations=iteration,
                mean_residual=mean_residual,
                flow_residual=flow_residual,
                max_mass_error=mass_error,
                min_density=min_density,
            )

    raise RuntimeError(
        "liquidity mean-field iteration did not converge after "
        f"{params.max_iterations} iterations "
        f"(mean residual={mean_residual:.3e}, flow residual={flow_residual:.3e})"
    )


def write_summary_csv(solution: LiquiditySolution, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time",
                "mean_inventory",
                "inventory_variance",
                "aggregate_flow",
                "liquidity",
                "price_drift",
                "cumulative_price_impact",
            ]
        )
        writer.writerows(
            zip(
                solution.time,
                solution.mean_inventory,
                solution.variance,
                solution.aggregate_flow,
                solution.liquidity,
                solution.price_drift,
                solution.cumulative_price_impact,
                strict=True,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/liquidity_feedback.csv")
    )
    args = parser.parse_args()
    result = solve_liquidity_game()
    write_summary_csv(result, args.output)
    print(f"converged in {result.iterations} iterations")
    print(f"mean residual: {result.mean_residual:.3e}")
    print(f"flow residual: {result.flow_residual:.3e}")
    print(f"maximum probability-mass error: {result.max_mass_error:.3e}")
    print(f"terminal mean inventory: {result.mean_inventory[-1]:.6f}")
    print(f"peak selling flow: {np.min(result.aggregate_flow):.6f}")
    print(f"minimum liquidity: {np.min(result.liquidity):.6f}")
    print(f"terminal price impact: {result.cumulative_price_impact[-1]:.6f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
