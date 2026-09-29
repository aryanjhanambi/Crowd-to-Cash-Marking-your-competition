"""Two-population endogenous-liquidity mean-field game.

Fast and slow traders have distinct trading costs, inventory penalties, terminal
penalties, and population weights.  They share a price and endogenous liquidity,
so their independently optimal controls combine into one aggregate order flow.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .grid_solver import FloatArray, GridParameters, _solve_fokker_planck, _space_derivatives
from .liquidity_model import LiquidityParameters, liquidity_from_flow, price_drift_from_flow


@dataclass(frozen=True)
class TraderType:
    name: str
    weight: float
    trading_cost: float
    inventory_penalty: float
    crowding_strength: float
    terminal_penalty: float
    noise: float
    initial_mean: float
    initial_variance: float

    def validate(self) -> None:
        if not self.name:
            raise ValueError("trader type name cannot be empty")
        if self.weight <= 0 or self.trading_cost <= 0:
            raise ValueError("weight and trading_cost must be positive")
        if min(
            self.inventory_penalty,
            self.crowding_strength,
            self.terminal_penalty,
            self.noise,
            self.initial_variance,
        ) < 0:
            raise ValueError("type penalties, noise, and variance must be non-negative")
        if self.initial_variance == 0:
            raise ValueError("initial_variance must be positive")


@dataclass(frozen=True)
class HeterogeneousParameters:
    horizon: float = 1.0
    time_steps: int = 1_600
    inventory_min: float = -4.0
    inventory_max: float = 4.0
    inventory_points: int = 161
    price_feedback: float = 0.35
    base_liquidity: float = 1.0
    minimum_liquidity: float = 0.25
    liquidity_sensitivity: float = 2.0
    mean_relaxation: float = 0.30
    flow_relaxation: float = 0.15
    tolerance: float = 8e-7
    max_iterations: int = 1_000
    fast: TraderType = TraderType(
        name="fast",
        weight=0.35,
        trading_cost=0.55,
        inventory_penalty=0.50,
        crowding_strength=0.90,
        terminal_penalty=2.5,
        noise=0.18,
        initial_mean=1.0,
        initial_variance=0.12,
    )
    slow: TraderType = TraderType(
        name="slow",
        weight=0.65,
        trading_cost=1.50,
        inventory_penalty=0.16,
        crowding_strength=0.55,
        terminal_penalty=1.5,
        noise=0.22,
        initial_mean=1.0,
        initial_variance=0.20,
    )

    def validate(self) -> None:
        if self.horizon <= 0 or self.time_steps < 2 or self.inventory_points < 5:
            raise ValueError("invalid time or space grid")
        if self.inventory_max <= self.inventory_min:
            raise ValueError("inventory_max must exceed inventory_min")
        if self.price_feedback < 0 or self.liquidity_sensitivity < 0:
            raise ValueError("feedback parameters must be non-negative")
        if not 0 < self.minimum_liquidity <= self.base_liquidity:
            raise ValueError("invalid liquidity bounds")
        if not 0 < self.mean_relaxation <= 1 or not 0 < self.flow_relaxation <= 1:
            raise ValueError("relaxation parameters must be in (0, 1]")
        self.fast.validate()
        self.slow.validate()
        if not math.isclose(self.fast.weight + self.slow.weight, 1.0, abs_tol=1e-12):
            raise ValueError("population weights must sum to one")


@dataclass(frozen=True)
class TypeSolution:
    trader_type: TraderType
    value: FloatArray
    control: FloatArray
    density: FloatArray
    mean_inventory: FloatArray
    variance: FloatArray
    aggregate_flow: FloatArray


@dataclass(frozen=True)
class HeterogeneousSolution:
    time: FloatArray
    inventory: FloatArray
    fast: TypeSolution
    slow: TypeSolution
    mean_inventory: FloatArray
    aggregate_flow: FloatArray
    liquidity: FloatArray
    price_drift: FloatArray
    cumulative_price_impact: FloatArray
    iterations: int
    mean_residual: float
    flow_residual: float
    max_mass_error: float
    min_density: float


def _liquidity_parameters(params: HeterogeneousParameters) -> LiquidityParameters:
    return LiquidityParameters(
        horizon=params.horizon,
        time_steps=params.time_steps,
        inventory_min=params.inventory_min,
        inventory_max=params.inventory_max,
        inventory_points=params.inventory_points,
        price_feedback=params.price_feedback,
        base_liquidity=params.base_liquidity,
        minimum_liquidity=params.minimum_liquidity,
        liquidity_sensitivity=params.liquidity_sensitivity,
    )


def _grid_parameters(
    params: HeterogeneousParameters, trader_type: TraderType
) -> GridParameters:
    return GridParameters(
        horizon=params.horizon,
        time_steps=params.time_steps,
        inventory_min=params.inventory_min,
        inventory_max=params.inventory_max,
        inventory_points=params.inventory_points,
        trading_cost=trader_type.trading_cost,
        inventory_penalty=trader_type.inventory_penalty,
        crowding_strength=trader_type.crowding_strength,
        terminal_penalty=trader_type.terminal_penalty,
        noise=trader_type.noise,
        initial_mean=trader_type.initial_mean,
        initial_variance=trader_type.initial_variance,
    )


def _solve_type_hjb(
    params: HeterogeneousParameters,
    trader_type: TraderType,
    inventory: FloatArray,
    population_mean: FloatArray,
    price_drift: FloatArray,
    dt: float,
    dq: float,
) -> tuple[FloatArray, FloatArray]:
    nt = params.time_steps
    value = np.empty((nt + 1, params.inventory_points), dtype=float)
    control = np.empty_like(value)
    value[-1] = 0.5 * trader_type.terminal_penalty * inventory**2
    diffusion = 0.5 * trader_type.noise**2

    for n in range(nt - 1, -1, -1):
        gradient, laplacian = _space_derivatives(value[n + 1], dq)
        state_cost = (
            0.5 * trader_type.inventory_penalty * inventory**2
            + 0.5
            * trader_type.crowding_strength
            * (inventory - population_mean[n]) ** 2
            - inventory * price_drift[n]
        )
        value[n] = value[n + 1] + dt * (
            state_cost
            - gradient**2 / (2.0 * trader_type.trading_cost)
            + diffusion * laplacian
        )

    for n in range(nt + 1):
        gradient, _ = _space_derivatives(value[n], dq)
        control[n] = -gradient / trader_type.trading_cost
    return value, control


def _solve_type_population(
    params: HeterogeneousParameters,
    trader_type: TraderType,
    inventory: FloatArray,
    control: FloatArray,
    dt: float,
    dq: float,
) -> TypeSolution:
    grid = _grid_parameters(params, trader_type)
    density, mean, variance, _, _ = _solve_fokker_planck(
        grid, inventory, control, dt, dq
    )
    flow = np.sum(control * density, axis=1) * dq
    return TypeSolution(trader_type, np.empty((0, 0)), control, density, mean, variance, flow)


def solve_heterogeneous_game(
    params: HeterogeneousParameters | None = None,
) -> HeterogeneousSolution:
    params = params or HeterogeneousParameters()
    params.validate()
    time = np.linspace(0.0, params.horizon, params.time_steps + 1)
    inventory = np.linspace(params.inventory_min, params.inventory_max, params.inventory_points)
    dt = params.horizon / params.time_steps
    dq = (params.inventory_max - params.inventory_min) / (params.inventory_points - 1)
    initial_mean = (
        params.fast.weight * params.fast.initial_mean
        + params.slow.weight * params.slow.initial_mean
    )
    mean_guess = initial_mean * (1.0 - 0.5 * time / params.horizon)
    flow_guess = np.full_like(time, -0.5 * initial_mean / params.horizon)
    liquidity_params = _liquidity_parameters(params)

    for iteration in range(1, params.max_iterations + 1):
        price_drift = price_drift_from_flow(flow_guess, liquidity_params)
        fast_value, fast_control = _solve_type_hjb(
            params, params.fast, inventory, mean_guess, price_drift, dt, dq
        )
        slow_value, slow_control = _solve_type_hjb(
            params, params.slow, inventory, mean_guess, price_drift, dt, dq
        )
        fast = _solve_type_population(params, params.fast, inventory, fast_control, dt, dq)
        slow = _solve_type_population(params, params.slow, inventory, slow_control, dt, dq)
        generated_mean = params.fast.weight * fast.mean_inventory + params.slow.weight * slow.mean_inventory
        generated_flow = params.fast.weight * fast.aggregate_flow + params.slow.weight * slow.aggregate_flow
        mean_residual = float(np.max(np.abs(generated_mean - mean_guess)))
        flow_residual = float(np.max(np.abs(generated_flow - flow_guess)))
        mean_guess = (1.0 - params.mean_relaxation) * mean_guess + params.mean_relaxation * generated_mean
        flow_guess = (1.0 - params.flow_relaxation) * flow_guess + params.flow_relaxation * generated_flow

        if max(mean_residual, flow_residual) < params.tolerance:
            price_drift = price_drift_from_flow(flow_guess, liquidity_params)
            fast_value, fast_control = _solve_type_hjb(
                params, params.fast, inventory, mean_guess, price_drift, dt, dq
            )
            slow_value, slow_control = _solve_type_hjb(
                params, params.slow, inventory, mean_guess, price_drift, dt, dq
            )
            fast_raw = _solve_type_population(params, params.fast, inventory, fast_control, dt, dq)
            slow_raw = _solve_type_population(params, params.slow, inventory, slow_control, dt, dq)
            fast = replace(fast_raw, value=fast_value)
            slow = replace(slow_raw, value=slow_value)
            mean = params.fast.weight * fast.mean_inventory + params.slow.weight * slow.mean_inventory
            flow = params.fast.weight * fast.aggregate_flow + params.slow.weight * slow.aggregate_flow
            liquidity = liquidity_from_flow(flow, liquidity_params)
            drift = price_drift_from_flow(flow, liquidity_params)
            impact = np.zeros_like(time)
            impact[1:] = np.cumsum(0.5 * (drift[:-1] + drift[1:]) * dt)
            fast_mass = np.sum(fast.density, axis=1) * dq
            slow_mass = np.sum(slow.density, axis=1) * dq
            return HeterogeneousSolution(
                time=time,
                inventory=inventory,
                fast=fast,
                slow=slow,
                mean_inventory=mean,
                aggregate_flow=flow,
                liquidity=liquidity,
                price_drift=drift,
                cumulative_price_impact=impact,
                iterations=iteration,
                mean_residual=mean_residual,
                flow_residual=flow_residual,
                max_mass_error=float(max(np.max(np.abs(fast_mass - 1)), np.max(np.abs(slow_mass - 1)))),
                min_density=float(min(np.min(fast.density), np.min(slow.density))),
            )

    raise RuntimeError(
        f"heterogeneous equilibrium did not converge after {params.max_iterations} iterations"
    )


def write_summary_csv(solution: HeterogeneousSolution, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "time", "population_mean", "fast_mean", "slow_mean", "aggregate_flow",
            "fast_flow", "slow_flow", "liquidity", "price_drift", "cumulative_price_impact",
        ])
        writer.writerows(zip(
            solution.time, solution.mean_inventory, solution.fast.mean_inventory,
            solution.slow.mean_inventory, solution.aggregate_flow,
            solution.fast.aggregate_flow, solution.slow.aggregate_flow,
            solution.liquidity, solution.price_drift,
            solution.cumulative_price_impact, strict=True,
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/heterogeneous.csv"))
    args = parser.parse_args()
    result = solve_heterogeneous_game()
    write_summary_csv(result, args.output)
    print(f"converged in {result.iterations} iterations")
    print(f"terminal population mean: {result.mean_inventory[-1]:.6f}")
    print(f"terminal fast mean: {result.fast.mean_inventory[-1]:.6f}")
    print(f"terminal slow mean: {result.slow.mean_inventory[-1]:.6f}")
    print(f"peak selling flow: {np.min(result.aggregate_flow):.6f}")
    print(f"terminal price impact: {result.cumulative_price_impact[-1]:.6f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
