"""Finite-difference solver for the baseline HJB--Fokker--Planck system.

Unlike :mod:`crowd_to_crash.baseline`, this module does not assume that the value
function is quadratic or that the population density is Gaussian.  Both are stored
on a time-by-inventory grid.  The quadratic baseline remains useful as an exact-ish
benchmark for this more general numerical method.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GridParameters:
    """Economic and numerical parameters for the grid MFG solver."""

    horizon: float = 1.0
    time_steps: int = 2_000
    inventory_min: float = -4.0
    inventory_max: float = 4.0
    inventory_points: int = 161
    trading_cost: float = 1.0
    inventory_penalty: float = 0.25
    crowding_strength: float = 0.75
    terminal_penalty: float = 2.0
    noise: float = 0.20
    initial_mean: float = 1.0
    initial_variance: float = 0.16
    relaxation: float = 0.35
    tolerance: float = 2e-7
    max_iterations: int = 500

    def validate(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.time_steps < 2 or self.inventory_points < 5:
            raise ValueError("the grid is too small")
        if self.inventory_max <= self.inventory_min:
            raise ValueError("inventory_max must exceed inventory_min")
        if self.trading_cost <= 0:
            raise ValueError("trading_cost must be positive")
        if min(
            self.inventory_penalty,
            self.crowding_strength,
            self.terminal_penalty,
            self.noise,
            self.initial_variance,
        ) < 0:
            raise ValueError("penalties, noise, and variance must be non-negative")
        if self.initial_variance == 0:
            raise ValueError("the grid solver requires positive initial_variance")
        if not 0 < self.relaxation <= 1:
            raise ValueError("relaxation must be in (0, 1]")


@dataclass(frozen=True)
class GridSolution:
    time: FloatArray
    inventory: FloatArray
    value: FloatArray
    control: FloatArray
    density: FloatArray
    mean_inventory: FloatArray
    variance: FloatArray
    iterations: int
    residual: float
    max_mass_error: float
    min_density: float


def _space_derivatives(values: FloatArray, dq: float) -> tuple[FloatArray, FloatArray]:
    """Return first and second inventory derivatives on a uniform grid."""

    gradient = np.gradient(values, dq, edge_order=2)
    laplacian = np.empty_like(values)
    laplacian[1:-1] = (values[2:] - 2.0 * values[1:-1] + values[:-2]) / dq**2

    # Copy the closest interior curvature at the artificial edges.  The inventory
    # domain is chosen wide enough that the density remains far from these edges.
    # We separately impose zero probability flux in the Fokker--Planck solve.
    laplacian[0] = laplacian[1]
    laplacian[-1] = laplacian[-2]
    return gradient, laplacian


def _initial_density(params: GridParameters, inventory: FloatArray) -> FloatArray:
    density = np.exp(
        -0.5 * (inventory - params.initial_mean) ** 2 / params.initial_variance
    ) / math.sqrt(2.0 * math.pi * params.initial_variance)
    dq = inventory[1] - inventory[0]
    density /= np.sum(density) * dq
    return density


def _solve_hjb(
    params: GridParameters,
    inventory: FloatArray,
    mean_path: FloatArray,
    dt: float,
    dq: float,
) -> tuple[FloatArray, FloatArray]:
    """Solve the HJB backward for a proposed population mean path."""

    nt = params.time_steps
    value = np.empty((nt + 1, params.inventory_points), dtype=float)
    control = np.empty_like(value)
    value[-1] = 0.5 * params.terminal_penalty * inventory**2
    diffusion = 0.5 * params.noise**2

    for n in range(nt - 1, -1, -1):
        gradient, laplacian = _space_derivatives(value[n + 1], dq)
        running_state_cost = (
            0.5 * params.inventory_penalty * inventory**2
            + 0.5
            * params.crowding_strength
            * (inventory - mean_path[n]) ** 2
        )
        # Minimizing eta*u^2/2 + u*V_q gives -V_q^2/(2*eta).
        minimized_hamiltonian = -(gradient**2) / (2.0 * params.trading_cost)
        value[n] = value[n + 1] + dt * (
            running_state_cost + minimized_hamiltonian + diffusion * laplacian
        )

    for n in range(nt + 1):
        gradient, _ = _space_derivatives(value[n], dq)
        control[n] = -gradient / params.trading_cost
    return value, control


def _solve_fokker_planck(
    params: GridParameters,
    inventory: FloatArray,
    control: FloatArray,
    dt: float,
    dq: float,
) -> tuple[FloatArray, FloatArray, FloatArray, float, float]:
    """Move the population forward with a conservative upwind flux."""

    nt = params.time_steps
    density = np.zeros((nt + 1, params.inventory_points), dtype=float)
    density[0] = _initial_density(params, inventory)
    diffusion = 0.5 * params.noise**2

    # Sufficient positivity condition for explicit upwind advection plus diffusion.
    max_speed = float(np.max(np.abs(control[:-1])))
    cfl = max_speed * dt / dq + 2.0 * diffusion * dt / dq**2
    if cfl > 0.95:
        raise ValueError(
            f"unstable grid: CFL number is {cfl:.3f}; increase time_steps or "
            "inventory spacing"
        )

    for n in range(nt):
        current = density[n]
        face_speed = 0.5 * (control[n, :-1] + control[n, 1:])
        advective_flux = np.where(
            face_speed >= 0.0,
            face_speed * current[:-1],
            face_speed * current[1:],
        )
        diffusive_flux = -diffusion * (current[1:] - current[:-1]) / dq

        # There are nq+1 cell faces. Boundary fluxes are zero, so probability
        # cannot leave the truncated inventory domain.
        total_flux = np.zeros(params.inventory_points + 1, dtype=float)
        total_flux[1:-1] = advective_flux + diffusive_flux
        density[n + 1] = current - (dt / dq) * (
            total_flux[1:] - total_flux[:-1]
        )

        if float(np.min(density[n + 1])) < -1e-12:
            raise ArithmeticError("density became negative despite the CFL check")
        density[n + 1] = np.maximum(density[n + 1], 0.0)

    mass = np.sum(density, axis=1) * dq
    means = np.sum(density * inventory[None, :], axis=1) * dq
    variances = (
        np.sum(density * (inventory[None, :] - means[:, None]) ** 2, axis=1) * dq
    )
    return (
        density,
        means,
        variances,
        float(np.max(np.abs(mass - 1.0))),
        float(np.min(density)),
    )


def solve_grid(params: GridParameters | None = None) -> GridSolution:
    """Solve the coupled HJB--Fokker--Planck fixed point on a grid."""

    params = params or GridParameters()
    params.validate()
    time = np.linspace(0.0, params.horizon, params.time_steps + 1)
    inventory = np.linspace(
        params.inventory_min, params.inventory_max, params.inventory_points
    )
    dt = params.horizon / params.time_steps
    dq = (params.inventory_max - params.inventory_min) / (
        params.inventory_points - 1
    )

    mean_guess = params.initial_mean * (1.0 - 0.5 * time / params.horizon)
    residual = math.inf

    for iteration in range(1, params.max_iterations + 1):
        value, control = _solve_hjb(params, inventory, mean_guess, dt, dq)
        density, generated_mean, variance, mass_error, min_density = (
            _solve_fokker_planck(params, inventory, control, dt, dq)
        )
        residual = float(np.max(np.abs(generated_mean - mean_guess)))
        mean_guess = (
            (1.0 - params.relaxation) * mean_guess
            + params.relaxation * generated_mean
        )
        if residual < params.tolerance:
            value, control = _solve_hjb(params, inventory, mean_guess, dt, dq)
            density, mean, variance, mass_error, min_density = (
                _solve_fokker_planck(params, inventory, control, dt, dq)
            )
            return GridSolution(
                time=time,
                inventory=inventory,
                value=value,
                control=control,
                density=density,
                mean_inventory=mean,
                variance=variance,
                iterations=iteration,
                residual=residual,
                max_mass_error=mass_error,
                min_density=min_density,
            )

    raise RuntimeError(
        f"grid mean-field iteration did not converge after {params.max_iterations} "
        f"iterations (residual={residual:.3e})"
    )


def write_summary_csv(solution: GridSolution, destination: Path) -> None:
    """Write time-dependent population moments without dumping the full grid."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "mean_inventory", "inventory_variance"])
        writer.writerows(
            zip(
                solution.time,
                solution.mean_inventory,
                solution.variance,
                strict=True,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/grid_baseline.csv")
    )
    args = parser.parse_args()
    result = solve_grid()
    write_summary_csv(result, args.output)
    print(f"converged in {result.iterations} iterations")
    print(f"fixed-point residual: {result.residual:.3e}")
    print(f"maximum probability-mass error: {result.max_mass_error:.3e}")
    print(f"minimum density: {result.min_density:.3e}")
    print(f"initial mean inventory: {result.mean_inventory[0]:.6f}")
    print(f"terminal mean inventory: {result.mean_inventory[-1]:.6f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

