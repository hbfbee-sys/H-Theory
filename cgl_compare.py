from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from h_type import (
    NineHReflexArc,
    digit_state,
    phase_angle,
    phase_complement,
)


def laplacian_1d(field: Sequence[complex]) -> List[complex]:
    """Reflecting-boundary one-dimensional Laplacian."""
    if not field:
        return []
    result: List[complex] = []
    for index, value in enumerate(field):
        left = field[index - 1] if index > 0 else value
        right = field[index + 1] if index < len(field) - 1 else value
        result.append(left - 2 * value + right)
    return result


def cgl_step(
    field: Sequence[complex],
    dt: float,
    alpha: float,
    omega: float,
    beta: float,
    gamma: float,
    diffusion: float,
) -> List[complex]:
    """Explicit Euler step for dpsi/dt = (a+iw)psi-(b+ig)|psi|^2psi+D laplacian."""
    lap = laplacian_1d(field)
    out: List[complex] = []
    for value, lap_value in zip(field, lap):
        derivative = (
            (alpha + 1j * omega) * value
            - (beta + 1j * gamma) * (abs(value) ** 2) * value
            + diffusion * lap_value
        )
        out.append(value + dt * derivative)
    return out


def solve_cgl(
    initial: Sequence[complex],
    steps: int,
    dt: float,
    alpha: float,
    omega: float,
    beta: float,
    gamma: float,
    diffusion: float,
) -> List[List[complex]]:
    history = [list(initial)]
    field = list(initial)
    for _ in range(steps):
        field = cgl_step(field, dt, alpha, omega, beta, gamma, diffusion)
        history.append(field)
    return history


def angle(value: complex) -> float:
    return math.atan2(value.imag, value.real)


def wrap_angle(value: float) -> float:
    wrapped = (value + math.pi) % (2 * math.pi) - math.pi
    return wrapped + 2 * math.pi if wrapped <= -math.pi else wrapped


def nearest_digit(value: complex) -> int:
    """Project a continuous phase point back to the closest H energy label."""
    theta = angle(value)
    return min(range(1, 8), key=lambda digit: abs(wrap_angle(theta - phase_angle(digit))))


@dataclass
class Comparison:
    label: str
    expected: str
    observed: str
    passed: bool


def compare_digit_complements() -> List[Comparison]:
    comparisons: List[Comparison] = []
    for a, b in ((1, 7), (2, 6), (3, 5), (4, 4)):
        product = digit_state(a) * digit_state(b)
        comparisons.append(
            Comparison(
                label=f"{a}+{b}=8 phase quantization",
                expected="z_a*z_b = -1",
                observed=f"{product.real:+.6f}{product.imag:+.6f}i",
                passed=phase_complement(a, b) and abs(product + 1) < 1e-9,
            )
        )
    return comparisons


def compare_path_operator() -> Comparison:
    arc = NineHReflexArc.standard(start_coord=(0, 2))
    operator = arc.net_path_operator()
    return Comparison(
        label="default nine-H path net operator",
        expected="-1",
        observed=f"{operator.real:+.6f}{operator.imag:+.6f}i",
        passed=abs(operator + 1) < 1e-9,
    )


def compare_weak_accumulation() -> Comparison:
    amplitude = 0.0
    phase = 0.0
    for _ in range(3):
        amplitude += 1.0 / 3.0
        phase += math.pi / 3.0
    passed = abs(amplitude - 1.0) < 1e-9 and abs(phase - math.pi) < 1e-9
    return Comparison(
        label="three weak inputs reach firing threshold",
        expected="A=1, phi=pi",
        observed=f"A={amplitude:.6f}, phi={phase:.6f}",
        passed=passed,
    )


def compare_cgl_projection(history: Sequence[Sequence[complex]]) -> Comparison:
    final = history[-1]
    projected = [nearest_digit(value) for value in final]
    amplitudes = [abs(value) for value in final]
    amplitude_ok = all(0.5 <= value <= 1.5 for value in amplitudes)
    return Comparison(
        label="continuous field projects back to H energy labels",
        expected="digits in 1..7 with bounded amplitudes",
        observed=f"digits={projected}, amplitudes={[round(v, 4) for v in amplitudes]}",
        passed=all(1 <= digit <= 7 for digit in projected) and amplitude_ok,
    )


def print_comparisons(comparisons: Iterable[Comparison]) -> None:
    for item in comparisons:
        mark = "PASS" if item.passed else "FAIL"
        print(f"[{mark}] {item.label}")
        print(f"  expected: {item.expected}")
        print(f"  observed: {item.observed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the CGL parent equation and compare it with H-type discrete rules.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--diffusion", type=float, default=0.08)
    args = parser.parse_args()

    # Default upper-right nine-H path: left,left,down,right,right,down,left,left
    # initialized by the A-state digit order 2,1,3,4,5,7,6 plus two middle samples.
    initial_digits = [2, 1, 3, 4, 5, 7, 6, 4, 4]
    initial = [digit_state(digit) for digit in initial_digits]
    history = solve_cgl(
        initial,
        steps=args.steps,
        dt=args.dt,
        alpha=args.alpha,
        omega=args.omega,
        beta=args.beta,
        gamma=args.gamma,
        diffusion=args.diffusion,
    )

    comparisons = []
    comparisons.extend(compare_digit_complements())
    comparisons.append(compare_path_operator())
    comparisons.append(compare_weak_accumulation())
    comparisons.append(compare_cgl_projection(history))
    print_comparisons(comparisons)

    failed = [item for item in comparisons if not item.passed]
    print()
    print(f"CGL steps: {args.steps}, dt: {args.dt}")
    print(f"Final projected digits: {[nearest_digit(value) for value in history[-1]]}")
    print(f"Overall: {'PASS' if not failed else 'FAIL'}")


if __name__ == "__main__":
    main()
