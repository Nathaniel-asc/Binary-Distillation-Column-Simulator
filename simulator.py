import math
from typing import cast

import matplotlib.pyplot as plt
import numpy as np


def y_equilibrium(x: float | np.ndarray, alpha: float) -> float | np.ndarray:
    # Binary VLE for a constant-relative-volatility system.
    return (alpha * x) / (1.0 + (alpha - 1.0) * x)


def x_from_y_equilibrium(y: float | np.ndarray, alpha: float) -> float | np.ndarray:
    # Invert the equilibrium relation so we can step from vapor composition back to liquid composition.
    return y / (alpha - (alpha - 1.0) * y)


def get_float(prompt: str) -> float:
    return float(input(prompt).strip())


def rectifying_line(x: float | np.ndarray, reflux_ratio: float, x_distillate: float) -> float | np.ndarray:
    # Operating line for the rectifying section above the feed stage.
    return (reflux_ratio / (reflux_ratio + 1.0)) * x + x_distillate / (reflux_ratio + 1.0)


def stripping_line(
    x: float | np.ndarray,
    x_intersection: float,
    y_intersection: float,
    x_bottoms: float,
) -> float | np.ndarray:
    # Operating line for the stripping section below the feed stage.
    slope = (y_intersection - x_bottoms) / (x_intersection - x_bottoms)
    return slope * (x - x_bottoms) + x_bottoms


def minimum_reflux_ratio(alpha: float, z_feed: float, x_distillate: float) -> float:
    # Minimum reflux is found from the feed pinch condition on the equilibrium curve.
    y_feed_eq = cast(float, y_equilibrium(z_feed, alpha))
    r_min_slope = (y_feed_eq - x_distillate) / (z_feed - x_distillate)
    return r_min_slope / (1.0 - r_min_slope)


def calculate_stages(
    alpha: float,
    z_feed: float,
    x_distillate: float,
    x_bottoms: float,
    reflux_ratio: float,
) -> tuple[float, list[tuple[tuple[float, float], tuple[float, float]]], float, float]:
    # The feed is assumed saturated liquid, so the q-line is vertical at z_feed.
    r_min = minimum_reflux_ratio(alpha, z_feed, x_distillate)

    if reflux_ratio <= r_min:
        raise ValueError(f"Reflux ratio must be greater than minimum reflux ratio ({r_min:.3f}).")

    # The rectifying and stripping lines meet at the feed composition for q = 1.
    x_intersection = z_feed
    y_intersection = cast(float, rectifying_line(x_intersection, reflux_ratio, x_distillate))

    # Step off stages by alternating between equilibrium and operating lines.
    stage_count = 0
    y_current = x_distillate
    x_current = x_distillate
    steps = []

    for _ in range(1000):
        # Horizontal move to the equilibrium curve: vapor and liquid are in equilibrium on each stage.
        x_eq = x_from_y_equilibrium(y_current, alpha)
        steps.append(((x_current, y_current), (x_eq, y_current)))
        stage_count += 1

        if x_eq <= x_bottoms:
            break

        # Vertical move to the relevant operating line for the next stage.
        if x_eq >= x_intersection:
            y_next = rectifying_line(x_eq, reflux_ratio, x_distillate)
        else:
            y_next = stripping_line(x_eq, x_intersection, y_intersection, x_bottoms)

        steps.append(((x_eq, y_current), (x_eq, y_next)))
        x_current = x_eq
        y_current = y_next
    else:
        raise RuntimeError("Stage stepping did not converge.")

    return stage_count, steps, x_intersection, y_intersection


def plot_mccabe_thiele(
    alpha: float,
    z_feed: float,
    x_distillate: float,
    x_bottoms: float,
    reflux_ratio: float,
    steps,
    x_intersection: float,
    y_intersection: float,
):
    # Build the equilibrium and operating curves on a mole-fraction basis.
    x = np.linspace(0.0, 1.0, 500)
    y_eq = y_equilibrium(x, alpha)
    y_rect = rectifying_line(x, reflux_ratio, x_distillate)
    y_strip = stripping_line(x, x_intersection, y_intersection, x_bottoms)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, y_eq, label="Equilibrium curve: y vs x for the light key")
    ax.plot(x, x, "--", label="Reference line: y = x")
    ax.plot(x, y_rect, label="Rectifying operating line")
    ax.plot(x, y_strip, label="Stripping operating line")
    ax.axvline(z_feed, linestyle=":", label="Feed composition line: zF (q = 1)")

    # Draw the stepped path used to count theoretical stages.
    for idx, ((x1, y1), (x2, y2)) in enumerate(steps):
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.2, label="Theoretical stage staircase" if idx == 0 else None)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x = liquid mole fraction of the light key")
    ax.set_ylabel("y = vapor mole fraction of the light key")
    ax.set_title("McCabe–Thiele Diagram for Binary Distillation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    print("McCabe–Thiele Binary Distillation Stage Estimator")
    print("This program estimates the number of theoretical stages for a binary distillation column.")
    print("Enter all compositions as mole fractions between 0 and 1.")
    # User inputs are all mole fractions except the reflux ratio.
    alpha = get_float("Relative volatility of the light key over the heavy key, alpha: ")
    # z_feed: feed composition, or the mole fraction of the more volatile component in the feed.
    z_feed = get_float("Feed composition, zF, or feed mole fraction of the light key: ")
    # x_distillate: desired distillate composition, usually the light-key purity at the top product.
    x_distillate = get_float("Distillate composition, xD, or desired top-product mole fraction of the light key: ")
    # x_bottoms: desired bottoms composition, usually the light-key mole fraction left in the bottoms.
    x_bottoms = get_float("Bottoms composition, xB, or light-key mole fraction remaining in the bottoms: ")

    r_min = minimum_reflux_ratio(alpha, z_feed, x_distillate)
    # Reflux ratio controls how much condensed distillate is returned to the column.
    if r_min >= 0.0:
        print(f"Minimum reflux ratio for these conditions: Rmin = {r_min:.4f}")
        reflux_ratio = get_float(
            f"Reflux ratio, R, or liquid returned to the top divided by distillate withdrawn (must be > {r_min:.4f}): "
        )
    else:
        reflux_ratio = get_float(
            "Reflux ratio, R, or liquid returned to the top divided by distillate withdrawn: "
        )

    # Basic physical consistency checks before solving.
    if alpha <= 1.0:
        raise ValueError("Relative volatility must be greater than 1.")
    if not (0.0 < x_bottoms < z_feed < x_distillate < 1.0):
        raise ValueError("Compositions must satisfy 0 < xB < zF < xD < 1.")
    if reflux_ratio < 0.0 or not math.isfinite(reflux_ratio):
        raise ValueError("Reflux ratio must be a finite positive number.")
    if reflux_ratio <= r_min:
        raise ValueError(f"Reflux ratio must be greater than the minimum reflux ratio ({r_min:.3f}).")

    # Compute the staircase, report the theoretical stage count, then plot the result.
    stage_count, steps, x_intersection, y_intersection = calculate_stages(
        alpha, z_feed, x_distillate, x_bottoms, reflux_ratio
    )
    print(f"Estimated number of theoretical stages: {stage_count}")
    print(f"Feed composition used: zF = {z_feed:.4f}")
    print(f"Distillate composition used: xD = {x_distillate:.4f}")
    print(f"Bottoms composition used: xB = {x_bottoms:.4f}")
    print(f"Reflux ratio used: R = {reflux_ratio:.4f}")

    plot_mccabe_thiele(
        alpha,
        z_feed,
        x_distillate,
        x_bottoms,
        reflux_ratio,
        steps,
        x_intersection,
        y_intersection,
    )


if __name__ == "__main__":
    main()
