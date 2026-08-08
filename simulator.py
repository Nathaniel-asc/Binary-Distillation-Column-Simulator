import math

import matplotlib.pyplot as plt
import numpy as np


def y_equilibrium(x: float, alpha: float) -> float:
    return (alpha * x) / (1.0 + (alpha - 1.0) * x)


def x_from_y_equilibrium(y: float, alpha: float) -> float:
    return y / (alpha - (alpha - 1.0) * y)


def get_float(prompt: str) -> float:
    return float(input(prompt).strip())


def rectifying_line(x: float, reflux_ratio: float, x_distillate: float) -> float:
    return (reflux_ratio / (reflux_ratio + 1.0)) * x + x_distillate / (reflux_ratio + 1.0)


def stripping_line(x: float, x_intersection: float, y_intersection: float, x_bottoms: float) -> float:
    slope = (y_intersection - x_bottoms) / (x_intersection - x_bottoms)
    return slope * (x - x_bottoms) + x_bottoms


def calculate_stages(alpha: float, z_feed: float, x_distillate: float, x_bottoms: float, reflux_ratio: float):
    y_feed_eq = y_equilibrium(z_feed, alpha)
    r_min_slope = (y_feed_eq - x_distillate) / (z_feed - x_distillate)
    r_min = r_min_slope / (1.0 - r_min_slope)

    if reflux_ratio <= r_min:
        raise ValueError(f"Reflux ratio must be greater than minimum reflux ratio ({r_min:.3f}).")

    x_intersection = z_feed
    y_intersection = rectifying_line(x_intersection, reflux_ratio, x_distillate)

    stage_count = 0
    y_current = x_distillate
    x_current = x_distillate
    steps = []

    for _ in range(1000):
        x_eq = x_from_y_equilibrium(y_current, alpha)
        steps.append(((x_current, y_current), (x_eq, y_current)))
        stage_count += 1

        if x_eq <= x_bottoms:
            break

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
    x = np.linspace(0.0, 1.0, 500)
    y_eq = y_equilibrium(x, alpha)
    y_rect = rectifying_line(x, reflux_ratio, x_distillate)
    y_strip = stripping_line(x, x_intersection, y_intersection, x_bottoms)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, y_eq, label="Equilibrium Curve")
    ax.plot(x, x, "--", label="y = x")
    ax.plot(x, y_rect, label="Rectifying Line")
    ax.plot(x, y_strip, label="Stripping Line")
    ax.axvline(z_feed, linestyle=":", label="q-line (q=1)")

    for idx, ((x1, y1), (x2, y2)) in enumerate(steps):
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.2, label="Stages" if idx == 0 else None)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x (liquid mole fraction)")
    ax.set_ylabel("y (vapor mole fraction)")
    ax.set_title("McCabe–Thiele Diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    print("McCabe–Thiele Binary Distillation Stage Estimator")
    alpha = get_float("Relative volatility (alpha): ")
    z_feed = get_float("Feed composition (zF): ")
    x_distillate = get_float("Distillate purity (xD): ")
    x_bottoms = get_float("Bottoms purity (xB): ")
    reflux_ratio = get_float("Reflux ratio (R): ")

    if alpha <= 1.0:
        raise ValueError("Relative volatility must be greater than 1.")
    if not (0.0 < x_bottoms < z_feed < x_distillate < 1.0):
        raise ValueError("Compositions must satisfy 0 < xB < zF < xD < 1.")
    if reflux_ratio <= 0.0 or not math.isfinite(reflux_ratio):
        raise ValueError("Reflux ratio must be a finite positive number.")

    stage_count, steps, x_intersection, y_intersection = calculate_stages(
        alpha, z_feed, x_distillate, x_bottoms, reflux_ratio
    )
    print(f"Estimated number of theoretical stages: {stage_count}")

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
