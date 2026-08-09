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
    # Backward-compatible default: q = 1 (saturated-liquid feed).
    return minimum_reflux_ratio_with_q(alpha, z_feed, x_distillate, 1.0)


def q_line(x: float | np.ndarray, z_feed: float, q_value: float) -> float | np.ndarray:
    # General feed line relation. For q = 1 it becomes a vertical line at x = zF.
    if math.isclose(q_value, 1.0, abs_tol=1e-9):
        raise ValueError("q-line is vertical when q = 1. Use x = zF directly.")
    return (q_value / (q_value - 1.0)) * x - z_feed / (q_value - 1.0)


def feed_pinch_point(alpha: float, z_feed: float, q_value: float) -> tuple[float, float]:
    # Find where the feed line intersects equilibrium; this pinch point defines Rmin.
    if math.isclose(q_value, 1.0, abs_tol=1e-9):
        x_pinch = z_feed
        y_pinch = cast(float, y_equilibrium(x_pinch, alpha))
        return x_pinch, y_pinch

    # Search for a sign change in f(x) = y_eq(x) - y_q(x), then refine with bisection.
    def f(x_value: float) -> float:
        y_eq = cast(float, y_equilibrium(x_value, alpha))
        y_q = cast(float, q_line(x_value, z_feed, q_value))
        return y_eq - y_q

    x_left = 0.0
    f_left = f(x_left)
    root_interval: tuple[float, float] | None = None

    for i in range(1, 1001):
        x_right = i / 1000.0
        f_right = f(x_right)
        if f_left == 0.0:
            root_interval = (x_left, x_left)
            break
        if f_left * f_right <= 0.0:
            root_interval = (x_left, x_right)
            break
        x_left = x_right
        f_left = f_right

    if root_interval is None:
        raise ValueError("Could not locate an equilibrium/feed-line intersection for the selected q value.")

    a, b = root_interval
    if a == b:
        x_pinch = a
    else:
        for _ in range(80):
            midpoint = 0.5 * (a + b)
            f_mid = f(midpoint)
            if f_mid == 0.0:
                a = midpoint
                b = midpoint
                break
            f_a = f(a)
            if f_a * f_mid <= 0.0:
                b = midpoint
            else:
                a = midpoint
        x_pinch = 0.5 * (a + b)

    y_pinch = cast(float, y_equilibrium(x_pinch, alpha))
    if not (0.0 < x_pinch < 1.0 and 0.0 < y_pinch < 1.0):
        raise ValueError("The feed line does not intersect equilibrium within the physical composition range.")
    return x_pinch, y_pinch


def minimum_reflux_ratio_with_q(alpha: float, z_feed: float, x_distillate: float, q_value: float) -> float:
    # Minimum reflux is defined by the rectifying line through (xD, xD) and the feed pinch point.
    x_pinch, y_pinch = feed_pinch_point(alpha, z_feed, q_value)
    r_min_slope = (y_pinch - x_distillate) / (x_pinch - x_distillate)
    if not 0.0 < r_min_slope < 1.0:
        raise ValueError(
            "The selected feed condition does not give a physically valid positive minimum reflux ratio."
        )
    return r_min_slope / (1.0 - r_min_slope)


def feed_line_intersection(
    z_feed: float,
    reflux_ratio: float,
    x_distillate: float,
    q_value: float,
) -> tuple[float, float]:
    # Intersection of rectifying operating line and feed line (q-line).
    if math.isclose(q_value, 1.0, abs_tol=1e-9):
        x_intersection = z_feed
        y_intersection = cast(float, rectifying_line(x_intersection, reflux_ratio, x_distillate))
        return x_intersection, y_intersection

    m_rect = reflux_ratio / (reflux_ratio + 1.0)
    b_rect = x_distillate / (reflux_ratio + 1.0)
    m_q = q_value / (q_value - 1.0)
    b_q = -z_feed / (q_value - 1.0)

    denominator = m_rect - m_q
    if math.isclose(denominator, 0.0, abs_tol=1e-12):
        raise ValueError("Rectifying line is parallel to the q-line for the selected inputs.")

    x_intersection = (b_q - b_rect) / denominator
    y_intersection = m_rect * x_intersection + b_rect
    return x_intersection, y_intersection


def calculate_stages(
    alpha: float,
    z_feed: float,
    x_distillate: float,
    x_bottoms: float,
    reflux_ratio: float,
    q_value: float,
) -> tuple[float, list[tuple[tuple[float, float], tuple[float, float]]], float, float]:
    if alpha <= 1.0 or not math.isfinite(alpha):
        raise ValueError("Relative volatility must be finite and greater than 1.")
    if not (0.0 < x_bottoms < z_feed < x_distillate < 1.0):
        raise ValueError("Compositions must satisfy 0 < xB < zF < xD < 1.")
    if reflux_ratio < 0.0 or not math.isfinite(reflux_ratio):
        raise ValueError("Reflux ratio must be finite and non-negative.")
    if not math.isfinite(q_value):
        raise ValueError("q must be finite.")

    r_min = minimum_reflux_ratio_with_q(alpha, z_feed, x_distillate, q_value)

    if reflux_ratio <= r_min:
        raise ValueError(f"Reflux ratio must be greater than minimum reflux ratio ({r_min:.3f}).")

    x_intersection, y_intersection = feed_line_intersection(z_feed, reflux_ratio, x_distillate, q_value)
    if not (x_bottoms < x_intersection < x_distillate and 0.0 < y_intersection < 1.0):
        raise ValueError(
            "The operating-line intersection is outside the physical composition range for these inputs."
        )

    # Step off stages by alternating between equilibrium and operating lines.
    stage_count = 0
    y_current = x_distillate
    x_current = x_distillate
    steps = []

    for _ in range(1000):
        # Horizontal move to the equilibrium curve: vapor and liquid are in equilibrium on each stage.
        x_eq = cast(float, x_from_y_equilibrium(y_current, alpha))

        if x_eq <= x_bottoms:
            # The final reboiler stage may be partially traversed. Report a whole
            # number of stages, so any nonzero fraction is counted as one stage.
            fraction = (x_current - x_bottoms) / (x_current - x_eq)
            steps.append(((x_current, y_current), (x_bottoms, y_current)))
            stage_count += math.ceil(fraction)
            break

        steps.append(((x_current, y_current), (x_eq, y_current)))
        stage_count += 1

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
    q_value: float,
    steps,
    x_intersection: float,
    y_intersection: float,
):
    # Build the equilibrium and operating curves on a mole-fraction basis.
    x = np.linspace(0.0, 1.0, 500)
    y_eq = y_equilibrium(x, alpha)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, y_eq, label="Equilibrium curve: y vs x for the light key")
    ax.plot(x, x, "--", label="Reference line: y = x")

    # Rectifying line segment: from q-line intersection to y=x intersection at (xD, xD).
    ax.plot(
        [x_intersection, x_distillate],
        [y_intersection, x_distillate],
        label="Rectifying operating line",
    )

    # Stripping line segment: from q-line intersection to y=x intersection at (xB, xB).
    ax.plot(
        [x_bottoms, x_intersection],
        [x_bottoms, y_intersection],
        label="Stripping operating line",
    )

    # q-line segment: from its y=x intersection at (zF, zF) to its rectifying-line intersection.
    if math.isclose(q_value, 1.0, abs_tol=1e-9):
        ax.plot([z_feed, z_feed], [z_feed, y_intersection], ":", label="Feed line (q-line): q = 1")
    else:
        ax.plot(
            [z_feed, x_intersection],
            [z_feed, y_intersection],
            ":",
            label=f"Feed line (q-line): q = {q_value:.3f}",
        )

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
    q_value = get_float("Feed thermal condition, q (1 = saturated liquid): ")

    # Basic physical consistency checks before solving.
    if alpha <= 1.0:
        raise ValueError("Relative volatility must be greater than 1.")
    if not (0.0 < x_bottoms < z_feed < x_distillate < 1.0):
        raise ValueError("Compositions must satisfy 0 < xB < zF < xD < 1.")
    if not math.isfinite(q_value):
        raise ValueError("q must be a finite number.")

    r_min = minimum_reflux_ratio_with_q(alpha, z_feed, x_distillate, q_value)
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

    if reflux_ratio < 0.0 or not math.isfinite(reflux_ratio):
        raise ValueError("Reflux ratio must be a finite positive number.")
    if reflux_ratio <= r_min:
        raise ValueError(f"Reflux ratio must be greater than the minimum reflux ratio ({r_min:.3f}).")

    # Compute the staircase, report the theoretical stage count, then plot the result.
    stage_count, steps, x_intersection, y_intersection = calculate_stages(
        alpha, z_feed, x_distillate, x_bottoms, reflux_ratio, q_value
    )
    print(f"Estimated number of theoretical stages: {stage_count}")
    print(f"Feed composition used: zF = {z_feed:.4f}")
    print(f"Distillate composition used: xD = {x_distillate:.4f}")
    print(f"Bottoms composition used: xB = {x_bottoms:.4f}")
    print(f"Feed thermal condition used: q = {q_value:.4f}")
    print(f"Reflux ratio used: R = {reflux_ratio:.4f}")

    plot_mccabe_thiele(
        alpha,
        z_feed,
        x_distillate,
        x_bottoms,
        reflux_ratio,
        q_value,
        steps,
        x_intersection,
        y_intersection,
    )


if __name__ == "__main__":
    main()
