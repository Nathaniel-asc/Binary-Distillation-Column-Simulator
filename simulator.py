import csv
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np


def y_equilibrium(x: float | np.ndarray, alpha: float) -> float | np.ndarray:
    # Binary VLE for a constant-relative-volatility system.
    return (alpha * x) / (1.0 + (alpha - 1.0) * x)


def x_from_y_equilibrium(y: float | np.ndarray, alpha: float) -> float | np.ndarray:
    # Invert the equilibrium relation so we can step from vapor composition back to liquid composition.
    return y / (alpha - (alpha - 1.0) * y)


@dataclass(frozen=True)
class Azeotrope:
    """An interior composition where liquid and vapor compositions are equal."""

    composition: float
    is_exact_data_point: bool


@dataclass(frozen=True)
class VLEModel:
    """Equilibrium relation and its inverse on a mole-fraction basis."""

    y_from_x: Callable[[float | np.ndarray], float | np.ndarray]
    x_from_y: Callable[[float | np.ndarray], float | np.ndarray]
    plot_x: np.ndarray
    plot_y: np.ndarray
    description: str
    azeotropes: tuple[Azeotrope, ...] = ()


def relative_volatility_vle(alpha: float) -> VLEModel:
    if alpha <= 1.0 or not math.isfinite(alpha):
        raise ValueError("Relative volatility must be finite and greater than 1.")

    x_values = np.linspace(0.0, 1.0, 500)
    return VLEModel(
        y_from_x=lambda x: y_equilibrium(x, alpha),
        x_from_y=lambda y: x_from_y_equilibrium(y, alpha),
        plot_x=x_values,
        plot_y=np.asarray(y_equilibrium(x_values, alpha)),
        description=f"constant relative volatility (alpha = {alpha:g})",
    )


def find_azeotropes(x_values: np.ndarray, y_values: np.ndarray) -> tuple[Azeotrope, ...]:
    """Find interior intersections with y=x, linearly interpolating between CSV points."""
    difference = y_values - x_values
    candidates: list[Azeotrope] = []

    for index, (x_value, difference_value) in enumerate(zip(x_values, difference)):
        if 0 < index < len(x_values) - 1 and math.isclose(difference_value, 0.0, abs_tol=1e-12):
            candidates.append(Azeotrope(float(x_value), True))

    for index in range(len(x_values) - 1):
        left_difference = difference[index]
        right_difference = difference[index + 1]
        if left_difference * right_difference < 0.0:
            fraction = -left_difference / (right_difference - left_difference)
            composition = x_values[index] + fraction * (x_values[index + 1] - x_values[index])
            candidates.append(Azeotrope(float(composition), False))

    unique: list[Azeotrope] = []
    for azeotrope in sorted(candidates, key=lambda item: item.composition):
        if not unique or not math.isclose(azeotrope.composition, unique[-1].composition, abs_tol=1e-10):
            unique.append(azeotrope)
    return tuple(unique)


def load_vle_csv(file_path: str) -> VLEModel:
    """Load monotonic VLE x,y data from a CSV with columns named x and y."""
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise ValueError(f"CSV file not found: {path}")

    try:
        with path.open(newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise ValueError("CSV must include a header row with columns named x and y.")

            columns = {name.strip().lower(): name for name in reader.fieldnames if name}
            if "x" not in columns or "y" not in columns:
                raise ValueError("CSV must include columns named x and y.")

            points = []
            for row_number, row in enumerate(reader, start=2):
                try:
                    x_value = float(row[columns["x"]].strip())
                    y_value = float(row[columns["y"]].strip())
                except (AttributeError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid numeric x,y data on CSV row {row_number}.") from exc
                points.append((x_value, y_value))
    except OSError as exc:
        raise ValueError(f"Could not read CSV file: {exc}") from exc

    if len(points) < 2:
        raise ValueError("CSV must contain at least two VLE data rows.")

    x_values = np.array([point[0] for point in points], dtype=float)
    y_values = np.array([point[1] for point in points], dtype=float)
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("All x and y values in the CSV must be finite.")
    if np.any((x_values < 0.0) | (x_values > 1.0) | (y_values < 0.0) | (y_values > 1.0)):
        raise ValueError("All x and y values in the CSV must be between 0 and 1.")
    if not np.all(np.diff(x_values) > 0.0) or not np.all(np.diff(y_values) > 0.0):
        raise ValueError("CSV x values and y values must each be strictly increasing.")
    if not (math.isclose(x_values[0], 0.0) and math.isclose(y_values[0], 0.0)
            and math.isclose(x_values[-1], 1.0) and math.isclose(y_values[-1], 1.0)):
        raise ValueError("CSV data must start with x,y = 0,0 and end with x,y = 1,1.")

    def interpolate(x_input: float | np.ndarray, xp: np.ndarray, fp: np.ndarray) -> float | np.ndarray:
        values = np.asarray(x_input)
        if np.any(values < xp[0]) or np.any(values > xp[-1]):
            raise ValueError("Composition is outside the VLE data range.")
        result = np.interp(values, xp, fp)
        return float(result) if np.ndim(x_input) == 0 else result

    return VLEModel(
        y_from_x=lambda x: interpolate(x, x_values, y_values),
        x_from_y=lambda y: interpolate(y, y_values, x_values),
        plot_x=x_values,
        plot_y=y_values,
        description=f"CSV VLE data ({path.name})",
        azeotropes=find_azeotropes(x_values, y_values),
    )


def get_float(prompt: str) -> float:
    return float(input(prompt).strip())


def prompt_float_with_validation(prompt: str, validator: Callable[[float], str | None]) -> float:
    while True:
        raw_value = input(prompt).strip()
        try:
            value = float(raw_value)
        except ValueError:
            print(f"Invalid input '{raw_value}'. Please enter a numeric value.")
            continue

        error_message = validator(value)
        if error_message is None:
            return value

        print(f"Invalid value: {error_message}")


def prompt_zero_one(prompt: str) -> bool:
    """Prompt for 0/1 and return True for 1."""
    while True:
        response = input(prompt).strip()
        if response == "1":
            return True
        if response == "0":
            return False
        print("Please enter 0 or 1.")


def separation_bounds_for_feed(vle: VLEModel, z_feed: float) -> tuple[float, float]:
    """Return the azeotropic interval that can be used by this light-key calculation."""
    lower_bound = 0.0
    upper_bound = 1.0
    for azeotrope in vle.azeotropes:
        if math.isclose(z_feed, azeotrope.composition, abs_tol=1e-10):
            raise ValueError(f"zF cannot equal the azeotrope at x = y = {azeotrope.composition:.6f}.")
        if azeotrope.composition < z_feed:
            lower_bound = azeotrope.composition
        elif azeotrope.composition > z_feed:
            upper_bound = azeotrope.composition
            break

    midpoint = 0.5 * (lower_bound + upper_bound)
    if cast(float, vle.y_from_x(midpoint)) <= midpoint:
        raise ValueError(
            "This feed region has y <= x. The simulator's light-key convention requires a region where y > x."
        )
    return lower_bound, upper_bound


def print_azeotrope_warning(vle: VLEModel) -> None:
    if not vle.azeotropes:
        return

    print("Warning: the VLE CSV contains an azeotrope (an intersection with y = x).")
    for azeotrope in vle.azeotropes:
        method = "an exact CSV point" if azeotrope.is_exact_data_point else "linear interpolation between CSV points"
        print(f"  x = y = {azeotrope.composition:.6f} ({method})")
    print(
        "An azeotrope cannot be crossed by ordinary binary distillation. "
        "After entering zF, the program will enforce the reachable composition bounds."
    )


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
    return minimum_reflux_ratio_with_q(relative_volatility_vle(alpha), z_feed, x_distillate, 1.0)


def q_line(x: float | np.ndarray, z_feed: float, q_value: float) -> float | np.ndarray:
    # General feed line relation. For q = 1 it becomes a vertical line at x = zF.
    if math.isclose(q_value, 1.0, abs_tol=1e-9):
        raise ValueError("q-line is vertical when q = 1. Use x = zF directly.")
    return (q_value / (q_value - 1.0)) * x - z_feed / (q_value - 1.0)


def feed_pinch_point(vle: VLEModel, z_feed: float, q_value: float) -> tuple[float, float]:
    # Find where the feed line intersects equilibrium; this pinch point defines Rmin.
    if math.isclose(q_value, 1.0, abs_tol=1e-9):
        x_pinch = z_feed
        y_pinch = cast(float, vle.y_from_x(x_pinch))
        return x_pinch, y_pinch

    # Search for a sign change in f(x) = y_eq(x) - y_q(x), then refine with bisection.
    def f(x_value: float) -> float:
        y_eq = cast(float, vle.y_from_x(x_value))
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

    y_pinch = cast(float, vle.y_from_x(x_pinch))
    if not (0.0 < x_pinch < 1.0 and 0.0 < y_pinch < 1.0):
        raise ValueError("The feed line does not intersect equilibrium within the physical composition range.")
    return x_pinch, y_pinch


def minimum_reflux_ratio_with_q(vle: VLEModel, z_feed: float, x_distillate: float, q_value: float) -> float:
    # Minimum reflux is defined by the rectifying line through (xD, xD) and the feed pinch point.
    x_pinch, y_pinch = feed_pinch_point(vle, z_feed, q_value)
    r_min_slope = (y_pinch - x_distillate) / (x_pinch - x_distillate)
    return r_min_slope / (1.0 - r_min_slope)


def maximum_q_for_nonnegative_rmin(vle: VLEModel, z_feed: float, x_distillate: float) -> float:
    # At Rmin = 0, the pinch point lies where y_eq = xD; the resulting feed-line slope defines q_max.
    x_at_y_equals_xd = cast(float, vle.x_from_y(x_distillate))

    if math.isclose(x_at_y_equals_xd, z_feed, abs_tol=1e-12):
        return 1.0

    feed_line_slope = (x_distillate - z_feed) / (x_at_y_equals_xd - z_feed)
    if math.isclose(feed_line_slope, 1.0, abs_tol=1e-12):
        raise ValueError("Could not determine a finite q limit for this set of compositions.")

    q_max = feed_line_slope / (feed_line_slope - 1.0)
    if not math.isfinite(q_max):
        raise ValueError("Could not determine a finite q limit for this set of compositions.")
    return q_max


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
    vle: VLEModel,
    z_feed: float,
    x_distillate: float,
    x_bottoms: float,
    reflux_ratio: float,
    q_value: float,
) -> tuple[float, list[tuple[tuple[float, float], tuple[float, float]]], float, float]:
    lower_bound, upper_bound = separation_bounds_for_feed(vle, z_feed)
    if not (lower_bound < x_bottoms < z_feed < x_distillate < upper_bound):
        raise ValueError(
            "Compositions must stay within the reachable azeotrope-bounded interval: "
            f"{lower_bound:.6f} < xB < zF < xD < {upper_bound:.6f}."
        )
    if reflux_ratio < 0.0 or not math.isfinite(reflux_ratio):
        raise ValueError("Reflux ratio must be finite and non-negative.")
    if not math.isfinite(q_value):
        raise ValueError("q must be finite.")

    r_min = minimum_reflux_ratio_with_q(vle, z_feed, x_distillate, q_value)

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
        x_eq = cast(float, vle.x_from_y(y_current))

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
    vle: VLEModel,
    z_feed: float,
    x_distillate: float,
    x_bottoms: float,
    reflux_ratio: float,
    q_value: float,
    steps,
    x_intersection: float,
    y_intersection: float,
    color_blind_mode: bool,
):
    fig, ax = plt.subplots(figsize=(8, 8))
    if color_blind_mode:
        # Distinct patterns keep the plot readable without relying on color.
        line_styles = {
            "equilibrium": {"color": "black", "linestyle": "-"},
            "reference": {"color": "black", "linestyle": "--"},
            "rectifying": {"color": "black", "linestyle": "-."},
            "stripping": {"color": "black", "linestyle": ":"},
            "feed": {"color": "black", "linestyle": (0, (5, 2, 1, 2))},
            "stages": {"color": "black", "linestyle": (0, (1, 1))},
        }
    else:
        line_styles = {
            "equilibrium": {}, "reference": {"linestyle": "--"}, "rectifying": {},
            "stripping": {}, "feed": {"linestyle": ":"}, "stages": {"color": "black"},
        }

    ax.plot(vle.plot_x, vle.plot_y, label=f"Equilibrium curve ({vle.description})", **line_styles["equilibrium"])
    ax.plot([0, 1], [0, 1], label="Reference line: y = x", **line_styles["reference"])

    # Rectifying line segment: from q-line intersection to y=x intersection at (xD, xD).
    ax.plot(
        [x_intersection, x_distillate],
        [y_intersection, x_distillate],
        label="Rectifying operating line",
        **line_styles["rectifying"],
    )

    # Stripping line segment: from q-line intersection to y=x intersection at (xB, xB).
    ax.plot(
        [x_bottoms, x_intersection],
        [x_bottoms, y_intersection],
        label="Stripping operating line",
        **line_styles["stripping"],
    )

    # q-line segment: from its y=x intersection at (zF, zF) to its rectifying-line intersection.
    if math.isclose(q_value, 1.0, abs_tol=1e-9):
        ax.plot(
            [z_feed, z_feed], [z_feed, y_intersection],
            label="Feed line (q-line): q = 1", **line_styles["feed"],
        )
    else:
        ax.plot(
            [z_feed, x_intersection],
            [z_feed, y_intersection],
            label=f"Feed line (q-line): q = {q_value:.3f}",
            **line_styles["feed"],
        )

    # Draw the stepped path used to count theoretical stages.
    for idx, ((x1, y1), (x2, y2)) in enumerate(steps):
        ax.plot(
            [x1, x2], [y1, y2], linewidth=1.2,
            label="Theoretical stage staircase" if idx == 0 else None, **line_styles["stages"],
        )

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
    color_blind_mode = prompt_zero_one(
        "Colour-blind-friendly plot with distinct line patterns (0 = no, 1 = yes): "
    )
    # The user can either generate VLE from constant relative volatility or provide measured/tabulated data.
    while True:
        vle_source = input("VLE source: enter 1 for relative volatility or 2 for a CSV file: ").strip()
        if vle_source == "1":
            alpha = prompt_float_with_validation(
                "Relative volatility of the light key over the heavy key, alpha (must be finite and greater than 1): ",
                lambda value: None if math.isfinite(value) and value > 1.0 else "alpha must be finite and greater than 1.",
            )
            vle = relative_volatility_vle(alpha)
            break
        if vle_source == "2":
            csv_path = input("Path to VLE CSV file (columns: x,y): ").strip().strip('"')
            try:
                vle = load_vle_csv(csv_path)
                break
            except ValueError as exc:
                print(f"Invalid VLE CSV: {exc}")
            continue
        print("Please enter 1 for relative volatility or 2 for a CSV file.")

    print_azeotrope_warning(vle)

    # z_feed: feed composition, or the mole fraction of the more volatile component in the feed.
    while True:
        z_feed = prompt_float_with_validation(
            "Feed composition, zF, or feed mole fraction of the light key (must be between 0 and 1): ",
            lambda value: None if 0.0 < value < 1.0 else "zF must be between 0 and 1.",
        )
        try:
            azeotrope_lower_bound, azeotrope_upper_bound = separation_bounds_for_feed(vle, z_feed)
            break
        except ValueError as exc:
            print(f"Invalid feed composition: {exc}")

    # x_distillate: desired distillate composition, usually the light-key purity at the top product.
    distillate_upper_bound_text = (
        f"the azeotrope [{azeotrope_upper_bound:g}]"
        if azeotrope_upper_bound < 1.0
        else "1"
    )
    x_distillate = prompt_float_with_validation(
        (
            "Distillate composition, xD, or desired top-product mole fraction of the light key "
            f"(must be greater than zF [{z_feed:.4f}] and less than "
            f"{distillate_upper_bound_text}): "
        ),
        lambda value: (
            None
            if z_feed < value < azeotrope_upper_bound
            else (
                f"xD must be greater than zF ({z_feed:.4f}) and less than "
                f"{azeotrope_upper_bound:.6f}."
            )
        ),
    )

    # x_bottoms: desired bottoms composition, usually the light-key mole fraction left in the bottoms.
    bottoms_lower_bound_text = (
        f"the azeotrope [{azeotrope_lower_bound:g}]"
        if azeotrope_lower_bound > 0.0
        else "0"
    )
    x_bottoms = prompt_float_with_validation(
        (
            "Bottoms composition, xB, or light-key mole fraction remaining in the bottoms "
            f"(must be less than zF [{z_feed:.4f}] and greater than "
            f"{bottoms_lower_bound_text}): "
        ),
        lambda value: (
            None
            if azeotrope_lower_bound < value < z_feed
            else (
                f"xB must be less than zF ({z_feed:.4f}) and greater than "
                f"{azeotrope_lower_bound:.6f}."
            )
        ),
    )

    while True:
        q_value = prompt_float_with_validation(
            "Feed thermal condition, q (1 = saturated liquid, must be finite): ",
            lambda value: None if math.isfinite(value) else "q must be a finite number.",
        )

        try:
            r_min = minimum_reflux_ratio_with_q(vle, z_feed, x_distillate, q_value)
            break
        except ValueError as exc:
            print(f"Invalid q value: {exc}")

    if r_min < 0.0:
        print(
            f"Warning: the minimum reflux ratio for these conditions is below 0 (Rmin = {r_min:.4f}). "
            "Continuing with a nonnegative reflux ratio requirement."
        )

    # Reflux ratio controls how much condensed distillate is returned to the column.
    print(f"Minimum reflux ratio for these conditions: Rmin = {r_min:.4f}")
    if r_min < 0.0:
        reflux_ratio = prompt_float_with_validation(
            "Reflux ratio, R, or liquid returned to the top divided by distillate withdrawn (must be finite and greater than or equal to 0): ",
            lambda value: None if math.isfinite(value) and value >= 0.0 else "R must be finite and greater than or equal to 0.",
        )
    else:
        reflux_ratio = prompt_float_with_validation(
            f"Reflux ratio, R, or liquid returned to the top divided by distillate withdrawn (must be finite and greater than {r_min:.4f}): ",
            lambda value: (
                None
                if math.isfinite(value) and value > r_min
                else f"R must be finite and greater than {r_min:.4f}."
            ),
        )

    # Compute the staircase, report the theoretical stage count, then plot the result.
    stage_count, steps, x_intersection, y_intersection = calculate_stages(
        vle, z_feed, x_distillate, x_bottoms, reflux_ratio, q_value
    )
    print(f"Estimated number of theoretical stages: {stage_count}")
    print(f"Feed composition used: zF = {z_feed:.4f}")
    print(f"Distillate composition used: xD = {x_distillate:.4f}")
    print(f"Bottoms composition used: xB = {x_bottoms:.4f}")
    print(f"Feed thermal condition used: q = {q_value:.4f}")
    print(f"Reflux ratio used: R = {reflux_ratio:.4f}")
    print(f"VLE source used: {vle.description}")

    plot_mccabe_thiele(
        vle,
        z_feed,
        x_distillate,
        x_bottoms,
        reflux_ratio,
        q_value,
        steps,
        x_intersection,
        y_intersection,
        color_blind_mode,
    )


if __name__ == "__main__":
    main()
