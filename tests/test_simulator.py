import math

import numpy as np
import pytest

import simulator


def test_equilibrium_relation_and_inverse_are_consistent():
    x_values = np.array([0.0, 0.2, 0.5, 0.9, 1.0])

    y_values = simulator.y_equilibrium(x_values, 2.5)
    recovered_x = simulator.x_from_y_equilibrium(y_values, 2.5)

    np.testing.assert_allclose(recovered_x, x_values)


@pytest.mark.parametrize("alpha", [1.0, 0.0, -2.0, math.inf, math.nan])
def test_relative_volatility_requires_finite_alpha_greater_than_one(alpha):
    with pytest.raises(ValueError, match="greater than 1"):
        simulator.relative_volatility_vle(alpha)


def test_find_azeotropes_detects_exact_and_interpolated_points():
    x_values = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    y_values = np.array([0.0, 0.35, 0.5, 0.70, 1.0])

    azeotropes = simulator.find_azeotropes(x_values, y_values)

    assert azeotropes == (simulator.Azeotrope(0.5, True),)

    interpolated = simulator.find_azeotropes(
        np.array([0.0, 0.2, 0.8, 1.0]),
        np.array([0.0, 0.35, 0.70, 1.0]),
    )
    assert interpolated[0].composition == pytest.approx(0.56)
    assert interpolated[0].is_exact_data_point is False


def test_load_vle_csv_reads_example_and_interpolates(tmp_path):
    csv_path = tmp_path / "vle.csv"
    csv_path.write_text("x,y\n0,0\n0.5,0.7\n1,1\n", encoding="utf-8")

    vle = simulator.load_vle_csv(str(csv_path))

    assert vle.description == "CSV VLE data (vle.csv)"
    assert vle.y_from_x(0.25) == pytest.approx(0.35)
    assert vle.x_from_y(0.35) == pytest.approx(0.25)
    assert len(vle.plot_x) == len(vle.plot_y) == 3


@pytest.mark.parametrize(
    "contents, message",
    [
        ("x,z\n0,0\n1,1\n", "columns named x and y"),
        ("x,y\n0.1,0.1\n1,1\n", "start with x,y"),
        ("x,y\n0,0\n0.5,0.8\n0.4,0.9\n1,1\n", "strictly increasing"),
    ],
)
def test_load_vle_csv_rejects_invalid_data(tmp_path, contents, message):
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        simulator.load_vle_csv(str(csv_path))


def test_q_line_and_saturated_liquid_feed_behavior():
    assert simulator.q_line(0.4, 0.3, 0.0) == pytest.approx(0.3)
    with pytest.raises(ValueError, match="vertical"):
        simulator.q_line(0.4, 0.3, 1.0)

    vle = simulator.relative_volatility_vle(2.4)
    x_pinch, y_pinch = simulator.feed_pinch_point(vle, 0.4, 1.0)
    assert x_pinch == pytest.approx(0.4)
    assert y_pinch == pytest.approx(simulator.y_equilibrium(0.4, 2.4))


def test_calculate_stages_returns_converged_staircase():
    vle = simulator.relative_volatility_vle(2.4)

    stage_count, steps, x_intersection, y_intersection = simulator.calculate_stages(
        vle,
        z_feed=0.4,
        x_distillate=0.9,
        x_bottoms=0.1,
        reflux_ratio=2.0,
        q_value=1.0,
    )

    assert stage_count == 10
    assert len(steps) > 0
    assert x_intersection == pytest.approx(0.4)
    assert y_intersection == pytest.approx(0.5666666667)
    assert steps[-1][1][0] == pytest.approx(0.1)


def test_calculate_stages_rejects_reflux_at_or_below_minimum():
    vle = simulator.relative_volatility_vle(2.4)
    r_min = simulator.minimum_reflux_ratio_with_q(vle, 0.4, 0.9, 1.0)

    with pytest.raises(ValueError, match="greater than minimum reflux ratio"):
        simulator.calculate_stages(vle, 0.4, 0.9, 0.1, r_min, 1.0)


def test_azeotropic_feed_region_cannot_cross_azeotrope():
    vle = simulator.load_vle_csv("example_azeotropic_vle_data.csv")

    with pytest.raises(ValueError, match="cannot equal the azeotrope"):
        simulator.separation_bounds_for_feed(vle, 0.5)

    assert simulator.separation_bounds_for_feed(vle, 0.3) == pytest.approx((0.0, 0.5))
