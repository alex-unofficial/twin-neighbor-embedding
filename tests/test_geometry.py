r"""`twin/geometry.py` — the curve definition shared by the drawing and the measurement.

Three things are checked here, and they are different in kind:

1. **The refactor was behaviour-preserving.** `curve_pins.npz` holds the curves
   `plots.draw_curved_edges` plotted *before* the kernel moved into `twin/geometry.py`, and the
   assertion is bit-for-bit equality. Without it "no behaviour change" is a claim rather than a
   check (CLAUDE.md, "Never mix a refactor with a behaviour change").
2. **The γ kernel really is a quadratic Bézier**, and `bezier_control_points` really is that
   conversion. The module derives $B_1$ from the kernel; the test compares it against the closed
   form $2\gamma P_1 + (1-2\gamma)(P_0+P_2)/2$ of `docs/CROSSING_GEOMETRY.md` §1. Neither
   expression is a copy of the other, so a wrong kernel row or a wrong conversion fails here.
3. **The exact AABB is exact and tight** — nothing sampled lies outside it, and it *equals* the
   sampled extent rather than merely containing it. The loose box on the control points would
   pass "contains"; only tightness distinguishes them.

Curves are sampled here with an evaluation written out in this module (`_sample`), not with a
helper from `twin.geometry`, so the oracle and the thing it checks do not share code — the trap
that hid a broken `twin_adjacency` for three sessions.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  - must follow the backend selection
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from twin import geometry  # noqa: E402
from twin.plots import draw_curved_edges  # noqa: E402

from tests.fixtures.make_curve_pins import GAMMAS, N_SAMPLES, fixture_inputs  # noqa: E402

PINS = Path(__file__).parent / "fixtures" / "curve_pins.npz"


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _sample(B: np.ndarray, n: int) -> np.ndarray:
    """Evaluate a Bézier from its control points, de Casteljau style. Independent of the module."""
    t = np.linspace(0.0, 1.0, n).reshape(-1, *([1] * (B.ndim - 2)), 1)
    b0, b1, b2 = B[..., 0, :], B[..., 1, :], B[..., 2, :]
    return (1 - t) ** 2 * b0 + 2 * t * (1 - t) * b1 + t**2 * b2


def _random_controls(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, 3, 2))


# --------------------------------------------------------------------------- #
#              1. the refactor did not move a single plotted point             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("gamma", GAMMAS)
def test_draw_curved_edges_is_bit_identical_to_its_pre_refactor_output(gamma):
    """The characterisation pin for step 5 commit 1. Exact equality, not `allclose`."""
    pinned = np.load(PINS)
    G, v_pos, e_pos = fixture_inputs()
    np.testing.assert_array_equal(v_pos, pinned["v_pos"])
    np.testing.assert_array_equal(e_pos, pinned["e_pos"])

    _, ax = plt.subplots()
    draw_curved_edges(
        G, 2, v_pos, e_pos, ax, n_samples=N_SAMPLES, alpha=0.5, linewidth=1.0, gamma=gamma
    )
    drawn = np.array([line.get_xydata() for line in ax.lines])

    np.testing.assert_array_equal(drawn, pinned[f"gamma_{gamma:g}"])


# --------------------------------------------------------------------------- #
#                  2. the kernel curve is exactly the Bézier                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("gamma", GAMMAS)
def test_bezier_control_points_match_the_closed_form(gamma):
    """`docs/CROSSING_GEOMETRY.md` §1: $B_1 = 2\\gamma P_1 + (1-2\\gamma)(P_0+P_2)/2$."""
    P = _random_controls(200, seed=1)
    p0, p1, p2 = P[:, 0], P[:, 1], P[:, 2]

    expected = np.stack(
        [p0, 2 * gamma * p1 + (1 - 2 * gamma) * (p0 + p2) / 2, p2],
        axis=1,
    )
    np.testing.assert_allclose(geometry.bezier_control_points(P, gamma), expected, atol=1e-15)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_the_bezier_reproduces_the_kernel_curve(gamma):
    """Sampling the converted control points gives the curve the kernel evaluates."""
    P = _random_controls(200, seed=2)
    t = np.linspace(0, 1, 17)

    coeffs = geometry.polynomial_coefficients(P, gamma)
    kernel_curve = np.einsum("tp,epd->etd", np.vander(t, 3, increasing=True), coeffs)
    bezier_curve = np.moveaxis(_sample(geometry.bezier_control_points(P, gamma), 17), 0, 1)

    np.testing.assert_allclose(bezier_curve, kernel_curve, atol=1e-14)


def test_gamma_zero_is_the_exact_chord():
    """The straight variant of §4.3 goes through the curved code path (DECISIONS (27))."""
    P = _random_controls(50, seed=3)
    B = geometry.bezier_control_points(P, 0.0)

    np.testing.assert_allclose(B[:, 1], (P[:, 0] + P[:, 2]) / 2, atol=1e-15)
    np.testing.assert_allclose(geometry.bulge(B), 0.0, atol=1e-15)
    # C2 == 0 exactly: the curve is a degree-1 polynomial, not a nearly-flat quadratic.
    np.testing.assert_array_equal(geometry.polynomial_coefficients(P, 0.0)[:, 2], 0.0)


def test_power_form_round_trips():
    P = _random_controls(100, seed=4)
    B = geometry.bezier_control_points(P, 0.75)
    C = geometry.polynomial_from_bezier(B)

    np.testing.assert_allclose(C, geometry.polynomial_coefficients(P, 0.75), atol=1e-14)
    # C_2 = -2D, the identity `crossings.py` leans on when it reads Q4 as -16 det(D_A, D_B)^2.
    np.testing.assert_allclose(C[:, 2], -2 * geometry.bulge(B), atol=1e-15)


def test_the_endpoints_are_the_edge_endpoints_for_every_gamma():
    """$C(0) = P_0$ and $C(1) = P_2$ — what the kernel was derived to guarantee."""
    P = _random_controls(20, seed=5)
    for gamma in GAMMAS:
        B = geometry.bezier_control_points(P, gamma)
        np.testing.assert_allclose(B[:, 0], P[:, 0], atol=1e-15)
        np.testing.assert_allclose(B[:, 2], P[:, 2], atol=1e-15)


# --------------------------------------------------------------------------- #
#                    3. the exact bounding box, and its tightness              #
# --------------------------------------------------------------------------- #


def test_the_exact_box_contains_the_curve_and_equals_its_extent():
    """Valid *and* tight: a looser box (the control-point hull) passes the first half only."""
    B = geometry.bezier_control_points(_random_controls(3000, seed=6), 0.9)
    boxes = geometry.exact_bounding_boxes(B)

    samples = _sample(B, 4001)  # (n_samples, m, 2)
    lo, hi = samples.min(axis=0), samples.max(axis=0)

    assert np.all(boxes[:, 0] <= lo + 1e-12)
    assert np.all(boxes[:, 1] >= hi - 1e-12)
    np.testing.assert_allclose(boxes[:, 0], lo, atol=1e-6)
    np.testing.assert_allclose(boxes[:, 1], hi, atol=1e-6)


def test_the_exact_box_is_smaller_than_the_control_point_box():
    """The reason it exists: candidate counts scale with box area (guide §2, ~37 % smaller)."""
    B = geometry.bezier_control_points(_random_controls(20000, seed=7), 0.9)
    exact = geometry.exact_bounding_boxes(B)

    hull_lo, hull_hi = B.min(axis=1), B.max(axis=1)
    exact_area = np.prod(exact[:, 1] - exact[:, 0], axis=1)
    hull_area = np.prod(hull_hi - hull_lo, axis=1)

    assert np.all(exact_area <= hull_area + 1e-12)
    assert np.median(exact_area / hull_area) < 0.7


def test_a_straight_edge_box_is_its_endpoints():
    """The stationary point's denominator vanishes; the endpoints are the whole range."""
    P = _random_controls(100, seed=8)
    B = geometry.bezier_control_points(P, 0.0)
    boxes = geometry.exact_bounding_boxes(B)

    np.testing.assert_allclose(boxes[:, 0], np.minimum(P[:, 0], P[:, 2]), atol=1e-15)
    np.testing.assert_allclose(boxes[:, 1], np.maximum(P[:, 0], P[:, 2]), atol=1e-15)
    assert np.isfinite(boxes).all()


def test_the_two_coordinates_turn_at_different_parameters():
    """The guide's worked example: the box is touched at four distinct parameters, not two."""
    B = np.array([[[0.0, 0.0], [4.0, 1.0], [3.0, 0.5]]])
    box = geometry.exact_bounding_boxes(B)[0]

    np.testing.assert_allclose(box[0], [0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(box[1], [3.2, 2 / 3], atol=1e-15)


def test_shapes_and_rejection_of_a_wrong_one():
    B = geometry.bezier_control_points(_random_controls(7, seed=9), 0.5)
    assert B.shape == (7, 3, 2)
    assert geometry.exact_bounding_boxes(B).shape == (7, 2, 2)
    assert geometry.bulge(B).shape == (7, 2)
    # a single curve, unbatched
    assert geometry.exact_bounding_boxes(B[0]).shape == (2, 2)

    with pytest.raises(ValueError, match=r"\(\.\.\., 3, d\)"):
        geometry.bezier_control_points(np.zeros((5, 2, 2)))
    with pytest.raises(ValueError, match=r"\(\.\.\., 3, d\)"):
        geometry.polynomial_from_bezier(np.zeros((5, 4, 2)))
