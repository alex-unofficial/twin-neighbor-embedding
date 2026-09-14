r"""The geometry of a γ-curved edge: one definition, shared by the drawing and the measurement.

`plots.draw_curved_edges` draws each edge as the curve

$$C(t) = \begin{bmatrix}1 & t & t^2\end{bmatrix}\, K_\gamma \begin{bmatrix}P_0\\P_1\\P_2\end{bmatrix},
\qquad P_0 = x_u,\; P_1 = x_{(u,v)},\; P_2 = x_v,$$

with $K_\gamma$ the kernel in :func:`curve_kernel`. That curve is **exactly** a quadratic Bézier
(``docs/CROSSING_GEOMETRY.md`` §1, verified to 5.1e-15), which is what makes the crossing metrics
of spec §2.4 algebraic rather than sampled: three control points describe an edge completely, two
quadratic Béziers meet in at most four points, and those points are the roots of one quartic.

⚠️ **The control points $(P_0, P_1, P_2)$ held by `plots.py` are not Bézier control points.**
Except at γ=0.5 they differ, and using them as if they were draws a different curve from the one
that is measured. :func:`bezier_control_points` is the conversion, and it lives here — in neither
`plots.py` nor `metrics/crossings.py` — because if it is duplicated the drawn curve and the
measured curve drift apart silently and no test notices (DECISIONS (27)).

**The kernel is the single authority.** Everything else in this module is derived from
:func:`curve_kernel`: :func:`polynomial_coefficients` applies it, and
:func:`bezier_control_points` converts the result. `docs/CROSSING_GEOMETRY.md` §1 also gives the
conversion in closed form, $B_1 = 2\gamma P_1 + (1-2\gamma)(P_0+P_2)/2$; that formula is
deliberately *not* what runs — it is what `tests/test_geometry.py` checks the derived value
against, so the two cannot drift.

**Orientation.** Everything here is in the ``(n_points, d)`` orientation, i.e. a control-point
array is ``(..., 3, d)`` with the three points along the second-to-last axis. `plots.py` is still
the legacy ``(d, n)`` surface (CLAUDE.md) and converts at its own boundary; nothing in this module
is transposed on entry.

Three representations of the same curve are used downstream, all convertible here:

* **control points** ``(..., 3, d)`` — what :func:`bezier_control_points` returns and what
  `edge_curves` hands to the crossing detector;
* **power form** ``[C_0, C_1, C_2]`` with $C(t) = C_0 + C_1 t + C_2 t^2$ — what the quartic is
  built from (:func:`polynomial_from_bezier`);
* the **bulge** $D = B_1 - (B_0+B_2)/2 = 2\gamma\bigl(P_1 - (P_0+P_2)/2\bigr)$
  (:func:`bulge`), a conditioning quantity only — it bounds nothing in this design (DECISIONS (30)).
"""

import numpy as np

__all__ = [
    "arc_length",
    "bezier_control_points",
    "bulge",
    "curve_kernel",
    "exact_bounding_boxes",
    "polynomial_coefficients",
    "polynomial_from_bezier",
]


def curve_kernel(gamma: float) -> np.ndarray:
    r"""The $3\times3$ curve kernel $K_\gamma$, verbatim from `plots.draw_curved_edges`.

    Rows are the coefficients of $1, t, t^2$; columns weight $P_0, P_1, P_2$. It is derived so
    that $C(0) = P_0$ and $C(1) = P_2$ for every γ. γ=0 is the straight chord, γ=0.5 the classic
    Bézier (a lerp of lerps), γ=1 the three-point interpolating curve with $C(1/2) = P_1$.
    """
    return np.array(
        [
            [1, 0, 0],
            [-1 - 2 * gamma, 4 * gamma, 1 - 2 * gamma],
            [2 * gamma, -4 * gamma, 2 * gamma],
        ],
        dtype=float,
    )


def polynomial_coefficients(controls: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    r"""Power-form coefficients $[C_0, C_1, C_2]$ of the γ-curve through ``controls``.

    Args:
        controls: ``(..., 3, d)`` — $(P_0, P_1, P_2)$, the vertex/edge-point triple, **not**
            Bézier control points.
        gamma: the curvature parameter.

    Returns:
        ``(..., 3, d)``, with $C(t) = C_0 + C_1 t + C_2 t^2$.
    """
    controls = np.asarray(controls, dtype=float)
    if controls.shape[-2] != 3:
        raise ValueError(f"controls must be (..., 3, d), got shape {controls.shape}")
    return curve_kernel(gamma) @ controls


def bezier_control_points(controls: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    r"""$(B_0, B_1, B_2)$ — the **Bernstein** control points of the γ-curve.

    Derived from :func:`polynomial_coefficients`, so the kernel remains the only definition of
    the curve in the tree. Equivalently, and this is what the test asserts,
    $B_0 = P_0$, $B_1 = 2\gamma P_1 + (1-2\gamma)(P_0+P_2)/2$, $B_2 = P_2$.

    At γ=0 the middle control point lands on the chord midpoint, which is an exact straight
    segment — that is how the straight variant of §4.3 is obtained, through the same code path
    as the curved one (DECISIONS (27), (15) superseded).

    Args:
        controls: ``(..., 3, d)`` — $(P_0, P_1, P_2)$.
        gamma: the curvature parameter.

    Returns:
        ``(..., 3, d)`` Bernstein control points.
    """
    c = polynomial_coefficients(controls, gamma)
    c0, c1, c2 = c[..., 0, :], c[..., 1, :], c[..., 2, :]
    return np.stack([c0, c0 + 0.5 * c1, c0 + c1 + c2], axis=-2)


def polynomial_from_bezier(B: np.ndarray) -> np.ndarray:
    r"""Power form $[C_0, C_1, C_2]$ of a Bézier given by its control points.

    $C_0 = B_0$, $C_1 = 2(B_1 - B_0)$, $C_2 = B_0 - 2B_1 + B_2$; note $C_2 = -2D$ with $D$ the
    bulge of :func:`bulge`.

    Args:
        B: ``(..., 3, d)`` Bernstein control points.

    Returns:
        ``(..., 3, d)`` power-form coefficients.
    """
    B = np.asarray(B, dtype=float)
    if B.shape[-2] != 3:
        raise ValueError(f"B must be (..., 3, d), got shape {B.shape}")
    b0, b1, b2 = B[..., 0, :], B[..., 1, :], B[..., 2, :]
    return np.stack([b0, 2.0 * (b1 - b0), b0 - 2.0 * b1 + b2], axis=-2)


def bulge(B: np.ndarray) -> np.ndarray:
    r"""$D = B_1 - (B_0 + B_2)/2$ — how far the curve departs from its chord.

    The curve's maximum perpendicular excursion from the chord is $\lVert D\rVert/2$, attained at
    $t=1/2$; in terms of the original triple, $D = 2\gamma(P_1 - (P_0+P_2)/2)$, so the excursion
    is exactly $\gamma h$ with $h$ the edge point's distance from the chord.

    $D$ is a **conditioning quantity only** (DECISIONS (30)): $\lVert D\rVert \approx 0$ says the
    curve is its chord and dispatches to a segment test. It bounds nothing — do not reintroduce a
    hull test on its account.

    Args:
        B: ``(..., 3, d)`` Bernstein control points.

    Returns:
        ``(..., d)``.
    """
    B = np.asarray(B, dtype=float)
    return B[..., 1, :] - 0.5 * (B[..., 0, :] + B[..., 2, :])


def exact_bounding_boxes(B: np.ndarray) -> np.ndarray:
    r"""The **exact** axis-aligned bounding box of each quadratic Bézier.

    Each coordinate of the curve is a scalar quadratic with a single stationary point, and the
    coordinates are treated **independently**: with $u = B_1 - B_0$ and $w = B_2 - B_1$ in one
    coordinate, $t^{*} = u/(u-w)$ and the extremal value is $B_0 + u^2/(u-w)$, which counts only
    when $t^{*} \in (0,1)$. $t^{*}_x$ and $t^{*}_y$ are generally different, so the box can be
    touched at up to four distinct parameters (``docs/CROSSING_GEOMETRY.md`` §2).

    This is not a nicety. The box on the control points is a valid bound but a loose one; the
    exact box is a **median 41 % smaller in area**, candidate counts scale with box area, and it
    costs ~123 ns per *edge* — per edge, not per pair, so 6 ms for all of tabula.

    A straight edge has $u = w$ in every coordinate; the division is guarded and the endpoints
    are then the whole range, which is correct.

    Args:
        B: ``(..., 3, d)`` Bernstein control points.

    Returns:
        ``(..., 2, d)`` — ``[..., 0, :]`` the lower corner, ``[..., 1, :]`` the upper one.
    """
    B = np.asarray(B, dtype=float)
    if B.shape[-2] != 3:
        raise ValueError(f"B must be (..., 3, d), got shape {B.shape}")
    b0, b1, b2 = B[..., 0, :], B[..., 1, :], B[..., 2, :]

    lower = np.minimum(b0, b2)
    upper = np.maximum(b0, b2)

    u = b1 - b0
    denom = u - (b2 - b1)  # == B0 - 2 B1 + B2, the coordinate of -2D
    interior = denom != 0
    safe = np.where(interior, denom, 1.0)
    t_star = u / safe
    extreme = b0 + u * u / safe

    counts = interior & (t_star > 0.0) & (t_star < 1.0)
    lower = np.where(counts, np.minimum(lower, extreme), lower)
    upper = np.where(counts, np.maximum(upper, extreme), upper)
    return np.stack([lower, upper], axis=-2)


def arc_length(B: np.ndarray) -> np.ndarray:
    r"""Exact arc length of each quadratic Bézier, in closed form — **no sampling**.

    With $u = B_1 - B_0$ and $w = B_0 - 2B_1 + B_2$ the speed is $\lVert B'(t)\rVert =
    2\lVert u + tw\rVert$, so with $a = \lVert w\rVert^2$, $b = 2\,u\!\cdot\!w$ and
    $c = \lVert u\rVert^2$ the length is $2\int_0^1\sqrt{at^2+bt+c}\,\mathrm dt$, which has an
    elementary antiderivative. This is what keeps ``detour_ratio`` — the one metric in the suite
    that is *about* arc length — free of a sampling parameter (spec §2.2).

    Three branches, and **the branch points are measured rather than assumed**, against 40-digit
    quadrature:

    * **Nearly no bend** ($\lVert w\rVert \le 10^{-3}\lVert u\rVert$): a three-term binomial
      series in $x = (at^2+bt)/c$. The closed form is not usable here — it divides by $a$ and by
      $a^{3/2}$, and its relative error grows as $10^{-16}/(\lVert w\rVert/\lVert u\rVert)$,
      reaching $2.6\times10^{-2}$ at a ratio of $10^{-14}$ and 2.8 at $10^{-16}$.
      ⚠️ **That band is not exotic — it is γ=0**, the straight variant of §4.3: $w$ is then not
      bitwise zero but a rounding residue of arbitrary *direction*, so a test for exact zero
      misses it and the collinear branch below reads the noise as a real direction. The series is
      accurate to $10^{-12}$ across this whole band and exact when $w$ really is zero.
    * **Collinear** ($4ac = b^2$, i.e. $u \parallel w$): the curve runs along a line, possibly
      doubling back through a cusp. The speed is $2\lvert\sqrt a\,t + b/(2\sqrt a)\rvert$ and
      the integral splits at the turning point — exact, and it correctly counts a doubled-back
      stretch twice. The test is relative ($\sin^2\theta$), so it fires only for genuine
      parallelism.
    * **Otherwise** the antiderivative, whose two logarithm arguments are strictly positive
      wherever $4ac > b^2$, so no branch cut is in play.

    Args:
        B: ``(..., 3, d)`` Bernstein control points.

    Returns:
        ``(...,)`` arc lengths. Accurate to ~1e-12 relative across every regime.
    """
    B = np.asarray(B, dtype=float)
    if B.shape[-2] != 3:
        raise ValueError(f"B must be (..., 3, d), got shape {B.shape}")
    b0, b1, b2 = B[..., 0, :], B[..., 1, :], B[..., 2, :]
    u = b1 - b0
    w = b0 - 2.0 * b1 + b2

    a = np.sum(w * w, axis=-1)
    b = 2.0 * np.sum(u * w, axis=-1)
    c = np.sum(u * u, axis=-1)

    tiny = np.finfo(float).tiny
    # 4ac - b^2 = 4(|u|^2|w|^2 - (u.w)^2) >= 0 by Cauchy-Schwarz, and 0 exactly when u || w.
    disc = np.maximum(4.0 * a * c - b * b, 0.0)
    nearly_straight = a <= 1e-6 * c
    collinear = ~nearly_straight & (disc <= 1e-14 * np.maximum(a * c, tiny))
    general = ~nearly_straight & ~collinear

    safe_a = np.where(general | collinear, a, 1.0)
    root_a = np.sqrt(safe_a)
    speed_0 = np.sqrt(np.maximum(c, 0.0))
    speed_1 = np.sqrt(np.maximum(a + b + c, 0.0))

    log_ratio = np.log(
        np.where(
            general,
            (2.0 * root_a * speed_1 + 2.0 * a + b) / np.where(general, 2.0 * root_a * speed_0 + b, 1.0),
            1.0,
        )
    )
    closed_form = ((2.0 * a + b) * speed_1 - b * speed_0) / (2.0 * safe_a)
    closed_form = closed_form + disc / (4.0 * safe_a * root_a) * log_ratio

    # Collinear: 2|sqrt(a) t + k| with k = b / (2 sqrt(a)); antiderivative sqrt(a) t^2 + 2 k t.
    k = b / (2.0 * root_a)
    turn = np.clip(-k / root_a, 0.0, 1.0)
    at_turn = root_a * turn * turn + 2.0 * k * turn
    at_one = root_a + 2.0 * k
    folded = np.abs(at_turn) + np.abs(at_one - at_turn)

    # sqrt(1+x) = 1 + x/2 - x^2/8 + x^3/16 integrated term by term over t in [0, 1].
    safe_c = np.where(c > 0, c, 1.0)
    first = (a / 3.0 + b / 2.0) / safe_c
    second = (a * a / 5.0 + a * b / 2.0 + b * b / 3.0) / (safe_c * safe_c)
    third = (a**3 / 7.0 + a * a * b / 2.0 + 3.0 * a * b * b / 5.0 + b**3 / 4.0) / safe_c**3
    series = speed_0 * (2.0 + first - second / 4.0 + third / 8.0)

    return np.where(nearly_straight, series, np.where(collinear, folded, closed_form))
