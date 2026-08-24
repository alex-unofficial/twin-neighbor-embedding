"""Shared metric machinery: orientation checks, the ideal-distance model, distance helpers.

Every layout array in this package is ``(n_points, d)``. Conversion from the ``(d, n)``
orientation the embedding layer returns happens exactly once, in
``twin.results.LayoutResult.from_raw``; see CLAUDE.md. Orientation is asserted at entry, never
inferred from ``shape`` — inference silently picks the wrong axis whenever ``n == d`` or the
input is square (spec §1.3).
"""

import numpy as np


def require_n_by_d(X: np.ndarray, n_expected: int, name: str = "X") -> np.ndarray:
    """Assert that ``X`` is ``(n_expected, d)``. Never reorients.

    Args:
        X: the layout array.
        n_expected: the number of points the caller's graph or label vector implies.
        name: used in the error message.

    Returns:
        ``X`` as a 2-D float array.

    Raises:
        ValueError: if ``X`` is not 2-D or its first axis is not ``n_expected``. If the
            transpose would have fitted, the message says so — that is the common call-site
            mistake and it should be fixed at the boundary, not here.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_points, d), got shape {X.shape}")
    if X.shape[0] != n_expected:
        hint = ""
        if X.shape[1] == n_expected:
            hint = (
                f" -- {name}.T would fit, so this is probably a (d, n) array. Convert it once "
                "at the boundary via LayoutResult.from_raw rather than here."
            )
        raise ValueError(
            f"{name} must be (n_points, d) with n_points={n_expected}, got {X.shape}{hint}"
        )
    return X
