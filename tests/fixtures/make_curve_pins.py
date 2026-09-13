"""Regenerate ``curve_pins.npz`` — the characterisation pins for step 5 commit 1.

Run from the repository root with the project interpreter::

    venv/bin/python tests/fixtures/make_curve_pins.py

The arrays are the **sampled curves that ``plots.draw_curved_edges`` actually plots**, lifted off
the Matplotlib lines it creates, for six values of γ on a fixed graph and a fixed pair of layouts.
They were generated from the code as it stood **before** the curve geometry moved into
``twin/geometry.py``, so ``tests/test_geometry.py`` can assert the move was bit-for-bit
behaviour-preserving rather than merely claiming it (CLAUDE.md: a refactor and a behaviour change
never share a commit, and "behaviour-preserving" needs a check).

Regenerating this file is only ever correct in a commit that *intends* to change the drawn curve
and says so.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  - must follow the backend selection
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from twin.plots import draw_curved_edges  # noqa: E402

OUT = Path(__file__).with_name("curve_pins.npz")

#: γ = 0 is the straight chord, 0.5 the classic Bézier, 1 the interpolating curve; 1.5 overshoots.
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)

#: Small enough to store, large enough that the quadratic is not determined by its endpoints.
N_SAMPLES = 7


def fixture_inputs():
    """A deterministic graph and a deterministic ``(d, n)`` / ``(d, m)`` layout pair.

    ``plots.py`` is still the legacy ``(d, n)`` surface (CLAUDE.md), and ``draw_curved_edges``
    indexes ``v_pos.T[u]`` and ``e_pos.T[e]`` — so these are ``(2, n)`` and ``(2, m)``.
    """
    G = nx.karate_club_graph()
    rng = np.random.default_rng(20260914)
    v_pos = rng.normal(size=(2, G.number_of_nodes()))
    e_pos = rng.normal(size=(2, G.number_of_edges()))
    return G, v_pos, e_pos


def sampled_curves(G, v_pos, e_pos, gamma: float) -> np.ndarray:
    """``(m, n_samples, 2)`` — the vertices of every polyline ``draw_curved_edges`` plots."""
    figure, ax = plt.subplots()
    try:
        draw_curved_edges(
            G,
            2,
            v_pos,
            e_pos,
            ax,
            n_samples=N_SAMPLES,
            alpha=0.5,
            linewidth=1.0,
            gamma=gamma,
        )
        return np.array([line.get_xydata() for line in ax.lines])
    finally:
        plt.close(figure)


def build() -> dict[str, np.ndarray]:
    G, v_pos, e_pos = fixture_inputs()
    arrays = {f"gamma_{gamma:g}": sampled_curves(G, v_pos, e_pos, gamma) for gamma in GAMMAS}
    arrays["v_pos"] = v_pos
    arrays["e_pos"] = e_pos
    return arrays


if __name__ == "__main__":
    arrays = build()
    np.savez_compressed(OUT, **arrays)
    for name, value in arrays.items():
        print(f"{name:12s} {value.shape}")
    print(f"\nwrote {OUT}")
