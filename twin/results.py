"""The layout result type (spec §3.1) — the single orientation boundary.

Layout arrays are ``(d, n)`` in :mod:`twin.embedding`, up to and including the ``embed`` call,
and ``(n, d)`` everywhere downstream. :meth:`LayoutResult.from_raw` is the *only* place that
conversion happens; nothing downstream of a ``LayoutResult`` ever sees ``(d, n)``.

Orientation is never re-derived by inspecting ``shape`` — that is wrong whenever ``n == d``,
and guessing per call site is the latent bug in spec §1.3.
"""

from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
import numpy as np

PointProvenance = Literal["primary", "derived"]

#: Point sets a layout can carry. "V" is the original graph's vertices, "E" its edges.
POINT_SETS = ("V", "E")


def _as_n_by_d(raw: np.ndarray, n_expected: int, what: str) -> np.ndarray:
    """Transpose one ``(d, n)`` array to ``(n, d)``, asserting rather than sniffing."""
    arr = np.asarray(raw)
    if arr.ndim != 2:
        raise ValueError(f"{what}: expected a 2-D (d, n) array, got shape {arr.shape}")
    if arr.shape[1] != n_expected:
        raise ValueError(
            f"{what}: expected (d, {n_expected}) from the embedding layer, got {arr.shape}. "
            "The embedding layer is (d, n); this is the only place that is converted."
        )
    return np.ascontiguousarray(arr.T)


@dataclass
class LayoutResult:
    """One layout of one graph by one method.

    Attributes:
        V: ``(n, d)`` vertex positions.
        E: ``(m, d)`` edge-point positions, or None for a method that produces none.
        slopes: ``(m,)`` edge orientations, or None.
        G: the graph that was laid out.
        representation: registry name of the ``GraphMatrixMethod`` used.
        embedding: registry name of the layout method used.
        params: the parameters of both, for the manifest.
        seed: the seed passed to the layout method, or None if it takes none.
        point_provenance: per point set, whether those points were laid out directly
            ("primary") or reconstructed from the other set ("derived"). See spec §0.5
            decision 5 — scoring derived sets is correct, but the asymmetry must be visible.
        runtime_s: wall time of the embed step only.
        representation_s: wall time of building the representation matrix only.
        n_points: rows of the matrix actually laid out — ``n`` for A_V, ``m`` for A_E,
            ``n + m`` for A_tne. Makes the cost of the transform visible in every row.
        peak_mem_mb: peak Python-side allocation during the embed, or None when not measured.
    """

    V: np.ndarray
    E: np.ndarray | None
    slopes: np.ndarray | None
    G: nx.Graph
    representation: str
    embedding: str
    params: dict = field(default_factory=dict)
    seed: int | None = None
    point_provenance: dict[str, PointProvenance] = field(default_factory=dict)
    runtime_s: float | None = None
    representation_s: float | None = None
    n_points: int | None = None
    peak_mem_mb: float | None = None

    def __post_init__(self):
        n, m = self.G.number_of_nodes(), self.G.number_of_edges()

        if self.V.ndim != 2 or self.V.shape[0] != n:
            raise ValueError(f"V must be (n, d) with n={n}, got {self.V.shape}")
        if self.E is not None and (self.E.ndim != 2 or self.E.shape[0] != m):
            raise ValueError(f"E must be (m, d) with m={m}, got {self.E.shape}")
        if self.E is not None and self.E.shape[1] != self.V.shape[1]:
            raise ValueError(
                f"V and E disagree on dimension: {self.V.shape[1]} vs {self.E.shape[1]}"
            )
        if self.slopes is not None and self.slopes.shape != (m,):
            raise ValueError(f"slopes must be ({m},), got {self.slopes.shape}")

        unknown = set(self.point_provenance) - set(POINT_SETS)
        if unknown:
            raise ValueError(f"unknown point sets in point_provenance: {sorted(unknown)}")
        bad = {k: v for k, v in self.point_provenance.items() if v not in ("primary", "derived")}
        if bad:
            raise ValueError(f"point_provenance values must be primary/derived, got {bad}")

    # ----------------------------------------------------------------- shapes

    @property
    def n(self) -> int:
        return self.V.shape[0]

    @property
    def m(self) -> int:
        return self.G.number_of_edges()

    @property
    def d(self) -> int:
        return self.V.shape[1]

    def provenance_of(self, point_set: str) -> PointProvenance:
        """Whether ``point_set`` was laid out directly or reconstructed from the other one."""
        try:
            return self.point_provenance[point_set]
        except KeyError:
            raise KeyError(
                f"point_provenance not declared for {point_set!r}; the representation class "
                "must declare it (spec §0.5 decision 5)"
            ) from None

    # ----------------------------------------------------------- construction

    @classmethod
    def from_raw(
        cls,
        *,
        V: np.ndarray,
        E: np.ndarray | None,
        slopes: np.ndarray | None,
        G: nx.Graph,
        representation: str,
        embedding: str,
        point_provenance: dict[str, PointProvenance],
        **kwargs,
    ) -> "LayoutResult":
        """Build from the ``(d, n)`` arrays the embedding layer returns.

        This is the one and only transpose. ``V`` must be ``(d, n)`` and ``E`` ``(d, m)``;
        anything else raises rather than being silently reoriented.
        """
        n, m = G.number_of_nodes(), G.number_of_edges()
        return cls(
            V=_as_n_by_d(V, n, "V"),
            E=None if E is None else _as_n_by_d(E, m, "E"),
            slopes=None if slopes is None else np.asarray(slopes).ravel(),
            G=G,
            representation=representation,
            embedding=embedding,
            point_provenance=dict(point_provenance),
            **kwargs,
        )
