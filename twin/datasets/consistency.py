"""Do a dataset's two stored sides agree about which pairs are close? (spec §3.4)

Storing both sides is the ideal — nothing has to be derived — but it introduces a hazard that a
single-sided dataset cannot have: two independent sources can disagree, silently and severely.
The layout optimises against the affinity while SNS and Sil* score against the distance, so a
method could look good or bad for reasons unconnected to the method.

Three statistics, the first two from `DECISIONS.md`:

* **global Spearman** over shared pairs — expected strongly negative but *not* $-1$, since
  tabula's per-row $\\rho_i, \\sigma_i$ vary;
* **median per-row Spearman**, on the symmetric forms the layout layer actually consumes;
* **median per-row Spearman over one-directional pairs only** — the pairs a symmetrisation
  could not have mixed. This one is the diagnostic: it separates "the two sources disagree"
  from "symmetrisation blurred an otherwise exact relationship", which the spec anticipated but
  could not distinguish with the first two alone.

Measured on tabula, and pinned in ``tests/test_datasets_registry.py``: global $-0.236$, per-row
$-0.782$, and one-directional **exactly $-1$ in every scored row**. So the relationship is exact
where it can be seen unmixed, and the degradation is fuzzy-union symmetrisation, not an
inconsistency between the two exports.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr

from twin.datasets.base import StoredMatrix

#: Rows shorter than this are not scored: a two-point rank correlation is always exactly +-1
#: and would swamp the median with an artifact of degree.
MIN_ROW = 3


@dataclass
class WeightConsistencyReport:
    """The outcome of comparing a dataset's two stored sides.

    Attributes:
        dataset: the dataset's name.
        n_shared: undirected pairs carrying both an affinity and a distance.
        n_affinity_only / n_distance_only: pairs present on one side only. Non-zero means the
            two sides do not even span the same edge set, which is a stronger fault than
            disagreeing about magnitudes.
        comparable: False when a side carries no rank information at all (an indicator, or a
            single constant weight). Nothing *could* disagree, so the correlations are NaN.
        global_spearman: rank correlation over all shared pairs.
        median_row_spearman: median over vertices of the within-row rank correlation, on the
            symmetric forms the layout layer consumes.
        median_row_spearman_unmixed: the same, restricted to pairs recorded in one direction
            only by the raw stored matrix — where a symmetrisation cannot have mixed two rows'
            parameters. NaN when neither side is a directed stored matrix.
        n_rows_scored / n_rows_scored_unmixed: rows with enough shared, non-constant entries.
    """

    dataset: str
    n_shared: int
    n_affinity_only: int
    n_distance_only: int
    comparable: bool
    global_spearman: float
    median_row_spearman: float
    median_row_spearman_unmixed: float
    n_rows_scored: int
    n_rows_scored_unmixed: int

    @property
    def spans_same_edge_set(self) -> bool:
        return self.n_affinity_only == 0 and self.n_distance_only == 0

    def summary(self) -> str:
        if not self.comparable:
            return f"{self.n_shared} shared pairs; not comparable (a side carries no ranks)"
        unmixed = (
            ""
            if np.isnan(self.median_row_spearman_unmixed)
            else (
                f", unmixed per-row rho={self.median_row_spearman_unmixed:+.4f} "
                f"over {self.n_rows_scored_unmixed} rows"
            )
        )
        return (
            f"{self.n_shared} shared pairs, global rho={self.global_spearman:+.4f}, "
            f"per-row rho={self.median_row_spearman:+.4f} over {self.n_rows_scored} rows"
            + unmixed
        )


def check_weight_consistency(dataset) -> WeightConsistencyReport | None:
    """Compare the two stored sides. Returns None unless **both** are stored.

    Both matrices are taken in the symmetric form the layout layer actually consumes, not in
    whatever directed form they were exported in — the question is whether the numbers that get
    *used* agree, not whether the raw files could be made to.
    """
    if dataset.affinity is None or dataset.distance is None:
        return None

    A = _clean(dataset.stored_matrix("affinity"))
    D = _clean(dataset.stored_matrix("distance"))

    a_pairs, d_pairs = _upper_pairs(A), _upper_pairs(D)
    shared = sorted(a_pairs.keys() & d_pairs.keys())

    comparable = _has_ranks(A) and _has_ranks(D) and len(shared) > 1
    if comparable:
        global_rho = float(
            spearmanr(
                np.array([a_pairs[p] for p in shared]),
                np.array([d_pairs[p] for p in shared]),
            ).statistic
        )
        row_rhos = _row_spearman(A, D)
        unmixed_rhos = _row_spearman(A, D, restrict=_one_directional_mask(dataset))
    else:
        global_rho = np.nan
        row_rhos = unmixed_rhos = np.array([])

    return WeightConsistencyReport(
        dataset=dataset.name,
        n_shared=len(shared),
        n_affinity_only=len(a_pairs.keys() - d_pairs.keys()),
        n_distance_only=len(d_pairs.keys() - a_pairs.keys()),
        comparable=comparable,
        global_spearman=global_rho,
        median_row_spearman=_median(row_rhos),
        median_row_spearman_unmixed=_median(unmixed_rhos),
        n_rows_scored=int(row_rhos.size),
        n_rows_scored_unmixed=int(unmixed_rhos.size),
    )


# --------------------------------------------------------------------------- #
#                                  internals                                   #
# --------------------------------------------------------------------------- #


def _clean(M) -> sp.csr_array:
    M = sp.csr_array(M).copy()
    M.eliminate_zeros()
    return M


def _has_ranks(M: sp.csr_array) -> bool:
    """False for an indicator or a single constant weight — nothing to rank."""
    return np.unique(M.data).size > 1


def _median(a: np.ndarray) -> float:
    return float(np.median(a)) if a.size else np.nan


def _upper_pairs(M: sp.csr_array) -> dict[tuple[int, int], float]:
    coo = sp.triu(M, k=1).tocoo()
    return {(int(i), int(j)): float(v) for i, j, v in zip(coo.row, coo.col, coo.data)}


def _one_directional_mask(dataset) -> sp.csr_array | None:
    """Row ``i``'s entries that the raw stored matrix recorded as ``i -> j`` only, or None.

    Deliberately **not** symmetric. A symmetrisation can only mix two rows' parameters where
    both directions were recorded, so within row ``i`` these are the entries whose affinity is
    ``i``'s own, unmixed — the ones on which an exactly-monotone relationship should survive
    intact. Adding the pairs ``j`` recorded one-directionally would put row ``j``'s parameters
    back in, which is the very thing being excluded.
    """
    for side in ("distance", "affinity"):
        source = getattr(dataset, side)
        if isinstance(source, StoredMatrix) and source.symmetrise != "none":
            raw = _clean(dataset.matrices[source.key]).astype(bool)
            both = raw.multiply(raw.T)
            return sp.csr_array((raw.astype(np.int8) - both.astype(np.int8)).astype(bool))
    return None


def _row_spearman(
    A: sp.csr_array, D: sp.csr_array, restrict: sp.csr_array | None = None
) -> np.ndarray:
    """Within-row rank correlation over each vertex's shared neighbours."""
    if restrict is not None:
        restrict = sp.csr_array(restrict)

    rhos = []
    for i in range(A.shape[0]):
        a_idx, a_val = _row(A, i)
        d_idx, d_val = _row(D, i)
        _, ia, id_ = np.intersect1d(a_idx, d_idx, return_indices=True)
        a, d = a_val[ia], d_val[id_]

        if restrict is not None:
            keep = np.isin(a_idx[ia], _row(restrict, i)[0])
            a, d = a[keep], d[keep]

        if a.size < MIN_ROW or np.unique(a).size < 2 or np.unique(d).size < 2:
            continue
        rhos.append(spearmanr(a, d).statistic)

    return np.asarray(rhos)


def _row(M: sp.csr_array, i: int) -> tuple[np.ndarray, np.ndarray]:
    s, e = M.indptr[i], M.indptr[i + 1]
    return M.indices[s:e], M.data[s:e]
