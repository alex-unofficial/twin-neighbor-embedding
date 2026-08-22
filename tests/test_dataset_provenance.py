"""Executable form of the claims in docs/DATASET_PROVENANCE.md.

These are assertions about the *data files*, not about our code. They exist because
``weight_kind`` is assigned from how a weight was produced, so if an underlying file is ever
replaced or regenerated, the classification silently stops being justified. Skipped when the
data is not present.
"""

import numpy as np
import pytest
import scipy.io as sio
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from twin.config import REAL_WORLD_DIR

TABULA = REAL_WORLD_DIR / "tabula-liver.mat"
pytestmark = pytest.mark.skipif(not TABULA.exists(), reason="tabula-liver.mat not present")


@pytest.fixture(scope="module")
def tabula():
    return sio.loadmat(str(TABULA))


@pytest.fixture(scope="module")
def tabula_A(tabula):
    return sp.csr_matrix(tabula["A"])


@pytest.fixture(scope="module")
def tabula_D(tabula):
    return sp.csr_matrix(tabula["D"])


def test_D_is_a_uniform_14nn_directed_graph(tabula_D):
    out_degree = np.diff(tabula_D.indptr)
    assert out_degree.min() == out_degree.max() == 14
    assert tabula_D.nnz == 70098


def test_A_and_D_span_the_same_edge_set(tabula_A, tabula_D):
    """70098 directed - 20730 reciprocal pairs == 49368 undirected, and the supports agree."""
    Db = tabula_D.astype(bool).astype(np.int8)
    reciprocal_pairs = Db.multiply(Db.T).nnz // 2

    assert reciprocal_pairs == 20730
    assert tabula_D.nnz - reciprocal_pairs == 49368
    assert sp.triu(tabula_A, k=1).nnz == 49368

    symmetrised_D = (Db + Db.T).astype(bool)
    assert (tabula_A.astype(bool) > symmetrised_D).nnz == 0, "edge in A but not in D"
    assert (symmetrised_D > tabula_A.astype(bool)).nnz == 0, "edge in D but not in A"


def test_distances_were_measured_in_X_scvi(tabula, tabula_D):
    """Not X_pca. The representation is part of the provenance and goes in the manifest."""
    rows = np.random.default_rng(0).choice(tabula_D.shape[0], size=100, replace=False)

    def max_abs_error(space):
        X = np.asarray(tabula[space], dtype=np.float64)
        errs = []
        for i in rows:
            s, e = tabula_D.indptr[i], tabula_D.indptr[i + 1]
            computed = np.linalg.norm(X[tabula_D.indices[s:e]] - X[i], axis=1)
            errs.append(np.abs(computed - tabula_D.data[s:e]))
        return np.concatenate(errs).max()

    assert max_abs_error("X_scvi") < 1e-5, "D is not Euclidean distance in X_scvi"
    assert max_abs_error("X_pca") > 1.0, "X_pca unexpectedly matches; provenance is ambiguous"


def test_D_neighbours_are_the_exact_14nn_in_X_scvi(tabula, tabula_D):
    X = np.asarray(tabula["X_scvi"], dtype=np.float64)
    rows = np.random.default_rng(1).choice(tabula_D.shape[0], size=50, replace=False)

    _, idx = NearestNeighbors(n_neighbors=15).fit(X).kneighbors(X[rows])
    for r, i in enumerate(rows):
        s, e = tabula_D.indptr[i], tabula_D.indptr[i + 1]
        assert set(tabula_D.indices[s:e]) == set(idx[r][1:])


def test_the_pile_up_at_one_is_umap_local_connectivity_not_a_clip(tabula_A, tabula_D):
    """Every point has membership 1 to its nearest neighbour by construction (rho_i)."""
    data = tabula_A.data
    assert data.max() == 1.0 and not (data > 1.0).any(), "values above 1 would suggest no cap"
    assert data.min() < 1e-6, "a clip would truncate the lower tail too"

    A_dense = tabula_A.toarray()
    exact_one = 0
    for i in range(tabula_D.shape[0]):
        s, e = tabula_D.indptr[i], tabula_D.indptr[i + 1]
        nearest = tabula_D.indices[s:e][np.argmin(tabula_D.data[s:e])]
        exact_one += A_dense[i, nearest] == 1.0

    assert exact_one > 0.85 * tabula_D.shape[0], (
        "A[i, nearest(i)] == 1 for most i is the signature of UMAP's rho_i; without it the "
        "kernel classification would need rechecking"
    )


def test_kernel_transform_would_zero_the_distance_for_most_vertices(tabula_A):
    """Why -log(w/w_max) is rejected for tabula: it is not an 8%-of-edges problem."""
    rows_at_max = np.unique(tabula_A.nonzero()[0][tabula_A.data == 1.0])
    fraction = rows_at_max.size / tabula_A.shape[0]
    assert fraction > 0.9, f"expected >90% of vertices touching a w=1 edge, got {fraction:.1%}"


@pytest.mark.parametrize(
    ("field", "n_distinct"),
    [("cell_ontology_class", 13), ("compartment", 4), ("donor", 2), ("method", 2)],
)
def test_label_fields_present_with_expected_cardinality(tabula, field, n_distinct):
    values = {str(x[0]) if len(x) else "" for x in tabula[field].ravel()}
    assert len(values) == n_distinct


@pytest.mark.parametrize("field", ["gender", "anatomical_information"])
def test_constant_fields_carry_no_information(tabula, field):
    values = {str(x[0]) if len(x) else "" for x in tabula[field].ravel()}
    assert len(values) == 1, f"{field} is no longer constant; reconsider registering it"
