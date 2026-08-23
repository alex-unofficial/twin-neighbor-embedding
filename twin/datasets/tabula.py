"""Tabula Sapiens — the one dataset that stores **both** sides (docs/DATASET_PROVENANCE.md).

`A` is UMAP's fuzzy simplicial set over an exact 14-NN graph built in a 50-d scVI latent space;
`D` is the measured kNN distance matrix from that same space. Two independently stored sides is
the ideal case: nothing is transformed in either direction, tabula's *measured* distances can
drive a Kamada-Kawai vertex layout while its adaptive affinities drive SG-t-SNE, and the same
geometry underlies both.

`A` is declared `adaptive_kernel`, whose derivation rule is to **raise**. UMAP membership is
$\\exp(-(d - \\rho_i)/\\sigma_i)$ with per-point $\\rho_i, \\sigma_i$ that are not stored, so no
global transform of `A` recovers $d$ — applying the `kernel` rule anyway would assign ideal
distance exactly 0 to one edge per vertex, requiring 93.1% of vertices to coincide. If `D` is
ever dropped, this dataset must fail loudly rather than fall back.

The price of two stored sides is that they can disagree; see
:func:`~twin.datasets.consistency.check_weight_consistency`.
"""

import warnings

import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from twin.config import REAL_WORLD_DIR
from twin.datasets.base import Dataset, DatasetStyle, LabelSet, StoredMatrix

#: Fields in the `.mat` that carry information. `gender` and `anatomical_information` are
#: constant across all 5007 cells and are deliberately not registered.
LABEL_FIELDS = ("cell_ontology_class", "compartment", "donor", "method")


def _string_field(mat, field: str) -> np.ndarray:
    """One MATLAB cell array of strings, as a flat array of Python strings."""
    return np.array([str(x[0]) if len(x) else "" for x in mat[field].ravel()])


def _coded(values: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    """Integer codes in alphabetical order, matching what the figures already produce."""
    names = np.unique(values)
    code_of = {name: i for i, name in enumerate(names)}
    return np.array([code_of[v] for v in values]), {i: str(n) for n, i in code_of.items()}


def tabula(organ: str = "liver") -> Dataset:
    with warnings.catch_warnings():
        # The export carries a stray duplicate variable name; scipy warns and takes the later
        # one. Nothing we read is affected.
        warnings.filterwarnings("ignore", category=sio.matlab.MatReadWarning)
        mat = sio.loadmat(str(REAL_WORLD_DIR / f"tabula-{organ}.mat"))

    A = sp.csr_array(mat["A"])
    D = sp.csr_array(mat["D"])
    graph = nx.from_scipy_sparse_array(sp.csc_matrix(mat["A"]))

    label_sets = {}
    for field in LABEL_FIELDS:
        values, names = _coded(_string_field(mat, field))
        label_sets[field] = LabelSet(
            name=field, values=values, display_names=names, derivation="ground truth"
        )

    # The fine-grained cell type is what the method claims to resolve, so it is the scoring
    # primary; the coarse compartment is its parent and supplies hue. This inverts football's
    # arrangement, which is why the two roles are separate fields.
    label_sets["cell_ontology_class"].scoring_role = "primary"
    label_sets["cell_ontology_class"].display_role = "shade"
    label_sets["cell_ontology_class"].parent = "compartment"
    label_sets["compartment"].scoring_role = "sub"
    label_sets["compartment"].display_role = "hue"
    # Batch covariates, scored *deliberately*: a layout that separates cells by donor rather
    # than by cell type is displaying an artifact, so a low score is the good outcome there.
    for field in ("donor", "method"):
        label_sets[field].scoring_role = "auxiliary"
        label_sets[field].display_role = "none"

    return Dataset(
        name="tabula",
        graph=graph,
        affinity=StoredMatrix("A", meaning="adaptive_kernel"),
        distance=StoredMatrix("D", meaning="distance"),
        matrices={"A": A, "D": D},
        label_sets=label_sets,
        composite_of={"compartment_cell_type": ("compartment", "cell_ontology_class")},
        reference_layouts={
            "X_umap": np.asarray(mat["X_umap"], dtype=float),
            "X_scvi_umap": np.asarray(mat["X_scvi_umap"], dtype=float),
        },
        # `custom_cmap(13, seed=7)` with rows 0 and 5 exchanged is what the teaser draws today;
        # the swap is hand-tuned and is declared here rather than repeated in the figure.
        style=DatasetStyle(palette="distinctipy", palette_seed=7, palette_swaps=((0, 5),)),
        params={"organ": organ},
        provenance={
            "organ": organ,
            "distance_space": "X_scvi",
            "distance_metric": "euclidean",
            "knn_k": 14,
            "affinity": "umap_connectivities",
            "affinity_symmetrisation": "fuzzy union",
        },
        description=(
            f"Tabula Sapiens {organ}: single-cell transcriptomes as a UMAP fuzzy simplicial "
            "set over an exact 14-NN graph in a 50-d scVI latent space. The only dataset that "
            "stores both sides — measured distances and adaptive affinities from the same "
            "source — which is what a point-cloud-derived graph should do."
        ),
    )
