"""digits — `sklearn.datasets.load_digits`, as a kNN network (spec §0.5).

It was registered as `mnist`, which it is not: 1797 samples of **8x8** digits, so neither large
nor MNIST. Renamed, and described accurately.

The stored side is the **distance** side — the kNN graph's weights are Euclidean distances in
pixel space, 5.3 to 37.8 — so the affinity form is the derived one. That is bug §1.6: those
distances were being fed to the representations as if larger meant *more* coupled.
"""

import sklearn.datasets as skdata

from twin.dataset import construct_knn_graph
from twin.datasets.base import Dataset, DatasetStyle, LabelSet, Stored
from twin.datasets.sparsify import KNearest

#: The batch and the provenance record both use 12; the old `dataset.mnist` defaulted to 100,
#: which makes a 1797-vertex graph with mean degree 125 and was never what got run.
DEFAULT_N_NEIGHBORS = 12


def digits(n_neighbors: int = DEFAULT_N_NEIGHBORS) -> Dataset:
    raw = skdata.load_digits()
    graph = construct_knn_graph(raw.data, n_neighbors)

    return Dataset(
        name="digits",
        graph=graph,
        affinity=None,
        distance=Stored(meaning="distance"),
        label_sets={
            "digit": LabelSet(
                name="digit",
                values=raw.target,
                display_names={i: str(i) for i in range(10)},
                derivation="ground truth",
                scoring_role="primary",
                display_role="hue",
            )
        },
        sparsity=KNearest(k=n_neighbors, symmetrise="or", space="raw 8x8 pixels"),
        style=DatasetStyle(palette="husl", palette_seed=0),
        params={"n_neighbors": n_neighbors},
        provenance={
            "source": "sklearn.datasets.load_digits",
            "distance_space": "raw 8x8 pixels",
            "distance_metric": "euclidean",
            "knn_k": n_neighbors,
        },
        description=(
            "sklearn's 8x8 handwritten digits as a symmetric k-NN network. Retained as the "
            "representative point-cloud-derived k-NN network, present to test whether the twin "
            "transform improves on t-SNE-style point-cloud embeddings. Contrast with tabula, "
            "which is the same kind of object built better: both come from a point cloud, but "
            "tabula persisted its distance matrix *and* its affinities, while digits persists "
            "only distances and must derive the rest."
        ),
    )
