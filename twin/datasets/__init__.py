"""Dataset objects: source of truth, labels, style (spec §3.4).

Adding a dataset means adding **one file** here and one line in :mod:`twin.datasets.registry`.
Nothing in the metric, batch, plotting or reporting layer may branch on a dataset name; every
dataset-specific behaviour is reachable through the :class:`Dataset` object.
"""

from twin.datasets.base import (
    Dataset,
    DatasetStyle,
    LabelSet,
    Meaning,
    Stored,
    StoredMatrix,
    StoredSource,
    Unweighted,
)
from twin.datasets.consistency import WeightConsistencyReport, check_weight_consistency
from twin.datasets.sparsify import KNearest, SparsityModel
from twin.datasets.registry import DATASET_REGISTRY, load_dataset

__all__ = [
    "DATASET_REGISTRY",
    "Dataset",
    "DatasetStyle",
    "KNearest",
    "LabelSet",
    "Meaning",
    "Stored",
    "StoredMatrix",
    "SparsityModel",
    "StoredSource",
    "Unweighted",
    "WeightConsistencyReport",
    "check_weight_consistency",
    "load_dataset",
]
