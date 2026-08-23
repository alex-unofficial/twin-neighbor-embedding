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

__all__ = [
    "Dataset",
    "DatasetStyle",
    "LabelSet",
    "Meaning",
    "Stored",
    "StoredMatrix",
    "StoredSource",
    "Unweighted",
]
