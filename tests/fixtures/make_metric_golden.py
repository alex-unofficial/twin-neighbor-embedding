"""Regenerate ``metric_golden.json`` — the golden metric values of spec §6.

Run from the repository root with the project interpreter::

    venv/bin/python tests/fixtures/make_metric_golden.py

Three choices, all recorded in `DECISIONS.md`:

* **`karate_weighted`**, not `karate`. §6 says `karate_club`, which is no longer a registry
  name; `karate_weighted` is its literal successor, since `batch_registries["graph"]
  ["karate_club"]` *was* `nx.karate_club_graph`. It is also the only choice that works on the
  merits: on the unweighted `karate`, ADR degenerates to eq. (29) and `weighted_recall` equals
  `knn_recall` by construction, so a golden file there would pin two metrics at values that
  cannot detect a weighting bug.
* **The `karate_structured_layout` fixture**, not a seeded layout run. A golden file that runs
  SG-*t*-SNE pins the backend, so every `sgtsnepi` bump would read as a metric regression.
* **Keyed by metric name**, so steps 4-6 *add* keys instead of invalidating the file. One that
  has to be regenerated wholesale three times is one that will be regenerated without being
  read.

Regenerating is only ever correct in a commit that intends to change a metric's numerical output
and says so (CLAUDE.md, "Commit granularity").
"""

import json
from pathlib import Path

import networkx as nx
import numpy as np

from twin.datasets import load_dataset
from twin.metrics import (
    adjacency_distance_ratio,
    knn_accuracy,
    knn_jaccard,
    knn_precision,
    knn_recall,
    normalized_stress,
    raw_stress,
    scale_normalized_stress,
    weighted_recall,
)
from twin.metrics.context import EvaluationContext

OUT = Path(__file__).with_name("metric_golden.json")

#: The two neighbourhood sizes: the dataset's own default (mean degree) and a round 10.
KS = (5, 10)


def karate_structured_layout(graph: nx.Graph) -> np.ndarray:
    """The `conftest.py` fixture, duplicated here so the generator stands alone.

    ``tests/test_metric_golden.py`` asserts the two agree, so a change to the fixture cannot
    silently diverge from the file it generated.
    """
    labels = np.array([0 if graph.nodes[i]["club"] == "Mr. Hi" else 1 for i in graph.nodes()])
    degrees = np.array([d for _, d in graph.degree()], dtype=float)
    jitter = np.random.default_rng(1).normal(scale=0.25, size=(graph.number_of_nodes(), 2))
    return np.column_stack([labels * 0.6, degrees * 0.2]) + jitter


def build() -> dict:
    dataset = load_dataset("karate_weighted")
    ctx = EvaluationContext.for_dataset(dataset)
    X = karate_structured_layout(dataset.graph)
    labels = dataset.labels("club")

    values: dict[str, float] = {
        "adjacency_distance_ratio": adjacency_distance_ratio(X, ctx.affinity),
        "scale_normalized_stress": scale_normalized_stress(X, ideal=ctx.ideal),
        "raw_stress": raw_stress(X, ideal=ctx.ideal),
        "normalized_stress": normalized_stress(X, ideal=ctx.ideal),
    }
    for k in KS:
        values[f"knn_recall@k={k}"] = knn_recall(X, dataset.graph, k)
        values[f"knn_precision@k={k}"] = knn_precision(X, dataset.graph, k)
        values[f"knn_jaccard@k={k}"] = knn_jaccard(X, dataset.graph, k)
        values[f"weighted_recall@k={k}"] = weighted_recall(X, ctx.affinity, k)
        values[f"knn_accuracy@k={k}"] = knn_accuracy(X, labels, k)

    return {
        "dataset": "karate_weighted",
        "layout": "tests/conftest.py::karate_structured_layout",
        "k": list(KS),
        "default_k": ctx.k,
        "values": {name: float(value) for name, value in sorted(values.items())},
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"wrote {OUT}")
