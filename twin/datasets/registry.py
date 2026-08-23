"""`DATASET_REGISTRY` — name -> factory (spec §3.4).

Replaces the raw graph factories in `pipelines.batch_registries["graph"]`. Batch specs keep the
same syntax — `("football_2023", {})`, `("small_world", {"n": 150, "k": 12, "p": 0.01})` — but
the runner now receives a `Dataset`, so labels and weight semantics arrive with the graph
instead of being looked up separately.

Adding a dataset is one file plus one line here. Nothing downstream may branch on a name.

Not registered, deliberately: `polblogs`, `lfr`, `homer`, `football` (the older season),
`mycielski`, `stick_sculpture`. They exist in `twin/dataset.py` but have no provenance record,
so their stored sides cannot be declared honestly. Registering one means writing down where its
weights came from first — see `docs/DATASET_PROVENANCE.md`.
"""

from collections.abc import Callable

from twin.datasets.base import Dataset
from twin.datasets.digits import digits
from twin.datasets.football import football_2023
from twin.datasets.karate import karate, karate_weighted
from twin.datasets.sbm_er_3modes import sbm_er_3modes
from twin.datasets.synthetic import barabasi_albert, ring_of_cliques, small_world, tree_of_cliques
from twin.datasets.tabula import tabula

DATASET_REGISTRY: dict[str, Callable[..., Dataset]] = {
    "ring_of_cliques": ring_of_cliques,
    "tree_of_cliques": tree_of_cliques,
    "small_world": small_world,
    "barabasi_albert": barabasi_albert,
    "karate": karate,
    "karate_weighted": karate_weighted,
    "football_2023": football_2023,
    "sbm_er_3modes": sbm_er_3modes,
    "digits": digits,
    "tabula": tabula,
}


def load_dataset(name: str, **params) -> Dataset:
    """Build a registered dataset. The (name, params) pair is what a manifest records.

    A persisted layout must store this **reference**, not the object: a `Dataset` holds the
    graph, every label vector, the style and — for tabula — two stored matrices of ~50k edges
    each, and serialising it would write all of that into every layout file, once per
    representation, per seed, per lambda (spec §4.6).
    """
    try:
        factory = DATASET_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown dataset {name!r}; registered: {sorted(DATASET_REGISTRY)}"
        ) from None

    dataset = factory(**params)
    if dataset.name != name:
        raise ValueError(
            f"registry name {name!r} does not match the dataset's own name {dataset.name!r}; "
            "the name is the manifest key and must round-trip"
        )
    return dataset
