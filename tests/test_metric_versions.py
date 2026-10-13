"""The rule that keeps a resume from serving stale numbers, made enforceable.

CLAUDE.md: a commit that intentionally changes a metric's output regenerates
``metric_golden.json`` *in the same commit*. Step 7 adds the other half: it also bumps that
metric's :attr:`~twin.metrics.registry.MetricSpec.version`, because the version is inside the
metric-set digest and the digest is what decides whether a persisted shard is still current.
Regenerate the golden file without bumping, and a resumed run silently keeps the old values.

**How this test enforces it.** :data:`HISTORY` records, per metric, every ``(version, digest of
that metric's golden values)`` it has had. The last entry must match the registry's version and
the golden file's current values; versions strictly increase, and two entries never share a
digest. So regenerating the golden file fails here until a new entry is appended — and an
appended entry with the old version number fails too.

To change a metric's output, in one commit: change the code, regenerate the golden file, bump
the metric's ``version`` in ``twin/metrics/registry.py``, and append ``(new_version,
new_digest)`` below — the failure message prints the digest.
"""

import hashlib
import json
from pathlib import Path

import pytest

from twin.metrics.registry import METRIC_REGISTRY, metric_set_digest

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "metric_golden.json").read_text())

#: name -> [(version, digest of the metric's golden values)], oldest first. Append only.
HISTORY = {
    "adjacency_distance_ratio": [(1, "9392024553c46123")],
    "knn_recall": [(1, "5d2299ddbd164abe")],
    "knn_precision": [(1, "ff9847a93bde87d0")],
    "knn_jaccard": [(1, "4f789d157c020287")],
    "weighted_recall": [(1, "e3a2e8657b47f9f0")],
    "scale_normalized_stress": [(1, "0bcdd44a03700d81")],
    "raw_stress": [(1, "802cc426d9ad0b01")],
    "normalized_stress": [(1, "c6fd0ca59227fc88")],
    "normalized_nn_distance": [(1, "c1cf35af41070738")],
    "node_resolution": [(1, "32405b65d2bee84c")],
    "node_resolution_vertexwise": [(1, "0db98595ade5f71b")],
    "pairwise_distance_distribution": [(1, "6313c9f863b7b85e")],
    "nn_distance_distribution": [(1, "f91bd3440b275ce0")],
    "edge_lengths": [(1, "61c5cde3aa5f4584")],
    "edge_length_cv": [(1, "451107a498d67122")],
    "knn_accuracy": [(1, "ed5a9dbb4be8d547")],
    "silhouette": [(1, "b246b23fe2d80c62")],
    "crossing_number": [(1, "3797b7f3df1b8657")],
    "crossings_per_edge_pair": [(1, "660cb613727965bf")],
    "length_weighted_crossings": [(1, "f2210030aab86b66")],
    "detour_ratio": [(1, "0ab5dd95514ff779")],
    "silhouette_graph": [(1, "2cbdc4065dda4757")],
    "graph_neighborhood_accuracy": [(1, "6b3486ab5d78aade")],
    "majority_class_rate": [(1, "7c8792076f3bfc49")],
}


def golden_digest(name: str) -> str:
    values = sorted(
        (key, repr(value)) for key, value in GOLDEN["values"].items() if key.split("@")[0] == name
    )
    return hashlib.blake2b(json.dumps(values).encode()).hexdigest()[:16]


def test_every_registered_metric_has_a_history():
    assert set(HISTORY) == set(METRIC_REGISTRY)


@pytest.mark.parametrize("name", sorted(METRIC_REGISTRY))
def test_a_changed_golden_value_comes_with_a_version_bump(name):
    version, digest = HISTORY[name][-1]
    current = golden_digest(name)
    assert current == digest, (
        f"{name}'s golden values changed. Bump its `version` in twin/metrics/registry.py to "
        f"{version + 1} and append ({version + 1}, {current!r}) to HISTORY[{name!r}] — a shard "
        "computed under the old version would otherwise be resumed onto."
    )
    assert METRIC_REGISTRY[name].version == version, (
        f"{name} is at version {METRIC_REGISTRY[name].version} but its history ends at "
        f"{version}; a version bump must come with the golden values it describes"
    )


@pytest.mark.parametrize("name", sorted(HISTORY))
def test_the_history_is_append_only(name):
    versions = [v for v, _ in HISTORY[name]]
    digests = [d for _, d in HISTORY[name]]
    assert versions == sorted(set(versions)), "versions must strictly increase"
    assert len(set(digests)) == len(digests), "a bump with unchanged values invalidates for nothing"


def test_every_metric_has_golden_values():
    """A metric with nothing pinned could change output without this file noticing."""
    for name in METRIC_REGISTRY:
        assert any(key.split("@")[0] == name for key in GOLDEN["values"]), name


# --------------------------------------------------------------------------- #
#                    The digest moves with what changes the rows               #
# --------------------------------------------------------------------------- #


def _with(name, **changes):
    import dataclasses

    specs = dict(METRIC_REGISTRY)
    specs[name] = dataclasses.replace(specs[name], **changes)
    return specs


def test_a_version_bump_moves_the_digest():
    assert metric_set_digest(_with("silhouette", version=2)) != metric_set_digest()


def test_a_variant_parameter_moves_the_digest():
    changed = _with("crossing_number", params={"gamma": 0.25, "include_adjacent": True})
    assert metric_set_digest(changed) != metric_set_digest()


def test_adding_a_metric_moves_the_digest():
    specs = dict(METRIC_REGISTRY)
    del specs["edge_length_cv"]
    assert metric_set_digest(specs) != metric_set_digest()


def test_an_explicit_k_moves_the_digest():
    """The rows of a k-sweep are not the rows of the default, so a shard must not be reused."""
    assert metric_set_digest(k=None) != metric_set_digest(k=(5, 10))


def test_the_digest_is_stable_and_not_a_function_of_identity():
    """Rebuilding an equal registry gives the same digest — no function ids, no source."""
    assert metric_set_digest(dict(METRIC_REGISTRY), k=None) == metric_set_digest(k=None)
