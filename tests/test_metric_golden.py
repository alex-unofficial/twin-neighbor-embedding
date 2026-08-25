"""The golden metric suite (spec §6).

Every metric in the package, on one fixed dataset and one fixed deterministic layout, against
values stored in ``fixtures/metric_golden.json``. Where `test_characterisation.py` pins what the
*pre-existing* code did so a refactor can be shown to preserve it, this pins what the suite
computes *now*, so that steps 4-11 cannot move a number without saying so.

Regenerate only in a commit that intends to change a metric's output::

    venv/bin/python tests/fixtures/make_metric_golden.py

The file is keyed by metric name so steps 4-6 add keys rather than invalidating it — a golden
file that has to be regenerated wholesale three times will be regenerated without being read.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from twin.datasets import load_dataset
from twin.metrics.context import EvaluationContext

GOLDEN = Path(__file__).parent / "fixtures" / "metric_golden.json"

# Loaded by path rather than imported: `tests/` is not a package, and the generator has to stay
# runnable on its own from the repository root.
_spec = importlib.util.spec_from_file_location(
    "make_metric_golden", Path(__file__).parent / "fixtures" / "make_metric_golden.py"
)
generator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generator)


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN.exists():
        pytest.skip("metric_golden.json not present; run tests/fixtures/make_metric_golden.py")
    return json.loads(GOLDEN.read_text())


def test_every_metric_reproduces_its_golden_value(golden):
    """One assertion per metric, so a failure names the metric rather than the suite."""
    computed = generator.build()["values"]
    stored = golden["values"]

    assert set(computed) == set(stored), (
        "the metric set changed; regenerate the golden file in a commit that says so"
    )
    for name, value in stored.items():
        assert computed[name] == pytest.approx(value, rel=1e-12), name


def test_the_generators_layout_is_the_conftest_fixture(karate_structured_layout):
    """The generator duplicates the fixture so it can stand alone; they must not diverge."""
    graph = load_dataset("karate_weighted").graph
    assert karate_structured_layout == pytest.approx(generator.karate_structured_layout(graph))


def test_the_golden_file_records_its_provenance(golden):
    assert golden["dataset"] == "karate_weighted"
    assert golden["layout"] == "tests/conftest.py::karate_structured_layout"
    assert golden["default_k"] == 5


def test_the_golden_dataset_is_weighted():
    """On unweighted `karate`, ADR degenerates to eq. (29) and weighted recall to recall.

    A golden file there would pin two of the metrics at values that cannot detect a weighting
    bug, which is why §6's `karate_club` is read as `karate_weighted`.
    """
    dataset = load_dataset("karate_weighted")
    weights = {w for _, _, w in dataset.graph.edges(data="weight")}
    assert len(weights) > 1


def test_the_golden_values_are_not_degenerate(golden):
    """A file full of 0, 1 and nan would reproduce under almost any regression."""
    values = golden["values"]
    assert all(np.isfinite(v) for v in values.values())
    assert all(0.0 < v < 1.0 for name, v in values.items() if "recall" in name or "jaccard" in name)
    assert 0.0 < values["scale_normalized_stress"] < 1.0

    # ADR above 1 is not a defect: this layout places vertices by (club, degree), so a hub's
    # edges run further than the average pair. It is pinned because it is informative.
    assert values["adjacency_distance_ratio"] > 1.0

    # The visual metrics (spec §2.2, step 4). NND is a mean of the k smallest distances over the
    # mean of all of them, so it lies in (0, 1) and rises with k.
    assert 0.0 < values["normalized_nn_distance@k=5"] < values["normalized_nn_distance@k=10"] < 1.0

    # NR is the foil VNR is defined against, not a reported metric. NR <= VNR under either
    # normaliser is an identity, not an observation: each vertex's own minimum is at least the
    # global minimum and its own maximum at most the diameter. The gap here is a factor of 35 —
    # this fixture's jitter put two vertices very close together, which is exactly the case that
    # sends NR to near zero while VNR stays informative.
    assert 0.0 < values["node_resolution"] < values["node_resolution_vertexwise@max"]
    assert values["node_resolution_vertexwise@max"] > 30 * values["node_resolution"]

    # The distributions' lower tails coincide: the closest pair in the layout is some vertex's
    # nearest neighbour, so the two minima are the same number.
    assert (
        values["pairwise_distance_distribution@min"]
        == values["nn_distance_distribution@min"]
    )
    assert values["pairwise_distance_distribution@max"] > 1.0  # normalised by the mean
    assert values["edge_length_cv"] > 0.0


def test_the_golden_file_is_keyed_by_metric_name(golden):
    """Steps 4-6 must be able to add keys without touching the existing ones."""
    assert "adjacency_distance_ratio" in golden["values"]
    assert any(name.startswith("knn_recall@k=") for name in golden["values"])
    # Step 4 added its keys without regenerating anything: the diff was 13 insertions and no
    # deletions, which is the only available evidence that the first fourteen did not move.
    assert "node_resolution" in golden["values"]
    assert any(name.startswith("normalized_nn_distance@k=") for name in golden["values"])


def test_the_context_is_the_one_the_golden_values_were_built_from():
    """The golden file scores against the shared target, not against `dataset.graph`."""
    ctx = EvaluationContext.for_dataset(load_dataset("karate_weighted"))
    assert ctx.affinity.graph["twin_side"] == "affinity"
    assert ctx.ideal.shape == (34, 34)
