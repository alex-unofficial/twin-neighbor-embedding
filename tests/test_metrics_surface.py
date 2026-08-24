"""What ``from twin.metrics import *`` is allowed to mean.

``twin/pipelines.py`` imports the metric layer with a star import and ``twin/metrics.py`` has no
``__all__``, so today ``*`` means *every public module-level name*. When the module becomes a
package (spec §2) it will mean *whatever ``__init__.py`` binds* — a strictly different rule.
A name dropped in that move does not fail here in `pytest`; it fails as a `NameError` partway
through an overnight batch. Hence this file.

The pin is deliberately one-sided. `REQUIRED_EXPORTS` must be reachable; the incidental
re-exports (`np`, `nx`, ...) that the current no-``__all__`` module leaks are recorded as
*permitted*, not required, so the package move may drop them without editing this file.
"""

import importlib

import pytest

#: Every metric-layer name any caller in the tree resolves through the star import, plus the
#: helpers the tests import by name. Dropping one of these breaks a caller.
REQUIRED_EXPORTS = {
    "require_n_by_d",
    "KnnAccuracyReport",
    "majority_class_rate",
    "knn_accuracy_report",
    "knn_accuracy",
    "knn_neighbor_preservation_accuracy",
    # spec §2.0, added at step 3 commit 4.
    "ideal_distances",
    "finite_pair_mask",
    "condensed",
    "pairwise_distances",
    "mean_pairwise_distance",
    "clear_pairwise_cache",
}

#: Names the current ``import *`` leaks because the module has no ``__all__``. Permitted, never
#: required: nothing may start depending on `twin.metrics.np`.
PERMITTED_INCIDENTAL = {"np", "nx", "dataclass", "NearestNeighbors"}


def star_import_names() -> set[str]:
    namespace: dict = {}
    exec("from twin.metrics import *", namespace)  # noqa: S102 - that is the thing under test
    return {name for name in namespace if not name.startswith("__")}


def test_star_import_exports_every_required_name():
    missing = REQUIRED_EXPORTS - star_import_names()
    assert not missing, f"star import no longer reaches {sorted(missing)}"


def test_star_import_leaks_nothing_new():
    """New public names must be intentional, i.e. in ``REQUIRED_EXPORTS`` or a known leak."""
    unexpected = star_import_names() - REQUIRED_EXPORTS - PERMITTED_INCIDENTAL
    assert not unexpected, f"unexpected public names in the star surface: {sorted(unexpected)}"


@pytest.mark.parametrize("name", sorted(REQUIRED_EXPORTS))
def test_required_names_are_importable_by_name(name):
    assert hasattr(importlib.import_module("twin.metrics"), name)


def test_pipelines_imports():
    """The check that would actually have caught a dropped export.

    ``twin.pipelines`` resolves the metric names at call time, not import time, so this is
    necessary rather than sufficient — but it is free, and it fails immediately if the module
    itself stops importing.
    """
    assert importlib.import_module("twin.pipelines") is not None
