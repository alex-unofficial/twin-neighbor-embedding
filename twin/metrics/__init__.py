"""Layout quality metrics (spec §2).

The flat ``twin/metrics.py`` became this package at spec §7 step 3. The split follows §2's
layout: :mod:`~twin.metrics.common` for shared machinery, :mod:`~twin.metrics.structural` for
metrics of points against the graph, :mod:`~twin.metrics.label` for metrics of points against a
label vector. ``visual``, ``crossings`` and the registry arrive at steps 4-7.

Every layout array here is ``(n_points, d)``. Conversion from the ``(d, n)`` orientation the
embedding layer returns happens exactly once, in ``twin.results.LayoutResult.from_raw``; see
CLAUDE.md. Orientation is asserted at entry, never inferred from ``shape`` — inference silently
picks the wrong axis whenever ``n == d`` or the input is square (spec §1.3).

``__all__`` is explicit because ``twin/pipelines.py`` imports this package with ``*``. Without
it, ``*`` would re-export every name any submodule happens to bind, and the surface would drift
with the imports rather than with the design (`tests/test_metrics_surface.py`).
"""

from twin.metrics.common import (
    condensed,
    finite_pair_mask,
    ideal_distances,
    require_n_by_d,
)
from twin.metrics.label import (
    KnnAccuracyReport,
    knn_accuracy,
    knn_accuracy_report,
    majority_class_rate,
)
from twin.metrics.structural import knn_neighbor_preservation_accuracy

__all__ = [
    "KnnAccuracyReport",
    "condensed",
    "finite_pair_mask",
    "ideal_distances",
    "knn_accuracy",
    "knn_accuracy_report",
    "knn_neighbor_preservation_accuracy",
    "majority_class_rate",
    "require_n_by_d",
]
