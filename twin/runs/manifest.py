r"""The run manifest (spec §4.6) — what produced a run directory, in one JSON file.

Every field §4.6 lists, plus the ones the step-7 plan adds, each because the rows alone cannot
say it:

* ``disconnected`` — always ``"lcc"`` (decision 3); the runner refuses anything else.
* ``distribution_policy`` — ``off`` / ``histogram(n)`` / ``full``. Off by default:
  `tabula`'s pairwise distribution is 100 MB per case before compression.
* ``mem_measurement`` — ``"batch"``: memory was sampled inside a single-process batch, so
  ``peak_mem_mb`` is a lower bound on a baseline that drifts (decision 4). ``"isolated"`` is
  reserved for a later one-case-per-process pass that overwrites the columns that matter.
* ``k_per_dataset`` **and** ``k_source`` — ``default_k`` or ``explicit`` (decision 5). Both,
  because ``default_k`` is a function of the graph and a reader of the rows cannot tell which
  was used.
* ``metric_set_digest`` and the per-metric ``version`` map it was computed from.
* ``dropped_composites`` — §5.3 guard 3, no silent caps.
* ``contributions`` — a **list** of ``(sha, timestamp, …)``, one per invocation, rather than one
  SHA: a resumed run, and a run whose layouts came from the content-addressed store, were
  produced by more than one commit, and a single SHA would be a false record of all but one.
  Each case also records the SHA that produced its layout.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from twin.metrics.registry import CROSSING_GAMMA, METRIC_REGISTRY
from twin.runs.keys import KEY_VERSION, _encode

__all__ = ["MANIFEST_VERSION", "dataset_entry", "read_manifest", "write_manifest"]

MANIFEST_VERSION = 1


def _jsonable(value: Any) -> Any:
    """Dataclasses, NumPy scalars and arrays to plain JSON; anything else to ``repr``."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def dataset_entry(dataset) -> dict:
    """§4.6's per-dataset fields: stored sources, derivation rule (with ``d_scale``), weight
    consistency, and the declared sparsity."""
    from twin.datasets import check_weight_consistency
    from twin.layoutgraph import vertex_graph

    vg = vertex_graph(dataset)
    return {
        "name": dataset.name,
        "params": _encode(dict(dataset.params)),
        "n": dataset.n,
        "m": dataset.m,
        "stored_sources": dataset.describe_sources(),
        # The derived side's rule, including d_scale where exp(-d/d_scale) is the rule.
        "affinity_derivation": vg.derivation,
        "distance_derivation": vg.distance_derivation,
        "weight_consistency": _jsonable(check_weight_consistency(dataset)),
        "sparsity": dataset.sparsity.describe() if dataset.sparsity else None,
        "connected": dataset.is_connected(),
    }


def static_fields() -> dict:
    """What the metric code fixes rather than the run chooses: §4.6's γ, include_adjacent and
    VNR normalisation, and the version of every metric."""
    vnr = METRIC_REGISTRY["node_resolution_vertexwise"]
    return {
        "manifest_version": MANIFEST_VERSION,
        "key_version": KEY_VERSION,
        "gamma": CROSSING_GAMMA,
        "include_adjacent": True,
        "vnr_normalisation": {
            "default": vnr.variants[None]["normalize"],
            "variants": {str(v): p["normalize"] for v, p in vnr.variants.items() if v},
        },
        "metric_versions": {name: spec.version for name, spec in METRIC_REGISTRY.items()},
    }


def write_manifest(path: str | os.PathLike, manifest: dict) -> Path:
    """Atomically, as every other artifact here: a sibling temp file, then ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def read_manifest(path: str | os.PathLike) -> dict:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(
            f"{path}: manifest version {manifest.get('manifest_version')!r}, "
            f"expected {MANIFEST_VERSION}"
        )
    return manifest
