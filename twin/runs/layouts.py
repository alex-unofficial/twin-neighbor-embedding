r"""The layout store: one ``.npz`` per case id, shared by every run (spec §4.6, plan decision 1).

⚠️ **A departure from §4.6's tree**, which puts layouts at ``runs/<run_id>/layouts/``. With
``run_id`` containing the git SHA, adding a metric means a commit, a new ``run_id``, an empty
``layouts/`` — and every embed re-runs, which is the exact thing (22) says layouts are persisted
to prevent. So layouts are **content-addressed** at ``runs/layouts/<key_id>.npz``: the id is a
function of the case alone, so any run of that case finds it. Each file records the SHA that
produced it, and a run's manifest lists every SHA that contributed.

**What a file holds** — and what it deliberately does not:

* ``V``, ``E``, ``slopes`` — the arrays, ``(n, d)`` and ``(m, d)`` as in
  :class:`~twin.results.LayoutResult`;
* ``meta`` — a 0-d string array of JSON: the dataset's registry **name and factory params**,
  the case's names and params, the derivation chain, the point provenance, the seed, the scalar
  instrumentation, the producing SHA and a timestamp.

⚠️ **Never the** ``Dataset`` **object, and never** ``allow_pickle``. A `Dataset` holds the graph,
every label vector, the style and — for `tabula` — two ~50k-edge matrices; written into every
layout file it would be duplicated once per representation, seed and λ. It is rehydrated through
``DATASET_REGISTRY`` by name and params on load.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from twin.datasets import load_dataset
from twin.results import LayoutResult
from twin.runs.keys import _decode, _encode

__all__ = ["LAYOUT_FORMAT", "layout_path", "load_layout", "read_layout_meta", "save_layout"]

#: Bumped if the file's layout changes; a reader refuses a format it does not know.
LAYOUT_FORMAT = 1

#: The scalar instrumentation fields carried over from the `LayoutResult`.
INSTRUMENTATION = (
    "runtime_s", "representation_s", "n_points", "peak_mem_mb", "rss_baseline_mb", "rss_peak_mb",
)


def layout_path(root: str | os.PathLike, key_id: str) -> Path:
    """``<root>/layouts/<key_id>.npz`` — the store is beside the run directories, not in one."""
    return Path(root) / "layouts" / f"{key_id}.npz"


def save_layout(
    path: str | os.PathLike,
    result: LayoutResult,
    *,
    key_id: str,
    key_json: str,
    git_sha: str | None,
    created: str,
) -> Path:
    """Write ``result`` to ``path`` atomically: a sibling temp file, then :func:`os.replace`.

    ``os.replace`` is atomic only within one filesystem, which is why the temporary file is in
    the target's own directory; a kill mid-write leaves a stray temp file and no layout, so the
    case is simply embedded again.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "format": LAYOUT_FORMAT,
        "key_id": key_id,
        "key_json": key_json,
        "dataset": {"name": result.dataset.name, "params": _encode(dict(result.dataset.params))},
        "representation": result.representation,
        "embedding": result.embedding,
        "params": _encode(dict(result.params)),
        "derivation_chain": list(result.derivation_chain),
        "seed": result.seed,
        "point_provenance": dict(result.point_provenance),
        "instrumentation": {name: getattr(result, name) for name in INSTRUMENTATION},
        "git_sha": git_sha,
        "created": created,
    }
    arrays: dict[str, Any] = {"V": result.V, "meta": np.array(json.dumps(meta, sort_keys=True))}
    if result.E is not None:
        arrays["E"] = result.E
    if result.slopes is not None:
        arrays["slopes"] = result.slopes

    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def read_layout_meta(path: str | os.PathLike) -> dict:
    """The JSON metadata alone, without loading the arrays or the dataset."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"][()]))
    if meta.get("format") != LAYOUT_FORMAT:
        raise ValueError(f"{path}: layout format {meta.get('format')!r}, expected {LAYOUT_FORMAT}")
    return meta


def load_layout(path: str | os.PathLike, *, dataset=None) -> LayoutResult:
    """Rehydrate a :class:`~twin.results.LayoutResult` from ``path``.

    Args:
        dataset: the already-built `Dataset`, to avoid rebuilding it per case — the runner
            memoises one per dataset. It must match the file's name and params; otherwise the
            dataset is rebuilt from the registry.

    Raises:
        ValueError: when ``dataset`` is given and is not the one the layout is of.
    """
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"][()]))
        V = data["V"]
        E = data["E"] if "E" in data.files else None
        slopes = data["slopes"] if "slopes" in data.files else None
    if meta.get("format") != LAYOUT_FORMAT:
        raise ValueError(f"{path}: layout format {meta.get('format')!r}, expected {LAYOUT_FORMAT}")

    name = meta["dataset"]["name"]
    params = _decode(meta["dataset"]["params"])
    if dataset is None:
        dataset = load_dataset(name, **params)
    elif (dataset.name, dataset.params) != (name, params):
        raise ValueError(
            f"{path} is a layout of {name}{params}, not of {dataset.name}{dataset.params}"
        )

    return LayoutResult(
        V=V,
        E=E,
        slopes=slopes,
        dataset=dataset,
        representation=meta["representation"],
        embedding=meta["embedding"],
        params=_decode(meta["params"]),
        derivation_chain=list(meta["derivation_chain"]),
        seed=meta["seed"],
        point_provenance=dict(meta["point_provenance"]),
        **meta["instrumentation"],
    )
