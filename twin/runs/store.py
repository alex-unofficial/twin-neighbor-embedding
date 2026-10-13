r"""A run directory and the shared layout store — where every artifact of a run lives.

::

    runs/
        layouts/<key_id>.npz                  # shared by every run: content-addressed (B1)
        <run_id>/
            manifest.json
            metrics.parquet                   # the concatenated tables, rewritten per invocation
            diagnostics.parquet
            graph_side.parquet
            shards/<key_id>.metrics.parquet   # one per case: (22)'s metric-stage marker
            shards/<key_id>.diagnostics.parquet
            graph_side/<dataset digest>.parquet
            distributions/<key_id>.npz        # only under distribution_policy != "off"
            log.txt

⚠️ **Departs from §4.6's tree** in one place, deliberately: layouts are not under the run
directory (see :mod:`twin.runs.layouts`). Everything else is §4.6's, plus the per-case shards
(22) specifies and the per-dataset graph-side shards that §5.3 guard 2 makes necessary.

**Two stages, two markers** ((22)): a case skips its embed if its layout exists, and skips its
metrics if its metric shard's digest is current. The diagnostics shard is written *before* the
metric shard, so a metric shard on disk implies its diagnostics are too.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from twin.config import PROJ_ROOT
from twin.runs.keys import read_shard_metadata, write_shard
from twin.runs.layouts import layout_path
from twin.runs.manifest import read_manifest
from twin.runs.provenance import git_state, now

__all__ = ["DEFAULT_ROOT", "DistributionPolicy", "RunStore"]

DEFAULT_ROOT = PROJ_ROOT / "runs"

_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclasses.dataclass(frozen=True)
class DistributionPolicy:
    """``off`` / ``histogram(bins)`` / ``full`` — whether a distribution's raw array is kept.

    The summaries (``min``/``median``/``max``, ``mean``) reach the table under every policy;
    this decides only what is written beside them. ``off`` is the default because `tabula`'s
    pairwise distribution is 12.5 M float64 — 100 MB per case before compression — for a
    diagnostic §0.5 declines to score.
    """

    name: str
    bins: int | None = None

    @classmethod
    def parse(cls, text: str) -> "DistributionPolicy":
        if text in ("off", "full"):
            return cls(text)
        match = re.fullmatch(r"histogram(?:\((\d+)\))?", text)
        if match:
            bins = int(match.group(1) or 100)
            if bins < 1:
                raise ValueError("histogram needs at least one bin")
            return cls(f"histogram({bins})", bins)
        raise ValueError(f"distribution_policy must be off, full or histogram(n), got {text!r}")

    @property
    def keeps_arrays(self) -> bool:
        return self.name != "off"


def _atomic_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


class RunStore:
    """The paths and the reads and writes of one run, plus the shared layout store beside it.

    Open with :meth:`create` for a new run or :meth:`resume` for an existing one. Both record
    the invocation as a *contribution* — SHA, dirty flag, timestamp — because a resumed run, and
    one whose layouts came from the shared store, was produced by more than one commit.
    """

    def __init__(self, root: Path, run_id: str, *, previous: dict | None, git: dict):
        self.root = Path(root)
        self.run_id = run_id
        self.dir = self.root / run_id
        self.previous = previous
        self.git = git
        self.started = now()

    # ------------------------------------------------------------- opening

    @staticmethod
    def _git(allow_dirty: bool) -> dict:
        state = git_state()
        if state["dirty"] and not allow_dirty:
            raise RuntimeError(
                "the working tree has uncommitted changes to tracked files, so the run's git "
                "SHA would be a false record of the code that produced it (CLAUDE.md: start a "
                "results run only from a clean tree). Commit or stash first, or pass "
                "allow_dirty=True for a run whose numbers will not be reported."
            )
        return state

    @classmethod
    def create(cls, root=None, run_id: str | None = None, *, allow_dirty: bool = False):
        root = Path(DEFAULT_ROOT if root is None else root)
        git = cls._git(allow_dirty)
        if run_id is None:
            stamp = now().replace(":", "").replace("-", "").replace("+0000", "Z")
            short = (git["sha"] or "nogit")[:8]
            run_id = f"{stamp}-{short}" + ("-dirty" if git["dirty"] else "")
        if not _RUN_ID.match(run_id):
            raise ValueError(f"run_id may use only [A-Za-z0-9._-], got {run_id!r}")
        if (root / run_id / "manifest.json").exists():
            raise FileExistsError(
                f"run {run_id!r} already exists under {root}; pass resume={run_id!r} to "
                "continue it"
            )
        return cls(root, run_id, previous=None, git=git)

    @classmethod
    def resume(cls, run_id: str, root=None, *, allow_dirty: bool = False):
        root = Path(DEFAULT_ROOT if root is None else root)
        path = root / run_id / "manifest.json"
        if not path.exists():
            # A run killed before its first manifest still has shards worth keeping.
            if not (root / run_id).is_dir():
                raise FileNotFoundError(f"no run {run_id!r} under {root}")
            previous = None
        else:
            previous = read_manifest(path)
        return cls(root, run_id, previous=previous, git=cls._git(allow_dirty))

    # ---------------------------------------------------------------- paths

    def layout(self, key_id: str) -> Path:
        return layout_path(self.root, key_id)

    def shard(self, key_id: str, table: str) -> Path:
        return self.dir / "shards" / f"{key_id}.{table}.parquet"

    def graph_side(self, dataset_digest: str) -> Path:
        return self.dir / "graph_side" / f"{dataset_digest}.parquet"

    def distributions(self, key_id: str) -> Path:
        return self.dir / "distributions" / f"{key_id}.npz"

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    @property
    def log_path(self) -> Path:
        return self.dir / "log.txt"

    # ---------------------------------------------------------------- shards

    @staticmethod
    def read_rows(path: Path) -> list[dict]:
        return pq.read_table(path).to_pylist()

    def shard_digests(self, key_id: str) -> dict[str, str] | None:
        """Per-metric digests of a case's metric shard, or None when there is no usable shard."""
        meta = read_shard_metadata(self.shard(key_id, "metrics"))
        if meta is None or "metric_digests" not in meta:
            return None
        if not self.shard(key_id, "diagnostics").is_file():
            return None
        return json.loads(meta["metric_digests"])

    def write_case(self, key_id, rows, diagnostics, *, schemas, set_digest, digests) -> None:
        """Diagnostics first, so that a metric shard on disk implies its diagnostics are too."""
        metric_schema, diagnostic_schema = schemas
        extra = {"metric_digests": json.dumps(digests, sort_keys=True)}
        write_shard(self.shard(key_id, "diagnostics"), diagnostics, schema=diagnostic_schema,
                    metric_set_digest=set_digest, metadata=extra)
        write_shard(self.shard(key_id, "metrics"), rows, schema=metric_schema,
                    metric_set_digest=set_digest, metadata=extra)

    # --------------------------------------------------------- distributions

    def write_distributions(self, key_id: str, arrays: dict, policy: DistributionPolicy) -> None:
        """Merge ``arrays`` into the case's file — a partial recompute replaces only its own."""
        path = self.distributions(key_id)
        existing = {}
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                existing = {name: data[name] for name in data.files}
        for (point_set, label_set, metric, variant), array in arrays.items():
            name = f"{metric}|{variant or ''}|{label_set or ''}|{point_set or ''}"
            for stale in [n for n in existing if n.startswith(name)]:
                del existing[stale]
            if policy.bins is None:
                existing[name] = array
            else:
                finite = array[np.isfinite(array)]
                try:
                    counts, edges = np.histogram(finite, bins=policy.bins)
                except ValueError:
                    # A range too narrow to split into `bins` finite bins — the straight
                    # variant's detour ratios span 1.0 to 1.0000000000000004. Pad it rather
                    # than place a width threshold: numpy is the judge of "too narrow".
                    centre = float(finite.mean())
                    counts, edges = np.histogram(
                        finite, bins=policy.bins, range=(centre - 0.5, centre + 0.5)
                    )
                existing[f"{name}|counts"] = counts
                existing[f"{name}|edges"] = edges
        _atomic_npz(path, existing)

    # ---------------------------------------------------------------- tables

    def write_tables(self, tables: dict[str, pd.DataFrame]) -> None:
        for name, frame in tables.items():
            path = self.dir / f"{name}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
            os.close(fd)
            try:
                frame.to_parquet(tmp, index=False)
                os.replace(tmp, path)
            except BaseException:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise

    def contribution(self) -> dict:
        from twin.runs.provenance import host, package_versions

        return {
            "sha": self.git["sha"],
            "dirty": self.git["dirty"],
            "started": self.started,
            "finished": now(),
            "host": host(),
            "packages": package_versions(),
        }
