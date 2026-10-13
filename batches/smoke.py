"""The smoke batch: the whole metric registry over two small datasets, into one directory.

    python batches/smoke.py                 # writes runs/smoke/
    python batches/smoke.py --out /tmp/x    # anywhere else

`karate_weighted` and `football_2023`, all three representations, two cheap layout methods —
12 cases, about a minute. It is the project's first real metric table, and what makes Half A of
step 7 reviewable: every metric, every variant, the graph-side table and its reference levels,
through the same runner the overnight batches will use.

**Not a results run.** No shards, no persisted layouts, no manifest and no resume — those are
Half B — so nothing written here is a versioned contract and every file in the output directory
is regenerable by running this again. `runs/` is gitignored.

Three parquet files, one per table, because they are three different shapes: ``metrics`` (one
row per case × metric × variant × k), ``diagnostics`` ((18)'s counts and costs) and
``graph_side`` (§5.3 guard 2 — once per dataset, never per case).
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

from twin.config import PROJ_ROOT
from twin.runs.runner import batch_layout_metrics

BATCH = [
    {
        "graph": ["karate_weighted", "football_2023"],
        "representation": ["vertex", "edge", "twin"],
        "embedding": ["sgtsne", "spectral"],
    }
]


def run(out: Path, *, batch=BATCH, progress_fn=logger.info, **kwargs):
    """Run ``batch`` and write its three tables under ``out``. Returns the tables."""
    tables = batch_layout_metrics(batch, progress=True, progress_fn=progress_fn, **kwargs)
    out.mkdir(parents=True, exist_ok=True)
    tables.metrics.to_parquet(out / "metrics.parquet", index=False)
    tables.diagnostics.to_parquet(out / "diagnostics.parquet", index=False)
    tables.graph_side.to_parquet(out / "graph_side.parquet", index=False)
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=PROJ_ROOT / "runs" / "smoke")
    args = parser.parse_args(argv)

    tables = run(args.out)
    status = tables.metrics["status"].value_counts().to_dict()
    logger.info(
        f"wrote {args.out}: {len(tables.metrics)} metric rows {status}, "
        f"{len(tables.diagnostics)} diagnostics, {len(tables.graph_side)} graph-side rows"
    )
    if tables.dropped_composites:
        logger.info(f"dropped degenerate composites: {tables.dropped_composites}")
    return 0 if status.get("error", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
