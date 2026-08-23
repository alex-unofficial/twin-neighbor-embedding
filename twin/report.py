"""The dataset connectivity / weight / label report (spec §3.4).

    python -m twin.report                 # markdown to stdout
    python -m twin.report -o docs/DATASET_REPORT.md
    python -m twin.report tabula digits   # just these

One row per registered dataset. It exists because several decisions cannot be made without it:
§4.4's disconnected-graph policy needs to know which datasets are actually disconnected, §4.2's
lambda criterion is defined at *k* = mean degree, and whether a dataset's stored sides can be
trusted against each other is a measurement, not an assumption.

Nothing here branches on a dataset name; every column comes off the `Dataset` object. Adding a
dataset adds a row and nothing else.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np

from twin.datasets import DATASET_REGISTRY, WeightConsistencyReport, check_weight_consistency, load_dataset
from twin.layoutgraph import vertex_graph


@dataclass
class DatasetRow:
    """Everything the report knows about one dataset."""

    name: str
    n: int
    m: int
    density: float
    connected: bool
    n_components: int
    largest_component_fraction: float
    mean_degree: float

    affinity_source: str
    distance_source: str
    affinity_derivation: str
    distance_derivation: str

    weight_min: float
    weight_median: float
    weight_max: float
    weight_distinct: int

    zero_distance_pair_fraction: float
    labels: list[tuple[str, int, str, str, str]] = field(default_factory=list)
    composites: list[str] = field(default_factory=list)
    consistency: WeightConsistencyReport | None = None
    description: str = ""


def zero_distance_pair_fraction(distance: nx.Graph, n: int) -> float:
    """Fraction of **vertex pairs** whose ideal distance is exactly zero (spec §0.5).

    Not the fraction of edges at $w_{\\max}$: zero-weight edges compose, so whole groups can
    become mutually distance-zero. Since every derivation produces non-negative weights,
    $d(u, v) = 0$ exactly when $u$ and $v$ are joined by a path of zero-weight edges — so this
    is the connected components of the zero-weight subgraph, not an all-pairs shortest path.

    It matters because zero ideal distances damage different metrics differently: SNS tolerates
    them, silhouette is distorted, and shortest-path derivations collapse. A small fraction
    makes the question academic; a large one is evidence about the data.
    """
    weights = np.array([w for _, _, w in distance.edges(data="weight")], dtype=float)
    if (weights < 0).any():
        raise ValueError(
            "negative edge weights in the distance form; the zero-pair count assumes "
            "non-negativity, and shortest paths would not be well defined either"
        )
    if n < 2:
        return 0.0

    zero = nx.Graph()
    zero.add_nodes_from(range(n))
    zero.add_edges_from((u, v) for u, v, w in distance.edges(data="weight") if w == 0)

    coincident = sum(len(c) * (len(c) - 1) // 2 for c in nx.connected_components(zero))
    return coincident / (n * (n - 1) / 2)


def describe_dataset(name: str, **params) -> DatasetRow:
    """Build one row. Loads the dataset and its vertex graph; nothing is cached."""
    dataset = load_dataset(name, **params)
    vg = vertex_graph(dataset)

    weights = np.array([w for _, _, w in vg.distance.edges(data="weight")], dtype=float)
    sources = dataset.describe_sources()

    return DatasetRow(
        name=name,
        n=dataset.n,
        m=dataset.m,
        density=nx.density(dataset.graph),
        connected=dataset.is_connected(),
        n_components=dataset.n_components(),
        largest_component_fraction=dataset.largest_component_fraction(),
        mean_degree=dataset.mean_degree(),
        affinity_source=sources["affinity"],
        distance_source=sources["distance"],
        affinity_derivation=vg.derivation,
        distance_derivation=vg.distance_derivation,
        weight_min=float(weights.min()) if weights.size else float("nan"),
        weight_median=float(np.median(weights)) if weights.size else float("nan"),
        weight_max=float(weights.max()) if weights.size else float("nan"),
        weight_distinct=int(np.unique(weights).size),
        zero_distance_pair_fraction=zero_distance_pair_fraction(vg.distance, dataset.n),
        labels=[
            (
                label.name,
                label.n_classes,
                label.scoring_role,
                label.display_role,
                label.derivation,
            )
            for label in dataset.label_sets.values()
        ],
        composites=sorted(dataset.composite_of),
        consistency=check_weight_consistency(dataset),
        description=dataset.description,
    )


def build_report(names: list[str] | None = None) -> list[DatasetRow]:
    return [describe_dataset(name) for name in (names or list(DATASET_REGISTRY))]


# --------------------------------------------------------------------------- #
#                                  rendering                                   #
# --------------------------------------------------------------------------- #


def _fmt(x: float, places: int = 4) -> str:
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "—"
    return f"{x:.{places}g}"


def _derivation_note(row: DatasetRow) -> str:
    """Which side had to be derived, and by what rule. Empty when both are stored."""
    for side, text in (("affinity", row.affinity_derivation), ("distance", row.distance_derivation)):
        if text.startswith("derived"):
            return f"{side}: " + text.split(": ", 1)[1]
    return "both sides stored"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def to_markdown(rows: list[DatasetRow]) -> str:
    out = [
        "# Dataset report",
        "",
        "Generated by `python -m twin.report`. One row per entry in `DATASET_REGISTRY`; every",
        "column comes off the `Dataset` object, so adding a dataset adds a row and nothing else.",
        "",
        "## Shape and connectivity",
        "",
        _table(
            ["dataset", "n", "m", "density", "connected", "components", "largest cc", "mean degree"],
            [
                [
                    f"`{r.name}`",
                    f"{r.n}",
                    f"{r.m}",
                    _fmt(r.density, 3),
                    "yes" if r.connected else "**no**",
                    f"{r.n_components}",
                    _fmt(r.largest_component_fraction, 3),
                    _fmt(r.mean_degree, 4),
                ]
                for r in rows
            ],
        ),
        "",
        "`mean degree` is the *k* at which the λ selection criterion of spec §4.2 is defined.",
        "",
        "## Stored sides, and what is derived",
        "",
        _table(
            ["dataset", "affinity", "distance", "derivation"],
            [
                [
                    f"`{r.name}`",
                    r.affinity_source,
                    r.distance_source,
                    _derivation_note(r),
                ]
                for r in rows
            ],
        ),
        "",
        "`adaptive_kernel` has **no** derivation rule and raises: UMAP's per-point parameters",
        "are not stored, so no global transform of the affinity recovers a distance. Such a",
        "dataset must supply a stored distance side, and tabula fails loudly rather than falling",
        "back if `D` is ever dropped.",
        "",
        "## Ideal distances",
        "",
        "Distribution of the vertex graph's distance form over edges — the stored distance side",
        "where there is one, otherwise the single derivation from the stored affinity. This is",
        "the *ideal-distance source*: one target per dataset, shared by every representation.",
        "",
        _table(
            ["dataset", "min", "median", "max", "distinct", "pairs at d = 0"],
            [
                [
                    f"`{r.name}`",
                    _fmt(r.weight_min),
                    _fmt(r.weight_median),
                    _fmt(r.weight_max),
                    f"{r.weight_distinct}",
                    _fmt(100 * r.zero_distance_pair_fraction, 3) + " %",
                ]
                for r in rows
            ],
        ),
        "",
        "A zero ideal distance is not automatically a defect — under a probability or kernel",
        "model it is the model saying two vertices coincide — but it distorts silhouette and",
        "collapses shortest-path derivations, so the fraction of **vertex pairs** it affects is",
        "reported rather than the fraction of edges at the maximum weight.",
        "",
        "## Labels",
        "",
    ]

    label_rows = []
    for r in rows:
        if not r.labels:
            label_rows.append([f"`{r.name}`", "*none*", "", "", "", ""])
            continue
        for i, (name, n_classes, scoring, display, derivation) in enumerate(r.labels):
            label_rows.append(
                [
                    f"`{r.name}`" if i == 0 else "",
                    f"`{name}`",
                    f"{n_classes}",
                    scoring,
                    display,
                    derivation,
                ]
            )
    out += [
        _table(
            ["dataset", "label set", "classes", "scoring role", "display role", "derivation"],
            label_rows,
        ),
        "",
        "Class counts exclude the noise class. Scoring and display roles are independent:",
        "football scores the coarse label and takes hue from it, tabula inverts that. An",
        "`auxiliary` set is a technical covariate scored **deliberately** — a layout that",
        "separates cells by donor rather than by cell type is displaying an artifact, so a low",
        "score there is the good outcome.",
        "",
    ]

    composites = [(r.name, c) for r in rows for c in r.composites]
    if composites:
        out += [
            "Declared composites: "
            + ", ".join(f"`{name}` / `{c}`" for name, c in composites)
            + ".",
            "",
        ]

    both_stored = [r for r in rows if r.consistency is not None]
    checked = [r for r in both_stored if r.consistency.comparable]
    skipped = [r for r in both_stored if not r.consistency.comparable]

    out += ["## Weight-source consistency", ""]
    if not checked:
        out += ["No dataset stores two sides that carry rank information.", ""]
    else:
        out += [
            _table(
                [
                    "dataset",
                    "shared pairs",
                    "same edge set",
                    "global ρ",
                    "per-row ρ",
                    "unmixed per-row ρ",
                ],
                [
                    [
                        f"`{r.name}`",
                        f"{r.consistency.n_shared}",
                        "yes" if r.consistency.spans_same_edge_set else "**no**",
                        _fmt(r.consistency.global_spearman),
                        _fmt(r.consistency.median_row_spearman),
                        _fmt(r.consistency.median_row_spearman_unmixed),
                    ]
                    for r in checked
                ],
            ),
            "",
            "`unmixed per-row ρ` restricts to the pairs the raw stored matrix recorded in one",
            "direction only, where a symmetrisation cannot have mixed one row's parameters into",
            "another's. It is what separates *the two sources disagree* from *symmetrisation",
            "blurred an otherwise exact relationship*, and on tabula it settles the question:",
            "exactly −1 wherever it can be seen unmixed.",
            "",
        ]
    if skipped:
        out += [
            "Not comparable, because a side is an indicator or a single constant weight and so"
            " carries no ranks — nothing could disagree: "
            + ", ".join(f"`{r.name}`" for r in skipped)
            + ".",
            "",
        ]

    out += ["## Why each dataset is in the roster", ""]
    for r in rows:
        out += [f"**`{r.name}`** — {r.description}", ""]

    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m twin.report", description=__doc__)
    parser.add_argument("names", nargs="*", help="datasets to report on (default: all)")
    parser.add_argument("-o", "--out", type=Path, help="write markdown here instead of stdout")
    args = parser.parse_args(argv)

    unknown = [n for n in args.names if n not in DATASET_REGISTRY]
    if unknown:
        parser.error(f"unknown dataset(s) {unknown}; registered: {sorted(DATASET_REGISTRY)}")

    markdown = to_markdown(build_report(args.names or None))

    if args.out:
        args.out.write_text(markdown)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
