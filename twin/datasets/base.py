"""The `Dataset` layer — stored information only (spec §3.4).

A `Dataset` is a source of truth plus labels plus drawing instructions. It is **never laid
out**, and it holds **no derived information**: it declares which of the affinity and distance
sides are *stored* and what those numbers mean, and the `Dataset -> LayoutGraph` mapping in
:mod:`twin.graphmatrix` is the single place a missing side is derived.

The other half of the layering is that a `LayoutGraph` is never labelled. $A_E$'s nodes are
edges and $A_{tne}$'s are edges *and* vertices, so a length-$n$ label vector must never reach
them; `vertex_and_edge_points` returns points in the **original** indexing, so labels are
applied after layout. Hence:

    A metric or a figure is a function of the embedded points plus a `Dataset`.
"""

from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
import numpy as np
import scipy.sparse as sp

#: What a stored weight vector *means*. This names a model, not merely a direction: it is what
#: lets ``vertex_graph`` derive the missing side rather than guess it. ``count`` is an alias of
#: ``strength``; ``affinity`` was retired as a kind because it named the sign of a relationship
#: rather than a model (spec §0.5 decision 1).
Meaning = Literal["probability", "kernel", "adaptive_kernel", "strength", "count", "distance", "none"]

ScoringRole = Literal["primary", "sub", "auxiliary"]
DisplayRole = Literal["hue", "shade", "none"]

MEANINGS: tuple[str, ...] = (
    "probability",
    "kernel",
    "adaptive_kernel",
    "strength",
    "count",
    "distance",
    "none",
)


# --------------------------------------------------------------------------- #
#                          Which sides are stored                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Stored:
    """A side stored as the graph's own edge weights.

    Args:
        meaning: what the numbers are, which fixes the derivation of the *other* side.
        weight: the edge-attribute name.
    """

    meaning: Meaning
    weight: str = "weight"

    def __post_init__(self):
        _check_meaning(self.meaning)

    def matrix(self, dataset: "Dataset") -> sp.sparray:
        return nx.to_scipy_sparse_array(
            dataset.graph, nodelist=range(dataset.n), weight=self.weight, format="csr"
        )

    def describe(self) -> str:
        return f'graph edge attribute "{self.weight}" ({self.meaning})'


@dataclass(frozen=True)
class StoredMatrix:
    """A side stored as a separate matrix over the same edge set.

    Tabula is the case in hand: its affinity is UMAP's ``connectivities`` and its distance is
    the *measured* kNN distance matrix, from the same export. Two independently stored sides is
    the ideal case — nothing has to be derived — and it is also the case
    :func:`~twin.datasets.consistency.check_weight_consistency` exists to guard.

    Args:
        key: the key into :attr:`Dataset.matrices`.
        meaning: what the numbers are.
        symmetrise: how to fold a directed stored matrix onto the undirected edge set. ``"max"``
            is right for a kNN distance matrix, where a non-reciprocal neighbour appears in one
            direction only and the two directions agree whenever both are present.
    """

    key: str
    meaning: Meaning = "distance"
    symmetrise: Literal["max", "none"] = "max"

    def __post_init__(self):
        _check_meaning(self.meaning)

    def matrix(self, dataset: "Dataset") -> sp.sparray:
        try:
            M = dataset.matrices[self.key]
        except KeyError:
            raise KeyError(
                f"{dataset.name}: no stored matrix {self.key!r}; it declares "
                f"{sorted(dataset.matrices)}"
            ) from None
        M = sp.csr_array(M)
        if self.symmetrise == "max":
            M = M.maximum(M.T)
        return M

    def describe(self) -> str:
        return f'stored matrix "{self.key}" ({self.meaning})'


@dataclass(frozen=True)
class Unweighted:
    """A combinatorial side: the affinity is the 0/1 indicator, the distance is the hop count.

    Not a derivation — there is nothing to derive. The distance form must leave non-edges
    *absent* rather than materialising them at 1, so that shortest paths give hop counts.
    """

    meaning: Meaning = "none"

    def matrix(self, dataset: "Dataset") -> sp.sparray:
        return nx.to_scipy_sparse_array(
            dataset.graph, nodelist=range(dataset.n), weight=None, format="csr"
        )

    def describe(self) -> str:
        return "unweighted (combinatorial)"


StoredSource = Stored | StoredMatrix | Unweighted


def _check_meaning(meaning: str) -> None:
    if meaning not in MEANINGS:
        raise ValueError(f"unknown meaning {meaning!r}; expected one of {MEANINGS}")


# --------------------------------------------------------------------------- #
#                                   Labels                                     #
# --------------------------------------------------------------------------- #


@dataclass
class LabelSet:
    """A per-**original-vertex** label vector, plus how it was derived and what it is for.

    Scoring and display are separate roles on purpose. football scores `conference` and takes
    hue from it, with `division` an ordered sub-label supplying shade; tabula inverts that —
    the fine-grained `cell_ontology_class` is the scoring primary while the coarse `compartment`
    is its parent and supplies hue. One `role` field cannot express both, and collapsing them
    would force a dataset-name branch somewhere.

    Args:
        name: the key this set is registered under.
        values: integer-coded, one entry per vertex of the original graph.
        display_names: code -> human-readable name.
        derivation: ``"ground truth"`` | ``"generator"`` | ``"derived:<expr>"`` |
            ``"detected:<algo>"``.
        scoring_role: ``"auxiliary"`` marks a technical covariate (tabula's `donor`, `method`)
            which is scored *deliberately* — a layout separating cells by donor is displaying an
            artifact, so a low score is the good outcome there.
        display_role: which visual channel this set drives.
        parent: the label set this one subdivides.
        ordered: whether the codes have a meaningful order (West < none < East).
        noise_label: the code standing for "no class" (football's FBS independents), which the
            style renders in the noise colour rather than a hue.
    """

    name: str
    values: np.ndarray
    display_names: dict[int, str] = field(default_factory=dict)
    derivation: str = "ground truth"

    scoring_role: ScoringRole = "primary"
    display_role: DisplayRole = "hue"

    parent: str | None = None
    ordered: bool = False
    noise_label: int | None = None

    def __post_init__(self):
        self.values = np.asarray(self.values)
        if self.values.ndim != 1:
            raise ValueError(f"{self.name}: label values must be 1-D, got {self.values.shape}")

    @property
    def classes(self) -> np.ndarray:
        """Distinct codes, noise included."""
        return np.unique(self.values)

    @property
    def n_classes(self) -> int:
        """Distinct codes, **excluding** the noise class."""
        return int(sum(c != self.noise_label for c in self.classes))

    def class_sizes(self) -> dict[int, int]:
        codes, counts = np.unique(self.values, return_counts=True)
        return {int(c): int(n) for c, n in zip(codes, counts)}

    def display_name(self, code: int) -> str:
        return self.display_names.get(int(code), str(code))


# --------------------------------------------------------------------------- #
#                              Drawing instructions                            #
# --------------------------------------------------------------------------- #


@dataclass
class DatasetStyle:
    """How a dataset is drawn — declared here so no renderer has to know its name.

    At this step the style is *declarative*: it records the palette parameters that
    ``paper_figures`` currently hardcodes, so the plotting refactor (spec §3.6, step 10) has a
    single place to read them from. The palette itself is generated by :meth:`hue_table`, which
    reproduces today's two palettes exactly — the generalisation of the shade rule is a separate,
    later commit that declares it changes the figures (`DECISIONS.md` (26)).

    Args:
        palette: ``"husl"`` (seaborn, permuted) or ``"distinctipy"`` (`plots.custom_cmap`).
        palette_seed: seed for the permutation, and for `distinctipy`'s colour search.
        palette_swaps: index pairs exchanged after generation, for hand-tuned orderings.
        shade_factors: brightness multipliers for an ordered sub-label, in code order.
        noise_color: colour for a label set's ``noise_label``.
    """

    palette: Literal["husl", "distinctipy"] = "husl"
    palette_seed: int = 0
    palette_swaps: tuple[tuple[int, int], ...] = ()
    shade_factors: tuple[float, ...] = (0.7, 1.0, 1.3)
    noise_color: tuple[float, float, float] = (0.0, 0.0, 0.0)

    vertex_size: float = 15.0
    edge_alpha_intra: float = 0.6
    edge_alpha_inter: float = 0.1

    def hue_table(self, n_hues: int) -> np.ndarray:
        """``(n_hues, 3)`` base colours, before any shade adjustment."""
        if self.palette == "husl":
            import seaborn as sns

            base = sns.color_palette("husl", n_hues, desat=1)
            base = np.random.default_rng(seed=self.palette_seed).permutation(base)
        elif self.palette == "distinctipy":
            from twin.plots import custom_cmap

            # custom_cmap appends a trailing black row; keep it, it is load-bearing for the
            # tabula figure, where the inter-cluster edge label -1 indexes it by wrapping.
            base = custom_cmap(n_hues, seed=self.palette_seed)
        else:
            raise ValueError(f"unknown palette {self.palette!r}")

        base = np.asarray(base, dtype=float)
        for i, j in self.palette_swaps:
            base[[i, j]] = base[[j, i]]
        return base


# --------------------------------------------------------------------------- #
#                                  Dataset                                     #
# --------------------------------------------------------------------------- #


@dataclass
class Dataset:
    """Stored information about one graph. Owns identity. Never laid out.

    Args:
        name: the registry name.
        graph: the graph, with vertices ``0..n-1``.
        affinity: which affinity side is stored, or None if it must be derived.
        distance: which distance side is stored, or None if it must be derived.
        matrices: stored matrices referenced by :class:`StoredMatrix`.
        label_sets: per-**original-vertex** label vectors.
        composite_of: named composite labels, e.g. ``{"conference_division":
            ("conference", "division")}``. Declared rather than improvised, because the
            composite is a direct test of the claim that TNE reveals sub-conference structure
            (spec §3.4).
        reference_layouts: published 2-d embeddings shipped with the data (tabula's ``X_umap``),
            in ``(n, d)``.
        style: drawing instructions.
        params: the factory arguments, for the manifest.
        provenance: free-form facts for the manifest — where a distance was measured, in what
            space, with what k.
        description: why this dataset is in the roster. Goes in the generated dataset table.
    """

    name: str
    graph: nx.Graph

    affinity: StoredSource | None = None
    distance: StoredSource | None = None

    matrices: dict[str, sp.sparray] = field(default_factory=dict)
    label_sets: dict[str, LabelSet] = field(default_factory=dict)
    composite_of: dict[str, tuple[str, ...]] = field(default_factory=dict)
    reference_layouts: dict[str, np.ndarray] = field(default_factory=dict)

    style: DatasetStyle | None = None
    params: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    description: str = ""

    def __post_init__(self):
        if self.affinity is None and self.distance is None:
            raise ValueError(
                f"{self.name}: neither side is stored. A dataset must store at least one of "
                "affinity and distance — there is nothing to derive from otherwise "
                "(spec §3.4). Declare Unweighted() on both sides for a combinatorial graph."
            )
        if list(self.graph.nodes()) != list(range(self.graph.number_of_nodes())):
            raise ValueError(
                f"{self.name}: graph vertices must be 0..n-1; label vectors and every stored "
                "matrix are indexed by position."
            )
        for name, ls in self.label_sets.items():
            if ls.values.shape != (self.n,):
                raise ValueError(
                    f"{self.name}: label set {name!r} has {ls.values.shape[0]} entries for "
                    f"{self.n} vertices. Label sets are per original vertex."
                )
            if ls.parent is not None and ls.parent not in self.label_sets:
                raise ValueError(f"{self.name}: label set {name!r} names unknown parent {ls.parent!r}")
        for name, parts in self.composite_of.items():
            missing = [p for p in parts if p not in self.label_sets]
            if missing:
                raise ValueError(f"{self.name}: composite {name!r} names unknown label sets {missing}")

    # ------------------------------------------------------------------ shape

    @property
    def n(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def m(self) -> int:
        return self.graph.number_of_edges()

    def is_connected(self) -> bool:
        return self.n > 0 and nx.is_connected(self.graph)

    def n_components(self) -> int:
        return nx.number_connected_components(self.graph)

    def largest_component_fraction(self) -> float:
        if self.n == 0:
            return 0.0
        return max(len(c) for c in nx.connected_components(self.graph)) / self.n

    def mean_degree(self) -> float:
        return 2 * self.m / self.n if self.n else 0.0

    # `largest_component()` is deliberately absent. Subsetting a dataset means subsetting the
    # graph, every label vector and every stored matrix (tabula has two, ~50k edges each) while
    # keeping all of them index-consistent — and every registered dataset is connected, so none
    # of that code would run. Same call as `disconnected="components"` (spec §0.5 decision 3).
    # Add it when a disconnected dataset arrives, not in a hurry because SNS turned out to want
    # it. `is_connected()` stays; it is cheap and the report uses it.

    # ----------------------------------------------------------------- labels

    def labels(self, name: str) -> np.ndarray:
        """Integer-coded labels, per original vertex. Composites are built on demand."""
        if name in self.label_sets:
            return self.label_sets[name].values
        if name in self.composite_of:
            return self.composite_labels(name)
        raise KeyError(
            f"{self.name}: no label set {name!r}; it has {sorted(self.label_sets)} "
            f"and composites {sorted(self.composite_of)}"
        )

    def composite_labels(self, name: str) -> np.ndarray:
        """Codes for the tuple of label sets registered under ``name``, densely renumbered."""
        parts = self.composite_of[name]
        stacked = np.stack([self.label_sets[p].values for p in parts], axis=1)
        _, codes = np.unique(stacked, axis=0, return_inverse=True)
        return codes.astype(np.int64)

    def label_set(self, name: str) -> LabelSet:
        return self.label_sets[name]

    def label_sets_by_scoring_role(self, role: ScoringRole) -> list[LabelSet]:
        return [ls for ls in self.label_sets.values() if ls.scoring_role == role]

    def label_set_by_display_role(self, role: DisplayRole) -> LabelSet | None:
        for ls in self.label_sets.values():
            if ls.display_role == role:
                return ls
        return None

    # ------------------------------------------------------- stored sources

    def stored_matrix(self, side: Literal["affinity", "distance"]) -> sp.sparray | None:
        """The stored matrix for one side, in ``(n, n)`` and symmetric, or None if not stored."""
        source = getattr(self, side)
        return None if source is None else source.matrix(self)

    def describe_sources(self) -> dict[str, str]:
        return {
            side: (source.describe() if source is not None else "derived")
            for side, source in (("affinity", self.affinity), ("distance", self.distance))
        }
