# Step 6 — `label.py`: silhouette and the graph-side checks

> **Executed and closed in session 6. This file is history, not instructions.** The five
> departures are `DECISIONS.md` (39) and the measurements (40); `STATUS.md` has the summary.
> Read below for what was planned and, by difference, what the implementation had to change.

The executable sequence for spec §7 step 6. Read `CLAUDE.md`, then `docs/STATUS.md`, then spec
§2.3 and §4.4 before starting. This file is what to do, in what order, and how to know it worked;
the design record is `DECISIONS.md`.

---

## Context

`twin/metrics/label.py` exists and is half-built. The $k$-NN accuracy family landed early, at step
3, because §2's layout puts it in this module and moving it once inside a mechanical commit beat
moving it at step 6 alongside new functionality. What is missing is everything in spec §2.3 that
is not $k$-NN accuracy:

- `silhouette` — eqs. (49)–(52), the layout-side cluster-separation score;
- `silhouette_graph` — eqs. (53)–(55), the same thing over graph geodesics;
- `graph_neighborhood_accuracy` — eqs. (61)–(64), the 1-hop graph-side counterpart to
  `knn_accuracy`.

The two graph-side metrics are **label-validity checks, not baselines** (§2.3, §5.3): they answer
"are these labels reflected in the topology at all?", on a scale incomparable with the layout
score. That framing is the reason they exist, and it has to survive into the docstrings and,
later, the output schema.

After this step the metric suite of §2 is complete. Everything remaining is machinery: step 7's
registry and runner, step 8's sweep, steps 9–12.

**This is a small step.** The spec explicitly permits `sklearn` for the hard part, the module and
its test file already exist, and there are no new numerical subtleties of the kind step 5 was
full of. The measurements are the substantial part.

---

## What was settled before planning

### Two spec claims, checked against the data

**⚠️ §0.5 decision 9's premise does not hold.** It asks for a composite-label guard because
"14 conferences × 3 divisions over 132 teams produces many tiny classes, where silhouette is 0 by
convention". Measured on the actual roster:

| dataset | primary | composite |
|---|---|---|
| `football_2023` | **11** classes (not 14), min 4, median 13 | **15** classes, min **4**, median 7 |
| `tabula` | 13 classes, min 34, median 146 | **13** classes — *identical to the primary* |

There are no singletons and no class-size collapse. A minimum-size filter at 3 would remove
nothing; at 5 it would remove one football class. **And tabula's composite is redundant**:
`compartment` is a function of `cell_ontology_class`, so `compartment_cell_type` has exactly the
same 13 classes and scoring it produces a duplicate column. Both facts go in `DECISIONS.md`.

**`sklearn` implements Rousseeuw's singleton convention.** Verified directly:
`silhouette_samples` returns exactly `0.0` for a sample in a size-1 cluster, which is what §2.3
and §3.1.2 require. No wrapper needed for it.

### Decisions taken at review, so they are not re-litigated

1. **`sklearn`-backed, with a hand-rolled oracle in the test module.** §2.3 permits it, and unlike
   `knn_accuracy` — hand-rolled at step 3 to control the tie rule — silhouette has no knob worth
   fighting for. Per the session-5 rule, the oracle must not share code with what it checks, so
   the test module gets its own ~10-line $a(i)$/$b(i)$ implementation.
2. **The composite guard is a diagnostic, not a filter.** Report class count and min/median class
   size beside the score, as §0.5 asks; do not build the min-size filtering, which would be inert
   on every dataset in the roster. This follows §0.5 decision 3's precedent — untested code for a
   case that does not occur is worse than an honest refusal.
3. **All three metrics get a `*_report` sibling**, matching `knn_accuracy_report`,
   `*_report` in `structural.py` and `node_resolution_vertexwise_report`. §2.3's bare `-> float`
   is the scalar accessor, which the pattern preserves.
4. **Scope excludes `MetricSpec` / `METRIC_REGISTRY` / `evaluate_layout`, and excludes the
   "which label sets get scored" enumeration.** Steps 3, 4 and 5 all held that boundary. The
   enumeration is already one line for a caller — `dataset.label_sets_by_scoring_role(...)` and
   `dataset.composite_of` exist — so there is nothing to build, only to call, at step 7.

---

## Commit sequence

### 1. `metrics: silhouette — the layout side (spec §2.3, eqs. 49-52)`

```python
def silhouette(X, labels) -> float
def silhouette_report(X, labels) -> SilhouetteReport
```

`SilhouetteReport` carries the score plus what §0.5 decision 9 asks for: `n_classes`,
`min_class_size`, `median_class_size`, `n_singletons`. Those are the columns that would make a
future class-size collapse visible; the filtering is deliberately absent (decision 2 above).

- Backed by `sklearn.metrics.silhouette_samples`, which already gives singletons 0.
- `require_n_by_d(X, labels.size, "X")` at entry, like every other metric here.
- **A single-class label vector returns `nan`, not an exception** — `sklearn` raises, but a
  label vector with one class is a property of the *data*, not a caller error, and the package's
  convention is `nan` for that (`edge_length_cv` on an edgeless graph) and a raise for caller
  error (`require_n_by_d`). `n_classes=1` in the report makes it visible.
- Tests: the hand-rolled oracle on random and structured layouts; the singleton case constructed
  (spec §6 — `karate_weighted`'s only label set has 2 classes and no singletons, so the golden
  dataset cannot exercise it); the `karate_coincident_layout` fixture, whose behaviour is
  recorded rather than special-cased (§0.5 decision 1 predicts distortion, and no roster dataset
  has coincident vertices); bounds in $[-1, 1]$; invariance under translation/rotation/scaling.

### 2. `metrics: silhouette_graph — a label-validity check, not a baseline (eqs. 53-55)`

```python
def silhouette_graph(G=None, labels=..., *, ideal=None, **ideal_distance_kw) -> float
def silhouette_graph_report(...) -> SilhouetteReport
```

**Follow `structural._stress_pairs`' contract exactly** (`twin/metrics/structural.py:280`), which
is the established pattern for a metric that needs the ideal distances:

- `ideal` may be an `EvaluationContext`, and on the results path it should be — the context
  computes the closure once per dataset, where recomputing it per label set is 200 MB and
  seconds at tabula's size;
- supplying both `ideal` and `**ideal_distance_kw` is an error, because the kwargs configure a
  computation that has already happened;
- otherwise fall back to `ideal_distances(G, **kw)`.

This one wants the **square** `(n, n)` form for `sklearn`'s `metric="precomputed"`, i.e.
`ctx.ideal` rather than `ctx.ideal_condensed`. `twin/metrics/context.py:137` already anticipates
this in a comment.

Two things §4.4 requires, both load-bearing:

- ⚠️ **Refuse `disconnected="mask"`** with an explanatory error. Under `mask`, a vertex whose
  nearest other cluster lies in another component gets $b(i)=\infty$ and $s(i)\to1$, so a
  disconnected layout scores *better*. "Acceptable for SNS, dangerous for Sil\*."
- Under the `lcc` default, `ideal_distances` returns the full `(n, n)` with `inf` outside the
  component — it does not subset. So subset to the finite vertices before calling `sklearn`, and
  report `n_excluded`.

The docstring must state that this is a **label-validity check on an incomparable scale**, and
that §5.3 will keep it out of any table adjacent to the layout score. Column adjacency is an
invitation to difference two incomparable quantities.

Tests: agreement with the layout-side function when the "layout" is a metric MDS of the ideal
distances is *not* a useful check — instead, a constructed graph whose labels match its topology
scores high and a shuffled label vector scores near 0; the `mask` refusal; a disconnected
fixture exercising `n_excluded` (no roster dataset is disconnected, so construct one).

### 3. `metrics: graph_neighborhood_accuracy — the 1-hop check (eqs. 61-64)`

```python
def graph_neighborhood_accuracy(G, labels, *, weighting="uniform", tie_rule="fractional") -> float
def graph_neighborhood_accuracy_report(...) -> GraphAccuracyReport
```

The graph-side twin of `knn_accuracy`: the 1-hop neighbourhood of variable size $\deg(i)$ rather
than a fixed $k$, with edge-weight voting for weighted graphs (eq. 62). Same `weighting` and
`tie_rule` vocabulary, same fractional-tie semantics, and the same `majority_class_rate` chance
floor.

- **Isolated vertices are excluded and the count returned** (§2.3). `tests/conftest.py` already
  has `isolated_vertex_graph` for exactly this.
- ⚠️ **Take the valuation from the graph the caller passes, and document that the caller should
  pass `ctx.affinity`** — never `dataset.graph`. `digits`' stored `weight` is a *distance*, and
  `tabula`'s graph is unstamped, so a metric reading `dataset.graph.weight` would vote with an
  inverted sign on one dataset and an unchecked quantity on another (CLAUDE.md, STATUS traps).
  Consider a `twin.forms` stamp guard refusing a `"distance"`-stamped graph, as ADR and
  `weighted_recall` do — that is the existing backstop for this exact mistake.
- Reuse `knn_accuracy_report`'s vote-mass and tie machinery rather than restating it; the
  difference is the neighbourhood, not the voting.

### 4. `tests: golden keys for the label family (additive)`

Regenerate `tests/fixtures/metric_golden.json` via `make_metric_golden.py`, which already has
`labels = dataset.labels("club")` wired in and feeding only `knn_accuracy`. Add:

`silhouette`, `silhouette_graph`, `graph_neighborhood_accuracy` (both tie rules if cheap), and
**`majority_class_rate`**, which is missing today and is the chance floor the accuracy numbers
are read against.

**The diff must be additive only** — insertions, zero deletions. That is the only evidence that
the existing 34 values did not move.

### 5. `docs: what the label metrics say on real layouts`

The four measurements, into `DECISIONS.md`. Use SG-*t*-SNE twin layouts, seed 0, as steps 3–5 did.

1. **Is the auxiliary-label claim true?** `DECISIONS.md`, `DATASET_PROVENANCE.md` and spec §3.4
   all assert that a **low** silhouette on tabula's `donor` and `method` is the good outcome —
   a layout separating cells by batch covariate is displaying an artifact. Nobody has checked it
   *is* low. Report donor/method beside `cell_ontology_class` and `compartment`. If it is high,
   that is a finding about the data and it belongs in the write-up, not a defect in the metric.
2. **Does silhouette separate $A_V$ / $A_E$ / $A_{tne}$?** Matched seeds, one shared label vector,
   on `digits` and `football_2023`. Step 3 found the §1.6 fix visible in recall and nowhere else;
   the question here is whether silhouette can see the ablation at all, or is another quiet
   instrument like NND.
3. **The seed-noise band on `digits`**, seven seeds, as step 4 did for NND (0.4 %), VNR (1.2 %)
   and `edge_length_cv` (2.4 %). Without it, step 8's λ sweep could report differences smaller
   than noise.
4. **Sil\* as a label-validity check**, across the roster: high where labels match topology, low
   where they do not. This is what justifies the "validity check, not baseline" label, and it is
   currently an assertion.

By-product worth recording: **time Sil\* on tabula.** §2.5's cost class names SNS and Sil\* as the
metrics that will not run there, and §0.5 already corrects that in principle — a measured number
closes it, and step 7's run planner reads that cost class.

### 6. `docs: close step 6`

`STATUS.md` (step 6 done, step 7 next, what it inherits), a §0.5 session-6 corrections block, and
the §0.5 status table row for Sil\* (line ~160, still `✗`). Also fold in:

- the §0.5-decision-9 finding — premise measured, guard reduced to a diagnostic;
- tabula's redundant composite;
- §2.3's "14 conferences" → 11.

---

## Traps

- **`dataset.graph`'s `weight` is a stored *distance* on `digits`, and `tabula`'s is unstamped.**
  Any metric needing a valuation takes it from `EvaluationContext` — `ctx.affinity` or
  `ctx.ideal`. This is the single most likely way to get `graph_neighborhood_accuracy` quietly
  wrong.
- **A `Dataset` is never laid out and a `LayoutGraph` is never labelled.** Labels are length $n$;
  `vertex_and_edge_points` returns points in the **original** indexing, which is what lets a
  layout of $A_E$ or $A_{tne}$ be scored against the dataset's own labels. If a special case is
  needed to make that work, the layering has been broken somewhere else.
- **The leave-one-out cache must never learn to slice**, and `pairwise_distances` /
  `_loo_neighbors` return **shared cached arrays — treat them as read-only.**
- **`ctx.ideal` is `(n, n)` with `inf` outside the largest component under every policy.** It does
  not subset; the consumer does.
- ⚠️ **Check any tolerance's distribution on `digits` or `tabula`, never on `karate` or
  `football_2023`.** Session 5's shared-vertex tolerance looked well-placed on football — one
  clean empty decade — and was a continuum with no gap on `digits`.
- **Two places to update for every new export**: `twin/metrics/__init__.py`'s `__all__` and
  `tests/test_metrics_surface.py`'s `REQUIRED_EXPORTS`.
- **All plan files, including this one once it lands, are history rather than instructions.**
  Departures go in `DECISIONS.md`.

## Verification

```bash
venv/bin/python -m pytest -q          # 750 + step 6's; 2 skipped (sfdp), nothing existing moves
venv/bin/python -m pytest -q -m slow  # the tabula figure still renders
git status --short                    # clean
```

Then:

- the hand-rolled oracle agrees with `sklearn` on random and structured layouts, and **shares no
  code with the implementation**;
- a singleton class contributes exactly 0, on a constructed fixture;
- `silhouette_graph` **raises** on `disconnected="mask"`, and reports `n_excluded` on a
  constructed disconnected graph;
- `graph_neighborhood_accuracy` excludes isolated vertices and returns the count
  (`isolated_vertex_graph`);
- both graph-side docstrings say "label-validity check, not a baseline";
- `metric_golden.json`'s diff is **additive only**;
- all eight paper figures render.
