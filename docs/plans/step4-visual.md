# Step 4 — `visual.py`

The executable sequence for spec §7 step 4. **The design is not repeated here** — read
`EXPERIMENT_SETUP_SPEC.md` §2.2 and §4.3, §0.5's VNR `[THESIS EDIT]` and its session-3
corrections, and `DECISIONS.md` (18), (20) and the step-4 pins first. This file is what to do, in
what order, and how to know it worked.

Prerequisite: `docs/STATUS.md` for where things stand and the items it carries into this step.

**Scope is exactly NND$_k$, NR, VNR, the two distance distributions, `edge_lengths` and
`edge_length_cv`** — plus the shared machinery described below, which lands *first*.
`detour_ratio`, `edge_curves`, `crossings.py`, `MetricSpec`, `METRIC_REGISTRY` and
`evaluate_layout` are steps 5–7 and must not appear here. Nine commits.

**There is no characterisation-pin commit, and that is deliberate.** Step 3 opened with one
because the kNN family already existed and was about to be split. Nothing in the tree computes a
visual metric today, so there is nothing to preserve; the golden file (commit 8) is the only thing
in the step that can catch a silent move, which is why it is not optional either.

---

## Why the machinery goes first

Measured at tabula's size ($n = 5007$), which is the largest in the roster:

| artefact | time | memory | consumers | cached today? |
|---|---|---|---|---|
| condensed `pdist(X)` | 105 ms | 100 MB result | ADR, NND, NR, the distributions, three stresses | yes, 2 entries |
| `_loo_neighbors(X, k)` | 33 ms | 1.6 MB result | four kNN criteria, `knn_accuracy`, NND$_k$, VNR's min, the NN distribution — **eight after this step** | **no** |
| `condensed(ideal)` | 370 ms | 100 MB result, **301 MB transient peak** | SNS, `raw_stress`, `normalized_stress`, `finite_pair_mask` — each recomputes it | **no** |

Two consequences, neither of which is a micro-optimisation.

**The cost profiles are opposite, so one cache policy cannot serve both.** A leave-one-out
neighbourhood is ~1.6 MB and moderately expensive to build; a condensed distance vector is 100 MB
and cheap to rebuild. `PAIRWISE_CACHE_MAXSIZE = 2` is right for the second and badly wrong for the
first.

**The ideal-distance closure is condensed four times per case for a quantity that is constant per
dataset**, at a 301 MB transient peak each time — `np.triu_indices(5007, 1)` alone is 200 MB of
int64 index arrays. Row concatenation is bit-identical at 95 ms and a 101 MB peak.

Doing this now rather than at step 7 is the point. There are three consumers of the leave-one-out
neighbourhood today and eight after this step; at step 7 there are fifteen metrics behind a
registry, and changing how a metric obtains its neighbours then means touching all of them.

---

## Commit sequence

Each commit green and separable. Branch off `experiment-setup`, merge back when `pytest` passes
and all eight figure functions still render.

### Machinery

1. **`metrics:` `condensed()` stops building 200 MB of index arrays.**

   Row concatenation in place of `np.triu_indices`: `np.concatenate([D[i, i+1:] for i in
   range(n)])`. **Bit-identical** — assert it against the `triu_indices` expression on a
   *non-symmetric* fixture, so the test would catch a transposed triangle as well as a reordered
   one. 370 → 95 ms, 301 → 101 MB peak at $n = 5007$.

   Its own commit because it is a pure substitution provable by equality, and therefore the one
   thing in the step that can be reviewed without reading anything else.

2. **`metrics:` the context caches the condensed ideal and the finite-pair mask.**

   `EvaluationContext.ideal_condensed` and `.finite_pairs`, both lazy, both dropped by the
   existing `release()`. `_stress_pairs` uses them when the ideal came from a context.

   Today SNS, `raw_stress`, `normalized_stress` and `finite_pair_mask` each condense the same
   array — four times per case, for something that cannot vary within a dataset. They belong
   beside `ideal` for exactly the reason `ideal` is there: derived, per-dataset, expensive, shared
   by every representation.

3. **`metrics:` cache the leave-one-out neighbourhood.**

   The content-keyed pattern `pairwise_distances` already uses, with **its own capacity**, because
   1.6 MB per entry is not 100 MB per entry.

   ⚠️ **Keyed on `(digest, k_max)` exactly — never sliced out of a larger cached `k_max`.** Step 3
   established that slicing is not equivalent to re-querying when the layout has exact duplicates.
   A cache that served $k = 5$ out of a cached $k = 16$ would make `knn_recall(X, G, 5)` depend on
   which metric ran first — a cache that changes results, which is the worst kind and would be
   found the expensive way. The test is explicit: ask for $k = 16$, then $k = 5$, on
   `karate_coincident_layout`, and assert the second equals the standalone $k = 5$ value.

### The metrics

4. **`metrics:` `visual.normalized_nn_distance` — NND$_k$, eqs. (65)–(67).**

   Mean over vertices of the mean distance to the $k$ nearest embedded neighbours, over $\bar d$
   (eq. 66, already `common.mean_pairwise_distance` and already ADR's denominator). Built on the
   cached `_loo_neighbors`, so the package keeps one leave-one-out convention. Scalar `k` returns a
   float and array-like an array, as the kNN family does.

   Tests: a uniform grid against a hand computation; invariance under scaling, rotation, reflection
   and translation; swept-vs-scalar agreement; NND $= 0$ on a collapsed neighbourhood.

5. **`metrics:` `node_resolution` and `node_resolution_vertexwise` — eqs. (68), (69).**

   `normalize="mean"` is the default per §0.5's `[THESIS EDIT]`; `"max"` stays selectable and is
   **chunked** (see Traps). The per-vertex min is the $k = 1$ column of the cached query.

   The test that matters: the two normalisers order layouts alike when there is no outlier and
   **diverge** when one vertex is pushed far away. That divergence *is* the edit's justification,
   so it belongs in a test rather than in a paragraph.

6. **`metrics:` the two distance distributions — eqs. (70), (71).**

   Arrays normalised by $\bar d$, returned whole. No summarisation: §3.1.3 is explicit that these
   are diagnostics, not scores, and `DECISIONS.md` (20) says how step 7 persists them. Say in the
   docstring what they cost — tabula's all-pairs distribution is 12.5 M values, 100 MB.

7. **`metrics:` `edge_lengths` and `edge_length_cv` — eqs. (79), (80).**

   The chord $\lVert x_i - x_j\rVert$, in `G.edges()` order, and CV = std/mean over it. The
   chord-not-arc rule from `CLAUDE.md` goes in the module docstring, because step 5's LWCr depends
   on it and this is now where the function that computes it lives.

8. **`tests:` extend the golden file with the visual metrics.**

   New keys in `tests/fixtures/make_metric_golden.py`, regenerated. The diff on
   `metric_golden.json` must be **additive only** — the generator is both sides of the equality
   test, so an additive diff is the *only* evidence that the existing fourteen values did not move.

9. **`docs:` close step 4.**

   The seed-noise measurement for NND and VNR (≥ 5 seeds, matched init, median and IQR — step 3's
   protocol), including the mean-vs-max VNR comparison §0.5 promises will happen "once on real
   layouts"; then `STATUS.md`, the carried-forward table, and where the implementation departed
   from this plan. This file becomes history the moment it merges.

---

## Traps

- **`edge_lengths` must not go through `nx.adjacency_matrix`.** ADR already computes chord lengths
  (`twin/metrics/structural.py`, in `adjacency_distance_ratio`) but off the CSR upper triangle — a
  *different* order from `G.edges()`. Edge points, incidence columns and step 5's per-edge weights
  are all indexed by `G.edges()`. Do not unify the two without checking which order each consumer
  needs: ADR keeps its own because its lengths must align with its weights.
- **Interleaving layouts defeats a small cache.** Evaluating metric-by-metric across the vertex,
  edge and twin layouts of one case — or across the straight and curved variants at step 5 —
  evicts on every switch. Evaluate every metric of one layout before moving to the next. That is a
  step-7 loop-order decision; it is recorded now so that it gets made rather than inherited.
- **VNR's `"max"` normaliser is the only O(n²)-memory thing in the step.** Squaring up tabula's
  condensed vector is 200 MB on top of the 100 MB vector. Chunk the row-wise max.
- **NND$_k$ inherits the tie caveat and is the metric most likely to meet it.** Crowding is what it
  measures, and exact duplicates are where `_loo_neighbors`' choice becomes arbitrary. The
  distances are all zero either way, so the *value* is well defined; only the neighbour set is not.
- **§2.5's cost class is wrong here too, in the other direction.** NND, NR and VNR scale with $n$
  and run comfortably on tabula. The metrics that will hurt are step 5's, which scale with $m$.
- **`plots.py` is still `(d, n)`.** Nothing in `visual.py` may be wired into it before the step-10
  refactor.
- **A collapsed layout divides by zero** in every metric in this step — $\bar d = 0$ for NND and
  the distributions, `max = 0` for NR and VNR.
- **`pairwise_distances` holds at most two entries.** Four visual metrics over one layout is a hit
  each time; a metric that builds a derived array and calls again evicts one. Do not raise the cap
  to "fix" that — 100 MB per entry is why it is two.

## Verification

```bash
venv/bin/python -m pytest -q     # step 3's 487, plus step 4's
python -m twin.report            # unchanged; step 4 adds no dataset columns
git status --short               # clean; if notebooks appear modified, the jupyter filter is unset
```

Then:

- `condensed()` is bit-identical to the `triu_indices` expression on a **non-symmetric** fixture.
- A $k = 5$ request following a $k = 16$ request returns the standalone $k = 5$ value on
  `karate_coincident_layout` — the cache does not slice.
- `EvaluationContext.release()` drops the condensed ideal and the mask as well as the square array,
  and `ideal_is_computed` goes back to False.
- NND$_k$ and VNR are unchanged to floating point under scaling, rotation, reflection and
  translation of `X`.
- The two VNR normalisers agree in ordering without an outlier and disagree with one.
- `metric_golden.json`'s diff contains no modified lines — only additions.
- The mean-vs-max VNR measurement reports median **and** IQR over ≥ 5 seeds, and says whether the
  difference exceeds the seed-noise band.
- All eight `paper_figures` functions still render. Nothing in step 4 touches them, which is why a
  failure there means something moved that should not have.
