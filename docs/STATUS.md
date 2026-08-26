# Where the work stands

Read `CLAUDE.md` first, then this. Updated at the end of each session.

**Last updated:** end of session 4 (2026-08-26). It closed spec §7 step 4, then closed both
remaining open questions (`SpectralLayout`'s Laplacian, VNR's normaliser) and ran a pre-step-5
audit of the test suite. Branch `experiment-setup`; 22 commits.

---

## Position in the spec's §7 ordering

| Step | State |
|---|---|
| **1. Correctness and conventions groundwork** | **Done** — merged `b220cb8`, `e0428de` |
| **2. `Dataset` and `LayoutGraph` (§3.4, §3.5)** | **Done** — session 2, plus the step-2b follow-up; plan: [`docs/plans/step2-datasets.md`](plans/step2-datasets.md) |
| **3. `common.py` + `structural.py` — ADR, kNN family, SNS** | **Done** — session 3; plan: [`docs/plans/step3-metrics.md`](plans/step3-metrics.md), departures in `DECISIONS.md` |
| **4. `visual.py` — NND, NR/VNR, distributions, edge lengths** | **Done** — session 4; plan: [`docs/plans/step4-visual.md`](plans/step4-visual.md), departures in `DECISIONS.md` |
| **5. `crossings.py` — curves, detection, Cr/LWCr, detour** | **Next** |
| 6. `label.py` — silhouette, accuracy, graph-side checks | not started (step 2 unblocked it) |
| 7. Registry, `evaluate_layout`, `batch_layout_metrics`, persistence | not started |
| 8. λ sweep and selection, seeds, straight/curved variants | not started |
| 9. Bugs §1.4, §1.5, §1.7; graph *t*-SNE baseline | not started |
| 10. Plotting refactor (§3.6) | not started |
| 11. Reporting: aggregation, Typst export, schema guards | not started |
| 12. Dataset groups including the adversarial density case | not started |

**No results run has been produced.** `runs/` does not exist (and is now in `.gitignore`), so
nothing downstream is invalidated by a change that moves numbers. Spec §7: do not produce
Section 5 numbers before step 8.

## Read these before starting

1. `CLAUDE.md` — non-negotiable conventions and the version-control policy.
2. `docs/EXPERIMENT_SETUP_SPEC.md` **§0.5 first** — it supersedes the rest of the document. It
   now carries four correction blocks, one per session; the newest wins. Session 3's also holds
   the reconciliation table against `thesis.pdf` and the two new `[THESIS EDIT]`s; session 4's
   corrects §0.5's own VNR entry.
3. `docs/DECISIONS.md` — the open questions the implementer was asked to settle, with reasoning.
4. `docs/DATASET_PROVENANCE.md` — where each dataset's weights actually come from. Assertions
   here are executable: `tests/test_dataset_provenance.py`.
5. `docs/DATASET_REPORT.md` — the generated roster table (`python -m twin.report`).
6. There is **no plan file for step 5 yet** — write one first, as steps 2-4 each did.
   [`step2-datasets.md`](plans/step2-datasets.md), [`step3-metrics.md`](plans/step3-metrics.md)
   and [`step4-visual.md`](plans/step4-visual.md) are all **history** now, kept for their record
   of where the implementation departed from them — the departures themselves are in
   `DECISIONS.md`.

---

## What landed in session 4

Spec §7 step 4, in nine commits plus a merge — 571 tests at that point, up from 487; all eight
`paper_figures` functions still render. Three of the nine are shared machinery and contain no
metric, so they were separately mergeable. **The session then continued past step 4** — see the
two sections below — and ended at **656 tests**.

- **`twin/metrics/visual.py`** — NND$_k$ (eqs. 65-67), NR and VNR (68, 69), the two distance
  distributions (70, 71), `edge_lengths` (79) and `edge_length_cv`. `detour_ratio` stays with
  `edge_curves` at step 5, and `MetricSpec` / `METRIC_REGISTRY` / `evaluate_layout` are still
  step 7 — the same boundary steps 3 and 4 held.
- **`condensed()` stops building 200 MB of index arrays.** Row concatenation in place of
  `np.triu_indices`, bit-identical, asserted against the old expression on a *non-symmetric*
  fixture. Re-measured at $n = 5007$: 355 → 82 ms, 301 → 101 MB peak.
- **The evaluation context caches the condensed ideal and the finite-pair mask**, and
  `_stress_pairs` accepts the context itself as `ideal=`. Four condensings per case become one
  per dataset. `release()` now clears the cache wholesale.
- **The leave-one-out neighbourhood is cached**, keyed on `(digest, k_max)` **exactly**, with its
  own capacity (8, against the pairwise cache's 2). It never slices a small $k$ out of a large
  one; two tests, one of which asserts that slicing *would* have changed the answer.
- **The golden file grew by 13 keys, additively** — 13 insertions, 0 deletions, which is the only
  available evidence that the existing fourteen values did not move.

### What the measurements said

- **The crowding metrics are quiet instruments.** Seven seeds of SG-*t*-SNE on `digits`: NND's
  relative IQR is **0.4 %**, VNR's 1.3-1.8 %, `edge_length_cv`'s 2.4 %. Two orders of magnitude
  tighter than the effect step 3 had to strain to see in `knn_recall`.
- **NR is the foil for VNR, not a reported metric**, and the measurement corroborates the
  thesis's reason for that: $1\times10^{-5}$ on 1797 vertices with a **54 %** relative IQR across
  seeds — it reports which seed put two points nearest each other. Implemented and pinned because
  VNR is defined by contrast with it; **VNR is the point-crowding score**, with NND$_k$.
- **VNR(mean) was exactly NND$_1$** — an identity, not an approximation. That closed the
  normaliser question: **the default is now the per-vertex `"median"`** (author's decision), which
  is per-vertex, outlier-robust and bounded in $[0,1]$. `"max"` was not the fallback because it
  fails in *both* regimes — outlier-driven with a distant vertex, and within 0.5-5.8 % of a
  rescaled NND$_1$ on five real layouts without one.
- **⚠️ NND and VNR are not "higher is better" on their own.** They reward uniform spread, and
  density variation is the signature of a good layout of clustered data. On `digits` a random
  Gaussian control is **first** on both crowding metrics and **last** on ADR (0.9968) and recall
  (0.0090). They must be reported jointly with the structural metrics; §5 must not rank on them
  alone. In spec §0.5 as a reporting rule; **suggested thesis §3 wording is in `DECISIONS.md`,
  for the author to place — `thesis/` is read-only (CLAUDE.md).**
- **The edge-order trap is real.** `G.edges()` and the CSR upper triangle disagree in **12377 of
  14732** positions on `digits`, and agree exactly on `karate` and `football_2023` — so a test on
  either of those would have passed against a wrong implementation.

### After step 4, in the same session: `SpectralLayout`

The last genuinely open item in §0.5's reconciliation table, settled by the author and closed.

- **`variant="hall"` ($L^w = D - W$, eqs. 12-14) is the default.** The old code returned the
  eigenvectors of $L_{\text{sym}}$ directly — the spectral-*clustering* embedding, Koren's
  premultiplied by $D^{1/2}$. ⚠️ **Declared numerical change**: every spectral layout moves.
- **The argument was internal consistency, not baseline strength.** The old reading put
  high-degree vertices *outward* — Spearman(degree, radius) positive on **all ten** datasets —
  where thesis §2.2 says they collapse inward. The figures would have contradicted the prose.
- **§2.2's claim is now measured, and wants a qualifier**: true where there is hub structure
  ($\rho = -0.52$ on `barabasi_albert`, $-0.54$ on `karate_weighted`), false on the $k$-NN
  graphs ($+0.145$ on `digits`, $+0.009$ on `tabula`). **Thesis prose to change, not code.**
- **Koren's variant is selectable** (`variant="degree_normalized"`) and gets one reported line —
  it beats Hall's on `karate_weighted` (recall 0.566 vs 0.452), so this pre-empts
  "under-powered baseline" without touching the derivation.
- **§1.5's solver fix landed here, not at step 9**, because it is not independent: $D - W$'s
  spectrum runs to $2d_{\max}$ where $L_{\text{sym}}$'s is bounded, so `which="SM"` got worse
  exactly when the Laplacian changed. Shift-invert, overlap 1.0000 on all ten, 3.1× faster on
  tabula.
- **A determinism gap was found and closed**: ARPACK's random start made "deterministic"
  spectral runs differ by eigenvector sign. Invisible to every metric, fatal to a persisted
  layout.

### The pre-step-5 audit — what the test suite could and could not catch

Nineteen mutations, each an implementation deliberately broken in a throwaway process, then the
suite run against it. **Sixteen were caught.** The three that were not are now fixed, and the
mutants re-run to confirm it. The technique is cheap and worth repeating before any step that
will be built on top of existing code.

- ⚠️ **`twin_adjacency` — the method itself — had no test.** Forcing `alpha` to 0.5, or the
  polynomial order to 1, regardless of the caller left all 609 tests green while moving
  $A_{tne}$ by max\|diff\| 22.7 and 34.7 on `karate_weighted`. **Cause, and it generalises:**
  every test touching the twin representation compared `layoutgraph` against `graphmatrix`, and
  *both call the same function* — a self-referential comparison. The vertex and edge comparisons
  are fine, since those two modules implement that algebra independently.
  `tests/test_twin_embedding.py` now pins it against an independent dense reimplementation of
  $\sum_k \alpha^k A^k$.
- **The batch key could drop its swept parameters** undetected — the contract step 7's
  resumability rests on (`DECISIONS.md` (21)). Twelve tests in `tests/test_batch_key.py`.
- **No test rendered a figure**, although the merge rule requires it.
  `tests/test_paper_figures.py` now does, with the tabula one opt-in.

Two facts about the transform that fell out of writing the pins, both easy to get wrong:

- **The polynomial order is a reach in the *twin* graph, not in $G$.** The twin graph is
  bipartite, so a vertex-to-vertex walk alternates $v \to e \to v$: on the path 0-1-2, $v_0$
  and $v_2$ are two hops apart in $G$ but **four** in the twin graph, and the entry is zero at
  orders 1-3. So `k = 2` couples a vertex through one intermediate edge — one hop of the line
  graph — not two hops of $G$.
- **`twin_graph` strips the diagonal** the transform produces at even order, and it is large
  (1.44 against off-diagonal entries near 0.7). The stripping is in the wrapper, the diagonal is
  produced in the transform, and neither file's tests covered the seam.

### Where step 5 starts

No plan file exists yet; write one first. Three things it has to take account of:

- **§2.5's cost class is wrong in the direction that matters for step 5.** The expensive metrics
  scale with $m$ — 49368 edges on tabula — not with $n$. Nothing in step 4 hurt on tabula;
  crossings will.
- **`edge_lengths` is the per-edge order of record**, `G.edges()`, and LWCr's denominator
  $\bar L$ (eq. 80) is already pinned in the golden file. ADR's CSR order is a different one and
  the two must not be unified.
- **`count_intersections` diverges from eq. (78)** — it skips edge pairs sharing a vertex and
  counts only `Point` intersections — and step 5 reimplements it on layer 2 anyway. It is also
  the first of the two remaining legacy $(d, n)$ consumers to converge. ⚠️ It now has a caller
  in the test suite only indirectly, through the figure smoke test; a reimplementation needs its
  own pins before the old one is retired.
- **The self-referential-comparison trap generalises.** `twin_adjacency` was invisible because
  the two things compared shared an implementation. Step 5 will be tempted to check
  `crossings.py` against `plots.count_intersections` — that is legitimate *only* where the two
  do not share code, and §0.5 records that they should disagree on shared-endpoint pairs anyway.

## What landed in session 3

Spec §7 step 3, in one merge of nine commits plus four on top — **487 tests, up from 299**; all
eight `paper_figures` functions still render.

- **`twin/metrics/` is a package**: `common.py` (orientation, the ideal-distance model, the
  condensed pair ordering, the capped `pdist` cache, `default_k`, the one leave-one-out
  convention), `structural.py` (ADR, the four kNN criteria, `weighted_recall`, SNS and the two
  stresses it is contrasted with), `label.py` (the existing kNN accuracy family, moved two steps
  early because §2's layout puts it there), `context.py` (the evaluation context).
- **`twin/forms.py`** — which valuation a bare `nx.Graph` carries. Both the `Dataset` layer and
  the `LayoutGraph` layer stamp; ADR and `ideal_distances` refuse the wrong side rather than
  inverting silently. `digits` is stamped `"distance"`, `karate_weighted` `"affinity"`.
- **`EvaluationContext`** — one ideal-distance target per dataset, lazily computed, shared by
  $A_V$, $A_E$ and $A_{tne}$. The central ablation now has an object that enforces it.
- **The last `mode=` string is gone.** `knn_neighbor_preservation_accuracy` became
  `knn_recall` / `knn_precision` / `knn_jaccard` / `weighted_recall`, each with a `*_report`
  sibling carrying `n_excluded` and `n_scored`.
- **A golden metric suite** on `karate_weighted`, keyed by metric name so steps 4-6 add keys.

### What the measurements said

- **The §1.6 fix is visible in recall and nowhere else.** Seven seeds, matched init, one shared
  target: `knn_recall@16` 0.6332 → **0.6393**, *disjoint over all seven seeds*, while ADR and SNS
  do not separate. This is the first evidence that the bug mattered rather than merely changed
  something — Procrustes could not show it. `DECISIONS.md` has the table.
- **`exp_decay` beats `exp_decay_sq` on digits**, 6/7 paired seeds on recall and 7/7 on ADR, but
  one seed reverses — weaker than the §1.6 result. The default does not change on the strength
  of it; §3.4 keeps the choice configurable.
- **Two pinned values moved**, both on the coincident-points fixture added for exactly this, when
  the two metric families were put on one leave-one-out convention. Declared in the commit.

### Where step 4 starts

[`docs/plans/step4-visual.md`](plans/step4-visual.md) has the commit sequence; the eleven decisions
the spec left open at this step are pinned in
[`DECISIONS.md`](DECISIONS.md#step-4-decisions-pinned-before-they-are-discovered-the-hard-way).
Three things worth knowing before opening either:

- **Scope is exactly NND$_k$, NR/VNR, the two distance distributions and the edge lengths.**
  `detour_ratio` goes with `edge_curves` at step 5, and `MetricSpec` / `METRIC_REGISTRY` /
  `evaluate_layout` are still step 7 — the same boundary step 3 held.
- **The first three commits are shared machinery, not metrics, and the reason is measured.** The
  leave-one-out neighbourhood has three consumers today and eight after this step and is not
  cached at all; the condensed ideal-distance vector is rebuilt four times per case at a 301 MB
  transient peak. Retrofitting either at step 7 means touching fifteen metrics behind a registry.
  ⚠️ The neighbourhood cache must not slice a small *k* out of a large one — that would make a
  metric's value depend on evaluation order whenever the layout has duplicates.
- **Step 4 is greenfield, so there are no pins to preserve** — nothing in the tree computes a
  visual metric today. The golden file is the only thing in the step that can catch a silent
  move, and its one `[THESIS EDIT]` (VNR's normaliser) comes with an obligation to measure both
  options on real layouts.

## What landed in session 2

Spec §7 step 2, in two merges — `step2/datasets` (eleven commits against ten planned items) and
the `step2b/forms-and-sparsity` follow-up (five commits, prompted by review). **299 tests, up
from 94**; all eight `paper_figures` functions still render.

- **`twin/datasets/`** — `Dataset`, `LabelSet`, `DatasetStyle`, the stored-source types, one file
  per dataset, and `DATASET_REGISTRY`. Ten datasets: the `karate` / `karate_weighted` split, the
  `mnist` → `digits` rename, `sbm_er_3modes` newly reachable, and tabula loading `A`, `D`, all
  four label fields and `X_umap`.
- **`twin/layoutgraph.py`** — `LayoutGraph` plus `vertex_graph` / `edge_graph` / `twin_graph`.
  Derivation of a missing side happens in `vertex_graph` and nowhere else; the incidence matrix
  is built once and shared; the derivation chain is read off a `parent` link.
- **`twin/embedding.py`** — `embed(layout_graph, y0)`, `mode=` deleted, `EMBEDDING_REGISTRY` with
  `expects` / `stochastic` / `has_lambda`, and a conformance test that asserts each method reads
  exactly the side it declares.
- **`twin/results.py`** — `LayoutResult` holds the `Dataset`; `G` is a property; it records the
  derivation chain.
- **`twin/report.py`** — `python -m twin.report`; output committed as `docs/DATASET_REPORT.md`.
- **Bug §1.6 closed.** Digits stops being fed distances as if they were affinities.

### The step-2b follow-up

- **`LayoutGraph.combinatorial`** — the third form. A `LayoutGraph` is one support carrying up
  to three valuations, derived structurally rather than by thresholding the affinity.
- **Yifan Hu is a combinatorial method**, not a distance one, and the graphviz plumbing was
  inverted: `nx_pydot` writes `weight=`, which graphviz reads as a spring strength. Step 2's
  `expects="distance"` therefore had both graphviz methods reading *far apart means pull
  harder*. sfdp now gets a graph with no edge attributes; neato gets `len=`. The `expects`
  conformance test structurally cannot catch this, so three tests cover the plumbing directly.
- **Sparsity is declared and checked** — `KNearest` on the `Dataset`, verified against the graph
  (minimum degree at least *k*, nk/2 ≤ m ≤ nk; both hold exactly). The report gains a Sparsity
  section including the raw stored-matrix nnz.
- **The cost of sparsification is measured** for the two point-cloud datasets, whose source
  clouds are retained but never used to lay out or score.

### What the measurements said, where they contradicted the plan

- **Tabula's two stored sides do not disagree.** Global Spearman is −0.236 and per-row −0.782,
  both weaker than the spec expected — but restricted to pairs recorded in one direction only,
  where fuzzy union cannot have mixed two rows' parameters, it is **exactly −1 in all 3819
  scored rows**. Symmetrisation is the cause of the aggregate degradation. `DECISIONS.md` has the
  table; read the ≈ −0.9 threshold against the *unmixed* column.
- **The §1.6 fix is decisive at the matrix level and invisible at the layout level.** Spearman
  between the old and new digits edge weights is exactly −1 over all 14732 edges. But Procrustes
  disparity between the old and new layouts at matched init is 0.02–0.46, while *re-initialising
  the same configuration* moves it 0.46–0.83 — so the correction moves the layout less than seed
  noise does, and Procrustes cannot show that the bug mattered. Re-measure in quality terms
  (kNN recall, ADR) once step 3 lands.
- **Sparsification costs less than it appears to, for SNS specifically.** Raw relative stress
  between graph geodesics and the ambient metric is ~1.8 on digits and ~1.5 on tabula, which
  would be alarming to quote; after the optimal uniform rescaling — the same thing SNS's
  closed-form α\* does — it is ~0.20 for both. Nine tenths of the apparent distortion is a
  uniform inflation SNS absorbs. What is left is the ordering: rank ρ 0.70 and 0.79. All of it
  improves monotonically with *k* (digits at k = 5/12/30/60: ρ 0.61 → 0.85).
- **No dataset in the roster has a coincident vertex pair.** §0.5's zero-ideal-distance concern
  is inert for the current roster; the report has the column to catch a future dataset that
  reintroduces it.

---

## Environment facts not visible in the tree

- **`venv/` is the interpreter.** `venv/bin/python -m pytest -q`.
- **`sgtsnepi>=0.7` is required.** 0.5.1's `silent=True` closes fd 2 without restoring it, which
  breaks stderr — and so loguru — for the rest of the process. `tests/test_sgtsne_stdio.py` pins
  the floor. Layouts are bit-identical between 0.5.1 and 0.7, so the upgrade moved no numbers.
- **One-time setup per clone**, because git filters live in `.git/config` and cannot be set from
  the repository:

  ```bash
  git config filter.jupyter.clean venv/bin/nbstripout-fast
  ```

  Without it the `jupyter` clean filter cannot run, notebooks show as permanently modified, and
  the clean-tree rule for results runs is unenforceable.
- `pytest`, `pyarrow` and `psutil` are installed. `openTSNE` and `fa2_modified` are **not** — both
  are needed at step 9.
- **The notebooks under `notebooks/` were not updated** and import modules that have moved
  (`twin.metadata`, the old `graph_embedding` signature). They are not a record of results
  (outputs are stripped) and nothing in the test suite touches them.

## Health check

```bash
venv/bin/python -m pytest -q          # 656 passing, 1 deselected (`-m slow`)
venv/bin/python -m pytest -q -m slow  # the opt-in tabula figure, ~2 min
python -m twin.report            # the roster table; regenerates docs/DATASET_REPORT.md with -o
git status --short               # clean; if notebooks appear modified, the filter is unset
```

Figures: the six fast `paper_figures` functions render in seconds; `tabula_sapiens_teaser` takes
about three minutes and internally calls `tabula_sapiens_blockadj`, so rendering it covers both.
All eight render as of step 4's close. (`barabasi_albert` is commented out in the source and is not
one of the eight.)

---

## Carried forward — each with where it bites

| Item | Bites at |
|---|---|
| ~~Ideal distances have no consumer~~ — `common.ideal_distances`, the `max_n` guard and `EvaluationContext` landed in step 3 | discharged |
| ~~`exp_decay` vs `exp_decay_sq` never compared~~ — measured on digits, `DECISIONS.md` | discharged |
| ~~A metric needs the vertex `LayoutGraph`~~ — `twin/metrics/context.py` | discharged |
| SNS on a Kamada-Kawai or neato layout is close to a measure of optimiser success. The caveat is in the SNS docstring; the **reporting flag that acts on it** is not built | step 11 |
| `EvaluationContext.release()` exists but nothing calls it, and it now drops three arrays rather than one — tabula's closure is 200 MB and its condensed form another 100 MB. `clear_pairwise_cache()` and `clear_loo_cache()` are the same story | step 7 |
| The neighbourhood metrics are not sweep-consistent on a layout with exact duplicates (`DECISIONS.md`). No registered dataset produces one; a crowding-heavy layout at large λ might | whenever a layout first collapses two points |
| ~~Re-measure the §1.6 fix in quality terms~~ — done: recall separates disjointly over seven seeds, ADR and SNS do not | discharged |
| ⚠️ **Spec §2.5's cost class is wrong.** It says SNS and Sil\* are what will not run on tabula; the largest n is 5007, so the O(n²) closure is 200 MB and `max_n=20000` never trips. The expensive metrics scale with **m** — crossings, 49368 edges on tabula | a run planner built on it would size the wrong thing: **steps 5 and 7** |
| Memory runs need **one case per process**: `peak_mem_mb` is a delta from the entry baseline, and glibc does not always return freed memory, so later cases in a batch measure from a higher floor | step 7 |
| A persisted layout must store the dataset's registry **name and params**, never the `Dataset` object | step 7 |
| `common_init` is not in the batch runner — deliberately deferred, since it needs the seed axis and the paired protocol; the column-slicing convention it depends on is already tested | step 8 |
| `openTSNE` and `fa2_modified` not installed | step 9 |
| One legacy `(d, n)` surface: `graph_embedding_all` → `paper_figures` → `plots`, and with it `twin/graphmatrix.py`'s pre-restructure classes and their `.affinity` / `.distance` shim | converges at steps 5 and 10; tracked in `DECISIONS.md` (2) |
| Tabula's Leiden partition is still computed inside `tabula_sapiens_teaser` rather than registered as a `detected:leiden` label set | step 10, where the consumer and the cost land together |
| ~~**Open:** VNR's normaliser~~ — **closed session 4**: the per-vertex median. Still one `[THESIS EDIT]`; eq. (69) never changed | discharged |
| ~~**Open:** `SpectralLayout`'s Laplacian~~ — **closed session 4**: Hall's $D-W$ is the default, Koren's is selectable and reported once, §1.5's solver fix landed with it. §2.2's "hubs collapse inward" is now measured — true on hub graphs, false on the $k$-NN graphs, so **the thesis prose wants a qualifier** | thesis prose, before submission |
| ⚠️ **`SpectralLayout` still does not handle a disconnected graph** — the zero eigenvalue's multiplicity is the component count, so column 0 is not the only trivial vector and the layout would keep component indicators as coordinates. §1.5's other half; §4.4's policy covers metrics, not the layout side | whenever a disconnected dataset joins the roster |
| Two scheduled code divergences from the thesis: `count_intersections` versus eq. (78), and `fa2` → `fa2_modified` — spec §0.5's table | **steps 5 and 9** |
| Step 9 is lighter than the table says: **§1.5 is done** (session 4), so what remains there is §1.4, §1.7 and the graph *t*-SNE baseline | step 9 |
| Cache loop order: evaluate every metric of one layout before moving to the next, or a two-entry cache evicts on every switch. Nothing enforces it yet | **step 7**, the batch runner |
| **Open, needs the author's call:** VNR's `"mean"` normaliser is **exactly** NND$_1$, so under the `[THESIS EDIT]` the point-crowding column is NND at two values of $k$; the thesis's `"max"` is a genuinely second measurement but is distorted by outlier vertices. Both are implemented, both are in the golden file, and the comparison §0.5 asked for is done — `DECISIONS.md` | before the results tables are designed: **step 11**, but the decision can be taken now |
| The three visual metrics have a measured seed-noise band on `digits` (NND 0.4 %, VNR 1.2 %, `edge_length_cv` 2.4 %). Nothing reads it yet; a λ sweep that reports differences smaller than it would be reporting noise | step 8 |
| ⚠️ **NND and VNR reward uniformity, which a good layout of clustered data does not have** — a random control tops both and comes last on ADR and recall. The reporting layer must not rank on them alone, and the joint reading is not enforced anywhere yet | step 11 |
| `NodeResolutionReport.n_zero_denominator` joins `n_ties` / `n_zero_distance` in the diagnostics table of `DECISIONS.md` (18); nothing collects that table yet | step 7 |
| ⚠️ **`notebooks/metrics.ipynb` cannot run at all** — it imports `twin.metadata`, deleted at step 2. It is also the sole consumer of `embedding_metrics_all` / `metrics_to_typ` / `metrics_to_df`, so notebook and functions share a fate | step 11 |
| The other notebooks import modules that have moved; `twin-gui.ipynb` is the only consumer of `plots.application_plot`, which is **wanted** and needs porting to the new interface | step 10 |
| `twin/batching.py` is 1562 lines with one smoke test plus the twelve key-contract tests added in session 4. The rest of the engine is unexercised; CLAUDE.md says not to rewrite it, so the question is what step 7 leans on | step 7 |
| The figure check is a **smoke** test — it catches a raise, not a wrong picture. Real pixel baselines wait for the §3.6 refactor | step 10 |

## Traps

- **All three plan files are history, not instructions** — `step2-datasets.md`,
  `step3-metrics.md` and now `step4-visual.md`. Each records where the implementation departed
  from it; the departures are in `DECISIONS.md`. Step 5 has no plan yet.
- **Two per-edge orderings coexist and look identical.** `G.edges()` is the order of record for
  per-edge arrays; ADR alone uses the CSR upper triangle, because its lengths must align with
  the weights from the same matrix. They agree on `karate` and `football_2023` and disagree in
  12377 of 14732 positions on `digits`, so **a test on the small graphs proves nothing**.
- **The leave-one-out cache must never learn to slice.** Serving `k=5` out of a cached `k=16` is
  free, and it would make a metric's value depend on evaluation order whenever a layout has
  exact duplicates. `tests/test_metrics_common.py` asserts both that it does not and that
  slicing would have changed the answer.
- **`docs/plans/step2-datasets.md` is now history, not instructions.** Where the implementation
  departed from it — `LayoutGraph`'s module, the commit ordering, the shim on the legacy classes
  — `DECISIONS.md` records why.
- **`twin/graphmatrix.py` is not the representation layer any more.** `twin/layoutgraph.py` is.
  The old classes survive for `paper_figures` only; adding a caller re-entangles the two layers.
- **A `Dataset` is never laid out and a `LayoutGraph` is never labelled.** If a metric or figure
  needs a special case to score $A_E$ or $A_{tne}$ against the dataset's labels, the layering has
  been broken somewhere — `vertex_and_edge_points` returns points in the **original** indexing.
- **Declaring which form a method reads is not the same as it arriving intact.** The graphviz
  regression passed the `expects` conformance test the whole time, because the test checks which
  form was *read*, not what the backend took it to mean. Any new backend that serialises its
  input needs its plumbing tested separately.
- **`dataset.graph`'s `weight` attribute is stamped now, and `tabula`'s is not.** A metric that
  needs a valuation should take it from `EvaluationContext` — `ctx.affinity` or `ctx.ideal` —
  rather than from `dataset.graph`, which is stamped only where the dataset declares the
  attribute.
- **Pin derived data before refactoring it.** The step-2 pins (`tests/fixtures/step2_labels.npz`
  plus the digest table in `tests/test_step2_characterisation.py`) are what made the label and
  colour move provably behaviour-preserving. Regenerating the fixture without updating the
  digests fails loudly, which is the point.
