# Where the work stands

Read `CLAUDE.md` first, then this. Updated at the end of each session.

**Last updated:** end of session 6 (2026-09-15). It closed spec §7 step 6 — `label.py` now holds
the silhouette family and the 1-hop graph-side accuracy, so **the metric suite of §2 is
complete**. Branch `experiment-setup`; tree clean. **830 pass and 2 skip** — the two `sfdp` cases,
behind a runtime probe of the broken backend (see below).

> **Step 7 is next** — the registry, `evaluate_layout`, `batch_layout_metrics` and persistence.
> There is no plan file for it yet. Everything from steps 3-6 that was deferred to it is
> collected under "Where step 7 starts" below; the carried-forward table is the authority.

---

## Position in the spec's §7 ordering

| Step | State |
|---|---|
| **1. Correctness and conventions groundwork** | **Done** — merged `b220cb8`, `e0428de` |
| **2. `Dataset` and `LayoutGraph` (§3.4, §3.5)** | **Done** — session 2, plus the step-2b follow-up; plan: [`docs/plans/step2-datasets.md`](plans/step2-datasets.md) |
| **3. `common.py` + `structural.py` — ADR, kNN family, SNS** | **Done** — session 3; plan: [`docs/plans/step3-metrics.md`](plans/step3-metrics.md), departures in `DECISIONS.md` |
| **4. `visual.py` — NND, NR/VNR, distributions, edge lengths** | **Done** — session 4; plan: [`docs/plans/step4-visual.md`](plans/step4-visual.md), departures in `DECISIONS.md` |
| **5. `crossings.py` — curves, detection, Cr/LWCr, detour** | **Done** — session 5; plan: [`docs/plans/step5-crossings.md`](plans/step5-crossings.md), departures in `DECISIONS.md` (32)–(36) |
| **6. `label.py` — silhouette, accuracy, graph-side checks** | **Done** — session 6; plan: [`docs/plans/step6-labels.md`](plans/step6-labels.md), departures in `DECISIONS.md` (39), measurements (40) |
| **7. Registry, `evaluate_layout`, `batch_layout_metrics`, persistence** | **Next** — no plan file yet |
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
6. `docs/CROSSING_GEOMETRY.md` — **for step 5 specifically**: the crossing algorithm in full,
   with the derivation, the measurements and the degeneracy analysis. Spec §2.4 is the summary;
   this is the reference.
7. **All six plan files are history**, kept for their record of where the implementation
   departed from them — the departures themselves are in `DECISIONS.md`.
   [`step2-datasets.md`](plans/step2-datasets.md), [`step3-metrics.md`](plans/step3-metrics.md),
   [`step4-visual.md`](plans/step4-visual.md), [`step5-crossings.md`](plans/step5-crossings.md)
   and [`step6-labels.md`](plans/step6-labels.md). **Step 7 has no plan yet.**

---

## What landed in session 6

Spec §7 step 6, in six commits — **830 tests, up from 750**; the tabula figure still renders. The
plan ([`plans/step6-labels.md`](plans/step6-labels.md)) held on substance; the five departures are
`DECISIONS.md` (39) and the measurements (40). **The metric suite of §2 is now complete** —
everything remaining is machinery.

- **`twin/metrics/label.py` is finished**: `silhouette` / `silhouette_report` (eqs. 49-52),
  `silhouette_graph` / `silhouette_graph_report` (53-55), `graph_neighborhood_accuracy` /
  `graph_neighborhood_accuracy_report` (61-64). The $k$-NN accuracy family had landed at step 3.
- **`sklearn`-backed, validated against a hand-rolled oracle** in `tests/test_silhouette.py` that
  shares no code with it — one oracle for both sides, since eqs. (53)-(55) differ from (49)-(52)
  only in which $d$ they read. The graph-side accuracy has its own dictionary-count oracle.
- **Sil\* refuses `disconnected="mask"` on all three paths a policy can arrive by** — an explicit
  kwarg, an `EvaluationContext` built under it, and a bare array bearing its footprint — and
  subsets to the largest component itself, reporting `n_excluded`.
- **`_vote_outcome` is shared** by Acc$^k$ and Acc\*; the difference between them is the
  neighbourhood, not the voting. Bit-identical — `LABEL_PINS` and the golden file both pin
  `knn_accuracy`.
- **The golden file grew by 6 keys, additively** (now 44), including `majority_class_rate`, which
  was missing and is the floor every accuracy number is read against.

### What the measurements said

- ⚠️ **The auxiliary-label claim is false for `donor` and true for `method`.** A *low* silhouette
  on tabula's batch covariates is asserted to be the good outcome in three places. `donor` scores
  **0.206**, double `cell_ontology_class`'s 0.106 and first of the four — and first on Sil\* too,
  so the structure is in the graph before any layout. The cause is **confounding**: the two donors
  sampled almost disjoint cell populations, NMI(donor, cell type) = **0.413** against 0.059 for
  `method`. A low donor score is not an achievable target; `method` is the covariate the argument
  applies to. **Thesis and §3.4 prose to change, not code** — `DECISIONS.md` (40) has the wording.
- ⚠️ **Sil separates the ablation decisively and orders it differently from every other label
  metric.** Disjoint across five seeds for $A_V$ against both others on both datasets. But
  `knn_accuracy` is *identical to the digit* for $A_V$ and $A_{tne}$ on football (0.9697 every
  seed) where Sil separates them 0.758 to 0.617, and on `digits` the two **invert** $A_E$ against
  $A_{tne}$. Sil is distance-based, the accuracy family rank-based, and the twin graph stretches
  within-class neighbourhoods without changing who is whose neighbour. **Sil is not "which
  representation recovers labels best".**
- ⚠️ **Sil is the noisiest metric in the project so far**: rel IQR **3.6 %** and a **19 %** spread
  over seven seeds on `digits`, against NND's 0.4 %, VNR's 1.2 %, `edge_length_cv`'s 2.4 % and
  `knn_accuracy`'s 0.24 %. That explains the inversion — the gap is inside the band — and sets
  step 8's floor.
- **Sil\* works as a label-validity check, and its scale is not the one anyone assumes.** Every
  real label set is clearly above its shuffled level, and the one that is not is football's
  `division`, which should not be. But the **ceiling is ~0.53** even where labels *are* the
  topology (`ring_of_cliques`), and **chance is negative** by a class-count-dependent amount
  (−0.32 at 10 classes, −0.0004 at 2). Acc\* exceeds 0.94 everywhere and is near-useless without
  its floor: 0.987 against a **0.958** floor on tabula's `method`.
- **§2.5's cost class is wrong about Sil\*, now by a measured margin.** 0.58 s on tabula given the
  shared closure, 2.07 s for all four label sets, against crossings' 40 s. The label family scales
  with $n$, and the largest $n$ is 5007.
- ⚠️ **A wrong justification caught by measuring it.** `silhouette` shipped in commit 1 on
  `sklearn`'s `metric="euclidean"` with a claim that its chunked reduction never materialises
  $(n, n)$. It does, because `working_memory` defaults to 1024 MB — the exact `pdist` route is
  **faster, leaner and eight digits more accurate** (0.60 s / +273 MB against 0.91 s / +373 MB at
  $n = 5007$). Corrected in its own commit, numerically neutral and verified so: the golden value
  is bit-identical. ⚠️ **The tolerance would not have been**: an invariance test pinned at
  $10^{-12}$ from `karate` ($7\times10^{-14}$) fails at tabula's $n$ ($3\times10^{-12}$).

### Where step 7 starts

**There is no plan file.** What steps 3-6 deferred to it, in one place:

- **`MetricSpec`, `METRIC_REGISTRY`, `evaluate_layout`, `batch_layout_metrics`, persistence.**
  The boundary held at steps 3, 4, 5 *and* 6. Two concessions to it exist in the metric layer:
  the `crossings=` argument on the three layer-3 crossing metrics, and `ideal=` accepting an
  `EvaluationContext` on the stresses and on Sil\*.
- **The label-set enumeration is one line, not a thing to build.**
  `dataset.label_sets_by_scoring_role(...)` and `dataset.composite_of` already exist. ⚠️ But
  **tabula's composite is redundant** — `compartment_cell_type` is exactly
  `cell_ontology_class` — so a naive enumeration emits a duplicate column.
- ⚠️ **§4.4 permits `mask` for SNS only, and Sil\* now refuses it.** A runner that builds one
  `EvaluationContext` per dataset with `disconnected="mask"` will be told rather than get an
  inflated Sil\*. Two contexts, or `lcc` throughout.
- **The reference levels the measurements say are mandatory are not computed anywhere.** Sil\*
  needs a shuffled-label control (three lines) and Acc\* needs its floor carried alongside;
  `majority_class_rate` exists and is now in the golden file. §5.3 guard 2 — the separate table —
  is still unbuilt.
- **The cost model is now fully measured** and §2.5 is wrong in both directions: crossings is 40 s
  and +428 MB per tabula layout, the whole label family is under 4 s, and the `ideal_distances`
  closure is 14.9 s paid **once per dataset**. The loop order matters more than the metric set.

## What landed in session 5

Spec §7 step 5, in nine commits — **750 tests, up from 656**; all eight `paper_figures` functions
still render and Figure 3 is bit-identical to its pre-rewrite output. The plan
([`plans/step5-crossings.md`](plans/step5-crossings.md)) held; the five departures are
`DECISIONS.md` (32)–(36).

- **`twin/geometry.py`** — the γ kernel, the γ→Bézier conversion, the power form, the bulge, the
  exact AABB and the closed-form arc length, shared by `plots.draw_curved_edges` and the metrics.
  The refactor is pinned bit-for-bit against the curves `draw_curved_edges` plotted before it.
- **`twin/metrics/crossings.py`** — `edge_curves` (layer 1), `crossing_pairs` /
  `crossing_pairs_report` (layer 2), `crossing_number` / `crossings_per_edge_pair` /
  `length_weighted_crossings` (layer 3), and `detour_ratio`. No sampling anywhere; `n_samples`
  does not exist.
- **`plots.count_intersections` is built on layer 2**, behaviour-preserving against a pin taken
  first. The tree now holds one intersection routine.
- **`common.edge_index_array`** — the `G.edges()`-ordered endpoint array, lifted out of
  `edge_lengths` so the crossing layer indexes edges through the same function.
- **The golden file grew by 10 keys, additively.** Both §4.3 variants of all three crossing
  metrics, plus `detour_ratio`.

### What the measurements said

- **The guide reproduces to the digit.** $K$ is *identical* to the pre-work measurement on all
  three datasets (27 058 / 1 380 736 / 11 375 494) and $Q_4$'s distribution matches to three
  digits. Two of its figures were understatements: the Bernstein filter rejects **46–59 %** on
  real layouts rather than 35 %, and §4.3 costs **1.7×** rather than 2× because at γ=0 no quartic
  is built at all.
- **0 disagreements over 19 586 non-adjacent candidate pairs** of a real football twin layout,
  against a 1200-segment polyline oracle, plus 1200 random pairs at three γ. Two adjacent pairs
  disagree and **the exact method is the one that is right**: re-solved at 50 digits, both have a
  genuine second root distinct from the shared-vertex root — one at $s = 1.4956546183\times10^{-6}$
  against $1.4956546170\times10^{-6}$ from the double-precision solve — which a 1200-segment
  polyline cannot separate from the endpoint.
- ⚠️ **Tolerances do not survive contact with the large dataset.** Two of them were designed
  against `football_2023`, where the data shows a clean gap, and `digits` turned out to be a
  **continuum** in the same quantity. That killed the distance-based shared-vertex rule outright
  (DECISIONS (32)); the replacement divides the known root out of the quartic and has no
  tolerance at all. **Check a tolerance's distribution on `digits` or `tabula` before believing
  a gap on a small graph.**
- **The coupled $\lVert D\rVert$ / $Q_4$ band that the guide predicts is empty**, and not because
  of $\varepsilon_{\text{lin}}$ — the deflation closes it before the conditioning can matter
  ((33)). Verified across twelve decades with constructed pairs, since no real layout populates
  the band at all.
- **Cost, and (31)'s call: stay in NumPy.** Tabula is **40.2 s curved / 28.9 s straight** end to
  end, peak +428 MB, which is 1.3× over (31)'s line rather than the 1.5–2× the plan estimated.
  Rung 2 of the escalation — parallelism across layouts — is untouched and **belongs to step 7's
  runner**. The stage split and **two rejected optimisations** are `DECISIONS.md` (37): the solve
  is 47 % of the runtime, no pair deflates below degree 3 so there are no cheap ones, and a
  vectorised Ferrari is 5.4× faster while changing the crossing count on 0.02 % of quartics —
  concentrated at near-tangency, which is the regime under study. **Do not re-propose either.**
  One real ~9 s win is identified and deliberately not taken: `_swept_segments` is redundant for
  an exactly straight curve.
- **γ=0 is a numerically special case that does not announce itself.** `arc_length`'s first
  implementation was out by a factor of 2 on the entire straight variant, because at γ=0 the
  quadratic coefficient is not bitwise zero but rounding noise *with an arbitrary direction* —
  so a collinearity guard reads the noise as a real direction. Three branches, not the spec's
  two ((35)).

### Where step 6 started — history; step 6 is done

> Superseded by "What landed in session 6" above. Kept because the three items below were all
> load-bearing and two of them turned out differently than this section expected.

What step 5 left for it:

- **Scope is silhouette (eqs. 49–52) and the graph-side variants (53–55, 61–64).** The kNN
  accuracy family already landed at step 3, so `label.py` exists and is half-built.
- ⚠️ **§0.5 decision 9's composite-label guard rests on a premise that does not hold.** It cites
  "14 conferences × 3 divisions over 132 teams" producing many tiny classes; `football_2023`
  actually has **11** conferences and its composite has 15 classes with a **minimum size of 4**,
  no singletons. And `tabula`'s composite is **redundant** — `compartment` is a function of
  `cell_ontology_class`, so it has the same 13 classes. The guard is reduced to a diagnostic;
  the plan has the numbers.
- **`MetricSpec` / `METRIC_REGISTRY` / `evaluate_layout` are still step 7** — the boundary steps
  3, 4 and 5 all held. Step 5's one concession to it is the `crossings=` argument on the three
  layer-3 metrics, which exists so the registry can solve once and summarise three times.
- **The crossing metrics are the expensive ones and the runner must know it.** 40 s per tabula
  layout, 428 MB peak, and the grid is ~150 layouts. §2.5's cost class still says the wrong thing
  (it names SNS and Sil\*), which matters at step 7, not step 6. **Session 6 measured the other
  half**: the whole label family is under 4 s on tabula, Sil\* 0.58 s.

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

> **Read [`plans/step5-crossings.md`](plans/step5-crossings.md) — it is approved and current,
> and it supersedes the summary in this section wherever they differ.** The guide's mathematics
> was independently re-verified before planning (it holds), three of its passages were corrected
> in place as **Revision 3**, and $K$ has been measured, so the section below understates what
> is already known.

**The algorithm is settled.** Read [`docs/CROSSING_GEOMETRY.md`](CROSSING_GEOMETRY.md) first — it
is the specification for this step. Spec §2.4 Layers 1–2 were rewritten to match it, §0.5 carries
the change log, and `DECISIONS.md` (27)–(31) record the choices. Crossings are computed
**exactly**, by finding the roots of one quartic per candidate pair: no sampling, no subdivision,
no geometric tolerance. `edge_curves` returns control points, not polylines, and `n_samples` no
longer exists.

The plan is [`plans/step5-crossings.md`](plans/step5-crossings.md), nine commits. What it takes
account of, in short:

- ~~**The one unmeasured quantity is $K$**~~ — **measured, before implementation.** SG-*t*-SNE
  twin layouts, seed 0, γ=0.5, unit-box normalised: `football_2023` **27 058**, `digits`
  **1 380 736**, `tabula` **11 375 494** (guide predicted 1e7 — the cost model is sound). Straight
  and curved differ by 0.2 %. The table is in the guide §6 and the plan. Commit 3 still prints it
  as a check: **if tabula is not within ~2× of 11.4 M, stop and find out why.** The *decision*
  (31) feeds — NumPy versus compiled — is deferred to commit 4 and an end-to-end timing, not
  these per-stage estimates.
- ⚠️ **The broad-phase query is not free and the cost model omitted it**: 9.5 s and a **365 MB**
  return array on tabula, 27 % of the runtime. Blocking the solve does **not** cap that memory —
  the bulk query materialises every pair first, so capping it needs *edge-chunked queries*.
- **§2.5's cost class is wrong in the direction that matters for step 5.** The expensive metrics
  scale with $m$ — 49368 edges on tabula — not with $n$. Nothing in step 4 hurt on tabula;
  crossings will.
- **`twin/geometry.py` is a new module and it is shared with `plots.py`.** The γ→Bézier conversion
  must live in exactly one place or the drawn curve and the measured curve drift apart with
  nothing to catch it. Moving it out of `draw_curved_edges` is a *refactor*; `crossings.py` is a
  *behaviour change*. Per CLAUDE.md those are separate commits, in that order.
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
  The independent oracle to check against is a **dense polyline reference** (~1200 segments per
  curve), kept in the test module only. It agreed 200/200 with the exact method in validation.
- **Two things about the filter and the guards were wrong in the pre-work and are now fixed in
  the guide (Revision 3); the plan carries the detail.**
  - ⚠️ The Bernstein filter has **one real false negative** — a root exactly at $s=0$ or $s=1$,
    which is 23 % of rejections on real data because adjacent edges share a vertex. Harmless
    there, a lost crossing for a *non-adjacent* pair. The fix is three lines in the filter and
    the non-adjacency gate on it is load-bearing.
  - ⚠️ The $Q_4$ guard's trigger is **near-straight curves**, not parallel bulges: $Q_4$ goes as
    $\det(D_A,D_B)^2$, which is small when *either* bulge is small. So its threshold is
    **coupled** to the straight-edge one, and finding the bad band needs a *joint* measurement of
    $\lVert D\rVert$ against the companion matrix's condition number — not a count below a fixed
    cutoff. `DECISIONS.md` (30), corrected in the guide.
- **Constructed tests, not random ones**, for every degeneracy. No-false-negatives for
  transversal crossings is a **theorem** (the hull property), so a random-quartic sweep confirms
  something that cannot fail and has no power. Do not write one.


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

## ⚠️ Broken in the environment, not in the tree: `sfdp` / `YifanHuLayout`

**`venv/bin/python -m pytest` reports 830 passed, 2 skipped. The two skips are an upstream
graphviz crash, and they un-skip by themselves once the installed graphviz is new enough.**

```
SKIPPED tests/test_embedding.py::test_every_method_touches_exactly_the_form_it_declares[yifanhu]
SKIPPED tests/test_embedding.py::test_every_method_lays_out_every_representation[yifanhu]
```

`sfdp` corrupts the heap and aborts — `Fatal glibc error: malloc.c:2199 (sysmalloc): assertion
failed`, returncode −6. **graphviz 16.0.0-1 was installed on 2026-09-06**, after this project's
last green run.

Characterised in session 5, in enough detail that nobody need repeat it:

- **Only `sfdp`.** `neato`, `dot`, `fdp`, `twopi` and `circo` handle every input that kills it, so
  only `YifanHuLayout` is affected.
- **It is in the multilevel coarsening.** `-Glevels=0` and `-Glevels=1` survive; `-Glevels≥2`
  aborts. No other attribute changes anything — `quadtree`, `mode`, `overlap`, `smoothing`, `K`
  and `maxiter` all still crash.
- **Minimal reproducer: 5 nodes, 5 edges** — `0--1, 1--3, 1--4, 2--3, 2--4`. 157 of the connected
  graphs on ≤ 7 nodes crash it.
- **It depends on the order graphviz *creates* the nodes in, not on the graph.** That same graph
  survives creation order 0,1,3,4,2 and aborts on 0,1,2,3,4. `nx_pydot` always writes node
  statements in sorted order, so every call from `YifanHuLayout` lands on the failing order — and
  that is why it first looked size-dependent (`complete_graph(10)` fine, `karate_club_graph` not).
- **graphviz 14.1.4 handles all of it**, verified by extracting the cached package and running its
  `sfdp` against every failing input. It is a regression, not a tightening.
- **Upstream has it**: [forum thread](https://forum.graphviz.org/t/sfdp-crash-regression-in-15-1-1/3392),
  GitLab issue #2852 — introduced in 15.1.1, **fixed in 16.1.0**. Arch `extra` still ships
  16.0.0-1. Nothing to report; we are just early. (Our 5-node reproducer is smaller than the
  upstream report's 24-node one, if anyone wants to add it.)

### What was done about it, and what it costs

`tests/test_embedding.py` **probes the backend at collection time** and skips only the `yifanhu`
parameters of the two tests that actually invoke it. Three behaviours, all verified:

| state of `sfdp` | result |
|---|---|
| graphviz 16.0.0 (broken) | 2 skipped, with the reason and the upstream pointer in the message |
| graphviz 14.1.4 on `PATH` | **35 passed, 0 skipped** |
| `sfdp` absent entirely | probe declines to skip; the tests **fail loudly** |

The middle row is the important one: run against a working binary, `YifanHuLayout` passes both
tests unchanged. The layout code was never the problem, and that is now demonstrated rather than
asserted.

The probe is deliberately narrow. It runs the 5-node trigger through the exact `nx_pydot` path
and treats **only a signal death** (`returned code: -N`) as grounds to skip, so a missing binary
or a broken `pydot` fails the way it already does for the `neato` tests instead of quietly
disabling a method forever. And if `sfdp` handles the trigger but a real layout still fails, the
tests run and fail — that would be a *different* bug and should be loud.

⚠️ **What it costs:** while the skip is active, `YifanHuLayout`'s backend has no coverage, so a
regression in it would go unnoticed. Partial mitigation: `test_yifan_hu_is_handed_the_structure
_and_nothing_else` inspects the dot source rather than running the binary, and still runs — the
plumbing is still guarded, only the end-to-end layout is not. **Do not "fix" the skip by changing
`YifanHuLayout`**; the conformance test is doing its job.

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
venv/bin/python -m pytest -q          # 830 passing, 2 skipped (sfdp — see above), 1 deselected
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
| ~~`label.py` is half-built — the silhouette family and the graph-side checks are missing~~ — landed in session 6; **the metric suite of §2 is complete** | discharged |
| `EvaluationContext.release()` exists but nothing calls it, and it now drops three arrays rather than one — tabula's closure is 200 MB and its condensed form another 100 MB. `clear_pairwise_cache()` and `clear_loo_cache()` are the same story | step 7 |
| The neighbourhood metrics are not sweep-consistent on a layout with exact duplicates (`DECISIONS.md`). No registered dataset produces one; a crowding-heavy layout at large λ might | whenever a layout first collapses two points |
| ~~Re-measure the §1.6 fix in quality terms~~ — done: recall separates disjointly over seven seeds, ADR and SNS do not | discharged |
| ⚠️ **Spec §2.5's cost class is wrong, and now it is quantified on both sides.** It says SNS and Sil\* are what will not run on tabula; the largest n is 5007, so the O(n²) closure is 200 MB and `max_n=20000` never trips. Measured: Sil\* **0.58 s** given the shared closure, all four tabula label sets **2.07 s**, Sil 0.60 s, Acc\* 0.49 s, the closure itself 14.9 s **once per dataset** — against crossings' **40 s and +428 MB per layout**. The expensive metrics scale with **m** | a run planner built on it would size the wrong thing: **step 7** |
| Memory runs need **one case per process**: `peak_mem_mb` is a delta from the entry baseline, and glibc does not always return freed memory, so later cases in a batch measure from a higher floor | step 7 |
| A persisted layout must store the dataset's registry **name and params**, never the `Dataset` object | step 7 |
| `common_init` is not in the batch runner — deliberately deferred, since it needs the seed axis and the paired protocol; the column-slicing convention it depends on is already tested | step 8 |
| `openTSNE` and `fa2_modified` not installed | step 9 |
| ⚠️ **Crossing output is big.** A tabula layout produces 2.9 M crossing rows (139 MB as `(k, 6)`). The three layer-3 metrics take `crossings=` so one solve serves all three, but nothing yet *enforces* that, and persisting the array rather than the scalars would be a mistake | **step 7** |
| `QUERY_CHUNK_EDGES = 1024` is the crossing pipeline's memory bound, measured on tabula. A runner that parallelises across layouts multiplies it by the worker count | step 7 |
| A **~9 s, accuracy-free** saving on both crossing variants is measured and left undone: `_swept_segments` rediscovers $(B_0, B_2)$ for curves already flagged as exactly straight — `DECISIONS.md` (37) | whenever someone is in `crossings.py` anyway; otherwise step 7 |
| One legacy `(d, n)` surface: `graph_embedding_all` → `paper_figures` → `plots`, and with it `twin/graphmatrix.py`'s pre-restructure classes and their `.affinity` / `.distance` shim. **Half discharged**: `count_intersections` is now layer 2 and transposes at its own boundary; `drawedges` / `draw_curved_edges` and `graph_embedding_all` are what remain | the rest converges at **step 10**; tracked in `DECISIONS.md` (2) |
| Tabula's Leiden partition is still computed inside `tabula_sapiens_teaser` rather than registered as a `detected:leiden` label set | step 10, where the consumer and the cost land together |
| ~~**Open:** VNR's normaliser~~ — **closed session 4**: the per-vertex median. Still one `[THESIS EDIT]`; eq. (69) never changed | discharged |
| ~~**Open:** `SpectralLayout`'s Laplacian~~ — **closed session 4**: Hall's $D-W$ is the default, Koren's is selectable and reported once, §1.5's solver fix landed with it. §2.2's "hubs collapse inward" is now measured — true on hub graphs, false on the $k$-NN graphs, so **the thesis prose wants a qualifier** | thesis prose, before submission |
| ⚠️ **`SpectralLayout` still does not handle a disconnected graph** — the zero eigenvalue's multiplicity is the component count, so column 0 is not the only trivial vector and the layout would keep component indicators as coordinates. §1.5's other half; §4.4's policy covers metrics, not the layout side | whenever a disconnected dataset joins the roster |
| ~~Two scheduled code divergences from the thesis~~ — **one left**: `fa2` → `fa2_modified`. `count_intersections` versus eq. (78) was reclassified at step 5 as a **standing intentional divergence** for Figure 3, not a fix; §0.5's table now says so | `fa2`: **step 9** |
| Step 9 is lighter than the table says: **§1.5 is done** (session 4), so what remains there is §1.4, §1.7 and the graph *t*-SNE baseline | step 9 |
| Cache loop order: evaluate every metric of one layout before moving to the next, or a two-entry cache evicts on every switch. Nothing enforces it yet | **step 7**, the batch runner |
| **Open, needs the author's call:** VNR's `"mean"` normaliser is **exactly** NND$_1$, so under the `[THESIS EDIT]` the point-crowding column is NND at two values of $k$; the thesis's `"max"` is a genuinely second measurement but is distorted by outlier vertices. Both are implemented, both are in the golden file, and the comparison §0.5 asked for is done — `DECISIONS.md` | before the results tables are designed: **step 11**, but the decision can be taken now |
| The three visual metrics have a measured seed-noise band on `digits` (NND 0.4 %, VNR 1.2 %, `edge_length_cv` 2.4 %). Nothing reads it yet; a λ sweep that reports differences smaller than it would be reporting noise | step 8 |
| ⚠️ **NND and VNR reward uniformity, which a good layout of clustered data does not have** — a random control tops both and comes last on ADR and recall. The reporting layer must not rank on them alone, and the joint reading is not enforced anywhere yet | step 11 |
| `NodeResolutionReport.n_zero_denominator` joins `n_ties` / `n_zero_distance` in the diagnostics table of `DECISIONS.md` (18); nothing collects that table yet. Step 6 adds `SilhouetteReport`'s four class-size columns, and `GraphAccuracyReport`'s `n_isolated` beside its `n_excluded` | step 7 |
| ⚠️ **§4.4 permits `disconnected="mask"` for SNS only, and `silhouette_graph` now refuses it unconditionally** — including on a connected graph, where `lcc` and `mask` give the identical array. A runner sharing one `EvaluationContext` between SNS and Sil\* under `mask` will be told rather than get an inflated number. Two contexts, or `lcc` throughout | **step 7**, the runner |
| ⚠️ **The label-set enumeration is one line but must not be naive**: tabula's `compartment_cell_type` is *exactly* `cell_ontology_class` (`compartment` is a function of it), so scoring every composite emits a duplicate column | step 7 |
| ⚠️ **Sil\* and Acc\* need a reference level printed beside them, and nothing computes one.** Sil\*'s chance is **negative** by a class-count-dependent amount (−0.32 at 10 classes, −0.0004 at 2) and its ceiling is ~0.53 where labels *are* the topology; Acc\* exceeds 0.94 on every dataset and is near-useless without its floor (0.987 against a 0.958 floor on tabula's `method`). A shuffled-label control is three lines; `majority_class_rate` exists | step 7 for the column, **step 11** for the table |
| ⚠️ **§5.3 guard 2 is unbuilt**: Sil\*/Acc\* must be emitted in a *separate* table from Sil/Acc. The docstrings say "label-validity check, not a baseline" and two tests assert the sentence exists, which is all that can be enforced from here | step 11 |
| ⚠️ **Sil needs seed replication to be read at all** — rel IQR 3.6 % and a 19 % spread over seven seeds on `digits`, the noisiest metric measured, against `knn_accuracy`'s 0.24 %. A λ sweep reporting Sil differences below ~4 % on one seed is reporting noise | step 8 |
| ⚠️ **Sil must not be tabulated as "which representation recovers labels best".** It separates the ablation disjointly but orders it *differently* from `knn_accuracy` and `knn_recall` — identical Acc for A_V and A_tne on football where Sil separates them 0.758/0.617, and an inverted A_E/A_tne on digits. Distance-based versus rank-based; §5 must say which question each answers | step 11 |
| ⚠️ **The tabula `donor` silhouette contradicts the "low is good" reading for batch covariates**, because `donor` is confounded with cell type at NMI 0.41. `method` is the covariate the argument applies to. `DATASET_PROVENANCE.md`, spec §3.4 and `DECISIONS.md`'s `LabelSet` note all want the qualifier; the wording is in `DECISIONS.md` (40) | thesis and docs prose, before submission |
| ⚠️ **`notebooks/metrics.ipynb` cannot run at all** — it imports `twin.metadata`, deleted at step 2. It is also the sole consumer of `embedding_metrics_all` / `metrics_to_typ` / `metrics_to_df`, so notebook and functions share a fate | step 11 |
| The other notebooks import modules that have moved; `twin-gui.ipynb` is the only consumer of `plots.application_plot`, which is **wanted** and needs porting to the new interface | step 10 |
| `twin/batching.py` is 1562 lines with one smoke test plus the twelve key-contract tests added in session 4. The rest of the engine is unexercised; CLAUDE.md says not to rewrite it, so the question is what step 7 leans on | step 7 |
| The figure check is a **smoke** test — it catches a raise, not a wrong picture. Real pixel baselines wait for the §3.6 refactor | step 10 |

## Traps

- **All six plan files are history, not instructions** — `step2-datasets.md` through
  `step6-labels.md`. Each records where the implementation departed from it; the departures are
  in `DECISIONS.md`. Step 7 has no plan yet.
- **`sklearn` refuses both degenerate ends of the silhouette**, not just the single-class one:
  its range is $2 \le |\mathcal{L}| \le n-1$. `label.py` returns `nan` for one class and
  **0.0** for all-singletons, which Rousseeuw's convention already implies.
- **The two graph-side label metrics take `labels` first**, departing from spec §2.3's
  `(G, labels)`, because `G` is optional for Sil\* once `ideal` is supplied. Both check that
  `labels` is 1-D, so passing a layout in that slot raises rather than returning a number.
- **`graph_neighborhood_accuracy`'s weighted mode is `"affinity"`, not `"distance"`.** Eq. (62)
  votes with $w_{ij}$; `knn_accuracy`'s `"distance"` applies $f(d) = 1/d$ to a length. Opposite
  quantities, and the metric refuses a `"distance"`-stamped graph under `"affinity"`.
- **Not every plausible memory argument survives measurement.** `silhouette` shipped on
  `sklearn`'s chunked Euclidean route to avoid an $(n, n)$ matrix; `working_memory` defaults to
  1024 MB, so it builds one anyway and the exact `pdist` route is faster *and* leaner at both
  $n = 5007$ and $n = 12000$. Measure one process per route — `ru_maxrss` is a high-water mark,
  so two routes in one process give the second one a free pass.
- **Two per-edge orderings coexist and look identical.** `G.edges()` is the order of record for
  per-edge arrays; ADR alone uses the CSR upper triangle, because its lengths must align with
  the weights from the same matrix. They agree on `karate` and `football_2023` and disagree in
  12377 of 14732 positions on `digits`, so **a test on the small graphs proves nothing**.
- **The leave-one-out cache must never learn to slice.** Serving `k=5` out of a cached `k=16` is
  free, and it would make a metric's value depend on evaluation order whenever a layout has
  exact duplicates. `tests/test_metrics_common.py` asserts both that it does not and that
  slicing would have changed the answer.
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
