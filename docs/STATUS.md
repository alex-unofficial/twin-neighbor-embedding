# Where the work stands

Read `CLAUDE.md` first, then this. Updated at the end of each session.

**Last updated:** end of session 3 (2026-08-24), which closed spec §7 step 3. Branch
`experiment-setup` @ `0974282`.

---

## Position in the spec's §7 ordering

| Step | State |
|---|---|
| **1. Correctness and conventions groundwork** | **Done** — merged `b220cb8`, `e0428de` |
| **2. `Dataset` and `LayoutGraph` (§3.4, §3.5)** | **Done** — session 2, plus the step-2b follow-up; plan: [`docs/plans/step2-datasets.md`](plans/step2-datasets.md) |
| **3. `common.py` + `structural.py` — ADR, kNN family, SNS** | **Done** — session 3; plan: [`docs/plans/step3-metrics.md`](plans/step3-metrics.md), departures in `DECISIONS.md` |
| **4. `visual.py` — NND, NR/VNR, distributions, edge lengths** | **Next** |
| 5. `crossings.py` — curves, detection, Cr/LWCr, detour | not started |
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
2. `docs/EXPERIMENT_SETUP_SPEC.md` **§0.5 first** — it supersedes the rest of the document.
3. `docs/DECISIONS.md` — the open questions the implementer was asked to settle, with reasoning.
4. `docs/DATASET_PROVENANCE.md` — where each dataset's weights actually come from. Assertions
   here are executable: `tests/test_dataset_provenance.py`.
5. `docs/DATASET_REPORT.md` — the generated roster table (`python -m twin.report`).
6. [`docs/plans/step2-datasets.md`](plans/step2-datasets.md) and
   [`step3-metrics.md`](plans/step3-metrics.md) are **history**, kept for their record of where
   the implementation departed from them — the departures themselves are in `DECISIONS.md`.
   There is no plan for step 4 yet.

---

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

### Where step 3 starts

[`docs/plans/step3-metrics.md`](plans/step3-metrics.md) has the commit sequence; the five
decisions the spec left open at this step are pinned in
[`DECISIONS.md`](DECISIONS.md#step-3-decisions-pinned-before-they-are-discovered-the-hard-way).
Two things about the step worth knowing before opening either:

- **Scope is exactly ADR, the kNN family and SNS.** `twin/metrics.py` becomes a package, but
  `visual.py`, `crossings.py`, `label.py`'s content and the whole
  registry / `evaluate_layout` / persistence layer are steps 4–7. The plan states the boundary
  because it is the easiest thing to drift across.
- **Only one commit in it can move a number, and it touches all sixteen existing pins** — the one
  that splits `knn_neighbor_preservation_accuracy`'s `mode=` string into four named functions.
  The pin *keys* become function names there, which is exactly the cover a quiet value change
  would need.

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
venv/bin/python -m pytest -q     # 487 passing
python -m twin.report            # the roster table; regenerates docs/DATASET_REPORT.md with -o
git status --short               # clean; if notebooks appear modified, the filter is unset
```

Figures: the six fast `paper_figures` functions render in seconds; `tabula_sapiens_teaser` takes
about three minutes and internally calls `tabula_sapiens_blockadj`, so rendering it covers both.
All eight render as of `0974282`. (`barabasi_albert` is commented out in the source and is not
one of the eight.)

---

## Carried forward — each with where it bites

| Item | Bites at |
|---|---|
| ~~Ideal distances have no consumer~~ — `common.ideal_distances`, the `max_n` guard and `EvaluationContext` landed in step 3 | discharged |
| ~~`exp_decay` vs `exp_decay_sq` never compared~~ — measured on digits, `DECISIONS.md` | discharged |
| ~~A metric needs the vertex `LayoutGraph`~~ — `twin/metrics/context.py` | discharged |
| SNS on a Kamada-Kawai or neato layout is close to a measure of optimiser success. The caveat is in the SNS docstring; the **reporting flag that acts on it** is not built | step 11 |
| `EvaluationContext.release()` exists but nothing calls it. A batch runner over several datasets must drop the closure between them — tabula's is 200 MB | step 7 |
| The neighbourhood metrics are not sweep-consistent on a layout with exact duplicates (`DECISIONS.md`). No registered dataset produces one; a crowding-heavy layout at large λ might | whenever a layout first collapses two points |
| ~~Re-measure the §1.6 fix in quality terms~~ — done: recall separates disjointly over seven seeds, ADR and SNS do not | discharged |
| ⚠️ **Spec §2.5's cost class is wrong.** It says SNS and Sil\* are what will not run on tabula; the largest n is 5007, so the O(n²) closure is 200 MB and `max_n=20000` never trips. The expensive metrics scale with **m** — crossings, 49368 edges on tabula | a run planner built on it would size the wrong thing: **steps 5 and 7** |
| Memory runs need **one case per process**: `peak_mem_mb` is a delta from the entry baseline, and glibc does not always return freed memory, so later cases in a batch measure from a higher floor | step 7 |
| A persisted layout must store the dataset's registry **name and params**, never the `Dataset` object | step 7 |
| `common_init` is not in the batch runner — deliberately deferred, since it needs the seed axis and the paired protocol; the column-slicing convention it depends on is already tested | step 8 |
| `openTSNE` and `fa2_modified` not installed | step 9 |
| One legacy `(d, n)` surface: `graph_embedding_all` → `paper_figures` → `plots`, and with it `twin/graphmatrix.py`'s pre-restructure classes and their `.affinity` / `.distance` shim | converges at steps 5 and 10; tracked in `DECISIONS.md` (2) |
| Tabula's Leiden partition is still computed inside `tabula_sapiens_teaser` rather than registered as a `detected:leiden` label set | step 10, where the consumer and the cost land together |
| The notebooks import modules that have moved | whenever someone opens one |

## Traps

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
