# Where the work stands

Read `CLAUDE.md` first, then this. Updated at the end of each session.

**Last updated:** end of session 1 (2026-08-23). Branch `experiment-setup` @ `e0428de`.

---

## Position in the spec's §7 ordering

| Step | State |
|---|---|
| **1. Correctness and conventions groundwork** | **Done** — merged `b220cb8`, `e0428de` |
| **2. `Dataset` and `LayoutGraph` (§3.4, §3.5)** | **Next** |
| 3. `common.py` + `structural.py` — ADR, kNN family, SNS | not started |
| 4. `visual.py` — NND, NR/VNR, distributions, edge lengths | not started |
| 5. `crossings.py` — curves, detection, Cr/LWCr, detour | not started |
| 6. `label.py` — silhouette, accuracy, graph-side checks | not started (needs step 2) |
| 7. Registry, `evaluate_layout`, `batch_layout_metrics`, persistence | not started |
| 8. λ sweep and selection, seeds, straight/curved variants | not started |
| 9. Bugs §1.4, §1.5, §1.7; graph *t*-SNE baseline | not started |
| 10. Plotting refactor (§3.6) | not started |
| 11. Reporting: aggregation, Typst export, schema guards | not started |
| 12. Dataset groups including the adversarial density case | not started |

**No results run has been produced.** `runs/` does not exist, so nothing downstream is invalidated
by a change that moves numbers. Spec §7: do not produce Section 5 numbers before step 8.

## Read these before starting

1. `CLAUDE.md` — non-negotiable conventions and the version-control policy.
2. `docs/EXPERIMENT_SETUP_SPEC.md` **§0.5 first** — it supersedes the rest of the document, and
   records that §3.4/§3.5 were rewritten to the two-source model.
3. `docs/DECISIONS.md` — the open questions the implementer was asked to settle, with reasoning.
4. `docs/DATASET_PROVENANCE.md` — where each dataset's weights actually come from. Assertions
   here are executable: `tests/test_dataset_provenance.py`.

---

## What landed in session 1

Spec §7 step 1, on `session1/correctness-and-conventions`, merged at `b220cb8`; the `sgtsnepi`
upgrade merged after it at `e0428de`.

- **`twin/results.py`** — `LayoutResult`, the single `(d, n) → (n, d)` conversion point.
  `from_raw()` transposes unconditionally and asserts against the graph; orientation is never
  inferred from `shape`, which is wrong when `n == d`.
- **`twin/metrics.py`** — `require_n_by_d` replaces transpose sniffing; `point_distance_metric`
  deleted (§1.2); `knn_accuracy` / `knn_accuracy_report` reimplemented from thesis eqs. (57)–(60)
  with tie rules, a `d = 0` guard and a majority-class floor.
- **`twin/instrumentation.py`** — `RSSSampler`. Memory is sampled as process RSS, not tracked with
  `tracemalloc`, because the primary backend is a C extension and `tracemalloc` cannot see into it.
- **`twin/pipelines.py`** — `graph_embedding` returns a `LayoutResult` and records `runtime_s`,
  `representation_s`, `n_points`, `peak_mem_mb`.
- **`twin/graphmatrix.py`** — `point_provenance` declared per representation (spec §0.5 decision 5).
- **94 tests**, including characterisation pins that make "behaviour-preserving" checkable.

Three of the spec's six §1 bug claims did not reproduce; §0.5 records the corrections. §1.8 (the
`sgtsnepi` fd-2 bug) was found during the work and is now fixed upstream in 0.7.

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

## Health check

```bash
venv/bin/python -m pytest -q     # 94 passing
git status --short               # clean; if notebooks appear modified, the filter is unset
```

Figures: the six fast `paper_figures` functions render in seconds; `tabula_sapiens_teaser` takes
about three minutes and internally calls `tabula_sapiens_blockadj`, so rendering it covers both.
All eight render as of `e0428de`.

---

## Carried forward — each with where it bites

| Item | Bites at |
|---|---|
| `dataset.tabula()` reads only `A`, discarding `D` and every label field in the same `.mat` | **step 2** — `D` is the measured distance source and the labels are needed by step 6 |
| `karate` / `karate_weighted` split not in the registry; `batch_registries` exposes only the *weighted* graph under the name `karate_club` | **step 2** |
| `twin/report.py` does not exist, though `CLAUDE.md` advertises `python -m twin.report` | **step 2** |
| `runs/` is **not** in `.gitignore`, although `CLAUDE.md` says it should be | **step 7**, when the first run is written |
| Memory runs need **one case per process**: `peak_mem_mb` is a delta from the entry baseline, and glibc does not always return freed memory, so later cases in a batch measure from a higher floor | step 7 |
| `common_init` is not in the batch runner — deliberately deferred, since it needs the seed axis and the paired protocol; the column-slicing convention it depends on is already tested | step 8 |
| `openTSNE` and `fa2_modified` not installed | step 9 |
| One legacy `(d, n)` surface: `graph_embedding_all` → `paper_figures` → `plots` | converges at steps 5 and 10; tracked in `DECISIONS.md` (2) |

## Traps

- **`batches/batch.py` breaks three separate ways in step 2** — it names `karate_club` and `mnist`
  in its graph axis and sweeps `mode: ['distance','similarity']` on Kamada–Kawai and neato. All
  three change. It is the least obvious dependency in the tree and it fails only at run time.
- **`paper_figures.py` imports both `metadata` and `dataset`**, and derives labels inline — the
  `stand[-5:-1]` division slicing, and a second ad-hoc `loadmat` for tabula cell types. Those
  derivations move into `Dataset` in step 2; the plotting refactor itself is step 10.
- **Pin derived data before refactoring it.** The characterisation pins in session 1 caught an
  inverted sort in the leave-one-out neighbour selection that would otherwise have looked like a
  plausible metric. Do the same for the label vectors and colour arrays before moving them.
