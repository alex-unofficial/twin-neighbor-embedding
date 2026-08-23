# Step 2 — `Dataset` and `LayoutGraph`

The executable sequence for spec §7 step 2. **The design is not repeated here** — read
`EXPERIMENT_SETUP_SPEC.md` §3.4/§3.5 (rewritten to the two-source model) and `DECISIONS.md` (11)
first. This file is what to do, in what order, and how to know it worked.

Prerequisite: `docs/STATUS.md` for where things stand and what is carried forward.

---

## Commit sequence

Each commit green and separable; none mixes a refactor with a behaviour change. Branch off
`experiment-setup`, merge back when `pytest` passes and all eight figure functions render.

1. **`tests:` characterisation pins before the restructure.** Derived label vectors (football
   conference/division, tabula cell types, digits targets), the figure colour arrays, and
   per-dataset graph invariants (n, m, weight distribution).

   This is the safety net for everything below, and it is not optional: the equivalent pins in
   session 1 caught an inverted sort in the leave-one-out neighbour selection that would otherwise
   have read as a plausible metric. Pin the data *before* the code that produces it moves.

2. **`datasets:` package skeleton.** `Dataset`, `LabelSet`, `DatasetStyle`, and the stored source
   types (`Stored`, `StoredMatrix`, `Unweighted`). Pure addition, no consumers.

3. **`datasets:` one file per dataset + `DATASET_REGISTRY`.** The `karate` / `karate_weighted`
   split, the `mnist` → `digits` rename, and tabula loading `A`, `D` **and** all four label fields
   (`cell_ontology_class`, `compartment`, `donor`, `method`) plus `X_umap`. Add
   `check_weight_consistency` and its tests here — tabula is the only dataset that exercises it.

4. **`graphmatrix:` `LayoutGraph` and the three mappings.** `vertex_graph(dataset)` is
   `Dataset → LayoutGraph` and is where a missing side is derived; `edge_graph(vertex_graph)` and
   `twin_graph(vertex_graph, alpha, k)` are `LayoutGraph → LayoutGraph`. No labels on the layout
   layer. Carry over `vertex_and_edge_points` and `point_provenance` from session 1, add the
   `parent` link and `is_faithful`, and build the incidence matrix **once** on the vertex graph
   rather than three times.

5. **`results:` `LayoutResult` holds the `Dataset`.** `G` becomes a property. Small and mechanical,
   but it is session-1 code changing, so keep it its own commit and let the existing
   `tests/test_results.py` prove nothing else moved.

6. **`embedding:` `embed(layout_graph, y0)`, delete `mode=`, add `affinity_matrix_to_distance`**
   (normalise, **symmetrise**, then `-log` — decision (12); the un-symmetrised version in
   `KamadaKawaiLayout(mode="similarity")` today is a live bug).

   ⚠️ **Behaviour change.** This is where bug §1.6 actually closes: digits stops being fed
   distances as if they were affinities, so its layouts and figures move. Declare it in the commit
   message and regenerate any affected pins in the *same* commit. Nothing has been written to
   `runs/` yet, so no results are invalidated — say so.

7. **`pipelines:` registry swap.** `batch_registries["graph"]` → `DATASET_REGISTRY`;
   `graph_embedding` takes a `Dataset` and builds the `LayoutGraph`.

8. **`figures:` source labels from `Dataset`, delete `metadata.py`.** A refactor — the label
   vectors and colour arrays pinned in commit 1 must not move. The plotting refactor proper is
   spec step 10; this commit only removes the duplicate label derivation.

9. **`report:` `twin/report.py`** (`python -m twin.report`, which `CLAUDE.md` already advertises).
   Per registered dataset: n, m, density, components, largest-component fraction, **mean degree**
   (the λ criterion in §4.2 is defined at k = mean degree, so it must be readable here), stored
   sources with their `meaning`, the derivation rule and chain for any derived side, weight
   distribution, label sets with **both** roles, `check_weight_consistency` results, and the
   **fraction of vertex pairs at d = 0**. Commit its output — it is a reportable artifact and
   feeds the thesis's experimental-setup section.

10. **`chore:` add `runs/` to `.gitignore`.**

## Traps

- **`batches/batch.py` breaks three ways and only at run time.** It names `karate_club` and `mnist`
  in its graph axis, and sweeps `mode: ['distance','similarity']` on Kamada–Kawai and neato. Fix it
  in commits 6 and 7. It is the least obvious dependency in the tree.
- `paper_figures.py` imports both `metadata` and `dataset`. `dataset.py` cannot be deleted until
  its callers move, and the plotting refactor is step 10 — so it survives this step.
- Nothing in `twin/report.py` may branch on a dataset name (`CLAUDE.md`); every column comes off
  the `Dataset` object.
- Derived graphs must expose **no** labels. A length-`n` label vector reaching a length-`m` node
  set is the failure mode the layering exists to prevent.

## Verification

```bash
venv/bin/python -m pytest -q     # session 1's 94, plus the new tests
python -m twin.report            # the connectivity / weight / label table
```

**The load-bearing check:** metrics run on an `A_E` and an `A_tne` layout using the dataset's own
labels, with no length mismatch. That is the whole point of the layering; if it needs a special
case at the call site, the layering is wrong.

Then:

- `check_weight_consistency` on tabula — global Spearman strongly negative; median per-row Spearman
  at **the measured value**. The argument says −1, but the stored `A` is fuzzy-union symmetrised,
  so measure first and pin what holds, recording symmetrisation as the cause if it is not exactly
  −1. Worse than ≈ −0.9 is the genuine inconsistency signal. Do not hardcode −1 and then loosen the
  test to make it pass.
- Dropping tabula's `D` makes `vertex_graph` **raise** (`meaning="adaptive_kernel"` has no
  derivation rule), not silently transform.
- Round-trip: for a derived side, the two forms compose back to the stored side.
- `Unweighted` distance gives **hop counts** — non-edges absent, not 1.
- Kamada–Kawai on `vertex_graph(tabula)` uses the measured `D`; on `twin_graph(...)` the generic
  fallback, with the chain recoverable by walking `parent`.
- The incidence matrix is constructed once per vertex graph, not once per representation.
- All eight `paper_figures` functions render: the six fast ones, plus `tabula_sapiens_teaser`
  (~3 min), which internally calls `tabula_sapiens_blockadj` and so covers both.
- `git status --short` clean — if notebooks appear modified, the `jupyter` filter is unset
  (`CLAUDE.md`, one-time setup).
