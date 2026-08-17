# TWIN — working notes

Twin Neighbor Embedding: a pre-embedding transform that builds a graph containing both the
vertices and the edges of the input graph as nodes, then hands it to a standard layout method.
See `EXPERIMENT_SETUP_SPEC.md` for the full background.

## Non-negotiable conventions

- **Array orientation.** Layout arrays are `(d, n)` in `twin/embedding.py` and `(n, d)`
  everywhere in `twin/metrics/`. The single conversion point is `LayoutResult.from_raw()`.
  Never re-derive orientation by inspecting `shape` — it is wrong when `n == d`.
- **Twin block ordering.** `twin_adjacency` produces `[[0, Bᵀ], [B, 0]]`, so rows are
  **[edges (m); vertices (n)]**. Vertices are `X[:, ne:]`, edge points are `X[:, :ne]`.
- **Representations consume affinities.** `VertexMatrix`, `EdgeMatrix` and
  `TwinEmbeddingMatrix` are always built from `dataset.as_affinity()`. Feeding distances to
  `A_tne` inverts the meaning of every term in the polynomial.
- **Edge length in crossing metrics is the chord**, never the arc. This is what keeps the
  straight-vs-curved comparison interpretable.
- **No dataset names in logic.** Nothing outside `twin/datasets/` may branch on a dataset name.
  Dataset-specific behaviour goes on the `Dataset` object.
- **`[THESIS EDIT]` in the spec** means the code intentionally differs from `thesis.pdf`. Do not
  reconcile the code to the PDF.

## Layout of the package

- `twin/datasets/` — `Dataset` objects: graph, weight semantics, labels, style
- `twin/graphmatrix.py` — the three representations (the method; treat as given)
- `twin/twin_embedding.py` — `twin_adjacency` (the method; treat as given)
- `twin/embedding.py` — layout method wrappers
- `twin/metrics/` — the metric suite
- `twin/batching.py` — generic batch-spec engine; domain-agnostic, do not rewrite
- `twin/pipelines.py` — runners tying the above together
- `twin/plots.py`, `twin/paper_figures.py` — rendering
- `batches/` — experiment definitions
- `runs/` — experiment output (manifests, metrics, layouts); not in version control

## Commands

```bash
pytest                       # test suite
python -m twin.report        # dataset connectivity / weight / label report
python batches/<name>.py     # run an experiment batch
```

## Version control

**Branching — `master` is off limits**

- **Never commit to, merge into, rebase onto, reset or force-push `master`.** Merging anything
  into `master` is done manually by the repository owner, never by an agent. Reading `master`
  and branching from it is fine; writing to it is not.
- All work lands on the long-lived integration branch **`experiment-setup`**, branched from
  `master`. If it does not exist yet, create it from `master` and switch to it before making any
  change.
- If separate feature branches are useful (e.g. one per phase of the spec), they branch **from**
  `experiment-setup` and merge back **into** `experiment-setup` — never into `master`.
- Merge a feature branch into `experiment-setup` only when the test suite passes and the
  existing paper figures still reproduce. `experiment-setup` should be usable at all times,
  because it is what gets reviewed before the manual promotion to `master`.
- Land the bug fixes (spec §1) first, as their own branch merged into `experiment-setup`. They
  are independently useful, and several change numerical results — keeping them separate makes
  the before/after attributable.
- When work is ready for promotion, say so and stop. Do not open, merge or land it into
  `master`; leave that to the owner.

**Commit granularity**

- Commit at every known-good state: one self-contained change, tests passing, then commit. The
  point is a working state to revert to when the next change breaks something.
- One logical unit per commit, and name the spec section in the message:
  `metrics: implement SNS (spec §2.1)`. This keeps the spec↔code mapping auditable and makes
  `git bisect` meaningful.
- **Never mix a refactor with a behaviour change.** The dataset restructure (§3.4) and the
  plotting refactor (§3.6) are meant to preserve behaviour; if they share a commit with new
  functionality, a later bisect cannot tell "the refactor broke it" from "the new code is
  wrong."
- When a commit **intentionally changes a metric's numerical output**, say so explicitly in the
  message and update the golden-file fixtures in the *same* commit. Otherwise an intended change
  is indistinguishable from a regression later.

**Experiment runs and reproducibility**

- **Start a results run only from a clean working tree.** The run manifest records the git SHA,
  and a dirty tree makes that SHA a false record of what produced the numbers. Commit or stash
  first.
- **Never rewrite history that a manifest points at.** Once a SHA appears in `runs/*/manifest.json`,
  no `commit --amend`, `rebase` or force-push on that commit — it orphans the provenance.
- Tag the commit whose results are reported (`thesis-results-YYYY-MM-DD`) so the write-up can
  cite an exact state.

**What is and is not committed**

- `runs/` is **output, not source** — add it to `.gitignore`. Batch definitions in `batches/`
  *are* source and must be committed.
- ⚠️ **`.gitignore` currently contains `/data/`.** The existing `.mat` files are tracked (they
  predate the rule, via Git LFS per `.gitattributes`), but **any new dataset file added under
  `data/` will be silently ignored.** When adding a dataset, either `git add -f` it or narrow
  the ignore rule — and verify with `git status --ignored` that it was actually staged.
- Commit the visual-regression baseline PNGs; they are test fixtures. Do not commit ad-hoc
  figures.
- Notebook outputs are stripped by the `jupyter` filter in `.gitattributes`, so a committed
  notebook is not a record of results. Results live in `runs/`.
- `.env` is already ignored; keep it that way.
