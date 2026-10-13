# TWIN — working notes

Twin Neighbor Embedding: a pre-embedding transform that builds a graph containing both the
vertices and the edges of the input graph as nodes, then hands it to a standard layout method.
**Where things stand: [`docs/STATUS.md`](docs/STATUS.md)** — position in the spec's ordering,
environment facts, and what is carried forward. Read it before starting anything.

Background and design record, in reading order:

- `docs/EXPERIMENT_SETUP_SPEC.md` — the specification. **Read its §0.5 first**: it supersedes the
  rest of the document, and records which of its own claims did not survive verification.
- `docs/DECISIONS.md` — the questions the spec left open, settled, with reasoning.
- `docs/DATASET_PROVENANCE.md` — where each dataset's weights actually come from. Its claims are
  executable (`tests/test_dataset_provenance.py`), not prose.

## Non-negotiable conventions

- **Array orientation.** Layout arrays are `(d, n)` up to and including the `embed` call in
  `twin/embedding.py`, and `(n, d)` from `LayoutResult` onwards — which is everything in
  `twin/metrics/`. `LayoutResult.from_raw()` is the conversion point. Never re-derive
  orientation by inspecting `shape` — it is wrong when `n == d`.
  - ⚠️ **One legacy `(d, n)` surface remains**, tracked in `docs/DECISIONS.md` (2):
    `pipelines.graph_embedding_all` still returns the old dict of `(d, n)` arrays, and
    `paper_figures.py` plus `plots.py` (`count_intersections`, `drawedges`,
    `draw_curved_edges`) consume it. It converges in two steps: `count_intersections` is
    reimplemented on `crossings.py` at spec step 5, and the rest disappears with the plotting
    refactor at step 10. Until then that path is `(d, n)` end to end — do not mix the two, and
    do not add new callers of `graph_embedding_all`.
- **Twin block ordering.** `twin_adjacency` produces `[[0, Bᵀ], [B, 0]]`, so rows are
  **[edges (m); vertices (n)]**. Vertices are `X[:, ne:]`, edge points are `X[:, :ne]`.
- **`Dataset` and `LayoutGraph` are separate layers, and only one of them has labels.**
  A `Dataset` is stored information — source of truth, labels, drawing instructions — and is never
  laid out. A `LayoutGraph` is an affinity form, a distance form, and a way to derive vertex/edge
  points; it is laid out and is **never labelled**. A_E's nodes are edges and A_tne's are edges
  *and* vertices, so a length-`n` label vector must never reach them. Labels are applied after
  layout, because `vertex_and_edge_points` returns points in the **original** indexing.
  - **A metric or a figure is a function of the embedded points plus a `Dataset`.**
  - Derivation happens in `Dataset → LayoutGraph` (the vertex graph) and nowhere else. `Dataset`
    holds no derived information.
- **Derived graphs consume affinities.** `edge_graph` and `twin_graph` are built from the vertex
  graph's *affinity* form, always. Feeding distances to `A_tne` inverts the meaning of every term
  in the polynomial.
- **One support, three valuations.** A `LayoutGraph` exposes `combinatorial` (1), `affinity` (w)
  and `distance` (d) over the same edge set, and a layout method declares which it reads.
  ⚠️ Declaring it right is **not enough**: a backend that serialises its input can still
  misread it — graphviz takes `weight` as a spring strength and `len` as a target length, so a
  distance sent as `weight` inverts the sign. Check what arrives, not just what was asked for.
- **Sparsity is declared in the dataset factory, never in `vertex_graph`.** The graph's edge set
  defines the original edge indexing, so a layout-layer knob must not change it. `k` is a
  declared budget parameter, and `KNearest.check()` verifies the declaration against the graph.
- **A metric takes its graph from the `EvaluationContext`, not from `dataset.graph`.**
  `ctx.affinity` is affinity-valued on every dataset; `dataset.graph`'s `weight` attribute is a
  stored *distance* on digits. ADR and `weighted_recall` refuse a graph stamped
  `twin_side="distance"`, and `ideal_distances` refuses one stamped `"affinity"` — but `tabula`'s
  graph is unstamped, so the refusal is a backstop and not a substitute for asking the context.
  - ⚠️ **That is for a metric that reads a valuation. A per-edge metric takes `result.G`.**
    `result.E` is indexed by `dataset.graph.edges()`, and `ctx.affinity` is rebuilt from a CSR
    matrix, so its `edges()` are in CSR order — they disagree on `digits` in over 10 000 of
    14 732 positions and agree on `karate`. Handing the context's graph to the crossing family,
    `detour_ratio` or `edge_lengths` pairs each edge's curve with another edge's point, silently.
    None of them reads a weight. `DECISIONS.md` (45c).
- **Ideal distances come from `vertex_graph(dataset).distance`** — the least-derived form —
  computed once per dataset by the runner and shared by every representation. Never from
  `edge_graph`/`twin_graph`'s generic fallback: that would score A_V, A_E and A_tne against three
  *different* targets and make the central ablation meaningless. The all-pairs closure belongs
  in `common.ideal_distances`, deliberately **not** as a `LayoutGraph` property.
- **The graph is the stipulated ground truth.** For a point-cloud dataset the target is a graph
  geodesic, not the ambient metric; a layout is not expected to recover distances the method was
  never shown. What that costs is measured per dataset, not asserted — `DECISIONS.md`.
- **Edge length in crossing metrics is the chord**, never the arc. This is what keeps the
  straight-vs-curved comparison interpretable. It is computed by `metrics.visual.edge_lengths`;
  the arc-over-chord ratio is `detour_ratio`, a separate metric at step 5.
- **Per-edge arrays are indexed by `G.edges()`, not by the CSR upper triangle.** Edge points,
  incidence-matrix columns, `edge_lengths` and step 5's per-edge crossing weights all use
  `G.edges()` order. ADR deliberately uses the CSR order instead, because its lengths must align
  with the weights it reads out of the same matrix — the two are **not** interchangeable: on
  `digits` they disagree in 12377 of 14732 positions, the same multiset in a different sequence.
  They agree on `karate` and `football_2023`, so a test on either proves nothing.
- ⚠️ **NND and VNR are never read as standalone "higher is better" scores.** They reward
  *uniform* point spacing, and density variation — tight clusters with space between them — is
  the signature of a good layout of clustered data, so they penalise it by construction.
  Measured on `digits`: a random Gaussian control is **first** on both (NND$_{16}$ 0.0952, VNR
  0.0306) and **last** on ADR (0.9968, the value expected of a layout unrelated to the graph)
  and recall (0.0090, chance). Report them beside the structural metrics; never rank on them
  alone. `node_resolution` (NR) is not a reported metric at all — it is the foil VNR is defined
  against.
- **No dataset names in logic.** Nothing outside `twin/datasets/` may branch on a dataset name.
  Dataset-specific behaviour goes on the `Dataset` object.
- **`[THESIS EDIT]` in the spec** means the code intentionally differs from `thesis.pdf`. Do not
  reconcile the code to the PDF.
- 🚫 **`thesis/` is READ ONLY.** Every file under it — the `.typ` sources, `ref.yaml`, `images/`,
  `thesis.pdf` — is the author's writing. Read it freely, cite it, measure the code against it,
  and quote it in the design record. **Never create, edit, delete or reformat a file there**, and
  do not build it in place (a compile writes `thesis.pdf`; if you must compile to check
  something, send the output to a scratch path). This holds even when a finding clearly belongs
  in the thesis and even when asked to "put it in §N": write the finding, the numbers and the
  suggested wording into `docs/DECISIONS.md` and say where it belongs, and leave the prose to
  the author. An agent drafting in someone's thesis is the one edit that cannot be reviewed by
  running the tests.

## Layout of the package

Present:

- `twin/datasets/` — `Dataset` objects: source of truth, labels, style, declared sparsity; one
  file per dataset plus `registry.py`. Adding a dataset is one file and one registry line.
- `twin/layoutgraph.py` — `LayoutGraph` and the three mappings `vertex_graph` / `edge_graph` /
  `twin_graph`; the single place a missing affinity or distance side is derived
- `twin/results.py` — `LayoutResult`; the single `(d, n) → (n, d)` conversion point
- `twin/report.py` — the connectivity / weight / label report (`python -m twin.report`)
- `twin/instrumentation.py` — `RSSSampler`; runtime and memory for the embed step
- `twin/twin_embedding.py` — `twin_adjacency` (the method; treat as given)
- `twin/embedding.py` — layout methods and `EMBEDDING_REGISTRY`
- `twin/forms.py` — which valuation a bare `nx.Graph` carries (`twin_side` / `twin_source`),
  stamped by both the `Dataset` and `LayoutGraph` layers and read by the metric guards.
  Dependency-free on purpose: it must not pull in `sgtsnepi`
- `twin/metrics/` — the metric suite as a package: `common` (ideal distances, the condensed pair
  ordering, the `pdist` and leave-one-out caches, `default_k`), `structural` (ADR, the kNN
  family, SNS), `visual` (NND, NR/VNR, the two distance distributions, edge lengths),
  `label` (silhouette, kNN and graph-side accuracy), `crossings` (curves, the exact crossing
  solve, Cr/LWCr, detour), `context` (`EvaluationContext`, and the condensed ideal it shares),
  `reference` (the permutation nulls, Cramér's V and the compactness ceiling Sil\*/Acc\* are
  read against), `registry` (`MetricSpec`, `METRIC_REGISTRY`, the metric-set digest) and
  `evaluate` (`evaluate_layout` per case, `evaluate_graph_side` per dataset).
  Each cache family has its own capacity and its own `clear_*` — 1.6 MB per leave-one-out entry
  is not 100 MB per condensed vector — and **the leave-one-out cache never serves a small `k`
  out of a larger cached one**, because slicing is not re-querying on a layout with duplicates.
- `twin/batching.py` — generic batch-spec engine; domain-agnostic, do not rewrite
- `twin/pipelines.py` — `graph_embedding` (one case) and the legacy runners
- `twin/runs/` — persistence: `keys` (the `v1-` case id, row schemas, atomic shards — the
  versioned on-disk contract), `layouts` (the content-addressed layout store), `store` (a run
  directory), `manifest`, `provenance`, and `runner` (`batch_layout_metrics`: the batch grammar
  through layouts and the registry, in memory or persisted and resumable). **Source**; the
  repository-root `runs/` is its output
- `twin/plots.py`, `twin/paper_figures.py` — rendering
- `twin/graphmatrix.py` — the **pre-restructure** representation classes, plus `compute_slopes`.
  Superseded by `twin/layoutgraph.py`; alive only because `graph_embedding_all` and
  `paper_figures` still use it. **Do not add callers**; deleted at step 10.
- `twin/dataset.py` — raw graph factories, called by `twin/datasets/`. Not a dataset layer.
- `batches/` — experiment definitions
- `tests/`, `docs/`

Output, not source, and ignored by git:

- `runs/` — `runs/layouts/<key_id>.npz` (shared by every run) and one directory per run
  (manifest, the three tables, per-case shards, log). Tests write to `tmp_path`, **never** here.

## Commands

```bash
venv/bin/python -m pytest    # test suite (excludes `-m slow`)
venv/bin/python -m pytest -m slow   # the opt-in extras: the ~3 min tabula figure
python batches/<name>.py     # run an experiment batch
python batches/smoke.py      # the whole registry on two small datasets → runs/smoke/ (~30 s)
python -m twin.report        # dataset connectivity / weight / label report
```

**One-time setup per clone.** Git filters live in `.git/config` and cannot be set from the
repository, so a fresh clone must point the notebook filter at the installed binary:

```bash
git config filter.jupyter.clean venv/bin/nbstripout-fast
```

Without it git cannot run the `jupyter` clean filter from `.gitattributes`, every notebook shows
as permanently modified, and the "start a results run only from a clean working tree" rule below
becomes unenforceable.

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
  existing paper figures still reproduce — the second half is now `tests/test_paper_figures.py`
  rather than a manual step, though the three-minute tabula figure stays opt-in (`-m slow`).
  `experiment-setup` should be usable at all times, because it is what gets reviewed before the
  manual promotion to `master`.
- ~~Land the bug fixes (spec §1) first~~ — done, merged at `b220cb8`.
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

- `runs/` is **output, not source**, and is in `.gitignore`. Batch definitions in `batches/`
  *are* source and must be committed. A persisted layout stores the dataset's registry **name
  and factory params**, never the `Dataset` object — that would write every label vector and,
  for tabula, two ~50k-edge matrices into each layout file, once per representation, per seed,
  per λ.
- ⚠️ **`.gitignore` currently contains `/data/`.** The existing `.mat` files are tracked (they
  predate the rule, via Git LFS per `.gitattributes`), but **any new dataset file added under
  `data/` will be silently ignored.** When adding a dataset, either `git add -f` it or narrow
  the ignore rule — and verify with `git status --ignored` that it was actually staged.
- ⚠️ **There are no visual-regression baseline PNGs, and there is no pixel comparison.** An
  earlier version of this line said to commit them as test fixtures; none exist, and every
  tracked PNG is coil-20 *input data*. What exists is `tests/test_paper_figures.py`, a **smoke**
  test: each figure function must return a figure without raising, which catches the failure
  mode that actually occurs here (a signature change, a renamed dataset, a deleted parameter)
  but would pass an empty or wrong picture. Real baselines belong at **step 10**, with the §3.6
  plotting refactor — committing them now would pin output that is about to be rewritten. Do not
  commit ad-hoc figures in the meantime.
- Notebook outputs are stripped by the `jupyter` filter in `.gitattributes`, so a committed
  notebook is not a record of results. Results live in `runs/`.
- `.env` is already ignored; keep it that way.
