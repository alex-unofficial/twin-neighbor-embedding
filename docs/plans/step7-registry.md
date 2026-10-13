# Step 7 — the registry, `evaluate_layout`, and persistence

The executable sequence for spec §7 step 7. Read `CLAUDE.md`, then `docs/STATUS.md`, then spec
§2.5, §3.7, §4.1, §4.6 and §5.3, then `DECISIONS.md` (18), (20), (21), (22) and (41). This file is
what to do, in what order, and how to know it worked; the design record is `DECISIONS.md`.

**Approved and not yet started.** Five decisions were taken by the author at review, four more
questions were resolved by inspection rather than preference, and four facts were checked against
the tree; all three sets are below, so none of it is re-litigated during implementation. **There
are no open items.**

---

## Context

**The §2 metric suite is complete and nothing consumes it.** Every metric exists, every one has a
`*_report` sibling carrying its diagnostics, `EvaluationContext` was built at step 3 explicitly as
this step's noun — and there is still no way to produce a table. `batches/batch.py` builds a dict
in memory and exits. **No results run has ever been produced**, so nothing downstream is
invalidated by what this step decides; after it, everything is.

Step 7 turns the suite into rows on disk. It is the widest step so far and the first that writes
a **versioned** artifact: once `v1-` key ids are in a run directory, changing the encoding
invalidates every run that used it. That is why the contract comes first and why the step splits.

Three things from session 6 constrain it, and none of them existed when §2.5 was written:

- **Reference levels must travel with the scores.** Sil\* 0.33 is meaningless without the
  permutation null, the Leiden compactness ceiling and Cramér's V against the primary; Acc\*
  exceeds 0.94 everywhere and is near-useless without `majority_class_rate`. `twin/report.py`
  computes all four. If step 7's rows do not carry them, step 11 inherits unreadable numbers.
- **The registry must key *variants*, not just metrics** — `@uniform` / `@affinity` / `@first`,
  `@straight` / `@curved`. That is what defers the Acc\* headline question to step 11 ((41)) and
  it is **already the convention in `tests/fixtures/metric_golden.json`**, whose 44 keys include
  `detour_ratio@curved_mean` and `edge_lengths@mean`. The registry reproduces those keys; it does
  not invent a naming scheme.
- **`Dataset.degenerate_composites()` must be applied at enumeration**, or tabula emits
  `compartment_cell_type` as a duplicate column of `cell_ontology_class`.

---

## What was settled before planning

### Five decisions taken by the author at review

1. **Layouts are a shared content-addressed store.** `runs/layouts/<key_id>.npz`, with the
   producing git SHA recorded per layout; `runs/<run_id>/` holds metrics, manifest and log only.
   §4.6's tree puts layouts inside the run dir, which contradicts (22) — adding a metric means a
   new SHA, a new `run_id`, an empty `layouts/`, and every embed re-runs, which is the exact
   thing (22) says layouts exist to prevent. **Departure from §4.6, recorded in DECISIONS.**
2. **Two merges, A then B**, split at "does it create a versioned on-disk artifact". A is
   runnable and produces the project's first real metric table; B owns every irreversible
   mistake.
3. **`disconnected="lcc"` only; refuse everything else at the runner.** Every registered dataset
   is connected, so §4.4's SNS-only `mask` allowance is inert, and `silhouette_graph` already
   refuses `mask` unconditionally — building two-contexts-per-dataset plumbing for it would be
   untested code for a case that does not occur. §0.5 decision 3's precedent. Record the policy
   in the manifest; the allowance arrives with the first disconnected dataset.
4. **Memory is recorded in the batch loop, with the mode flagged.** §3.2 wants peak memory per
   embed inside the run loop; `STATUS.md` says memory needs one case per process, because
   `peak_mem_mb` is a delta from an entry baseline and glibc does not return freed memory, so
   later cases measure from a higher floor. ⚠️ Those cannot both hold in a single-process
   sequential runner, and reusing persisted layouts cannot fix it — memory has to be measured
   *during* the embed, so an isolated pass is a second full embed pass, the expensive half of a
   run. **So: keep per-embed sampling, and add `rss_baseline_mb` and `rss_peak_mb` beside the
   delta plus `mem_measurement: "batch" | "isolated"` in the manifest.** The drift becomes
   visible post hoc instead of averaged into a thesis table, and a later isolated pass can
   overwrite the columns for the cases that matter. Two additive fields on `LayoutResult`.
5. **`k` is `default_k(dataset)` only; a sweep is an opt-in flag.** §0.5 decision 6 fixes
   $k$ = mean degree of $G$ and §4.2's λ criterion is defined at that $k$ — 5 on
   `karate_weighted`, 11 on `football_2023`, 16 on `digits`, 20 on `tabula`, 55 on
   `sbm_er_3modes`. The headline run uses the single value per dataset; `k=(...)` is a flag for a
   robustness appendix. ⚠️ Recorded in the manifest **either way**, since a reader cannot tell
   from the rows which it was. A fixed ladder was rejected because mean degree is 55 on
   `sbm_er_3modes`, so the same ladder means different things per dataset.

### Four questions resolved by inspection, not preference

- ⚠️ **Moving the reference levels out of `twin/report.py` is *forced*, not optional.**
  `report.py` imports *from* `twin.metrics` (at three call sites), so a `twin/metrics/registry.py`
  that imported `twin.report` would be metrics → report → metrics — a cycle that survives today
  only because report's imports are function-local, and backwards layering regardless: the metric
  layer must not depend on the reporting layer. Hence commit A4 below, and it is a **refactor
  commit of its own** (CLAUDE.md: never mix a refactor with a behaviour change). It is provable
  because `tests/test_report_numbers.py` pins every value it moves.
- **`distribution_policy="off"` is safe**, and the worry that it would leave §5 with nothing to
  tabulate is unfounded: the golden file already pins `@min` / `@median` / `@max` for **both**
  distributions. They get `reductions=("min", "median", "max")`, so the summary rows always land
  in the table and the policy governs only whether the raw 100 MB array is *also* written.
- **`run_id` is caller-settable**, defaulting to timestamp + short SHA — forced by decision 1,
  since `--resume <run_id>` cannot work otherwise.
- **The smoke output goes to `runs/`**, which is gitignored at `/runs/`. ⚠️ With one rule: **tests
  use `tmp_path` and never `runs/`**, or a test run pollutes a real run directory.

### Four facts checked against the tree before planning

- **Every metric's `X` is the vertex layout.** Nothing in `twin/metrics/` takes the edge point
  cloud as its `X`; `E` is consumed only as *geometry*, by the crossing family. So
  `applies_to=("V",)` is right, and it is load-bearing: a pairwise metric on tabula's `E`
  (m = 49368) is a 9.7 GB condensed vector, i.e. a hard OOM rather than a slow case.
- **`detour_ratio` returns `(m,)` and is a reported metric** (§4.3 — it is what separates "the
  transform reduces crossings" from "the drawing style does"). (20)'s binary swept-or-distribution
  cannot express it, and the golden file already pins `@curved_mean` / `@straight_max`. Hence
  declared **reductions**.
- **Fixed kwargs are deliberately absent from `BatchKey`** — pinned by
  `tests/test_batch_key.py::test_a_fixed_kwarg_is_not_in_the_key`. ⚠️ So two runs at different
  seeds produce **identical** keys today, and `("sgtsne", {"lambda_par": 10})` is fixed while
  `("sgtsne", {"lambda_par": [0.01, 0.1]})` is swept. This is why (21) has `extra`, and why
  `extra` must be the **whole fixed set**, not the two names someone remembered.
- **`twin/batching.py` has no persistence, no resume, no `key_id`, no file I/O.** `key_id` and
  two-stage resume are specified in (21)/(22) and referenced by test docstrings; nothing
  implements them.

---

## Half A — the suite runs and produces a table

Mergeable on its own. Its only durable output is a regenerable parquet.

### A1. `metrics: the on-disk contract — key_id, row schemas, atomic shards`

New module `twin/runs/keys.py` (and `twin/runs/__init__.py`). Everything versioned under `v1-`
lands here together, so nothing downstream can drift from it.

```python
def key_id(key: BatchKey, **extra) -> str            # "v1-{slug}-{digest}", (21) verbatim
def canonical_json(key, extra) -> str                # digest input only
def key_to_json(key) -> str                          # type-preserving, round-trips
def key_from_json(s) -> BatchKey
METRIC_ROW_SCHEMA: pa.Schema
DIAGNOSTIC_ROW_SCHEMA: pa.Schema
def write_shard(path, rows, *, schema) -> None       # temp file in the SAME dir, os.replace
def shard_is_current(path, *, metric_set_digest) -> bool
```

⚠️ **Two serialisers, not one.** (21)'s canonical form is a *digest input*: `repr(float(v))` maps
`1` and `1.0` both to `"1.0"` on purpose, and `bool` to `"true"`. Decoding that back gives
`embedding.lambda_par == "1.0"`, a string, and silently stringifies the results table. The
type-preserving encoder is a separate function and the docstring says which is which.

⚠️ **Shards carry a *fixed* schema.** `key_to_flat_record` emits `embedding.linlog` only for the
`forceatlas2` cases, so per-case shards would each have a different column set and concatenation
would promote to `object`. The shard columns are fixed — `key_id`, `slug`, `key_json`, the three
axis-name columns, `point_set`, `label_set`, `metric`, `variant`, `k`, `value`, `status`,
`skip_reason`, `n_excluded`, `metric_set_digest` — and the per-axis parameter columns are
materialised **at concat time** by decoding `key_json` and running `key_to_flat_record`, where the
full column union is known. That is §3.7's "use them; do not hand-roll" honoured, without ragged
shards.

`extra` is built mechanically, not enumerated: per axis,
`{k: v for k, v in choice["kwargs"].items() if k not in dict(choice["key_params"])}` namespaced as
`embedding.lambda_par`, plus the runner scalars (`seed`, `d`, `normalize`, `common_init`). That
closes the whole collision class rather than two instances of it, and is compatible with (21),
which specifies `extra`'s *encoding* and not its membership.

**Tests, in this commit:** (21)'s normalisation table case by case; `1` and `1.0` producing the
same id; two cases whose slugs truncate identically getting different ids; a round-trip preserving
types; a shard whose digest no longer matches being re-run; a half-written shard not satisfying
`shard_is_current`.

### A2. `metrics: MetricSpec and METRIC_REGISTRY (spec §2.5)`

New `twin/metrics/registry.py`. **Declaration only — no driver.**

```python
@dataclass(frozen=True)
class MetricSpec:
    fn: Callable
    bind: Callable              # (ctx, result, variant, params) -> (args, kwargs)
    kind: Literal["structural", "visual", "label", "crossing"]
    needs: tuple[str, ...]      # providers, not argument names
    report_fn: Callable | None = None
    value_field: str | None = None
    sweeps: tuple[str, ...] = ()
    returns: Literal["scalar", "distribution"] = "scalar"
    reductions: tuple[str, ...] = ()
    swept_diagnostics: tuple[str, ...] = ()
    variants: tuple[str, ...] = ()
    applies_to: tuple[str, ...] = ("V",)
    cost: Literal["cheap", "quadratic", "crossing"] = "cheap"
    within_graph_only: bool = False
    graph_side: bool = False
    version: int = 1
```

Five fields the spec's sketch does not have, each forced by something measured:

- **`bind`** — a signature match on `needs` is impossible. The argument orders are deliberately
  inconsistent and pinned that way: `adjacency_distance_ratio(X, G)`, `silhouette(X, labels)`,
  `graph_neighborhood_accuracy(labels, G)` (labels-first is (39c), decided on purpose),
  `crossing_number(G, V, E)`. The binder is the adapter, and the conformance test asserts
  `inspect.signature(fn).bind(*args, **kwargs)` succeeds on its output.
- **`report_fn` / `value_field`** — bind the `*_report` sibling wherever one exists. It is free
  (`knn_accuracy` literally calls `knn_accuracy_report` and returns one field) and it carries
  `n_excluded`, `n_ties` and `majority_class_rate`, which (40) says is mandatory beside Acc.
- **`swept_diagnostics`** — for the k-swept family the report's diagnostic fields are *arrays over
  k* (`NeighborPreservationReport.n_excluded`) while others are scalars. Sniffing by shape is what
  (20) forbids in spirit; declare it.
- **`reductions`** — see above, and they are **read off the golden file rather than chosen**:
  `detour_ratio` gets `("mean", "max")`, `edge_lengths` `("mean",)`, and both distance
  distributions `("min", "median", "max")`. That reproduces 11 of the 44 pinned keys exactly, and
  it is what makes `distribution_policy="off"` safe.
- **`version`** — see A6's digest. ⚠️ Not a hash of the function source: a docstring edit would
  invalidate every shard.

**`needs` names providers, not arguments.** `crossing_number`, `crossings_per_edge_pair` and
`length_weighted_crossings` all take `crossings=` precisely so one solve feeds three — 40 s and
+428 MB on tabula, so getting it wrong triples the dominant cost. A flat registry of independent
entries cannot express that; `needs=("crossing_pairs",)` can.

**Tests:** binder-versus-signature conformance for every entry; **every name in
`twin.metrics.__all__` is either registered or on an explicit exclusion list with a reason** (the
mirror of `tests/test_metrics_surface.py`'s one-sided pin — otherwise a metric is simply never run
and nobody notices until §5); and **every key in `metric_golden.json` is produced by some registry
entry**, which catches a renamed variant immediately.

### A3. `metrics: the per-case provider cache and evaluate_layout (§3.7)`

```python
def evaluate_layout(result: LayoutResult, ctx: EvaluationContext, *, specs=METRIC_REGISTRY,
                    k=None) -> tuple[list[dict], list[dict]]
```

⚠️ **Signature departs from §3.7, which takes `(result, dataset, metric_spec)`.**
`LayoutResult.dataset` *is* the dataset, so passing it again creates a path where a rehydrated
layout is scored against a different dataset object — exactly what resume introduces. Take the
context instead: it carries the dataset, the faithful vertex graph, the shared `k` and the shared
ideal, and it is where the policy refusal lives. Assert `ctx.dataset is result.dataset`, or
name-and-params equality after rehydration.

A small **provider cache** sits beside it, per `(case, variant)` — the per-case analogue of
`EvaluationContext`, and a *second object*, because adding a per-case field to the frozen
per-dataset context re-couples the layers step 2 separated. First provider is `crossing_pairs`.
Variant encoding: `straight` is `E=None`, `curved` is `E=result.E, gamma=0.5`.

Also here: `status` of `ok` / `skipped` / `distribution` / **`error`**, with a `skip_reason`
column. A 40 s crossing metric raising on one case must not abort an eight-hour batch and must not
be silently absent either; and §5.3 guard 3 ("no silent caps") requires the exporter to say *why*
rows are missing, which one token cannot carry. Per-metric `elapsed_s` always (free); RSS sampling
only when `cost != "cheap"` — the first real use of §2.5's cost class, and the justification for
the field. Both go to the **diagnostics** frame, not the metric frame ((18)).

Label-set enumeration is `label_sets_by_scoring_role` plus `composite_of`, **minus
`degenerate_composites()`**, with the drop recorded (§5.3 guard 3 again).

### A4. `metrics: move the reference levels out of report.py (refactor, no behaviour change)`

⚠️ **Required, not optional** — see the inspection note above. `PermutationNull`, `cramers_v`,
`compactness_ceiling`, `score_label_sets` and `_permutation_count` move from `twin/report.py` to
new `twin/metrics/reference.py`; `report.py` imports them from there.

**A refactor commit with no behaviour change**, and `tests/test_report_numbers.py` is what makes
that provable: it pins every Sil\*, Acc\*, ceiling and Cramér's V the module produces, so the move
either reproduces them or fails loudly. Do not take the opportunity to change anything — in
particular the Leiden seed stays `CEILING_SEED = 10` matching `paper_figures`, and
`_permutation_count` keeps its `200000 // n` clamp.

### A5. `metrics: the dataset-level graph-side table (§5.3 guard 2)`

⚠️ **Sil\*, Acc\* and the four reference levels are functions of `(dataset, label_set, policy)`
and of nothing in the layout.** Emitting them per case would repeat the identical value once per
representation × embedding × seed — and at tabula's 39 permutation draws across four label sets
that is ~145 s *per case* instead of once per dataset. They get their own table keyed by
`(dataset, params, label_set, metric, variant)`, computed once alongside the context. §5.3 guard 2
then falls out of the structure rather than being enforced by a rule.

**Do not reimplement them** — import from `twin/metrics/reference.py`, which A4 just created.
All five are pure over `(dataset, ideal, affinity)` and return dataclasses.

### A6. `runs: the metric-set digest, and a one-dataset smoke script`

The digest over `{name, variant, params, version}` for every registered entry — what A1's
`shard_is_current` compares. ⚠️ **`version` is why this is not just names:** without it, editing
`silhouette`'s internals and re-running resumes onto stale values, which is the failure mode the
golden-file rule exists to prevent, except the stale numbers reach the results table. Tie bumping
it to the existing rule — a commit that intentionally changes a metric's output regenerates
`metric_golden.json` *and* bumps that metric's version — and make it enforceable with a test
cross-referencing the two.

Then `batches/smoke.py`: `karate_weighted` and `football_2023`, all three representations, two
cheap embeddings, writing **one** parquet. No shards, no npz, no run dir, no resume. This is the
first real metric table the project has produced and it is what makes Half A reviewable.

---

## Half B — it survives being killed

### B1. `runs: layout persistence, content-addressed`

`runs/layouts/<key_id>.npz` — `V`, `E`, `slopes`, the scalar instrumentation (now
`runtime_s`, `representation_s`, `n_points`, `peak_mem_mb` **and** `rss_baseline_mb` /
`rss_peak_mb` per decision 4), **plus a JSON metadata blob as a 0-d string array**: `point_provenance` (§0.5 decision 5 needs it in the rows),
the derivation chain (§4.6's manifest needs it), the dataset registry **name and factory params**,
and the producing git SHA. ⚠️ Never the `Dataset` object, and never `allow_pickle`. Temp file in
the **same directory** — `os.replace` is atomic only within a filesystem.

Rehydration goes through `DATASET_REGISTRY` by name + params.

### B2. `runs: the manifest`

Every field §4.6 lists, plus the ones this plan adds:

- `disconnected: "lcc"` — always, per decision 3, and the runner raises on anything else;
- `distribution_policy: "off" | "histogram(n)" | "full"`, default `off`;
- `mem_measurement: "batch" | "isolated"` — per decision 4, so a drifting baseline is
  attributable rather than invisible;
- `k_per_dataset: {name: [...]}` **and** `k_source: "default_k" | "explicit"` — per decision 5.
  Both, because the rows cannot say which was used and `default_k` is a function of the graph;
- `metric_set_digest`, and the per-metric `version` map it was computed from;
- `dropped_composites: [{dataset, composite, collapses_to}]` — §5.3 guard 3, no silent caps;
- the **list** of contributing `(sha, timestamp)` pairs rather than one, which is what the
  content-addressed store makes necessary and honest.

⚠️ **Distribution policy defaults to `off`.** `pairwise_distance_distribution` on tabula's `V` is
12.5 M float64 = 100 MB per case before compression; three representations × several embeddings ×
five seeds is tens of gigabytes of npz for a diagnostic §0.5 explicitly declines to score.
`off` / `histogram(bins)` / `full`, recorded in the manifest.

### B3. `runs: batch_layout_metrics — the runner`

```python
def batch_layout_metrics(batch_input, *, registries=batch_registries, run_id=None,
                         resume=None, specs=METRIC_REGISTRY, ...) -> pd.DataFrame
```

**Drops one level below `run_batch`, and that is *using* `batching.py`, not rewriting it.**
`run_batch_cases` computes the key inside its loop and immediately calls the runner, so there is
no hook at which to skip a persisted case; and `batch_graph_embedding` accumulates every
`LayoutResult` in a dict, which is the opposite of what an overnight batch wants. So:
`expand_batch` → sort → `make_batch_key(axis_order=["graph","representation","embedding"])` → own
loop. ⚠️ Two things are lost and must be re-implemented deliberately: the `duplicate_policy`
check (a real guard) and progress reporting.

⚠️ **Order the cases dataset-major.** `expand_batch` concatenates groups, and `batches/batch.py`
has two groups per block over the same `common.graph` list — so a naive "current dataset" cache of
size 1 builds, releases and rebuilds tabula's 14.9 s closure. Results are order-independent, so a
stable sort by `(graph name, graph params)` is free. Memoise the `Dataset` too:
`resolve_object_choice` calls the registry *factory* per case, so today every case reloads tabula's
`.mat`.

⚠️ **The cache loop unit is `(case, point set)`, not `(case)`.** `PAIRWISE_CACHE_MAXSIZE` is 2 and
its docstring says the pattern it serves is several metrics over one layout. Call
`clear_pairwise_cache()` and `clear_loo_cache()` when releasing a context — content-keyed entries
from the previous dataset stay resident otherwise. Budget ~1 GB for a tabula case: `ctx.ideal`
200 MB + condensed 100 MB + two pairwise entries 200 MB + crossings' +428 MB transient.

### B4. `tests: resume, end to end`

The expensive failure in this step is a resume that silently serves stale data, so these tests
land here rather than trailing: a killed-and-restarted run producing the same rows; adding a
metric re-running metrics without re-embedding (which is (22)'s entire promise, asserted); a
bumped `version` invalidating exactly its own shard; a layout produced under one SHA reused under
another.

### B5. `docs: close step 7`

`STATUS.md`, a §0.5 session-7 corrections block, and DECISIONS departures: §3.7's signature,
§4.6's directory layout, (20)'s reductions, (22)'s `version` field, and §2.5's cost class being
measured-wrong in both directions.

---

## Traps

- ⚠️ **Do not raise `PAIRWISE_CACHE_MAXSIZE`.** It will look like the cause of a slow batch; its
  own docstring predicts the 12 GB footgun. Fix the loop order.
- ⚠️ **Do not teach the leave-one-out cache to slice** a small `k` out of a larger `k_max`. It is
  free and it would move numbers on layouts with duplicate points.
- ⚠️ **Do not add per-case state to `EvaluationContext`.** It is per-dataset, frozen and lazy by
  design; the provider cache is a second object.
- ⚠️ **Do not regenerate `metric_golden.json` to make a registry test pass.** Failing in that
  situation is its entire purpose.
- **Do not add persistence or key logic to `graph_embedding`.** It stays `Dataset → LayoutResult`;
  the runner persists.
- **Do not update `graph_embedding_all` / `embedding_metrics_all` / `metrics_to_df` to the new
  frame.** Dead `(d, n)` legacy, tracked in (2), deleted at step 10. Add no callers.
- **Do not make `RSSSampler` nestable** for per-metric memory — nested samplers on one process
  measure the same process. Use the cost class to decide whether to sample at all. ⚠️ And note
  the standing conflict: `peak_mem_mb` is a delta from a baseline that drifts upward under glibc,
  so a single-process batch makes it a lower bound. Record absolute baseline and peak beside the
  delta and a `mem_measurement: "batch" | "isolated"` flag, so the drift is visible post hoc
  rather than averaged into a thesis table.
- **Do not repurpose `batches/batch.py`** — it is the only worked example of the batch grammar.
  Add a new file.
- ⚠️ **Check any tolerance on `digits` or `tabula`, never on `karate` or `football_2023`.** Four
  sessions running.

## Verification

```bash
venv/bin/python -m pytest -q          # 864 + step 7's; 0 skipped
venv/bin/python -m pytest -q -m slow  # the tabula figure and the score pins, ~4 min
python batches/smoke.py               # Half A: writes one parquet
git status --short                    # clean
```

Then, specifically:

- **A2** — every `twin.metrics.__all__` export is registered or excluded with a reason; every one
  of `metric_golden.json`'s 44 keys is produced by a registry entry; every binder satisfies its
  function's signature.
- **A3** — a case with no labels emits `status="skipped"` with a `skip_reason`, not a NaN; a
  deliberately raising metric yields `status="error"` and does not abort the batch;
  `crossing_pairs` is solved **once** per `(case, variant)` for three metrics (assert the call
  count).
- **A4** — `tests/test_report_numbers.py` passes **unchanged**, which is the whole proof that the
  refactor preserved behaviour; and `grep -r "twin.report" twin/metrics/` is empty, which is the
  proof the cycle is gone.
- **A5** — tabula's Sil\* appears **once**, not once per case; `compartment_cell_type` does not
  appear at all and the manifest says why.
- **A6** — the smoke parquet's row count equals the predicted
  `cases × point_sets × metrics × variants × k`, computed independently in the test.
- **B1** — an npz round-trips to a `LayoutResult` that validates, with no pickle; the rehydrated
  dataset is the registry one.
- **B4** — the four resume tests above. **Adding a metric must re-run metrics and not embeds** —
  assert by counting `embed` calls, not by timing.
- **B2** — the manifest records `k_source` and `mem_measurement`; a run with an explicit `k=(...)`
  and one without are distinguishable from the manifest alone.
- **Cost** — one tabula case through the full registry stays under ~1 GB peak and the crossing
  metrics are solved once.

## Open

**None.** The two items this plan carried at approval — whether `runs/` holds the smoke output,
and whether `run_id` is caller-settable — were resolved at the follow-up review (both yes, see
"resolved by inspection"), along with the memory and `k` decisions. Anything discovered during
implementation goes to `DECISIONS.md` as a departure, as steps 2-6 did.
