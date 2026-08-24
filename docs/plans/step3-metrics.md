# Step 3 — `common.py` and `structural.py`

The executable sequence for spec §7 step 3. **The design is not repeated here** — read
`EXPERIMENT_SETUP_SPEC.md` §2.0/§2.1 (and §0.5 decisions 3, 4, 6, plus the session-2 corrections)
and `DECISIONS.md` (8), (10), (18), (20) and "A metric needs the vertex graph" first. This file is
what to do, in what order, and how to know it worked.

Prerequisite: `docs/STATUS.md` for where things stand and the items it carries into this step.

**Scope is exactly ADR, the kNN family and SNS.** `visual.py`, `crossings.py`, `label.py`'s new
content, `METRIC_REGISTRY`, `MetricSpec`, `evaluate_layout` and persistence are steps 4–7 and must
not appear here. Thirteen commits, more than step 2's ten — the natural split is after commit 9:
**1–9 are the code and leave the tree green; 10–13 are measurement and docs.**

---

## Commit sequence

Each commit green and separable; exactly one of them can move a number, and it says so. Branch off
`experiment-setup`, merge back when `pytest` passes and all eight figure functions still render.

1. **`tests:` pins for the export surface and the two edges the existing pins cannot see.**
   Three things: the exact set of names `from twin.metrics import *` exports; the NaN/exclusion
   behaviour of `knn_neighbor_preservation_accuracy` on a graph with a zero-degree vertex; and its
   behaviour on a layout with coincident points.

   The surface pin is not bureaucracy. `twin/pipelines.py` does `from twin.metrics import *` and
   `twin/metrics.py` has no `__all__`, so today `*` means "every public module-level name"; after
   the package move it means "whatever `__init__.py` binds" — a strictly different rule. A dropped
   export surfaces as a `NameError` in an overnight batch, not in `pytest`.

   The other two guard the places the split can move numbers while all sixteen existing pins still
   reproduce. No registered dataset has an isolated vertex and karate has no coincident points, so
   `tests/test_characterisation.py` cannot see either. Pin the behaviour you are about to share a
   core with, so that "it changed" is a fact rather than a discovery.

2. **`metrics:` `twin/metrics.py` becomes `twin/metrics/`. Pure move, zero diff to any function
   body.** `common.py` takes `require_n_by_d`; `structural.py` takes
   `knn_neighbor_preservation_accuracy` **with its `mode=` intact**; `label.py` takes
   `KnnAccuracyReport`, `majority_class_rate`, `_loo_neighbors`, `knn_accuracy_report`,
   `knn_accuracy`. `__init__.py` re-exports all of them behind an explicit `__all__`.

   The proof that this commit is behaviour-preserving is that **no test file changes**. Every
   existing import is `from twin.metrics import X`, which still resolves. If a test needs editing
   here, it was not a move.

   Creating `label.py` now, two steps early, is deliberate: the label family is already written
   and already pinned, and §2's file layout puts it there. Moving it once inside a mechanical
   commit beats moving it at step 6 in the same commit that adds silhouette. **Do not implement
   anything from §2.3 here**, and **do not create `registry.py`** — a registry written against
   three metrics gets rewritten against fifteen.

   Keep the §1.2 tombstone comment for `point_distance_metric`, relocated beside ADR where it now
   points at real code.

3. **`layoutgraph:` stamp the side and the source on every graph the layer hands out.**
   `G.graph["twin_side"]` and `G.graph["twin_source"]`, set in `_weighted_graph` and in the derived
   graphs' `_build_distance`.

   Two step-3 guards need to know where a bare `nx.Graph` came from, and both guard against a
   *silent* misuse: `ideal_distances` must raise when handed a `LayoutGraph` distance form
   together with an explicit `weight_transform` (`DECISIONS.md` (10)), and ADR must raise when
   handed a distance-valued graph (commit 6). An `nx.Graph` carries no provenance on its own. One
   commit ahead of both, so each guard commit is one idea.

   ⚠️ **The assertion test ships in this commit, not in the one that first reads the stamp.**
   Metadata nothing reads is metadata that drifts — the exact failure mode `EmbeddingSpec.expects`
   had, and the reason `DECISIONS.md` has an entry called "`expects` must be asserted".

4. **`metrics:` `common.ideal_distances`, `finite_pair_mask`, and the O(n²) guard.**

   Use `scipy.sparse.csgraph.dijkstra` over `nx.to_scipy_sparse_array(G, nodelist=range(n),
   weight=weight)`, not networkx's all-pairs generators: it returns exactly what `DECISIONS.md` (8)
   specifies — a dense float array with `np.inf` in unreachable entries — and does tabula's 5007
   sweeps in seconds rather than minutes. `finite_pair_mask(D)` returns the **condensed** boolean
   vector.

   **Declare one pair ordering for the whole package and never deviate: scipy's condensed
   ordering.** `pdist` produces it, `squareform` converts into it, SNS sums over it. Every "which
   triangle, in what order" bug in a stress implementation comes from having two.

   `disconnected="lcc"` is the default, `"mask"` is available, `"components"` raises
   `NotImplementedError` with the §0.5-decision-3 message. Both policies return the **full
   `(n, n)` array** and differ only in where the infinities are (`DECISIONS.md` records why).

   The guard — raise unless `n <= max_n`, default 20000 — is **the first statement in the
   function**, before the sparse conversion and before any allocation, and its message quotes the
   projected allocation. Test it two ways: an explicit small `max_n` on karate (the mechanism) and
   a 20001-node path graph (the default). A test that only passes `max_n=10` still passes if
   someone changes the default to `None`.

   Testing the disconnected policies needs a fixture, since every registered dataset is connected:
   add `disjoint_paths` to `conftest.py` — `nx.disjoint_union(nx.path_graph(4), nx.cycle_graph(3))`,
   7 nodes, 6 finite pairs under `"lcc"` and 9 under `"mask"`. Both numbers are computable by hand.

   **Do not add `largest_component()` to `Dataset` to implement `"lcc"`.** It was dropped at step 2
   and the reasoning named this exact moment — "not in a hurry at step 3 because SNS turned out to
   want it". `csgraph.connected_components` gives the labels; nothing needs a subset dataset.

5. **`metrics:` `common.pairwise_distances` — the condensed vector and mean $\bar d$ (eq. 66).**

   `scipy.spatial.distance.pdist`, condensed form kept, mean returned alongside. Key the cache on a
   digest of `X.tobytes()` plus shape and dtype, **not** on `id(X)` — ids are reused after garbage
   collection and would serve one layout's distances for another's. Hashing 80 KB is microseconds
   against a 12.5-million-pair `pdist`.

   ⚠️ **Cap the cache at one or two entries, and say why in the docstring.** Tabula's condensed
   vector is 100 MB. An `lru_cache(maxsize=128)` here is a 12 GB footgun that will look like a leak
   in the step-7 batch runner and be found the expensive way. This is its own commit precisely
   because it is the one piece of step 3 with a memory-footprint failure mode, and therefore the
   one worth being able to revert alone.

6. **`metrics:` `structural.adjacency_distance_ratio` — eqs. (27)/(29).**

   One expression, no branch: eq. (29) is eq. (27) with $w_{ij}\in\{0,1\}$, and
   `nx.adjacency_matrix(G, weight="weight", nodelist=range(n))` already returns unit weights when
   the attribute is absent. **Do not write an unweighted branch**; write the eq.-(29) *test*
   instead, `karate` against `karate_weighted`.

   ⚠️ **ADR's $w$ is an affinity, and `dataset.graph`'s `weight` attribute is not always one.** On
   digits it is the stored Euclidean *distance*, 5.3–37.8. Passing `dataset.graph` straight in
   makes ADR reward putting graph-distant pairs close together — bug §1.6 recurring one layer up,
   this time with no Spearman −1 to make it visible. ADR refuses a graph stamped
   `twin_side="distance"` (commit 3) and takes its affinity graph from the evaluation context
   (commit 9). A bare unstamped graph is accepted, because §6's analytic tests build $K_4$ and a
   path by hand and must not need a `Dataset`.

   The docstring states, in these words, that ADR is **comparable across layouts of a fixed graph
   under a fixed weighting and across nothing else** — not across datasets, not across $w$ versus
   $w^2$. §5.3 enforces it at step 11; the docstring is what makes that enforcement read as a rule
   rather than a restriction.

   Tests: $K_4$, `path5`, `star5`, `triangle` against hand-computed values in the test file;
   invariance under uniform scaling, rotation, reflection and translation.

   Same commit, **close the point-distance slot in prose**: update the three stale comments in
   `twin/pipelines.py` to say ADR exists and that this path is not getting it. `DECISIONS.md`
   records why.

7. **`metrics:` split the kNN family into four functions, and delete the last `mode=`.**

   `knn_recall`, `knn_precision`, `knn_jaccard`, `weighted_recall` over a shared private core, plus
   a `NeighborPreservationReport` carrying `k`, `value` and `n_excluded` — mirroring
   `KnnAccuracyReport`, which `DECISIONS.md` (18) already names as the shape. Scalar `k` returns a
   float, array-like `k` an array, matching `knn_accuracy`; that is all step 3 owes
   `DECISIONS.md` (20), since `MetricSpec` decides sweep-versus-distribution at step 7 and
   **nothing here may infer it from shape**.

   Query at `max(k)` **once** and slice, replacing the recursive re-fit. Use `np.atleast_1d`, not
   `isinstance(x, (list, np.ndarray))` — a tuple currently falls through to the scalar path and
   into an obscure sklearn error.

   ⚠️ **This is the only commit in step 3 that can move a pinned number, and it touches all
   sixteen.** `tests/test_characterisation.py` pins `knn_neighbor_preservation_accuracy` at
   `rel=1e-12` across four modes and $k \in \{5, 10\}$. The pin **keys** become function names
   here; the pin **values must not change**. A changed value is either the duplicate-drop
   divergence (see Traps) or a regression, and in the first case it must be declared in the commit
   message with the new value and the reason, per `CLAUDE.md`. Editing a value quietly while
   renaming the keys is the worst thing that can happen in this step, because the rename supplies
   the cover.

   ⚠️ **The last `mode=` in the codebase disappears here**, so this commit also edits
   `twin/pipelines.py`: the three `knn_neighbor_preservation_accuracy(..., mode=mode)` call sites
   become `knn_recall`, and `embedding_metrics_all`'s own `mode=` goes. **No deprecation shim** —
   there is one caller, it is in-tree, and `mode=` was deleted from the embedding layer at step 2
   without one.

   Same commit: `common.default_k(G)` = mean degree of the **input** graph,
   `int(np.floor(2 * m / n + 0.5))`, floor 1 (§0.5 decision 6). Not `round()` — see Traps. And the
   eq.-(39) test: `weighted_recall == knn_recall` on `karate` and on `football_2023`.

8. **`metrics:` `scale_normalized_stress`, `raw_stress`, `normalized_stress` — eqs. (40), (41),
   (43)–(45).**

   $\alpha^{*}$ in closed form. **No optimiser, and no "just in case" iterative fallback.** All
   three consume the condensed ideal vector and the condensed `pdist` from commit 5, masked by
   `finite_pair_mask` so that `"mask"` and any future disconnected dataset need no second code
   path.

   Keep §2.1's signature and add `ideal=None`. When `ideal` is supplied the function does not touch
   `G`; when it is not it computes one, guarded. The runner always supplies it — that is the whole
   content of "one source, one owner" — but the primitive has to stand alone for the analytic
   tests, so the docstring must name which caller is the correct one.

   Tests: SNS invariant under $X\mapsto\beta X$ and under $d\mapsto\beta d$; `raw_stress` and
   `normalized_stress` demonstrably **not** invariant. That asymmetry is why all three exist —
   §2.1 wants the §3 argument demonstrated rather than asserted.

   ⚠️ **Put the KK caveat in the SNS docstring now, not when the first table is read.** SNS on a
   Kamada–Kawai or neato layout is close to a measure of optimiser success, because KK optimises
   approximately what SNS measures. The reporting-layer flag that acts on it is step 11; the
   sentence has to exist before anyone reads a number.

9. **`metrics:` the per-dataset evaluation context.**

   A frozen bundle — the `Dataset`, the un-normalised faithful `vertex_graph(dataset)`, the
   affinity-valued graph ADR and `weighted_recall` read, and a **lazy** `ideal` property — built
   once per dataset and threaded through. `DECISIONS.md`: a metric is a function of points +
   `Dataset` + the **vertex** `LayoutGraph`, and this is a context, **not a field on
   `LayoutResult`** — attaching it per case duplicates it per representation, per seed, per λ and
   re-couples the two layers step 2 separated.

   Lazy `ideal` matters twice: a metric set of ADR and kNN recall must not pay for a closure it
   never reads, and at step 7 the resume path builds contexts for cases whose shards already exist.

   **No `METRIC_REGISTRY`, no `evaluate_layout`, no `MetricSpec`, no persistence.** Those are step 7
   and they are what this object exists to make trivial. Build the noun; leave the verb.

   Without it, commits 6 and 8 have no legitimate caller and "one target per dataset, shared by
   every representation" stays prose in three documents.

10. **`docs:` re-measure the §1.6 digits fix in quality terms.**

    Session 2 established that Procrustes cannot show the fix mattered — old-versus-new disparity
    at matched init is 0.02–0.46 while **re-initialisation alone moves it 0.46–0.83**. So this
    measurement decides whether the most consequential correction in the spec is reportable at all,
    and it has to be read against that noise floor: **at least five seeds, median and IQR** (§4.1),
    reporting `knn_recall@k=16`, ADR and SNS. One seed is noise, and a single-seed table here would
    be worse than none.

    $k = 16$ because digits' mean degree is 16.4 and §0.5 decision 6 fixes $k$ at the input graph's
    mean degree.

    Reproducing the pre-fix layout means building a `LayoutGraph` whose affinity *is* the stored
    distance. Do that in the measurement script only; **do not add a "legacy" flag to
    `vertex_graph`.** Record the result in `DECISIONS.md` beside the Procrustes paragraph it
    corrects.

11. **`docs:` `exp_decay` versus `exp_decay_sq` on digits** (§3.4's deliberately-unresolved list).

    After commit 10, because commit 10 establishes the seed-noise floor this has to be read
    against — the lesson of the §1.6 measurement is that a difference smaller than
    re-initialisation is not a difference. Same protocol: five seeds, median and IQR,
    `knn_recall@k=16` as the headline with ADR and SNS beside it.

    SNS is legitimate here specifically because digits' distance side is **stored**, so changing
    the affinity rule does not move the ideal distances and both rules are scored against an
    identical target. Say so; it is what makes the comparison clean.

    ⚠️ **Do not change the default on the strength of this.** §3.4 lists the choice as deliberately
    unresolved and configurable. Record the measurement; if it favours `exp_decay_sq`, that is a
    `DECISIONS.md` entry and a separate commit, not a side effect of a measurement.

12. **`tests:` the golden-file metric suite (§6).** On `karate_weighted`, using the deterministic
    `karate_structured_layout` fixture rather than a seeded layout run, keyed by metric name.
    `DECISIONS.md` records all three choices and why.

13. **`docs:` update `STATUS.md` and `DECISIONS.md`.** Step 3 done, the carried-forward items
    discharged or re-assigned with their measured numbers, and — following step 2's precedent —
    **where the implementation departed from this plan and why**. This file becomes history the
    moment it merges.

## Traps

- **`dataset.graph`'s `weight` attribute means different things on different datasets.** On
  `digits` it is a Euclidean distance; on `karate_weighted` and `tabula` an affinity; on the six
  unweighted datasets it is absent and `nx.adjacency_matrix` fills 1. ADR and `weighted_recall`
  both read it and both invert silently on digits — and karate, the only graph the pins use, is on
  the safe side.
- **The two kNN families disagree about which duplicate to drop.** sklearn's
  `kneighbors_graph(X=None)` masks column 0 for rows whose duplicates outnumber `k`;
  `_loo_neighbors` masks the last column by design. Sharing a core between the preservation family
  and the accuracy family changes coincident-point behaviour, and karate has no coincident points,
  so all sixteen pins reproduce while the behaviour moves. Commit 1 exists for this.
- **`nx.adjacency_matrix(G)` uses `G.nodes()` order; `X` rows are 0…n−1.** Every current graph
  happens to be inserted in order, so `nodelist=range(n)` is a no-op today and its omission stays
  invisible until a dataset relabels. `Stored.matrix` already passes it; the metric layer does not.
- **A zero-weight edge disappears from a CSR matrix**, so `csgraph.dijkstra` reads it as absent
  rather than as length 0. No dataset produces one today (the report says 0 % of pairs sit at
  d = 0), but §0.5 is explicit that a zero ideal distance is legitimate under `probability` and
  `kernel`. Detect and raise with the reason; do not build the masked-array path for a case that
  does not occur.
- **`round()` is banker's rounding** — `round(16.5)` is 16. §0.5 decision 6 says round to nearest.
  No current dataset lands on a half, so the bug ships silently and appears when one does.
- **`max_n = 20000` admits a 3.2 GB allocation.** Kept deliberately (`DECISIONS.md`), but it means
  the guard does not by itself protect a workstation — which is why the message quotes the number.
- **The condensed `pdist` for tabula is 100 MB and the square ideal array 200 MB**, and condensing
  the square transiently needs both.
- **`rel=1e-12` on the pins has four orders of magnitude of headroom, not infinite headroom.**
  $A_V$ reproduces $w_{ij}$ through a sqrt round-trip at ~1e-16 relative — fine at 1e-12, not fine
  if someone tightens the pins "for rigour".
- **The `karate` test *fixture* is `nx.karate_club_graph()`, which is registry `karate_weighted`,
  not registry `karate`.** Renaming the fixture to match the registry, or repointing it at
  `load_dataset("karate")`, silently changes every `weighted recall` pin.
- **`np.nanmean` over an all-NaN row warns and returns NaN**, which reaches a table as a blank
  rather than an error. Assert `n_excluded + n_scored == n` in the report constructor so the two
  cannot disagree.
- **`"mask"` and `finite_pair_mask` are unexercised by the roster.** Without `disjoint_paths`,
  `finite_pair_mask` returns an all-true array in every test that runs it and an inverted
  comparison inside it is invisible.

## Verification

```bash
venv/bin/python -m pytest -q     # session 2's 299, plus step 3's
python -m twin.report            # unchanged; step 3 adds no dataset columns
git status --short               # clean; if notebooks appear modified, the jupyter filter is unset
```

**The load-bearing check:** one `ideal_distances` array, built once from
`vertex_graph(dataset).distance`, scores an $A_V$, an $A_E$ and an $A_{tne}$ layout of the same
dataset — same target, three representations, no recomputation — so the three SNS values are on a
common scale. If any representation reaches for its own closure, or a metric recomputes one per
call, the central ablation is measuring three different things.

Then:

- `ideal_distances` on `disjoint_paths` gives **6** finite pairs under `"lcc"` and **9** under
  `"mask"`; `"components"` raises `NotImplementedError`, not `ValueError`.
- `ideal_distances(nx.path_graph(20001))` raises at the default `max_n` and the message names the
  projected allocation; `max_n=10` on karate raises too. Both, not either.
- Handing `ideal_distances` a graph stamped `twin_side="distance"` **and** an explicit
  `weight_transform` raises; the same graph without the transform does not.
- ADR on unweighted `karate` equals the eq.-(29) hand computation, from the eq.-(27) code path,
  with no unweighted branch present in the source.
- ADR refuses `load_dataset("digits").graph` and accepts the context's affinity graph, and the
  refusal names the side it got and the side it needed.
- ADR is unchanged to floating point under scaling, rotation, reflection and translation of `X`.
- All sixteen `NEIGHBORHOOD_PINS` reproduce at `rel=1e-12` under the four named functions, with
  only the keys renamed — `git show` on commit 7 shows no numeric literal changing in
  `test_characterisation.py`. If one did, the commit message says which and why.
- `weighted_recall == knn_recall` exactly on `karate` and `football_2023`, and **differs** on
  `karate_weighted` — the second half is what proves the first is not vacuous.
- The four kNN functions at `k=[5, 10]` agree elementwise with two scalar calls, and
  `NearestNeighbors` is fitted **once** for the pair, not twice.
- `n_excluded` is non-zero on the isolated-vertex fixture, and `n_excluded + n_scored == n`.
- SNS unchanged under $X \mapsto 7X$ and $d \mapsto 7d$; `raw_stress` changes under both,
  `normalized_stress` under one — pin which.
- `common.default_k` gives 5 on karate, 16 on digits, 20 on tabula, 1 on a graph of isolated
  vertices.
- `from twin.metrics import *` exports the pre-move names plus the new ones, and
  `python -c "import twin.pipelines"` succeeds — the second is the one that would have caught it.
- The digits re-measurement reports median **and** IQR over ≥ 5 seeds, and says whether the fix's
  effect exceeds session 2's 0.46–0.83 re-initialisation band.
- All eight `paper_figures` functions still render. Nothing in step 3 touches them, which is why a
  failure there means something moved that should not have.
