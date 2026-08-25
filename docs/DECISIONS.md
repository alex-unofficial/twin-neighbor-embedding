# Implementation decisions

Decisions on the points `docs/EXPERIMENT_SETUP_SPEC.md` leaves open, made by the implementer and
recorded here so they are auditable. Numbering follows the review list; §0.5 of the spec records
the ones the author decided. Where a decision applies to a later step, it is a commitment for
that step, not something already built.

---

## Spec §1.8 — resolved upstream, workaround removed

§1.8 of the spec describes `sgtsnepi(silent=True)` closing fd 2 without restoring it, and calls
for a save/`dup`/restore guard. That guard existed briefly as `twin/sgtsne_compat.py`.

**Fixed in `sgtsnepi` 0.7; the wrapper is gone and both call sites import the library directly.**
Verified against the raw library, bypassing any wrapper: no descriptor damage in any mode across
repeated calls, silent mode still emits zero bytes to stdout, and no standard descriptor is freed
during a call, so the race window that made concurrent sampling unsafe is closed at the source.

`pyproject.toml` pins `sgtsnepi>=0.7` and `tests/test_sgtsne_stdio.py` pins the requirement in
both directions — descriptors survive a silent embed, *and* silent mode stays silent. Reading §1.8
in the spec without this note would suggest an unimplemented item.

The one design choice the bug prompted that is **kept**:
`twin.instrumentation.RSSSampler` holds a single descriptor and `pread`s it rather than opening
per sample. That is better regardless — fewer syscalls in a polling loop — and the same hazard
applies to any C extension that closes descriptors.

## (2) Where the last `(d, n)` surface converges — tracked

Two orientations are live, and CLAUDE.md's "the single conversion point" is not yet literally
true. The exception is bounded and named here so it is tracked rather than assumed away.

**The legacy surface:** `pipelines.graph_embedding_all` returns the pre-`LayoutResult` dict of
`(d, n)` arrays. Its only consumer is `paper_figures.py`, which passes those arrays to
`plots.count_intersections`, `plots.drawedges` and `plots.draw_curved_edges` — all of which index
as `y[:, i]` and so require `(d, n)`.

This is safe today because `require_n_by_d` fails loudly rather than silently reorienting, so a
`(d, n)` array cannot leak into a metric. It is still a convention violation with a deadline:

| Step | What converges |
|---|---|
| **5** (`crossings.py`) | `count_intersections` is reimplemented on Layer 2 and takes `(n, d)`; `paper_figures` adapts at the call. |
| **10** (plotting refactor, §3.6) | `graph_embedding_all` and the bespoke figure functions are replaced by `layout_grid` over `LayoutResult`. Renderers convert at their own boundary per spec §0.5 (3). The legacy surface is deleted here. |

**Until then:** do not add callers of `graph_embedding_all`, and do not mix orientations within a
function. When step 10 lands, this section and the CLAUDE.md warning both come out.

## (5) Argument order in `crossings.py`

**Every function in `crossings.py` takes `(G, V, E=None, *, ...)`** — graph, vertex points, edge
points, and all parameters keyword-only. That includes `detour_ratio`, whose spec signature
(`detour_ratio(X, E, G, gamma, n_samples)`) is reordered to match `edge_curves`.

The spec writes two orders for the same three arguments, which is how argument-order bugs happen.
`edge_curves` sets the pattern; keyword-only parameters remove the remaining hazard. Metrics
elsewhere keep the `(X, G, **params)` shape declared at the head of §2 — the two conventions
answer different questions (a curve helper is not a layout metric) and each is internally
consistent.

## (8) What `ideal_distances` returns for unreachable pairs

**A dense float array with `np.inf` in unreachable entries**, plus a companion
`finite_pair_mask(D)` helper.

`inf` is the correct value and it is *loud*: it propagates through arithmetic, so a consumer that
forgets to mask produces `inf` or `nan` rather than a plausible number. `nan` would be quietly
absorbed by `np.nanmean` — which is exactly how `point_distance_metric` (spec §1.2) came to
return a number that meant nothing. Under the default `"lcc"` policy no infinite entries arise at
all; this only matters for `"mask"`.

## (10) `weight_transform=` as an escape hatch only

**`ideal_distances(G, weight_transform=...)` keeps the parameter as a low-level escape hatch; the
pipeline never passes it.** On the path that produces results, ideal distances come from
`LayoutGraph.distance` — which is the dataset's stored distance side, or the one derived for it by
`vertex_graph` (see (11)). Passing a graph that came off a `LayoutGraph` *and* an explicit
`weight_transform` raises, rather than silently overriding a declared source.

The primitive stays testable in isolation — the analytic cases in spec §6 need it without
constructing a `Dataset` — while the dataset remains the single source of truth everywhere else.

## (11) The `Dataset` / `LayoutGraph` layering, and stored sources

The spec's §3.4 originally made a dataset "a graph plus affinities plus distances" and typed both
accessors as `nx.Graph | sp.spmatrix`. Two problems: a return type that varies pushes a branch
into every consumer, and — the serious one — it left no answer to *what are $A_E$'s labels?*

**Two layers.** A `Dataset` is stored information: source of truth, labels, drawing instructions.
It is never laid out. A `LayoutGraph` is an affinity form, a distance form, and a way to derive
vertex/edge points; it is laid out and **never labelled**. Labels do not need to reach the layout
layer, because `vertex_and_edge_points` returns points in the **original** indexing — so a layout
of $A_E$ or $A_{tne}$ is scored against the dataset's own labels with no length mismatch possible.

> A metric or a figure is a function of the embedded vertex/edge points plus a `Dataset`.

**One type each, unconditionally.** `LayoutGraph.affinity -> sp.sparray` (what the representations
and `sgtsnepi` consume); `LayoutGraph.distance -> nx.Graph` (the `Unweighted` case *requires*
absent non-edges so shortest paths are hop counts, and a graph expresses absence naturally where a
matrix must encode it). Different types for two different questions is fine; a type that varies
with the data is not.

**`Dataset` holds no derived information.** It declares which sides are *stored* — `Stored`,
`StoredMatrix`, `Unweighted` — and what the numbers mean. Derivation of a missing side happens in
the `Dataset → LayoutGraph` mapping and nowhere else. `edge_graph` and `twin_graph` are then
`LayoutGraph → LayoutGraph` off the vertex graph, which matches the algebra ($A_E = B^{\top}B$ and
$A_{tne}$ are both functions of the affinity incidence matrix) and buys two things: the derivation
chain is readable off a `parent` link instead of maintained as a string, and the incidence matrix
is built once rather than three times.

`weight_kind` was doing two jobs — saying what stored weights mean *and* prescribing how to derive
the other side. Those coincide only for a dataset with one weight vector. Tabula has both `A` and
`D`, so it satisfies the interface directly and no derivation arises; the vocabulary survives as
the `meaning` of a stored source.

**Why a measured distance source matters.** `A` is **not invertible to distances at all**: UMAP's
per-point $\rho_i$ and $\sigma_i$ are unstored, so no global transform recovers $d$ — true of any
per-point-adaptive kernel. Hence `meaning="adaptive_kernel"`, whose derivation rule is *raise*:
such a dataset must supply a stored distance side, and tabula fails loudly rather than silently
applying $-\log(w/w_{\max})$ if `D` is ever dropped. See
[DATASET_PROVENANCE.md](DATASET_PROVENANCE.md#tabula-liver). Worth a line in thesis §3.1.1, which
lists global transforms as interchangeable options.

**`Dataset` does not own the representations.** They are the *method*; it is the *data*. Keeping
them mappings keeps two open sets independent — adding a dataset touches no representation, adding
a representation touches no dataset — and avoids forcing `Dataset` to carry `alpha`, `k` and
`normalize`, which are method knobs. Caching an expensive $A_{tne}$ across a λ sweep belongs in
the runner.

## Step 2 decisions the plan did not cover

Six choices forced during the implementation of spec §7 step 2, recorded because a later reader
would otherwise have to reconstruct them.

**`LayoutGraph` lives in `twin/layoutgraph.py`, not `graphmatrix.py`.** The plan said
`graphmatrix:`. That file holds the pre-restructure classes, which `paper_figures` needs until
step 10; keeping the new layer separate makes that deletion a file removal rather than a
bisection of a mixed file. `compute_slopes` is imported from there, not duplicated.

**Plan commits 5 and 7 are one commit, and it comes after commit 6.** There is no green state in
between: `graph_embedding` is `LayoutResult`'s only constructor and `batch_registries` is its
only source of graphs, so `LayoutResult` cannot hold a `Dataset` before the registry supplies
one. Bridging with a placeholder `Dataset` would mean declaring stored sources for a graph whose
provenance is unknown, which is the guess the layering exists to remove.

**`GraphMatrixMethod` gains `.affinity` / `.distance` / `.n_points`.** Deleting `mode=` would
otherwise break `graph_embedding_all`, which is step-10 work. The shim's `distance` is
`nx.from_scipy_sparse_array(adj)` — bit-identical to what `mode="distance"`, the default, did —
so removing the flag moves nothing on the legacy path. It is not a principled distance form, and
a test pins it as the shim it is. Nothing new may route through it.

**`digits` defaults to `n_neighbors=12`, not 100.** The old `dataset.mnist` default was 100,
giving a 1797-vertex graph with mean degree 125; nothing ever ran it that way — the batch passed
12, and `DATASET_PROVENANCE.md` describes 12. Making the default the value that is actually used
means a bare `load_dataset("digits")` reproduces the roster.

**No Leiden label set on tabula, yet.** `DATASET_PROVENANCE.md` notes that the teaser's Leiden
partition is a `derivation="detected:leiden"` label set that belongs on the dataset. It is not
registered here because computing it on every load costs time for every consumer — the report,
every batch case — for something only one figure uses. It moves with the figure at step 10, where
the consumer and the cost land together.

**`sbm_er_3modes` is registered; `polblogs`, `lfr`, `homer`, `football`, `mycielski` and
`stick_sculpture` are not.** Only sbm-er-3modes has a provenance record (`DATASET_PROVENANCE.md`:
one constant weight, hence `Unweighted()` on both sides), so only it can have its stored sides
declared honestly. Registering another means writing down where its weights came from first.

## Three forms, not two — and what reaches the backend

A `LayoutGraph` is **one support carrying up to three valuations**: combinatorial (1), affinity
($w$), distance ($d$). The third was missing, and its absence produced a wrong-sign bug.

**Yifan Hu is not a distance method.** Hu's is a multilevel combinatorial force method —
attractive $d^2/K$ along edges, repulsive $-CK^2/d$ between all pairs, with $K$ a *global*
constant. There is no per-edge target length in the formulation. It declares
`expects="combinatorial"` and receives a graph with no edge attributes.

**`expects` being right is not sufficient.** Both graphviz methods go through `nx.nx_pydot`,
which writes edge attributes into the dot source as `weight=`; graphviz reads `weight` as a
spring-strength multiplier and `len` as the desired edge length. So a distance passed as `weight`
means *far apart, pull harder*. The step-2 commit that switched both methods to `.distance`
introduced exactly that; before it they received the affinity as `weight`, which was at least the
right sign. `neato` now writes `len=`.

The conformance test cannot catch this class of error — it checks *which form* was read, not what
the backend then takes it to mean. Three separate tests cover the plumbing: the sfdp input
carries no attributes, the neato input carries `len` and not `weight`, and neato's output moves
when the lengths change. **Any new backend that serialises its input needs the same treatment.**

## Sparsity is the operating regime, and it is declared in the factory

**Not a compromise for speed.** The twin transform *requires* sparse input: on a complete graph
over tabula's 5007 vertices, $A_E$ would carry $\sum_v \deg(v)(\deg(v)-1)\approx 10^{11}$
nonzeros and $A_{tne}$ would span $n+m\approx 12.5$ M nodes. A dense-input comparison is not an
experiment that could be run at any budget, so the dense metric is not a yardstick any of these
methods could be *built* to serve. Argue it there — at n ≈ 5000 a dense distance matrix is only
200 MB, so the general "dense is slow" version is attackable and the transform-level version is
not.

**Sparsification happens in the dataset factory**, before any representation exists. Two reasons:
the graph's edge set defines the *original edge indexing* that `vertex_and_edge_points` returns
into, so a layout-layer knob must not be able to change it; and $k$ then sweeps through the
existing batch axis for free as `("digits", {"n_neighbors": [5, 12, 30]})`, which is right
semantically — a different $k$ is a different dataset, not a different view of one.

$k$-NN over $r$-neighbourhoods: it bounds degree and cannot isolate a vertex. Sparsify on the
**distance** side only — for a globally monotone derivation like $\exp(-d/d_{\text{scale}})$,
the $k$ nearest by distance and the $k$ largest by affinity are the same set. And $k$ is
**declared, not derived from a target density**: §4.2's λ criterion is defined at $k$ = mean
degree, so it is already a first-class experimental quantity, and choosing it through a density
proxy would hide the axis one most wants visible.

`KNearest.check()` verifies the declaration against the graph: after or-symmetrisation every
vertex has degree at least $k$, and $nk/2 \le m \le nk$. Both hold exactly (minimum degree 12 for
digits, 14 for tabula). Neither is implied by $n$ and $m$, so neither would show up in the
existing pins — which is the point.

A general sparsifier for a dense stored distance is **not built**. No dataset has one; same call
as `largest_component()`.

## The graph is the stipulated ground truth, and what that costs is measured

Sparsifying to a $k$-NN graph means the target SNS and Sil\* score against is the shortest-path
closure of that graph — a **graph geodesic** (Isomap distance, for Euclidean edge weights), not
the ambient metric. It is *exact* on the $k$-NN edges, since an edge's weight is the ambient
distance and no path can beat it, and strictly larger elsewhere. So the distortion is a one-sided
inflation of **long-range** distances: local metrics (kNN recall, NND, edge lengths) are
undistorted, and the global ones score against a proxy. It is monotone in $k$.

**A layout is therefore not expected to recover distances the method was never shown, and is not
scored as though it should be.** Every method sees the same graph, so the comparison between
$A_V$, $A_E$ and $A_{tne}$ — which is the point of the experiment — is internally valid. Two
things worth stating rather than leaving to be discovered:

- Sparsification does not *disadvantage* stress methods inside the comparison. It relocates their
  advantage: KK and neato optimise approximately the functional SNS measures, so **SNS on a KK
  layout is closer to a measure of optimiser success than of the representation.** Report it
  alongside metrics that are not KK's objective.
- The equal constraint is on what each method is *given*, not on what it spends. KK and neato
  expand to all-pairs internally regardless. That asymmetry is not assumed away — `runtime_s`,
  `representation_s` and `peak_mem_mb` measure it per case.

**Measured, not merely acknowledged.** For `digits` and `tabula` the unsparsified source is
available and is declined, so `sparsification_distortion` quantifies the stipulation's cost:

| | rank ρ | stress | stress after optimal rescaling | median $d_{geo}/d_{full}$ |
|---|---|---|---|---|
| digits ($k$=12) | 0.6955 | 1.784 | **0.2027** | 2.63 |
| tabula ($k$=14) | 0.7876 | 1.489 | **0.2102** | 2.19 |

**Quote the rescaled column for SNS.** It fits its own global scale, so a uniform inflation costs
it nothing, and roughly nine tenths of the raw stress here is exactly that. What remains is the
ordering. On digits at $k = 5/12/30/60$ the rank correlation rises 0.61 → 0.85 and the median
ratio falls 3.78 → 1.74, so fidelity and tractability are the same knob and both ends of it are
in `docs/DATASET_REPORT.md`.

## Step 3 departures from `plans/step3-metrics.md`, and why

Following step 2's precedent. The plan was written without implementing anything; these are the
places it did not survive contact. Fourteen commits landed against thirteen planned.

**The form stamp had to reach `dataset.graph`, so it does not live in `layoutgraph`.** The plan
puts the stamp in `layoutgraph` (commit 3) and then expects ADR to refuse
`load_dataset("digits").graph` (Verification). Those two cannot both hold: a dataset's graph
never passes through the `LayoutGraph` layer. The declaration already existed on the `Dataset`
— a side stored as `Stored` *is* the graph's `weight` attribute — so `Dataset.graph_weight_side()`
reads it and `__post_init__` stamps it. The primitives moved to a new dependency-free
`twin/forms.py`, because `twin.layoutgraph` pulls in `sgtsnepi` through `twin.twin_embedding`
and neither the dataset layer nor the metric layer should load a C extension to ask what a
`weight` attribute means. `tabula` is stamped None: both its sides are separate matrices, so
nothing declares the attribute, and unstamped is accepted everywhere as before.

A consequence worth stating: `ideal_distances` now refuses `karate_weighted.graph`, whose
weights are Zachary's interaction strengths. That is correct — they are affinities, and the
ideal-distance source is `vertex_graph(dataset).distance`, which applies the declared $1/w$
resistance model.

**`LayoutGraph.affinity_graph()` landed with ADR, not with the context.** ADR refuses a
distance-valued graph, and a guard whose legitimate alternative does not exist yet is an
obstruction rather than a guard. It is cached like `.distance` and stamped `"affinity"`.

**Query-at-max(k)-and-slice is not equivalent to re-querying, on a layout with exact
duplicates.** Neither the plan nor the spec anticipated this. Which of nine coincident points
count as "the five nearest" depends on how many neighbours were asked for, and no leave-one-out
convention fixes it, because the arbitrariness is in the data rather than in the code. Pinned as
behaviour in `test_characterisation.py`; on a layout without duplicates — every registered
dataset, every real layout so far — slicing and re-querying agree exactly.

**Two pinned values moved, both on the coincident fixture, and they are the reason commit 1
existed.** Adopting `_loo_neighbors` for both metric families — one leave-one-out convention in
the package instead of two — changed `("jaccard", 5)` from 0.08954566675154911 to
0.09206667515491045 and `("weighted recall", 5)` from 0.1460194654084813 to 0.1482819088473953.
All sixteen karate pins reproduce at `rel=1e-12` with only the keys renamed.

**`weighted_recall` uses unit weights on a graph stamped `"combinatorial"`.** `football_2023`
declares `Unweighted()` on both sides and *still* carries a stray `weight` attribute from its
`.mat` — 732 edges at 1 and three at 2. The declaration is what the representation is built
from, so it is what the metric reads; otherwise eq. (39) fails on a graph the dataset says is
unweighted, and fails by three edges in 735, which is exactly the size of error that gets
explained away.

**Normalised stress is scale-dependent in *both* arguments, not one.** The plan predicts
"`raw_stress` changes under both, `normalized_stress` under one — pin which". Eq. (41) is eq.
(43) with $\alpha$ fixed at 1, so it is invariant only when $X$ and $d$ are scaled *together*.
That joint invariance, which raw stress does not have, is the real asymmetry between eqs. (40)
and (41), and it is what the tests pin.

**The evaluation context is `twin/metrics/context.py`, not `common.py`.** §2's file list does
not name it, but it is a noun about evaluation rather than a distance helper, and step 7's
`evaluate_layout` belongs beside it.

**A collapsed layout gives SNS `nan`, not 1.** $\alpha^{*}$ is $0/0$ there. The limit is 1, but
reporting a limit as a score is how an undefined quantity reaches a table.

**Seven seeds, not five, for both measurements.** §4.1's five is a floor and the runs are cheap
on `digits`; the §1.6 result is disjoint over all seven, which is a stronger statement than any
comparison of medians would have been.

## The §1.6 fix, re-measured in quality terms

Session 2 could show only that the correction *moved* the layout: Procrustes disparity between
the old and new digits layouts at matched initialisation was 0.02–0.46 while re-initialisation
alone moved it 0.46–0.83, so the effect sat below the seed-noise floor of the instrument. That
paragraph is corrected here, not withdrawn: Procrustes was the wrong instrument, and the right
question is whether the layout is *better*, not whether it is different.

**Protocol.** SG-*t*-SNE on $A_V$ of `digits`, seven seeds, matched initialisation per seed
across the two arms, $k = 16$ (§0.5 decision 6: digits' mean degree is 16.4). Both arms are
scored against **one** target — the corrected dataset's evaluation context — so only the matrix
that is laid out differs. The pre-fix arm is the stored Euclidean distance used directly as the
affinity, which is what `VertexMatrix` did when it built from $\sqrt{B_w}$ over the graph's
`weight` attribute; Spearman between the two affinity vectors is **exactly −1** over all 14732
edges, as in session 2. It is built in the measurement script and nowhere else — `vertex_graph`
has no "legacy" flag.

| metric | pre-fix, median [IQR] | corrected, median [IQR] | separates? |
|---|---|---|---|
| `knn_recall` @ *k*=16 | 0.6332 [0.6324, 0.6343] | **0.6393 [0.6387, 0.6401]** | **yes** |
| ADR | 0.0823 [0.0821, 0.0829] | 0.0835 [0.0828, 0.0836] | no |
| SNS | 0.3475 [0.3433, 0.3527] | 0.3452 [0.3417, 0.3509] | no |

**Read it against the noise floor, which is the only way this measurement is worth anything.**
For recall the two arms are *disjoint over all seven seeds* — the worst corrected seed (0.6364)
beats the best pre-fix seed (0.6352) — so the effect is larger than re-initialisation, and this
is the first evidence that §1.6 mattered rather than merely changed something. The gain is small
in absolute terms, ≈ +0.006 recall, ≈ 1 % relative.

ADR and SNS do not separate, and both ranges overlap across most of their IQR. That is not a
failure of the fix; it is what those two measure. ADR compares edge lengths to the layout's own
spread, and both arms compress edges by about as much relative to their own scale. SNS scores
against a shortest-path closure that neither arm optimises, and an inverted affinity perturbs a
*t*-SNE layout's local structure long before it perturbs its global stress. **The correction is
visible where the corrupted quantity is used — in the neighbourhoods — and nowhere else.**

Consequence for reporting: quote `knn_recall` when quoting §1.6, with the disjointness rather
than the difference of medians, and do not claim an improvement in ADR or SNS. Consequence for
the protocol: five seeds is the floor (§4.1) and seven were used; a single-seed table here would
have shown a difference of 0.006 against an unstated spread of 0.004 and been worthless.

## `exp_decay` versus `exp_decay_sq` on digits — measured, and the default stands

§3.4 lists the distance → affinity rule as deliberately unresolved and configurable, and it had
never been compared. Same protocol as the §1.6 re-measurement above and read against the same
noise floor: SG-*t*-SNE on $A_V$ of `digits`, seven seeds, **matched initialisation per seed**,
$k = 16$, one shared target for both arms — the default rule's evaluation context. Matched
inits make this a *paired* experiment, which is what makes a difference this small readable at
all.

SNS is legitimate here specifically because digits' distance side is **stored**: changing the
affinity rule does not move the ideal distances, so both arms are scored against an identical
target. ADR's affinity target is the default rule's, for the same reason.

| metric | `exp_decay`, median [IQR] | `exp_decay_sq`, median [IQR] | paired wins |
|---|---|---|---|
| `knn_recall` @ *k*=16 | **0.6393** [0.6387, 0.6401] | 0.6361 [0.6351, 0.6368] | 6 / 7 |
| ADR | **0.0835** [0.0828, 0.0836] | 0.0842 [0.0841, 0.0853] | 7 / 7 |
| SNS | **0.3452** [0.3417, 0.3509] | 0.3524 [0.3463, 0.3614] | 5 / 7 |

`exp_decay` is better or equal on all three. The evidence is strongest for ADR — every seed,
and the ranges almost disjoint — and weakest for SNS. For recall the IQRs are disjoint but the
*ranges* are not: one seed out of seven reverses, so this is a weaker result than the §1.6 one
above, where all seven were disjoint.

Mechanically the two rules differ in the tail: at digits' $d_{\text{scale}} = 21.84$ the
squared rule spreads the affinities over 0.050–0.943 where the plain one spans 0.177–0.785, both
with the same median by construction. The squared rule therefore sharpens the contrast between
near and far edges — and on a graph that is *already* a $k$-NN sparsification, sharpening it
again costs neighbourhood recovery rather than buying it.

**The default does not change on the strength of this**, and did not need to: `exp_decay` is
already the default, and §3.4 keeps the choice configurable. Recorded so that the assumption is
now a measurement.

## Step 4 decisions, pinned before they are discovered the hard way

Eleven choices settled in advance so that [`plans/step4-visual.md`](plans/step4-visual.md) can be
executed without re-deriving them. Each records the alternative and why it lost.

**`detour_ratio` is step 5, not step 4.** §2.2 gives its signature under `visual.py`; §7 assigns it
to step 5, and it needs the `edge_curves` layer. Building the curve geometry in `visual.py` would
either duplicate it or make `visual.py` import a module that does not exist yet — and §2.4's layer 1
exists precisely so that the drawn picture and the measured crossings come from one implementation.
The alternative, pulling layer 1 forward into step 4, buys the straight-versus-curved cost one step
earlier at the price of putting geometry in the module that does not consume it.

**The visual metrics take `X` alone; only `edge_lengths` takes a graph, and that graph is
`dataset.graph`.** No `EvaluationContext`: no ideal distances, no affinities, no labels. Only the
edge *set* and its *order* are read, so **no stamp guard is needed** — worth stating explicitly,
because every other graph-consuming metric in the package has one and the absence here should read
as deliberate rather than forgotten.

**Shared results are cached module-level and content-keyed, not on a new layout-scoped object.** A
`LayoutView` bundling the pairwise distances, the neighbourhood and the per-vertex reductions would
read better and would mirror `EvaluationContext` — but §2 fixes three metric signatures and step 7's
`MetricSpec` calls them uniformly, so a required `view=` parameter would break the registry before
it is written, and an optional one would be a second path that is never the tested one. The existing
`pairwise_distances` cache is the pattern; the runner gets one `clear_*` per family, mirroring
`EvaluationContext.release()`.

**Each cache family sets its own capacity, from its own cost profile.** Measured at $n = 5007$:
leave-one-out arrays are 1.6 MB and 33 ms, condensed distance vectors are 100 MB and 105 ms, and
step 5's edge curves will be ~50 MB per variant and far more expensive than either. So: cache
several neighbourhoods, two distance vectors, one curve set. A single global `maxsize` is what turns
a cache into a memory incident — `lru_cache(maxsize=128)` over tabula's condensed vector is 12 GB.

**The leave-one-out cache is keyed on `(digest, k_max)` exactly and never slices.** Serving $k = 5$
out of a cached $k = 16$ is free and wrong: step 3 established that slicing is not equivalent to
re-querying when the layout has exact duplicates, so the value of `knn_recall(X, G, 5)` would depend
on which metric ran first. A cache that changes results is the worst kind available, and this one
would be found the expensive way.

The other alternative is genuinely attractive and is recorded as open: a canonical tie-break by
`(distance, index)` inside `_loo_neighbors` would make slicing order-independent *and* make every
neighbourhood metric reproducible under duplicates, which nothing currently is. It moves
`knn_accuracy`'s coincident-point behaviour and the two coincident pins, so it is a declared-change
commit of its own — never a side effect of adding a cache.

**The condensed ideal and the finite-pair mask belong to `EvaluationContext`**, beside `ideal`, for
the same reason `ideal` is there: derived, per-dataset, expensive, and shared by every
representation. Today the three stress functions and `finite_pair_mask` condense the same array four
times per case at a 301 MB transient peak each.

**VNR's `"max"` normaliser is chunked, never squared up**, and the `"mean"` default is also the
cheap path — it needs only $\bar d$ and the per-vertex min, both of which are already computed.
§0.5's `[THESIS EDIT]` argues the modelling case; that the default is *also* the cheaper computation
is recorded here so that nobody optimises back toward `max`.

**NND$_k$ reuses `_loo_neighbors`** — one leave-one-out convention in the package, as of step 3 —
and inherits its documented tie arbitrariness. It is the metric most likely to meet it, since
crowding is what NND measures. The distances among coincident points are all zero either way, so the
*value* is well defined; only the neighbour set is not.

**A collapsed layout gives `nan`, not 0**, in NND, NR and VNR, as it does in SNS. The limit exists
in each case and reporting a limit as a score is how an undefined quantity reaches a table.

**The distributions are normalised by $\bar d$ inside the function and returned whole**, per eqs.
(70)/(71) and §3.1.3's "diagnostics, not scores". Summarising them here would pre-empt the choice
step 11 has to make from the data.

**The golden file grows additively, and there is still no `registry.py`.** Step 4 adds keys to
`make_metric_golden.py` and regenerates; the diff on `metric_golden.json` must contain no modified
lines. Since the generator is both sides of the equality test, an additive diff is the *only*
evidence the existing fourteen values did not move. `MetricSpec`, `METRIC_REGISTRY` and
`evaluate_layout` remain step 7 — the same boundary step 3 held.

## Step 4 departures from `plans/step4-visual.md`, and why

Following the precedent of steps 2 and 3. Nine commits landed against nine planned, and the
plan's own timing table reproduced on re-measurement — 355 → 82 ms and a 301 → 101 MB peak for
`condensed()` at $n = 5007$, against the plan's 370 → 95 ms and 301 → 101 MB — so the argument
for putting the machinery first stands as written. These are the places the plan did not survive
contact.

**VNR under the `"mean"` normaliser is *exactly* `normalized_nn_distance(X, 1)`.** Neither the
plan nor §0.5 says so. Both are $\frac{1}{n}\sum_i \min_{j\neq i}\lVert x_i-x_j\rVert$ divided by
$\bar d$; the sum and the division commute. So the `[THESIS EDIT]` does not merely put NND and
VNR into one family with one normaliser — it makes VNR the $k = 1$ member of that family. The
`"mean"` form therefore carries no information NND$_k$ does not, while the thesis's `"max"` form
does. Pinned as an identity in `tests/test_visual.py`, and confirmed on real layouts below. This
is **the author's call to make**, not the implementer's: the edit is still defensible on the
grounds §0.5 gives, but "report NND$_k$ and VNR" is, under it, reporting one metric at two values
of $k$.

**"Collapses back toward the global NR" is a claim about the normaliser, not about the value.**
Measured on a 4×4 grid plus one distant vertex: VNR$_{\max}$/NR *rises*, 1.21 → 3.74, because NR
falls off as the diameter while VNR$_{\max}$ is held up by the outlier's own well-separated row.
What does collapse is the *spread* of the per-vertex maxima — 0.140 → 0.023 — which is the actual
mechanism: the normaliser stops being per-vertex. The test asserts that, and nothing about the
ratio.

**`edge_length_cv` is not a thesis equation.** The plan cites "eqs. (79), (80)" for it; eq. (79)
is the chord and eq. (80) is the *mean* edge length $\bar L$ — the quantity LWCr normalises by.
The coefficient of variation is spec §2.2's own addition. Both are now in the golden file, so
step 5 inherits a fixed $\bar L$ rather than establishing one.

**`_stress_pairs` takes the `EvaluationContext` itself as `ideal=`.** The plan says only that it
"uses them when the ideal came from a context". A new parameter would have broken §2's three
metric signatures before step 7's registry is written; `ideal=` already admitted a square array
and a condensed vector, so admitting the object that owns both costs no surface. `release()`
also clears the cache wholesale rather than popping `"ideal"`: a condensed vector outliving the
array it was condensed from is a stale copy of the one target the whole ablation rests on.

**The golden file pins the distributions by order statistics.** The plan does not say how to pin
an array-valued metric, and the metric deliberately does not summarise itself (§3.1.3). Three
order statistics each — a summary of the *fixture*, not of the metric — because otherwise a
silent move in either distribution is invisible until step 11 plots it. The diff was 13
insertions and 0 deletions.

**The edge-order trap is real on `digits` and latent everywhere else.** `G.edges()` and the CSR
upper triangle agree exactly on `karate` and on `football_2023`, because those graphs were built
in sorted order — so a test on either would have passed against a wrong implementation. On
`digits` they disagree in **12377 of 14732 positions**: the same multiset of lengths in a
different sequence, which is the shape §1.6 had one layer down. `edge_lengths` is tested there.

## The visual metrics on real layouts — the noise floor, and VNR's two normalisers

Step 4 is greenfield, so unlike step 3 there is no arm to compare against. What had to be
established is what a difference has to beat before it means anything at step 8, and the
mean-versus-max comparison §0.5 promises will happen "once on real layouts".

**Protocol.** SG-*t*-SNE on $A_V$ of `digits`, seven seeds, $k = 16$ (§0.5 decision 6), the same
protocol as step 3's two measurements. Median, IQR and range over seeds.

| metric | median | IQR | range | IQR/median |
|---|---|---|---|---|
| NND$_{16}$ | 0.0449 | [0.0448, 0.0450] | [0.0445, 0.0451] | **0.004** |
| NND$_1$ = VNR(mean) | 0.0123 | [0.0122, 0.0123] | [0.0119, 0.0124] | 0.013 |
| VNR(max) | 0.0069 | [0.0068, 0.0070] | [0.0068, 0.0070] | 0.018 |
| `edge_length_cv` | 1.5280 | [1.5062, 1.5425] | [1.4987, 1.6023] | 0.024 |
| **NR** | **0.0000** | [0.0000, 0.0001] | [0.0000, 0.0001] | **0.540** |

**The crowding metrics are quiet instruments.** NND's relative IQR is 0.4 % — two orders of
magnitude tighter than the effect step 3 had to strain to see in `knn_recall`, so a step-8
difference of a few percent in NND will be readable where the same difference in recall was not.

**NR is in the table as a control, not as a candidate.** It is **not one of the reported
metrics**: the thesis introduces it only to argue against it — "because it depends on the single
closest pair of vertices, it behaves as a worst-case overlap detector… for this reason, we use a
vertex-wise averaged version" — and its closing summary names NND$_k$ and VNR as the
point-crowding scores. §2.2 asks for `node_resolution` in code, and it is implemented and pinned,
because VNR is *defined* by contrast with it and because a metric one argues against should be
computable. What the measurement adds is that the thesis's argument is quantitatively right on a
real layout: NR is $1\times10^{-5}$ on 1797 vertices and swings by 54 % of its median between
seeds, so it reports which seed happened to place two points nearest each other. Read the row as
corroboration of the design decision, not as a warning about a metric anyone planned to report.

**Mean versus max, on five real layouts of one graph.** Five layouts of `digits` — SG-*t*-SNE,
spectral, spring, Kamada-Kawai and a random Gaussian control — ranked by each normaliser.

| layout | VNR(mean) | VNR(max) | NND$_{16}$ | NR | spread of per-vertex maxima |
|---|---|---|---|---|---|
| spectral | 0.0078 | 0.0019 | 0.0256 | 0.0000 | **0.062** |
| sgtsne | 0.0119 | 0.0068 | 0.0445 | 0.0001 | 0.142 |
| spring | 0.0171 | 0.0085 | 0.0486 | 0.0004 | 0.171 |
| kamada-kawai | 0.0278 | 0.0142 | 0.0740 | 0.0013 | 0.134 |
| random | 0.0324 | 0.0115 | 0.0952 | 0.0003 | 0.135 |

**The two normalisers disagree, and the disagreement is not noise.** They order the first three
identically and swap the last two: `"mean"` ranks the random control as the least crowded layout
(0.0319 over five seeds, range [0.0315, 0.0331]) and `"max"` ranks Kamada-Kawai above it
(0.0142 against 0.0109, range [0.0105, 0.0112]) — **disjoint over all five seeds in both
directions**. Spearman between the two columns is 0.90. So the choice of normaliser changes a
reported ordering on real data, which is what §0.5 wanted to know.

**The mechanism §0.5 predicted does occur, on `spectral`.** Its per-vertex maxima have a spread
of 0.062 against 0.13–0.17 for every other layout — a few extreme vertices, so every denominator
is approximately the diameter and `"max"` stops being a per-vertex quantity. It is also where
the two normalisers disagree most in *degree*: spectral is 1.5× below SG-*t*-SNE under `"mean"`
and 3.6× below it under `"max"`. Whether that exaggeration is a defect depends on what is being
asked, which is exactly why both are kept selectable and both are in the golden file.

**What is not settled here.** The Kamada-Kawai/random swap is not explained by the outlier
mechanism — both have ordinary spread — so on that pair neither normaliser is demonstrably
wrong, and this measurement discharges §0.5's promise to *compare* them without deciding which
to feature. Taken with the identity above, the decision the author faces is sharper than §0.5
framed it: `"mean"` is NND$_1$, so featuring it means the point-crowding column is NND at two
values of $k$; `"max"` is a genuinely second measurement, at the cost of being distorted by
outlier vertices on layouts like spectral's.

## Reconciling the code with `thesis.pdf` — the decision, not the list

The divergences themselves are a table in **spec §0.5's session-3 corrections**, which is their one
authoritative copy: `[THESIS EDIT]` is a spec concept, every existing marker lives in that document,
and two copies of a table like that drift inside a single session. What is settled here is what to
*do* about them, which is: nothing yet, deliberately.

- **Three are thesis prose**, and the code must not chase a draft. The `[THESIS EDIT]` mechanism
  exists precisely so that the implementation is not "fixed" toward whichever revision of the PDF
  happens to be on disk. They are the author's to make.
- **Two are code work already scheduled** — `count_intersections` versus eq. (78) at step 5, where
  §2.4 reimplements it anyway, and `fa2` → `fa2_modified` at step 9. Pulling either forward means
  touching the legacy $(d, n)$ path twice.
- **One is genuinely open**: `SpectralLayout` reads the symmetric-normalised Laplacian while the
  thesis defines Hall's formulation on $L_w = D - W$. It is not obvious which side should move, and
  the choice is a modelling one rather than a bug — so it wants the author's call before step 9
  touches that method, not a decision taken here to close a table row.

## Ideal distances: one source, one owner

> Ideal distances come from **`vertex_graph(dataset).distance`**, computed once per dataset per
> run by the **runner**, and shared by every representation of that dataset.

`vertex_graph`'s distance form is by construction the least-derived one available: the stored
distance side if there is one, otherwise the single derivation from the stored affinity. Never the
generic fallback that `edge_graph` and `twin_graph` expose.

**The reason is the central ablation, not tidiness.** SNS and Sil\* score a layout against the
graph's ideal distances. If those came from whichever `LayoutGraph` produced the layout, $A_V$,
$A_E$ and $A_{tne}$ would each be scored against a *different* target — and the comparison between
them, which is the point of the whole experiment, would mean nothing. One target per dataset is
what makes it legitimate.

Ownership is the runner's because the alternatives are both wrong: `Dataset` holds no derived
information (see (11)), and the metric would redo an $O(n^2)$ expansion on every call. The
`max_n` guard (default 20000) lives in `common.ideal_distances`, so there is exactly one place it
can be forgotten.

**Two distinct objects, and only one of them belongs on `LayoutGraph`.** The *edge-level*
distance model — one weight per existing edge — is `LayoutGraph.distance`, an `nx.Graph`, and it
stays sparse: that is what the layout methods consume, and `Unweighted`'s hop counts *require*
non-edges to be absent. The *all-pairs* ideal distances are its shortest-path closure, and they
belong in `common.ideal_distances`, **not** as a `LayoutGraph` property, even though that would
read better. Hanging them there invites `twin_graph(...).ideal_distances`, which is precisely
what one target per dataset forbids. Making the correct thing easy to reach and the wrong thing
require an import is worth the asymmetry.

The redundancy this leaves is real and cheap: `nx.kamada_kawai_layout` and graphviz `neato` both
expand to all-pairs internally, so a KK run does the expansion twice. Two Dijkstra sweeps against
a multi-minute SG-*t*-SNE is not worth coupling the layers to save.

## (5) A metric needs the vertex graph, not just labels

"A metric is a function of the embedded points plus a `Dataset`" is not sufficient. Metrics need
graph structure (kNN recall, crossings, edge lengths), ideal distances (SNS, Sil\*, ADR) and
sometimes affinities (weighted recall). Accurately:

> A metric is a function of the embedded points, the `Dataset`, and the **vertex** `LayoutGraph`.

Nothing is lost, because the vertex graph is a pure function of the dataset — it just has to be
*reachable*, and it must be the un-normalised faithful one rather than whatever the case was laid
out with.

**Build it as an evaluation context, not as a field on `LayoutResult`.** A bundle of
`(dataset, vertex_graph, ideal_distances)` constructed once per dataset and threaded through
`evaluate_layout` shares the expensive parts across every case of that dataset and gives the
`max_n`-guarded expansion a natural owner. Attaching the vertex graph to each `LayoutResult`
would duplicate it per case and re-couple the layers. Step 3/7 work; recorded here so it is not
reinvented.

## Step 3 decisions, pinned before they are discovered the hard way

Five choices the spec leaves open at step 3, settled in advance so that
[`plans/step3-metrics.md`](plans/step3-metrics.md) can be executed without re-deriving them. Each
records the alternative and why it lost.

**`ideal_distances` returns the full `(n, n)` array under every policy.** Under `"lcc"` the
infinities sit outside the largest component; under `"mask"` only genuinely unreachable pairs are
infinite. The alternative — returning the `(n_lcc, n_lcc)` submatrix plus an index vector — is
more honest about what was actually scored, but it pushes index bookkeeping into every consumer,
and every consumer is holding an `(n, d)` layout it would then have to subset in step. Keeping the
shape fixed makes alignment free, and the two policies still give documented, *different* answers,
which is what §6 asks of them.

**`max_n` stays at 20000, and the message quotes the projected allocation.** It is what §2.0 sets,
and the guard is an escape valve rather than a memory policy — the runner is expected to know what
it is asking for. But note what the default actually admits: $20000^2 \times 8$ B is **3.2 GB**, so
the guard does not by itself protect a workstation, which is the reason the error names the number
rather than just refusing. Lowering it to ~10000 (800 MB) would make it genuinely protective at
the cost of deviating from a spec-stated default; revisit if it ever bites, with the change
recorded here.

**`weighted_recall` reads the vertex graph's affinity side, not `dataset.graph`'s `weight`
attribute.** They agree on `karate_weighted` and `tabula` and **disagree on digits**, whose
`weight` attribute is a stored Euclidean distance. §2.1's literal signature implies the graph
attribute; the layering says the affinity form, and the layering wins for the same reason bug §1.6
does — a metric that reads whichever number happens to be on the edge is a metric that inverts
silently on the one dataset where the two differ. Recorded here rather than left to be inferred
from the plan, because it is a deviation from the written signature. ADR takes the same rule and
enforces it by refusing a graph stamped `twin_side="distance"`.

**ADR is not wired into `embedding_metrics_all`; the point-distance slot is closed in prose.**
Three comments in `twin/pipelines.py` promise ADR will fill the slot left by `point_distance_metric`
(spec §1.2). It will not, there. That function has zero callers and zero tests, is superseded by
`batch_layout_metrics` (§3.7), and its Typst emitters branch on the removed `VertexPointDistance`
/ `EdgePointDistance` keys and are rewritten against the long DataFrame at §5.2. Wiring ADR in
means adding untested code to a function scheduled for deletion *and* inventing a table shape that
step 11 discards. The comments are updated to say ADR exists and that this path is not getting it.
Wrong only if a thesis draft needs point-distance tables before step 7 lands.

**The golden file is built on `karate_weighted`, with a deterministic layout, keyed by metric
name.** §6 says `karate_club`, which is no longer a registry name; `karate_weighted` is its literal
successor, since `batch_registries["graph"]["karate_club"]` *was* `nx.karate_club_graph`. It is
also the only choice that works on the merits: on unweighted `karate`, ADR degenerates to eq. (29)
and `weighted_recall` equals `knn_recall` by construction, so a golden file there would pin two
metrics at values that **cannot detect a weighting bug**. Use the `karate_structured_layout`
fixture rather than a seeded layout run — `conftest.py` already states the principle, and a golden
file that runs SG-*t*-SNE pins the backend, so every `sgtsnepi` bump reads as a metric regression.
Key it by metric name so steps 4–6 add keys instead of invalidating the file; one that must be
regenerated wholesale three times will be regenerated without being read.

## `largest_component()` is not built

Dropped from the `Dataset` interface. Subsetting a dataset means subsetting the graph, *every*
label vector and *every* stored matrix — tabula has two, ~50k edges each — while keeping all of
them index-consistent, and **every registered dataset is connected**, so none of that code would
execute.

Same reasoning as `disconnected="components"` (§0.5 decision 3): untested code for a case that
does not occur is worse than an honest absence, and a method that exists will get written in a
hurry at step 3 when SNS turns out to want it. `is_connected()` stays — it is cheap and the
connectivity report uses it.

## `expects` must be asserted, since nothing branches on it

`EmbeddingSpec.expects` survives the §3.5 rewrite as declared metadata for the manifest and
report, but the runner no longer converts at the boundary, so no code path depends on it being
right. Metadata nothing depends on drifts.

One parametrised test over the embedding registry: hand each method a `LayoutGraph` that records
which of `.affinity` / `.distance` was touched, embed a small graph, assert the accessed set
matches the declaration. It costs almost nothing and fails the moment a method starts reading the
other side.

## Weight-source consistency

Two independently stored sides can disagree about which pairs are close, and the failure is silent
and severe: the layout optimises against the affinity while SNS and Sil\* score against the
distance, so a method could look good or bad for reasons unconnected to the method.

`check_weight_consistency(dataset)` runs whenever both sides are stored:

- **Global Spearman** over shared edges — strongly negative but *not* $-1$, since $\rho_i$ and
  $\sigma_i$ vary by row.
- **Median per-row Spearman** — the sharp test. Within a row, UMAP membership is a strictly
  decreasing function of distance, so this should be $-1$.

**Measure it before pinning it.** The stored `A` is fuzzy-union symmetrised, so `A[i,j]` mixes in
row $j$'s $\rho_j,\sigma_j$ and within-row monotonicity is not guaranteed a priori. If the observed
median is not exactly $-1$, record symmetrisation as the cause rather than treating it as a data
fault; a materially weaker correlation (worse than $\approx -0.9$) is the genuine signal. Do not
hardcode $-1$ into a test that later gets "fixed" by loosening it.

### Measured — and neither expectation held

| Statistic | Expected | Measured on tabula |
|---|---|---|
| Global Spearman over shared pairs | "strongly negative" | **−0.2364** |
| Median per-row Spearman | $-1$ | **−0.7820** over 5007 rows |
| Median per-row over one-directional pairs | *(not anticipated)* | **exactly −1.0** over 3819 rows |

The first two are weaker than the argument predicted, and the second is past the $\approx -0.9$
line that was supposed to signal genuine inconsistency. So a third statistic was added to settle
what the first two could not.

`check_weight_consistency` restricts to pairs the **raw** stored matrix recorded in one direction
only — where fuzzy union cannot have mixed row $j$'s parameters into row $i$, so `A[i,j]` is
$\mu_{ij}$ exactly. There the within-row correlation is $-1$ in *every* scored row. The
relationship is exact wherever it can be seen unmixed; the degradation is symmetrisation, and the
weak global figure is per-row $\sigma_i$ variation decoupling membership from distance across
rows, which was anticipated. Tabula's two sides do **not** disagree.

All four numbers are pinned in `tests/test_datasets_registry.py` and printed by
`python -m twin.report`. The $\approx -0.9$ threshold should be read against the *unmixed*
column, not the raw per-row one.

## `LabelSet` carries scoring and display roles separately

Not one `role` field. football scores `conference` and takes hue from it, with `division` as an
ordered sub-label supplying shade. Tabula **inverts** that: the fine-grained `cell_ontology_class`
(13 classes) is what the method claims to resolve and is the scoring primary, while the coarse
`compartment` (4 classes) is its parent and supplies hue. One field cannot express both
arrangements, and collapsing them would force a dataset-name branch somewhere.

`donor` and `method` are auxiliary batch covariates and are scored **deliberately** — a layout
separating cells by donor rather than by cell type is displaying an artifact, so a *low* silhouette
is the good outcome there. Reporting it is how that becomes visible instead of assumed away.

## (13) Endpoint exclusion in crossing detection

**Post-filter by vertex identity, do not trim the curves.** Compute the intersection geometry of
the two polylines, then discard intersection points lying within `tol` of the position of a vertex
the two edges *share by index*. Shared-ness comes from the edge index pair, never from comparing
coordinates.

This is what §2.4 asks for. Trimming the polyline endpoints would be cheaper but would also
discard genuine crossings that happen to occur near a coincident vertex — and two distinct
vertices landing on the same coordinate is the crowding case the convention exists for.

## (14) Mapping intersection geometry to `cr`

`cr(e_a, e_b)` = **number of connected components of the filtered intersection geometry**, after
de-duplicating points that lie within `tol` of one another:

| geometry | `cr` contribution |
|---|---|
| `Point` | 1 |
| `MultiPoint` | one per point, after `tol`-dedup |
| `LineString` / `MultiLineString` (collinear overlap) | 1 per component (spec §2.4) |

The `tol`-dedup is what makes the count independent of `n_samples`: a near-tangency sampled at 64
points can yield two intersection points a floating-point distance apart, which is one crossing,
not two. Degenerate incidents (overlap components) are counted and reported separately.

## (15) `gamma=0` short-circuits to two samples

**Yes.** `edge_curves(..., gamma=0)` emits 2 samples per edge regardless of `n_samples`.

At γ=0 the curve *is* the chord, so a 64-point polyline is 63 redundant segments, and both the
STRtree prefilter and the pairwise refinement scale with segment count. A test asserts that γ=0
gives identical crossing counts at `n_samples=2` and `n_samples=64`, so this is verified rather
than assumed. Both variants still go through one code path, as §2.4 requires.

## (18) Where the exclusion and diagnostic counts live

**Not in one column.** The tidy metric frame carries:

- `status` — `ok` / `skipped` / `distribution` (see (20));
- `n_excluded` — the metric's *own* exclusions only: zero-degree vertices for recall, isolated
  vertices for graph-side accuracy.

Everything else goes to a **separate diagnostics table** keyed by the same row key: `n_ties`,
`n_zero_distance`, `n_infinite_pairs`, `n_degenerate_crossings`.

They have different denominators — vertices, vertex pairs, edge pairs — so a single column would
be summing incommensurable things. `KnnAccuracyReport` already follows this shape.

## (20) Array-valued metrics: swept versus distribution

**`MetricSpec` declares which it is; nothing is inferred from the return shape.**

- `sweeps=("k",)` → the array is unpacked into one row per `k`, with the `k` column filled.
- `returns="distribution"` → the array goes to `distributions/<key>.npz`; its row in the frame
  carries `status="distribution"`, `value=NaN`, and the array's key.

Both cases return an `np.ndarray`, so shape cannot distinguish them. Declaring it also lets the
driver size the run before executing it.

## (21) Key encoding for persisted artifacts

Resumability depends on this, so it is a versioned on-disk contract.

```
key_id(key, **extra) -> "v1-{slug}-{digest}"
```

**`digest`** — first 16 hex characters of BLAKE2b over a canonical serialisation of the *complete*
key plus extras. Not Python's `hash()`, which is salted per process and would break resumability
across runs.

**Canonical form**, JSON with `sort_keys=True` and `separators=(",", ":")`, over:

```python
{"axes":  [[axis, name, [[k, v] for k, v in sorted(params)]] for axis, name, params in key],
 "extra": {k: v for k, v in sorted(extra.items())}}
```

with scalar normalisation applied to every value:

| type | encoding |
|---|---|
| `bool` | `"true"` / `"false"` — before the numeric rule, since `bool` is an `int` |
| `int`, `float` | `repr(float(v))` |
| `None` | `"null"` |
| everything else | `repr(v)` |

Real numbers collapse to one float form deliberately: `lambda_par=1` and `lambda_par=1.0` are the
same configuration and must resume onto each other. `repr(float(...))` is shortest-round-trip in
Python 3, so it is stable across runs and platforms.

**`slug`** — a readable prefix, `axis=name[p=v,p=v]` segments joined by `__`, restricted to
`[A-Za-z0-9._=,-]` with everything else replaced by `-`, truncated to 80 characters. It is
decoration for humans reading a directory listing. **Collision-freedom rests entirely on the
digest, which is taken over the untruncated key** — two cases whose slugs truncate identically
still get different filenames.

**`extra` carries `seed` and any λ**, which are not batch axes today but must be part of the
identity of a persisted artifact.

Bumping `v1-` invalidates every existing run directory, so it is changed only deliberately.

## (22) Two-stage resumability

Two stages, two markers:

- **Embed stage** — `layouts/<key>.npz`, written to a temporary file and `os.replace`d into
  position, so a partial file never looks complete after a kill.
- **Metric stage** — `metrics/<key>.parquet`, one shard per case, concatenated at the end.

A case skips the embed if its layout exists, and skips the metric stage if its shard exists **and**
the metric-set digest recorded inside the shard matches the current one. Adding a metric therefore
re-runs metrics against persisted layouts without re-running any embed — which is the entire
reason layouts are persisted (spec §4.6). Sharding avoids rewriting one large parquet on every
case.

## (26) `DatasetStyle` versus the visual-regression baselines

**Pixel baselines guard *refactors*, not figures.** They answer one question: did a change that
was supposed to be pure reorganisation quietly alter the data behind the picture? A commit that
*intentionally* changes that data — the digits affinity fix, a palette generalisation — declares
it and regenerates the baselines in the same commit. Figures changing is not, by itself, a
failure; figures changing in a commit that claimed to be a refactor is.

**Two phases, two commits, per CLAUDE.md's "never mix a refactor with a behaviour change".**

1. **Baselines first.** Render and commit the baseline PNGs from the *pre-refactor* code. Then the
   §3.6 refactor reproduces the existing `3*l + 1 + s` palette exactly and must match those
   baselines bit-for-bit. This is the commit that proves the decomposition is behaviour-preserving.
2. **Generalisation second.** A separate commit replaces the hardcoded three-shade arithmetic with
   the general ordered-sub-label rule, declares that it changes the figures, and regenerates the
   baselines in that same commit.

Doing it in one step would make the baselines unable to distinguish "the refactor broke it" from
"the new colour rule is different on purpose".

**Before step 10, use data-level pins instead.** Step 2 moves the label derivations out of
`paper_figures.py`, which is a refactor and needs a guard — but pixel output is sensitive to
matplotlib and freetype versions, so bit-for-bit comparison that early would produce false
failures. Pin the derived **label vectors and colour arrays** exactly, plus a renders-without-error
smoke test; those are precise about what the label move can actually break. Pixel baselines land
at step 10, where the renderer refactor is exactly the kind of change they are good at guarding.
