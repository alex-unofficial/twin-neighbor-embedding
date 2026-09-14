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
(`detour_ratio(X, E, G, gamma, n_samples)`) is reordered to match `edge_curves`. ⚠️ Amended at
step 5: `n_samples` is gone from both signatures — `edge_curves` returns control points and
`detour_ratio` uses the closed-form arc length. See (27) and spec §2.2.

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

> ⚠️ **Partly superseded.** The noise-floor table below stands. The mean-versus-max comparison
> was the input to a decision since taken — see "VNR's normaliser — closed" — and two of its
> readings did not survive later work: the `spectral` ordering flip was measured on the old
> $L_{	ext{sym}}$ spectral layout and does not reproduce against the corrected one, and the
> `random` > Kamada-Kawai ordering turned out to be a real property rather than an artefact of
> the global normaliser. Both are corrected there.

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

## VNR's normaliser — closed: the per-vertex median

Settled by the author after the identity below was found. **`normalize="median"` is the default**;
`"max"` (eq. 69) and `"mean"` stay selectable. This supersedes §0.5's original edit, and it is
still **one** `[THESIS EDIT]`, not two — eq. (69) has never changed, and the `"mean"` decision
only ever lived in §0.5.

**Why `"mean"` failed.** It divides a per-vertex numerator by a global constant, and a constant
factors out of a sum:
$\frac{1}{n}\sum_i \min_{j\neq i}\lVert x_i-x_j\rVert/\bar d = \mathrm{NND}_1(X)$, exactly. A
"vertex-wise node resolution" divided by one global number is definitionally not vertex-wise. The
outlier diagnosis that motivated the edit was right; the fix chosen made the two metrics identical.

**Why `"max"` is not the fallback.** It fails in *both* regimes, which is a stronger objection
than the outlier failure alone. Where the maxima are near-homogeneous — the ordinary case — it
reduces to $\mathrm{NND}_1\cdot\bar d/M$, a rescaled NND$_1$. Measured against that closed form on
five real layouts of `digits`: within **0.5–5.8 %** everywhere. So it is redundant when the layout
is compact and outlier-driven when it is not.

**Why the median.** Per-vertex, so it does not factor out; robust, because one distant vertex
moves an order statistic essentially not at all; bounded in $[0, 1]$, since the minimum is no
greater than any order statistic of the same set. On the motivating pair — a clean grid plus one
distant vertex against a grid plus a genuine near-overlap — it is the only per-vertex form that
gets the ordering right:

| outlier distance | mean | max | row-mean | **median** |
|---|---|---|---|---|
| 10 | 0.4643 | 0.1494 | 0.4213 | 0.5082 |
| 50 | 0.4876 | 0.0746 | 0.2443 | 0.5175 |
| 200 | 0.4963 | 0.0627 | 0.1237 | 0.5185 |
| 1000 | 0.4992 | 0.0596 | 0.0733 | **0.5188** |
| *near-overlap reference* | 0.4165 | 0.2577 | **0.4283** | 0.4575 |

A row-**mean** was considered and rejected on this table: it attenuates the outlier by $1/n$ and
nothing more, so it drops below the crowded reference already at an outlier distance of 10. (The
median column here includes the self-distance; excluding it, as the implementation does, gives
0.4860 against the reference's 0.4280 — same ordering, and the difference is why the diagonal
handling is tested.)

Seed-noise band, seven seeds of SG-*t*-SNE on `digits`: median **0.0126**, IQR [0.0125, 0.0126],
relative IQR **1.2 %** — in line with the other crowding metrics.

**Two claims from the earlier round that did not survive re-measurement.**

- **The "third strike on `spectral`" is withdrawn.** It was measured on the *old*
  $L_{\text{sym}}$ spectral layout, whose $\bar d$ was 3.06× its median pairwise distance. Against
  the corrected Hall layout the effect is gone: `median` and `mean` now order all five layouts
  identically, and both put spectral last. The case for the median rests on the identity, the
  both-regimes argument against `max`, robustness and boundedness — **not** on a changed ordering
  on real data, which no longer occurs.
- **The `random` > Kamada-Kawai ordering is not an artefact of globality.** $\bar d$ is not
  inflated for either (1.016 and 1.063 times the median pairwise distance) and the median
  normaliser preserves the ordering. It is real, and it means something — see the next section.

## VNR and NND penalise exactly what a good layout of clustered data does

The `random` > Kamada-Kawai result above is not an anomaly to be explained away. A uniform random
point set has a *lower* mean nearest-neighbour distance than a maximally spread one — hexagonal
packing beats Poisson by roughly 2× at equal density — so a random control beating KK on typical
spacing means **KK's layout is non-uniform in density**. Which is exactly what a structure-preserving
layout of clustered data produces: tight clusters, empty gaps, and therefore small typical
nearest-neighbour distances *inside* the clusters.

Five layouts of `digits`, one shared evaluation context, $k = 16$:

| layout | NND$_{16}$ | VNR (median) | ADR ↓ | recall ↑ |
|---|---|---|---|---|
| spectral | 0.0114 | 0.0092 | **0.0277** | 0.3398 |
| sgtsne | 0.0445 | 0.0122 | 0.0844 | **0.6398** |
| spring | 0.0486 | 0.0189 | 0.0952 | 0.3585 |
| kamada-kawai | 0.0740 | 0.0279 | 0.1627 | 0.2486 |
| **random** | **0.0952** | **0.0306** | **0.9968** | **0.0090** |

**The random control is first on both crowding metrics and last on both structural ones.** Its ADR
of 0.9968 is the value a layout unrelated to the graph is supposed to have, and its recall of
0.0090 is chance. Nothing is wrong with the implementations: uniformity is what NND and VNR
measure, and a structureless layout is uniform.

**Consequence, and it is a reporting rule rather than a caveat.** Neither NND nor VNR can be read
as "higher is better" standing alone — they reward the opposite of what ADR and recall reward, and
a layout can be driven to the top of both by discarding all structure. They must be reported
jointly with the structural metrics. This is the concrete demonstration of the degeneracy, on real
data rather than in the abstract, and it arrived as a by-product of the normaliser question.

### Suggested wording for thesis §3, for the author to place

`thesis/` is read-only to this project's agents (CLAUDE.md), so the prose is offered here rather
than inserted. It belongs after "We use both as representative point-crowding metrics", in
`metrics.typ`. Numbers are the table above; `#sgtsne` is the existing macro and there is no
`digits` macro.

> Neither metric can be read on its own as a measure of layout quality, and this is a property of
> what they measure rather than a limitation of how they are computed. Both reward *uniform* point
> spacing, whereas a layout that preserves the structure of clustered data is deliberately
> non-uniform: it places tight groups with empty space between them, and the typical
> nearest-neighbor distance inside such a group is consequently small. A uniformly random point
> set, by contrast, has no such variation in density. The two effects act in the same direction,
> and the resulting degeneracy is easily large enough to invert a ranking. On the digits graph, a
> random Gaussian layout scores highest of every layout we consider on both crowding metrics —
> $\mathrm{NND}_{16} = 0.095$ and $\mathrm{VNR} = 0.031$, against $0.045$ and $0.012$ for
> SG-$t$-SNE — while simultaneously scoring worst on both structural metrics, with
> $\mathrm{ADR} = 0.997$, essentially the value expected of a layout bearing no relation to the
> graph, and neighborhood recall of $0.009$, which is chance.
>
> We therefore treat $\mathrm{NND}_k$ and $\mathrm{VNR}$ as measurements of readability that are
> meaningful only alongside the structural metrics of the preceding section, and never as scores
> by which layouts are ranked on their own. This is the same reasoning that motivates the
> pluralistic treatment described at the end of this section, made concrete: a layout can be
> driven to the top of both crowding metrics by discarding all structure.

## `SpectralLayout`'s Laplacian — closed, and §2.2's claim measured

The one genuinely open divergence in the §0.5 reconciliation table. Settled by the author:
**`variant="hall"` is the default, `variant="degree_normalized"` is implemented and reported
once, and the old behaviour is removed.**

**The argument is internal consistency, not baseline strength.** The previous code returned the
eigenvectors of $L_{\text{sym}}$ directly — which is neither eqs. (12)-(14) nor Koren's
degree-normalised drawing, but the $L_{\text{sym}}$ eigenvector embedding used in spectral
clustering, i.e. Koren's premultiplied by $D^{1/2}$. (Not Ng-Jordan-Weiss proper: NJW
additionally row-normalises the eigenvector matrix to unit rows before clustering.) That
$D^{1/2}$ factor scales each vertex by the square root of its own degree and pushes hubs
outward. Thesis §2.2 says the opposite — "high-degree vertices tend to collapse toward the
center" — so the figures would have contradicted the prose describing them. A weak baseline is a
fairness problem and is arguable; a baseline that behaves opposite to its own description is an
inconsistency a reviewer can see in one glance at a figure.

### §2.2's claim, measured on the roster

The claim had never been checked here — it was repeated from the literature. Spearman(weighted
degree, radius from the centroid), $d = 2$, all ten registered datasets:

| dataset | *n* | degree ratio | (a) Hall | (b) Koren | (c) old code |
|---|---|---|---|---|---|
| ring_of_cliques | 40 | 1.3 | −0.867 | −0.867 | +0.867 |
| tree_of_cliques | 157 | 1.3 | −0.844 | −0.844 | +0.592 |
| small_world | 150 | 1.3 | −0.078 | −0.069 | +0.094 |
| **barabasi_albert** | 1000 | **20.1** | **−0.518** | −0.207 | +0.156 |
| karate | 34 | 17.0 | −0.535 | −0.194 | +0.496 |
| **karate_weighted** | 34 | **16.0** | **−0.543** | −0.080 | +0.658 |
| football_2023 | 132 | 1.3 | +0.178 | +0.228 | +0.272 |
| sbm_er_3modes | 210 | 1.8 | −0.612 | −0.064 | +0.342 |
| **digits** | 1797 | **8.4** | **+0.145** | +0.171 | +0.558 |
| tabula | 5007 | 4.0 | +0.009 | +0.014 | +0.305 |

**§2.2 is confirmed where there are hubs, and is not a general statement.** On the three graphs
with real degree heterogeneity and hub structure — `barabasi_albert` (the scale-free case, 20×),
`karate` and `karate_weighted` (17×, 16×) — Hall's formulation draws hubs inward at ρ ≈ −0.52 to
−0.54. On the two $k$-NN graphs it does not: `digits` is +0.145 and `tabula` +0.009, and there
weighted degree reports local density rather than hubness. So **§2.2 wants a qualifier rather
than a rewrite** — say it of graphs with hub structure, and note the $k$-NN graphs as the case
where it does not apply. Better found here than in the viva.

⚠️ **Do not read the near-regular rows as evidence either way.** At a degree ratio of 1.3 the
radii are nearly constant, so a 14 % modulation by $\sqrt{\deg}$ reorders every rank and
Spearman saturates on noise — which is exactly how `ring_of_cliques` manages ∓0.867 in adjacent
columns. Only the heterogeneous rows carry information.

The old column is the finding that closed the question: **positive on all ten**, without
exception.

### What it costs in quality, and why (b) is kept

Same graph, different layouts — within the ADR comparison rule (§2.1, §5.3); these are not
cross-graph aggregates.

| dataset (degree ratio) | variant | ADR ↓ | SNS ↓ | recall ↑ | Procrustes vs (a) |
|---|---|---|---|---|---|
| karate_weighted (16×) | (a) Hall | 0.310 | 0.397 | 0.452 | — |
| | (b) Koren | 0.334 | 0.434 | **0.566** | 0.107 |
| | (c) old | 0.553 | 0.517 | 0.354 | 0.301 |
| football_2023 (1.3×) | (a) | 0.311 | 0.542 | 0.530 | — |
| | (b) | 0.310 | 0.523 | 0.532 | 0.004 |
| | (c) | 0.311 | 0.523 | 0.534 | 0.005 |
| digits (8.4×) | (a) Hall | 0.028 | 0.656 | 0.340 | — |
| | (b) Koren | 0.028 | 0.666 | 0.341 | 0.005 |
| | (c) old | 0.174 | 0.679 | **0.124** | 0.037 |

On `football_2023` all three agree to three decimals — degrees run 10 to 13, so
$D^{1/2}\approx$ constant and the distinction vanishes. That is the control confirming the
mechanism is the degree scaling and nothing else.

(b) is at least as good as (a) everywhere and clearly better on `karate_weighted` (recall 0.566
against 0.452), which is why it is implemented and will be reported once: a single line in §5
pre-empts "the baseline was under-powered" without restructuring the derivation. (a) is the
default because it is what eqs. (12)-(14) derive, which keeps §2.1.2, §2.2 and §5 mutually
consistent at zero prose cost. Spectral is deterministic, so reporting the second variant costs
no seed axis.

### The eigensolver is not independent of the choice

$L_{\text{sym}}$'s spectrum is bounded in $[0, 2]$ for any graph; $L^w = D - W$ runs to
$2d_{\max}$ — 142 on `barabasi_albert`, 52 on `karate_weighted`. ARPACK converges on
well-separated extremal eigenvalues, and `which="SM"` on a spread spectrum is the case it is
worst at, so **switching to Hall makes §1.5's fragility worse rather than neutral**;
normalisation had been masking the conditioning. Shift-invert ($\sigma = 0$, `which="LM"`) is
insensitive to the spread, so §1.5's solver fix landed in the same commit rather than at step 9.

Measured on all ten datasets: subspace overlap **1.0000** against `which="SM"` everywhere, and
**3.1× faster on tabula** (1.90 s → 0.61 s), 1.8× on digits, *slower* on barabasi_albert
(0.12 s → 0.44 s). In fairness to §1.5's wording: no convergence *failure* was observed at this
roster's sizes, so the case for shift-invert here is conditioning and throughput, not an
observed break. $\sigma = 0$ factorises a singular matrix and could in principle fail; it did
not, on any of the ten.

### A determinism gap found while testing

ARPACK draws its starting vector at random. Successive spectral runs therefore differed by
eigenvector **sign**, while `EMBEDDING_REGISTRY` declares the method `stochastic=False`. No
metric in the suite could see it — every one is reflection-invariant — but a persisted layout
(step 7) or a committed figure would not have reproduced, and the declaration drives whether a
case gets the seed axis at all (§4.1). A seeded `v0` makes repeated runs bit-identical.

Not `v0 = ` $\mathbf{1}$: that is exactly $L^w$'s null eigenvector, and starting ARPACK on an
exact eigenvector is a degeneracy not worth relying on even where it happens to work.

## Reconciling the code with `thesis.pdf` — the decision, not the list

The divergences themselves are a table in **spec §0.5's session-3 corrections**, which is their one
authoritative copy: `[THESIS EDIT]` is a spec concept, every existing marker lives in that document,
and two copies of a table like that drift inside a single session. What is settled here is what to
*do* about them, which is: nothing yet, deliberately.

- **Five are thesis prose**, and the code must not chase a draft. The `[THESIS EDIT]` mechanism
  exists precisely so that the implementation is not "fixed" toward whichever revision of the PDF
  happens to be on disk. They are the author's to make, and `thesis/` is read-only to the
  implementation (CLAUDE.md). Three from session 3 — the tie convention in eq. (60), the
  disconnected-components sentence in §2.1.1, and `fa2`'s naming — plus two from session 4:
  - **§2.2's "high-degree vertices tend to collapse toward the center" wants a qualifier.**
    Measured on this roster it holds where there is hub structure ($\rho = -0.52$ on
    `barabasi_albert`) and not on the $k$-NN graphs (+0.145 on `digits`). Table above.
  - **§3 owes a sentence that NND$_k$ and VNR are not standalone scores**, since they reward
    uniformity and a random control tops both while coming last on ADR and recall. Suggested
    wording is in this file, beside the measurement.
- **Two are code work already scheduled** — `count_intersections` versus eq. (78) at step 5, where
  §2.4 reimplements it anyway, and `fa2` → `fa2_modified` at step 9. Pulling either forward means
  touching the legacy $(d, n)$ path twice.
- ~~**One is genuinely open**: `SpectralLayout`~~ — **closed in session 4** (`967d824`). The code
  moved to Hall's $L_w = D - W$; Koren's variant is selectable and reported once. What settled it
  was not that the old reading was weaker but that it contradicted §2.2's own description of the
  baseline. See "`SpectralLayout`'s Laplacian — closed" above.

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

**Mechanism updated at step 5, rule unchanged** (see (29)): exclusion is still by vertex identity
and never by comparing coordinates, but it now filters roots of the quartic rather than shapely
intersection geometry. There is no polyline left to trim, so the cheaper-but-wrong option this
decision rejected no longer exists as a temptation.

## (14) Mapping intersection geometry to `cr`

> **Superseded by (29).** Crossings are no longer computed by intersecting polylines, so there
> is no shapely geometry to classify. Kept for the record; the conventions it encodes — overlap
> counts as 1, endpoints excluded by identity — survive unchanged in (29).

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

> **Superseded by (27) and (30).** There are no samples to short-circuit. γ=0 places $B_1$ at the
> chord midpoint, and the $\lVert D\rVert \approx 0$ guard dispatches to a segment–segment test.
> The requirement it protected — both variants through one code path — is unchanged.

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
`n_zero_distance`, `n_infinite_pairs`, `n_degenerate_crossings`, `n_near_tangential` (double roots in
the crossing solve — see (29)).

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


## Step 5 pre-work: the crossing algorithm, settled before implementation

Settled in discussion *before* `crossings.py` was written, so the design is on record rather than
discovered during implementation. The derivation, the measurements and the degeneracy analysis are
in [`CROSSING_GEOMETRY.md`](CROSSING_GEOMETRY.md); spec §2.4 Layers 1–2 were rewritten to match and
§0.5 carries the change log. This section records the *choices* and what each one rules out.

### (27) Curves are represented by control points, not samples

`edge_curves` returns an `(m, 3, 2)` array of Bernstein control points. The drawn curve is exactly
a quadratic Bézier — the `plots.py` kernel with parameter γ is the Bézier with
$B_1 = 2\gamma P_1 + (1-2\gamma)(P_0+P_2)/2$ — so three points describe it completely and every
step downstream is algebraic.

`n_samples` is **removed, not defaulted.** Sampling was never a modelling choice; it was inherited
from the rendering code, where 64 points buy smoothness at display resolution. As a geometric
tolerance the same accuracy needs $N \ge \sqrt{\lVert D \rVert / 2\delta}$, typically 4 to 16.
Keeping the parameter would invite tuning a number that no longer means anything.

⚠️ The control points in `plots.py` are **not** Bézier control points except at γ=0.5. The
conversion is the one piece of geometry `plots.py` and `crossings.py` must share, which is why it
lives in `twin/geometry.py` and in neither of them. If it is duplicated, the drawn curve and the
measured curve drift apart silently and no test will notice.

### (28) Crossings are computed exactly, by root-finding, not by intersecting polylines

Two quadratic Béziers meet in at most four points, and all four are roots of one quartic obtained
by eliminating one curve's parameter. Per candidate pair: build the quartic (~30 multiply–adds),
reject on a Bernstein sign test, solve by companion-matrix eigenvalues, filter the roots.

Why this rather than the sampled-polyline approach the spec originally described:

- It is **exact.** There is no geometric tolerance, so the reported crossing number does not depend
  on a sampling parameter. "Your crossing count is an artifact of how finely you sampled" is the
  obvious objection to this metric, and it is worth being able to answer it flatly.
- It is **cheaper by orders of magnitude.** $4.8 \times 10^{12}$ segment tests on tabula at 64
  samples becomes $O(m \log m + K)$ at ~3.1 µs per candidate pair.
- The count is **independent of `n_samples`**, which is what (14)'s `tol`-dedup and (15)'s γ=0
  short-circuit existed to achieve. Both are therefore retired rather than reimplemented.

Validated before adoption: over 4000 random pairs the intersection counts were distributed
{0: 2211, 1: 1165, 2: 548, 3: 49, 4: 27}, confirming the bound of four, with every reported point
matching to $<10^{-7}$ except two near-tangential cases; against a 1200-segment polyline reference
on 200 further pairs the counts agreed 200/200.

### (29) `cr(e_a, e_b)` is the number of accepted roots

Replaces (14). A root is accepted when it is real, $s \in [0,1]$, **and** the recovered
$t = -P_{02}(s)/\beta$ is also in $[0,1]$ — filtering on $s$ alone is wrong, because the implicit
equation describes the whole parabola rather than the arc. Three cases are settled before
root-finding:

| case | detection | `cr` |
|---|---|---|
| identical or collinear-overlapping curves | every quartic coefficient ≈ 0 | 1, per spec §2.4 |
| tangency | double real root | 1, and counted in `n_near_tangential` |
| shared endpoint | root coincides with a vertex the two edges share **by index** | excluded |

The last is (13) unchanged in substance: exclusion is still by vertex identity, never by comparing
coordinates. Only the mechanism moved, from filtering shapely geometry to filtering roots.

### (30) Three degeneracy guards, each a determinant with a geometric reading

| guard | quantity | meaning | handling |
|---|---|---|---|
| straight edge | $\lVert D \rVert \approx 0$, where $D = 2\gamma\bigl(P_1 - \tfrac{P_0+P_2}{2}\bigr)$ | the curve is its chord | segment–segment test |
| collinear controls | $\beta = 2\det(B_1-B_0,\, B_2-B_0) \approx 0$ | the parabola collapses to a line | segment path over the swept interval |
| parallel bulges | $Q_4 = -16\det(D_A, D_B)^2 \approx 0$ | the two curves bow the same way | deflate to the true degree before the companion matrix |

**The third is the one that will actually bite.** Forming the companion matrix divides by $Q_4$,
and with a shared γ the guard trips whenever two edges bow in parallel — common in any layout with
orientational coherence, not an exotic case. It needs its own test, with constructed curves rather
than random ones, because random sampling will rarely hit it.

Note what these quantities are *not* doing. $D$ and the control-triangle area bound nothing in this
design; they appear only as conditioning quantities. An earlier draft used them for bounding-hull
rejection and for a subdivision stopping criterion, and both were removed (§0.5). Do not
reintroduce a hull test on their account.

### (31) Stay in NumPy until $K$ says otherwise

A compiled implementation would be a **different design**, not a translation: with early exit and
cheap branching the rejected bounding-triangle filter becomes worthwhile again and per-pair
processing becomes correct. That is two algorithms to test, not one algorithm in two languages.

The decision rests on $K$, the number of pairs surviving the broad phase — the one quantity in the
cost model that has not been measured. NumPy runtime is $\approx K \times 3.1$ µs, so $K=10^6$ is
3 s per layout and $K=10^8$ is 5 minutes. Below roughly 30 s per layout there is no case for
compiling: the metric runs once per layout, not in an inner loop.

Escalate in this order first:

1. the exact bounding box (already specified — cuts $K$ by about a third for 123 ns per edge);
2. parallelism across layouts, since the experiment grid is embarrassingly parallel;
3. the length-thresholded crossing curve, which shrinks $m$ directly and is what the thesis's
   claim (c) is actually about.

⚠️ **Print $K$ as soon as the broad phase runs, on the largest layout.** Nothing downstream should
be built on a guess about it.

### (31a) $K$ measured, and the two calls taken at plan review

Added when step 5 was planned, before any of it was written. `plans/step5-crossings.md` has the
detail; this is the record of what was settled.

**$K$ is no longer the unmeasured quantity.** SG-*t*-SNE twin layouts, seed 0, γ=0.5, layout
normalised to a unit box, exact boxes, `shapely.STRtree` bulk query:

| dataset | m | K | K/C(m,2) | query |
|---|---|---|---|---|
| football_2023 | 735 | 27 058 | 0.100 | 0.02 s |
| digits | 14 732 | 1 380 736 | 0.0127 | 1.1 s |
| tabula | 49 368 | **11 375 494** | 0.00934 | 9.5 s |

(31) guessed 1e6 and 1e7 for the last two; the cost model is sound. Straight and curved differ by
0.2 % in $K$ — but not necessarily in cost, since at γ=0 no quartic is built at all, so §4.3 may
cost ≈1.2× rather than 2×. Time them separately.

⚠️ **(31)'s cost model omits the broad-phase query** — 9.5 s and a 365 MB return array on tabula,
27 % of the runtime. And **blocking the solve does not cap that memory**: the bulk query
materialises every pair before any block loop begins, so capping it needs *edge-chunked queries*.

**Decision 1 — the NumPy/compiled call is deferred to an end-to-end measurement.** The estimate
assembled from per-stage costs puts tabula at ~35–45 s against (31)'s ~30 s line, but it assumes
3.1 µs/pair and the measured 35 % Bernstein rejection, neither of which has run through the real
code path. Take the reading at step-5 commit 4, then apply (31). Recorded so that the estimate is
not mistaken for the measurement.

⚠️ **Escalation rung 2 — parallelism across layouts — belongs to step 7's runner design**, not to
step 5 and not improvised at step 8 around a slow tabula run. Where the grid's parallelism lives
is a property of the runner.

**Decision 2 — `plots.py::count_intersections` keeps `include_adjacent=False`, permanently.**
Reimplementing it on layer 2 (§2.4) forces a choice, because `crossing_pairs` defaults to the
thesis convention `include_adjacent=True`. It keeps the old convention so the reimplementation is
**behaviour-preserving and the paper's Figure 3 does not move** — a refactor must not become a
silent result change.

This makes the `count_intersections`-versus-eq.-(78) row in §0.5's divergence table a **standing
intentional divergence for that figure**, not a scheduled fix. The table currently reads as the
latter ("step 5 reimplements it on layer 2 anyway"), which will be wrong the moment step 5 lands:
the reimplementation happens and the divergence deliberately survives it. Step 5's last commit
moves that row.

## Step 5 departures from `plans/step5-crossings.md`, and why

Nine commits, as planned, and the plan held. Four places where implementation contradicted it or
the guide, each found by measurement rather than by reading.

### (32) The shared-vertex root is **divided out**, not filtered out by distance

(13) says to discard intersection points lying within `tol` of the position of a vertex the two
edges share *by index*. Shared-ness by identity survives; **the `tol` does not**, because there is
nowhere safe to put it. On a `digits` twin layout the distance from a root to its pair's shared
vertex is a **continuum** with no gap:

| distance to the shared vertex | roots |
|---|---|
| $< 10^{-12}$ | 166 286 |
| $[10^{-12}, 10^{-9})$ | 208 |
| $[10^{-9}, 10^{-7})$ | 54 |
| $[10^{-7}, 10^{-6})$ | 326 |
| $[10^{-6}, 10^{-5})$ | 1 488 |
| $[10^{-5}, 10^{-4})$ | 3 246 |

`football_2023` happens to show a clean gap at $[10^{-7}, 10^{-6})$ and would have suggested a
threshold there — a good example of why the small graph proves nothing. Worse, a tight threshold
mislabels in *both* directions: a football pair whose root sat $9.7\times10^{-8}$ from the shared
vertex **was** the vertex (verified against a 50-digit solve), while two other pairs had genuine
second crossings at $1.5\times10^{-6}$ and $1.2\times10^{-4}$ in the curve parameter.

The replacement uses what identity already tells us. Both curves pass through the shared vertex,
so $s \in \{0, 1\}$ is a **known root** of the quartic: reflect $s=1$ to $u=1-s$ and drop the
constant term. Exact, no tolerance, no coordinate comparison at all — which is more faithful to
(13)'s intent than (13)'s own mechanism — and a genuine crossing arbitrarily close to the shared
vertex survives, which is what makes the $1.5\times10^{-6}$ case come out right.

The segment path gets the same rule for free and exactly: two distinct lines meet in at most one
point, both curves pass through the shared vertex, so an adjacent pair's proper intersection *is*
that vertex. Only a collinear overlap survives.

### (33) The $\lVert D\rVert$ / $Q_4$ band is empty, and `EPS_LIN` is not why

(30) as corrected predicts a band where $\lVert D_A\rVert$ sits just above the straight-edge
cutoff — so no segment dispatch — while $Q_4 = -16\det(D_A,D_B)^2$ is already small enough to
wreck the companion matrix. It was looked for with **constructed** pairs sweeping
$\lVert D\rVert$ across twelve decades, since no real layout populates it (minimum
$\lVert D\rVert$ on a football twin layout is $5.5\times10^{-4}$):

| $\lVert D_B\rVert$ | $\lvert Q_4\rvert/\max\lvert Q\rvert$ | companion cond. | count | point error |
|---|---|---|---|---|
| $10^{-1}$ | 4.4e-2 | 2.4e1 | 1 ✓ | 5.7e-16 |
| $10^{-4}$ | 4.0e-8 | 2.5e7 | 1 ✓ | 7.8e-16 |
| $10^{-7}$ | 4.1e-14 | 2.5e13 | 1 ✓ | 2.2e-15 |
| $10^{-9}$ | 4.0e-18 | 2.5e17 | 1 ✓ | 2.5e-16 |

**The deflation closes it.** By the time the conditioning could matter, $|Q_4|/\max|Q|$ is already
below `EPS_DEFLATE` and a well-conditioned cubic is solved instead — so the two thresholds are
coupled in principle and the coupling is broken in practice by a third one. `EPS_LIN`'s only real
job is routing an exactly-zero bulge (γ=0, or a vertex-only layout) to the segment path. The same
sweep with *both* curves near-straight, and with exactly parallel bulges ($Q_4 = 0$ exactly), also
comes out right.

$Q_4$'s own distribution reproduces the guide to three digits: **0 %** of football's 27 058
candidate pairs below $10^{-9}$, 0.0517 % below $10^{-6}$, 2.9566 % below $10^{-3}$ (guide: 0 %,
0.05 %, 2.96 %).

### (34) $K$ reproduces exactly, and the broad phase needs a box pad to do it

$K$ came out **identical to (31a)'s measurement on all three datasets** — 27 058 / 1 380 736 /
11 375 494 — but only with the exact boxes grown by $10^{-12}$ before the overlap query. Without
the pad, football gives 26 909 and karate loses 9 of 2154 pairs: two edges sharing a vertex touch
*exactly* there, and when that point is extremal in the same coordinate for both curves, the two
box coordinates come from different expressions — an endpoint on one side, the closed-form
stationary value on the other — and differ by a couple of ulp, so the query calls the boxes
disjoint. It is not a geometric tolerance: nothing downstream reads it and it cannot create a
crossing, it only restores "never discards a crossing pair".

### (35) `arc_length` needs three branches, and γ=0 lands in the one the spec did not name

§2.2 gives two guards: $a \approx 0$ (straight) and $4ac = b^2$ (the cusp). **An exact-zero test
for the first is wrong**, and wrong on the entire straight variant. At γ=0 the quadratic
coefficient $w$ is not bitwise zero; it is a rounding residue with an *arbitrary direction*, so
the collinear test reads the noise as a real direction and the antiderivative divides by
$a \sim 10^{-33}$. The first implementation was out by a factor of 2 on every straight edge.

Measured against 40-digit quadrature, the closed form's relative error grows as
$10^{-16}/(\lVert w\rVert/\lVert u\rVert)$ — $2.6\times10^{-2}$ at a ratio of $10^{-14}$, 2.8 at
$10^{-16}$ — while the $2\lVert u\rVert$ approximation errs by about half the ratio. A three-term
binomial series covers $\lVert w\rVert \le 10^{-3}\lVert u\rVert$ to $10^{-12}$ and is exact when
$w$ really is zero, so the whole range is now good to ~1e-12.

### (36) Timings, and (31)'s call: **stay in NumPy**

End-to-end on a tabula twin layout, single process, `QUERY_CHUNK_EDGES = 1024`:

| variant | $K$ | crossings | Bernstein rejects | time | peak |
|---|---|---|---|---|---|
| curved γ=0.5 | 11 375 494 | 2 893 169 | 58.6 % | **40.2 s** | +428 MB |
| straight γ=0 | 11 359 536 | 2 868 965 | — (segment path) | **28.9 s** | +436 MB |

40 s against (31)'s ~30 s line is 1.3× over, not the 1.5–2× the plan estimated from per-stage
costs. The ordered escalation of (31) says compiling is not the next rung: rung 1 (the exact box)
is in, and **rung 2 — parallelism across layouts — is untouched and belongs to step 7's runner**.
A compiled version is a different algorithm to test, for a metric that runs once per layout.
§4.3 costs **1.7×**, not 2×, because at γ=0 no quartic is built at all.

`QUERY_CHUNK_EDGES` is the memory bound and was measured, not guessed: 4096 → 40.2 s / +768 MB,
**1024 → 39.4 s / +428 MB**, 256 → 39.4 s / +492 MB, with identical counts at every setting.

Two smaller things the plan did not cover:

- **The Bernstein filter rejects 46–59 % on real layouts** (football 46.2 %, digits 56.3 %, tabula
  58.6 %), better than the guide's Revision-3 figure of 35 % and much better than R1's 21 %.
- **`include_adjacent` changes no count in a *straight* drawing**, except for collinear overlap.
  Two distinct straight edges sharing a vertex meet only there, and that is excluded by identity
  either way. So `count_intersections`' divergence from eq. (78), while real and standing, is
  vacuous for the figure it serves unless four vertices are collinear.

### (37) Where the 40 s actually goes, and two optimisations that were tested and rejected

(36) recorded totals. The stage split was taken afterwards, on the same tabula twin layout, by
mirroring `_solve_quartic_block` call for call with a timer on each stage — so this is the real
code path, not the guide's per-stage estimates. It is recorded because both of the obvious
follow-up optimisations were then measured and **neither pays**; without the numbers they will be
proposed again.

**Curved, γ=0.5, 41.6 s instrumented (40.2 s uninstrumented):**

| stage | time | share |
|---|---|---|
| **solve (`np.linalg.eigvals`)** | **19.63 s** | **47.1 %** |
| broad phase (query) | 9.01 s | 21.6 % |
| build quartic | 4.43 s | 10.6 % |
| Bernstein filter | 2.52 s | 6.1 % |
| root filter + assemble | 2.13 s | 5.1 % |
| pair bookkeeping | 1.40 s | 3.4 % |
| sort + rescale | 1.01 s | 2.4 % |
| deflate shared root | 0.90 s | 2.2 % |
| degree detection | 0.52 s | 1.2 % |
| setup (curves) | 0.09 s | 0.2 % |

#### Rejected: closed forms for the low-degree pairs

**There are none.** Of 4 705 511 live pairs, **86.8 % deflate to degree 4 and 13.2 % to degree 3;
nothing is below that.** Replacing `eigvals` with the quadratic formula where the quartic has
collapsed is worth exactly zero on real data.

The histogram does say something else, though: degree 4 costs 4.18 µs per matrix and degree 3
costs 4.03 µs. Near-identical cost for a 33 % smaller problem is the signature of per-call LAPACK
overhead — `eigvals` calls `dgeev` once per matrix and spends most of its time on setup, not on
the $O(n^3)$ work. Which points at the next idea.

#### Rejected: a vectorised closed-form quartic in place of the companion matrix

2 000 000 real degree-4 quartics taken from the tabula layout:

| solver | µs/pair | speedup | in-range **root count** matches `eigvals` | max \|Δs\| | p99.999 \|Δs\| |
|---|---|---|---|---|---|
| `eigvals` (companion) | 4.12 | — | — | — | — |
| Ferrari, vectorised | 0.76 | **5.4×** | 99.9800 % | 5.9e-1 | 8.2e-3 |
| Ferrari + 2 Newton steps | 3.19 | 1.3× | 99.9960 % | 2.2e-1 | 1.2e-8 |

The speedup is real and so is its cost. Ferrari alone changes the **crossing count** on 0.02 % of
quartics — roughly 940 pairs on tabula — and its tail errors are gross rather than small. Those
failures sit where the resolvent cubic cancels, which is near-tangency, which is the crowded
regime the thesis is about; this is exactly the objection the guide raises against Ferrari, now
quantified. Trading it for 15 s in a metric whose headline claim is "no sampling parameter, no
geometric tolerance" is the wrong trade.

Polishing to fix it costs the speedup back — Newton is memory-bound over a `(P, 4)` complex array
and leaves 1.3× overall, 0.9 s per layout. A hybrid (Ferrari, residual check, `eigvals` fallback)
would recover the 5.4×, but its correctness would then rest on a residual threshold, in the most
delicate code in the module — the same kind of tolerance (32) had just removed.

#### Not rejected, not done: the segment path's redundant swept-segment computation

The straight variant builds no quartic and runs no `eigvals`, and still costs 28.9 s. Its split
(`_segment_intersections` calls `_swept_segments` internally, so those two rows overlap):

| stage | time |
|---|---|
| segment intersect | 12.87 s (includes the row below) |
| **swept segments** | **9.40 s** |
| broad phase (query) | 8.65 s |
| shared vertices | 1.25 s |
| parameter recovery | 1.15 s |

`_swept_segments` computes a turning point, three candidate points, three pairwise distances and
an argmax — in order to rediscover that a straight curve's swept segment is $(B_0, B_2)$. For
$C_2 = 0$, which is exactly the per-edge flag the dispatch already computes, it is provably
redundant. A fast path there is worth **~9 s with no accuracy risk**, taking both variants from
69 s to about 60 s.

**Left undone deliberately** (author's call, after the measurement): it is out of step 5's scope,
and rung 2 of (31)'s escalation still buys 8× for free. Worth doing if anyone is in this file for
another reason, or as part of step 7's cost model.

#### What this does not change

The floor in NumPy is ~10 s of unavoidable array passes over 11.4 M pairs plus ~9 s of query, for
*either* variant. Optimising the curved solve cannot get it below what the straight variant
already costs, so "get under (31)'s 30 s line" is not a goal the solve can deliver on its own.
**(36)'s decision stands, now on measured rather than inferred grounds.**

## (38) The `sfdp` failures are skipped behind a probe, not fixed and not ignored

`STATUS.md` left four options open and called it the author's. Taken: **probe the backend and
skip the two `yifanhu` parameters when it is broken.** The other three were declined with reasons.

The crash was characterised first, because three of the facts changed what the right answer was:

- it is graphviz's **multilevel coarsening** (`-Glevels≤1` survives, `≥2` does not);
- the minimal trigger is **5 nodes, 5 edges**, and it depends on **node creation order**, not on
  the graph — `nx_pydot` writes nodes in sorted order, so every `YifanHuLayout` call hits it;
- it is a known upstream regression, introduced in 15.1.1 and **fixed in 16.1.0** (GitLab #2852).

That last one settles it: there is nothing to report, nothing to work around, and the fix arrives
with the next graphviz. What the repository needs is for its test suite to tell the truth in the
meantime, on any machine, without anyone having to read a paragraph in `STATUS.md` first.

| option | why not |
|---|---|
| downgrade graphviz to 14.1.4 | fixes it here (verified), but it is a system-wide pin two majors back, needs `IgnorePkg` to survive an upgrade, and encodes one machine's state in nothing the repository can see |
| report upstream | already reported and already fixed |
| leave the two red | every session re-reads why the suite is not green, and a *new* failure hides among the known ones |
| **probe and skip** | ✅ honest on any machine, un-skips by itself when graphviz updates, and touches no layout code |

**The probe is narrow on purpose**, and all three of its behaviours are verified rather than
assumed: with graphviz 16.0.0 two tests skip with the reason and the upstream pointer in the
message; with 14.1.4 on `PATH` **35 pass and none skip**; with `sfdp` absent it declines to skip
and the tests fail. Only a signal death (`returned code: -N`) counts as grounds to skip, so a
missing binary or a broken `pydot` still fails the way it does for the `neato` tests.

The middle case is the one worth keeping: run against a working binary, `YifanHuLayout` passes
both tests **unchanged**. "The layout code is correct and the conformance test is doing its job"
stops being an assertion and becomes a measurement.

⚠️ **The cost is real and is written into `STATUS.md`:** while the skip is active the method's
backend has no coverage. `test_yifan_hu_is_handed_the_structure_and_nothing_else` reads the dot
source rather than running the binary and still runs, so the plumbing stays guarded — the
end-to-end layout does not.

---

# Step 6 — the label metrics

## (39) Step 6 departures from `plans/step6-labels.md`, and why

The plan held on substance. Five departures, each forced by something measured.

### (39a) Both degenerate ends of the silhouette, not one

The plan named the single-class case. `sklearn`'s accepted range is
$2 \le |\mathcal{L}| \le n-1$, so it refuses the **all-singleton** case too — verified, not
inferred from the docs.

| case | returned | why |
|---|---|---|
| $\lvert\mathcal{L}\rvert = 1$ | `nan` | undefined, and a property of the *data* rather than a caller error; the package returns `nan` for the first and raises for the second |
| $\lvert\mathcal{L}\rvert = n$ | **0.0** | Rousseeuw's convention already covers it: every $s_i$ is 0, so the mean is 0 |

Returning `nan` for the second would have been a discontinuity of ours, not of the metric: a
vector with one pair and $n-2$ singletons goes through `sklearn` normally and scores near 0.

### (39b) `silhouette` reads `pdist`, not `sklearn`'s `metric="euclidean"`

Committed first with `metric="euclidean"` and a justification that was wrong on every count, then
corrected. The claim was that `silhouette_samples` chunks its Euclidean reduction and so never
materialises $(n, n)$, making a precomputed matrix the dearer option. Measured, one process per
route so each gets a clean high-water mark:

| $n$ | `metric="euclidean"` | `pdist` + `squareform` |
|---|---|---|
| 5 007 (`tabula`) | 0.91 s, +373 MB | **0.60 s, +273 MB** |
| 12 000 (extended) | 4.71 s, +2039 MB | **4.64 s, +1634 MB** |

It does chunk, but `working_memory` defaults to **1024 MB**, so at these sizes it takes the
matrix in one or two blocks and its reduction allocates more than the matrix itself. There was no
trade-off to weigh: the exact route is faster, leaner **and** eight digits more accurate.

The accuracy alone would not have justified it. `sklearn` computes distances as
$\lVert x\rVert^2+\lVert y\rVert^2-2x\cdot y$, which loses digits when a layout sits far from the
origin relative to its own extent — at an offset/extent ratio of 4000, $1.8\times10^{-8}$
relative on tabula's `X_umap` against $1.2\times10^{-14}$ over `pdist`. Real layouts sit within a
few multiples of their extent of the origin, where the deviation is $10^{-14}$ or below. ⚠️ But
the *tolerance* would have been wrong: an invariance test pinned at $10^{-12}$ from the small
graphs passes on `karate` ($7\times10^{-14}$) and **fails at tabula's $n$**
($3\times10^{-12}$) — the session-5 lesson again, on a different quantity.

Distances come through `common.pairwise_distances`, so in a metric sweep ADR, NND and the two
distributions have already paid for the condensed vector and the marginal cost is the
`squareform` copy. **Numerically neutral and verified so**: the golden `silhouette` is
bit-identical between the two routes, so `metric_golden.json` was not touched.

### (39c) `labels` comes first on both graph-side metrics

Spec §2.3 writes `silhouette_graph(G, labels)` and `graph_neighborhood_accuracy(G, labels)`.
Implemented as `(labels, G=None, *, ideal=None)` and `(labels, G)`.

`labels` is the only argument Sil\* always requires — the graph is optional once `ideal` is
supplied, as `_stress_pairs`' contract intends — so no sentinel-free ordering puts `G` first.
Making the two graph-side metrics disagree about their order would be precisely the hazard (5)
was written about: *the spec writes two orders for the same arguments, which is how
argument-order bugs happen*. `_as_labels` requires 1-D, so `silhouette_graph(X, labels)` from
muscle memory raises instead of returning a number computed from a layout.

### (39d) The second graph-vote mode is `weighting="affinity"`, not `"distance"`

The plan asked for `knn_accuracy`'s vocabulary. `tie_rule`'s is unchanged; `weighting`'s could
not be. Eq. (62) votes with $w_{ij}$ — a coupling — where `knn_accuracy`'s `"distance"` mode
applies $f(d) = 1/d$ to turn a *length* into a weight. They are opposite quantities, and naming
the graph mode `"distance"` would invite exactly the substitution the stamp guard exists to
refuse. Uniform weighting reads no valuation at all and therefore carries no guard, the same
reasoning as `visual.py`'s deliberate absence of one.

### (39e) Exclusion is by vote mass, and `n_isolated` is reported beside `n_excluded`

§2.3 says isolated vertices are excluded and the count returned. A vertex of positive degree
whose every incident affinity is zero also has nothing to vote with, and an all-zero score row
reads as *every class tied* — worth 1/|L| under the fractional rule, which is noise dressed as a
score. So the rule is "no vote mass", and the two counts are separated: `n_excluded` is what was
dropped, `n_isolated` the structural reason. They are equal under uniform weighting and on every
registered dataset. `majority_class_rate` is computed over the **scored** vertices, so the score
and its floor share a denominator.

## (40) What the label metrics say on real layouts

SG-*t*-SNE layouts, seed 0 unless stated, λ = 1, as steps 3-5 used. Four measurements the plan
asked for, and all four moved something.

### Sil\* behaves as a label-validity check — and its scale is not the one anybody assumes

Across every labelled dataset, graph side only (no layout involved):

| dataset | label set | Sil\* | shuffled labels | Acc\* | floor | classes |
|---|---|---|---|---|---|---|
| `ring_of_cliques` | `clique` | **0.532** | −0.337 … −0.272 | 1.0000 | 0.100 | 10 |
| `karate_weighted` | `club` | 0.377 | −0.014 … 0.007 | 0.9412 | 0.500 | 2 |
| `football_2023` | `conference` | 0.330 | −0.138 … −0.125 | 0.9697 | 0.106 | 11 |
| `football_2023` | `division` | **0.020** | −0.024 … −0.017 | 0.9886 | 0.591 | 3 |
| `football_2023` | `conference_division` | 0.327 | −0.178 … −0.153 | 0.9722 | 0.106 | 15 |
| `digits` | `digit` | 0.367 | −0.034 … −0.026 | 0.9853 | 0.102 | 10 |
| `tabula` | `cell_ontology_class` | 0.284 | −0.081 … −0.069 | 0.9615 | 0.311 | 13 |
| `tabula` | `compartment` | 0.298 | −0.021 … −0.018 | 0.9897 | 0.528 | 4 |
| `tabula` | `donor` | **0.310** | −0.0004 … 0.0006 | 0.9873 | 0.571 | 2 |
| `tabula` | `method` | 0.085 | −0.020 … 0.011 | 0.9868 | **0.958** | 2 |
| `tabula` | `compartment_cell_type` | 0.284 | −0.083 … −0.064 | 0.9615 | 0.311 | 13 |

**The check works**: every real label set scores clearly above its own shuffled level, and the one
that does not — football's `division`, at 0.020 against a shuffled −0.02 — is the one where it
*should* not, since a division is West/East/none and orthogonal to conference. That is the
justification for "validity check, not baseline" turned from an assertion into a measurement.

⚠️ **Two facts about the scale, both against the obvious reading.**

- **The ceiling is far below 1.** `ring_of_cliques`' labels *are* its topology and it scores
  0.532. Inside a clique $a^*_i = 1$; the nearest other clique is 2–3 hops away, so
  $b^*_i = 1.75$ for a vertex holding a ring edge and $s^*_i = 0.43$. Graph distances are small
  integers and $b^*$ is a *mean* over the nearest class. **A Sil\* of 0.5 is strong agreement,
  not mediocre**, and the write-up must not read it on a 0-to-1 scale.
- **Chance is negative, not zero**, and by a dataset-dependent amount: −0.32 on a 10-class
  graph, −0.13 on 11 classes, −0.0004 on 2. $b^*_i$ is a min over $|\mathcal{L}|-1$ class means
  and so sits below the overall mean while $a^*_i$ is at it, and the bias grows with the class
  count. **"Low means the labels are not in the topology" needs a per-dataset reference level**,
  the role `majority_class_rate` plays for the accuracy metrics. Nothing computes one yet; a
  shuffled-label control is three lines and belongs in step 11's reporting layer.

**Acc\* is above 0.94 everywhere and is nearly uninformative on its own.** A 1-hop neighbourhood
of a $k$-NN graph is almost perfectly label-homogeneous by construction, so the number to read is
the gap to the floor: 0.969 against 0.106 on football's conference is decisive, 0.987 against
**0.958** on tabula's `method` is not. Reporting Acc\* without the floor beside it would make the
weakest case look like the strongest.

### ⚠️ The auxiliary-label claim is false for `donor` and true for `method`

`DECISIONS.md`, `DATASET_PROVENANCE.md` and spec §3.4 all assert that a **low** silhouette on
tabula's `donor` and `method` is the good outcome — a layout separating cells by batch covariate
is displaying an artifact. Measured on an $A_{tne}$ layout, seed 0:

| label set | role | Sil | Sil\* | Acc@20 | Acc\* | floor |
|---|---|---|---|---|---|---|
| `cell_ontology_class` | primary | 0.106 | 0.284 | 0.947 | 0.962 | 0.311 |
| `compartment` | sub | 0.110 | 0.298 | 0.987 | 0.990 | 0.528 |
| **`donor`** | auxiliary | **0.206** | **0.310** | 0.983 | 0.987 | 0.571 |
| `method` | auxiliary | 0.057 | 0.085 | 0.985 | 0.987 | 0.958 |
| `compartment_cell_type` | composite | 0.106 | 0.284 | 0.947 | 0.962 | 0.311 |

**`donor` is first on Sil, at roughly double `cell_ontology_class`** — the outcome the claim says
would be bad. Controls confirm the number is real: a random Gaussian layout scores 0.0007 on
`donor` and −0.085 on `cell_ontology_class`, and shuffling `donor` on the real layout gives
−0.0008 … 0.0011.

**It is not a layout artifact, and the reason is confounding.** `donor` is first on **Sil\*** too,
so the structure is already in the graph before anything is laid out — and the two donors sampled
almost disjoint cell populations:

| | TSP14 | TSP6 |
|---|---|---|
| hepatocyte | **1491** | 67 |
| macrophage | 2 | **1379** |
| monocyte | 7 | 605 |
| endothelial cell | 267 | 0 |
| endothelial cell of hepatic sinusoid | 80 | 341 |
| *column total* | 2147 | 2860 |

TSP14 is 69 % hepatocytes and TSP6 48 % macrophages. $\mathrm{NMI}(\mathrm{donor},
\mathrm{cell\ type}) = 0.413$, against $\mathrm{NMI}(\mathrm{method}, \mathrm{cell\ type}) =
0.059$.

So the donor silhouette is largely **cell-type structure read through a confounded label**, and a
low score on it is not an achievable target. `method` is the clean covariate — nearly independent
of cell type — and it behaves exactly as the claim predicts, at 0.057 on the layout and 0.085 in
the graph.

**What to change, and it is prose rather than code.** The "low silhouette on a batch covariate is
the good outcome" reading is sound in general and unsound for tabula's `donor` specifically.
`DATASET_PROVENANCE.md`'s and §3.4's wording should say that `method` is the batch covariate the
argument applies to, and that `donor` is confounded with cell type at NMI 0.41 and cannot be read
that way. The `LabelSet` note above ("`donor` and `method` are auxiliary batch covariates… a
*low* silhouette is the good outcome there") is the sentence that needs the qualifier. Suggested
thesis placement: §3.1 where the tabula label sets are introduced. **Left to the author —
`thesis/` is read-only (CLAUDE.md).**

### Silhouette separates the three representations decisively — and orders them differently from every other label metric

Five matched seeds, one shared label vector per dataset:

| dataset | | $A_V$ | $A_E$ | $A_{tne}$ |
|---|---|---|---|---|
| `football_2023` | Sil | **0.758** [0.748, 0.771] | 0.540 [0.512, 0.550] | 0.617 [0.606, 0.633] |
| | Acc@11 | 0.9697 (all seeds) | 0.955 [0.947, 0.962] | 0.9697 (all seeds) |
| | recall@11 | 0.715 | 0.645 | 0.710 |
| `digits` | Sil | **0.480** [0.470, 0.488] | 0.431 [0.417, 0.447] | 0.419 [0.404, 0.458] |
| | Acc@16 | 0.985 [0.983, 0.986] | 0.965 [0.958, 0.970] | 0.983 [0.981, 0.985] |
| | recall@16 | 0.639 | 0.539 | 0.632 |

**It is not a quiet instrument.** Unlike NND, which step 4 found could barely see anything, Sil
separates $A_V$ from both others *disjointly across all five seeds* on both datasets, and all
three pairwise on football. Step 3 had to strain to see the §1.6 fix in recall; this has no such
problem.

⚠️ **But it is not measuring label recovery, and two results show it.**

- On football, `knn_accuracy` is **identical to the digit** for $A_V$ and $A_{tne}$ (0.9697 on
  every seed) while Sil separates them disjointly, 0.758 against 0.617.
- On digits the two metrics **invert the ordering**: Sil puts $A_{tne}$ *below* $A_E$ (1/5 seeds,
  overlapping) where `knn_accuracy` puts it clearly above (5/5, disjoint), as does `knn_recall`
  (0.632 against 0.539).

Sil is a distance-based compactness-and-separation score; `knn_accuracy` and `knn_recall` are
rank-based. The twin graph inserts an edge node between every pair of adjacent vertices, which
stretches within-class neighbourhoods geometrically without changing who is whose neighbour — and
Sil, being scale-invariant but not rank-based, reads that as lost separation. **So Sil must not
be tabulated as "which representation recovers the labels best".** Both belong in the results,
and §5 should say which question each answers rather than letting a reader average them.

### ⚠️ Silhouette is the noisiest metric measured in this project so far

Seven seeds of $A_{tne}$ on `digits`, against step 4's bands:

| metric | rel IQR | full spread |
|---|---|---|
| NND$_{16}$ | 0.4 % | — |
| VNR | 1.2 % | — |
| `edge_length_cv` | 2.4 % | — |
| `knn_accuracy@16` | **0.24 %** | 0.74 % |
| **`silhouette`** | **3.6 %** | **19.0 %** (0.380 … 0.458) |

Raw: 0.4577, 0.4079, 0.4070, 0.4038, 0.4190, 0.3934, 0.3802. And it is
representation-dependent — over the same five seeds $A_V$ spans 3.8 % where $A_{tne}$ spans 13 %.

**This explains the inversion above rather than excusing it**: the $A_{tne}$-versus-$A_E$ gap on
digits is about 3 %, *inside* the noise band, which is exactly why it came out 1/5 and
overlapping. It also sets a floor for step 8: **a λ sweep reporting silhouette differences below
~4 % on a single seed is reporting noise**, where the same sweep can trust `knn_accuracy` to
0.25 %. Sil needs seed replication to be read at all; the crowding metrics do not.

### §2.5's cost class is wrong about Sil\*, and now by a measured margin

§2.5 names SNS and Sil\* as the metrics that will not run on tabula. On `tabula` ($n = 5007$,
$m = 49368$):

| | time | notes |
|---|---|---|
| `ideal_distances` closure | 14.9 s, +254 MB | **shared per dataset** with SNS and the stresses |
| Sil\*, given the closure | **0.58 s** | |
| Sil\*, all four label sets off one closure | **2.07 s** | the results-path pattern |
| Sil on the layout | 0.60 s, +273 MB | shares its `pdist` with ADR and NND |
| Acc\* | 0.49 s | |

Against crossings' **40 s and +428 MB per layout**, Sil\* is among the cheapest things in the
suite. The expensive metrics scale with $m$, and the label family scales with $n$ — which is
5007. §0.5 corrects the cost class in principle; these are the numbers, and step 7's run planner
reads it.
