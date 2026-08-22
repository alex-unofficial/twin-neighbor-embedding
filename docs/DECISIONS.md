# Implementation decisions

Decisions on the points `docs/EXPERIMENT_SETUP_SPEC.md` leaves open, made by the implementer and
recorded here so they are auditable. Numbering follows the review list; §0.5 of the spec records
the ones the author decided. Where a decision applies to a later step, it is a commitment for
that step, not something already built.

---

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

## (10) `weight_transform=` versus the dataset's declared `weight_kind`

**`ideal_distances(G, weight_transform=...)` keeps the parameter as a low-level escape hatch; the
pipeline never passes it.** `Dataset.ideal_distances(**kw)` fills `weight_transform` from
`weight_kind` and is the only form the runner calls. Passing both a `Dataset` and an explicit
`weight_transform` raises.

The primitive stays testable in isolation — the analytic cases in §6 need it without constructing
a `Dataset` — while the dataset remains the single source of truth on the path that produces
results.

## (11) Return type of `as_distance()` / `as_affinity()`

Each returns **one type unconditionally**:

- `as_distance() -> nx.Graph`, edge attribute `weight` carrying the distance, non-edges **absent**.
- `as_affinity() -> sp.sparray`, which is what the representations and `sgtsnepi` consume.
- `as_distance_matrix()` added only if a dense form is ever actually needed.

**A distance source may be measured rather than derived.** `Dataset` carries an optional
`distance_source` that is *not* a transform of the weights, with its own provenance fields
(`distance_space`, `distance_metric`, `knn_k`). When present, `as_distance()` returns it and the
`weight_kind` transform is not consulted; the manifest records which path was taken.

This exists because tabula has both directions measured independently — `A` is a UMAP membership
and `D` is the Euclidean distance in `X_scvi` that `A` was built from — and because `A` is
**not invertible to distances at all**: UMAP's per-point $\rho_i$ and $\sigma_i$ are not stored,
so no global transform recovers $d$. That is true of any per-point-adaptive kernel. See
[DATASET_PROVENANCE.md](DATASET_PROVENANCE.md#tabula-liver); it is also worth a line in thesis
§3.1.1, which currently lists global transforms as interchangeable options.

The spec types both as `nx.Graph | sp.spmatrix`, which pushes a branch into every consumer. The
`none` weight kind *requires* absent non-edges so shortest paths are hop counts, and a graph
expresses absence naturally where a matrix has to encode it. Different types for the two
questions is fine; a type that varies with `weight_kind` is not.

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

**Two phases, two commits, per CLAUDE.md's "never mix a refactor with a behaviour change".**

1. **Baselines first.** Render and commit the baseline PNGs from the *pre-refactor* code. Then the
   §3.6 refactor reproduces the existing `3*l + 1 + s` palette exactly and must match those
   baselines bit-for-bit. This is the commit that proves the decomposition is behaviour-preserving.
2. **Generalisation second.** A separate commit replaces the hardcoded three-shade arithmetic with
   the general ordered-sub-label rule, declares that it changes the figures, and regenerates the
   baselines in that same commit.

Doing it in one step would make the baselines unable to distinguish "the refactor broke it" from
"the new colour rule is different on purpose".
