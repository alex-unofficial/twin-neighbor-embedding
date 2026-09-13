# Step 5 — `crossings.py` and `twin/geometry.py`

The executable sequence for spec §7 step 5. **The design is not repeated here** — read
[`CROSSING_GEOMETRY.md`](../CROSSING_GEOMETRY.md) first (it is the specification; spec §2.4 is the
summary), then §0.5's "Session 5 pre-work" and `DECISIONS.md` (27)–(31). This file is what to do,
in what order, and how to know it worked.

Prerequisite: `docs/STATUS.md` for where things stand and what it carries into this step.

**Scope: `twin/geometry.py`, `edge_curves`, `crossing_pairs`, the three layer-3 metrics, and the
reimplementation of `plots.py::count_intersections` on top of layer 2.** `MetricSpec` /
`METRIC_REGISTRY` / `evaluate_layout` remain step 7 — the boundary steps 3 and 4 both held.
`detour_ratio` lands here, since `edge_curves` now exists and the closed-form arc length needs no
sampling. Nine commits.

---

## 0. The guide was checked before planning, and it holds

Every load-bearing claim was re-derived and re-measured rather than taken on trust. Results, so
that implementation starts from verified ground:

| claim | result |
|---|---|
| the γ kernel is exactly the quadratic Bézier | ✅ max error **5.1e-15** over 1200 curves × 37 parameters, γ ∈ {0, .25, .5, .75, 1, 1.5} |
| the closed-form exact AABB | ✅ **0 failures / 3000**: no sampled point outside, and the box equals the sampled extent (i.e. it is tight, not merely valid) |
| the quartic coefficients and `t = −P02(s)/β` | ✅ every reported point lies on **both** curves to **9.2e-10**; root counts {0:332, 1:174, 2:80, 3:7, 4:7} |
| agreement with a 1200-segment polyline oracle | ✅ **600/600** |
| Bernstein sign test has no false negatives | ⚠️ **true for transversal crossings only** — and that case is a *theorem*, not a measurement (see below). The exact-endpoint case is a genuine false negative and is fixed in commit 4 |

Two of the guide's numbers come out better on real data than on the random curves it was
calibrated with: the Bernstein filter rejects **35 %** on a real `football_2023` layout (guide:
21 %) at **65 ns/pair** (guide: 108 ns). The exact box costs **8.4 ms for all 49 368 tabula
edges**, matching the guide's 123 ns/edge.

### K is measured — the guide's cost model was right, and tabula sits just over the line

Requirement (31)'s *measurement* is discharged in advance; the decision it feeds is taken at
commit 4, against an end-to-end number rather than these per-stage estimates. Twin layouts
(SG-*t*-SNE on $A_{tne}$, seed 0, γ=0.5),
layout normalised to a unit box, exact boxes, `shapely.STRtree` bulk query:

| dataset | m | **K** | K/C(m,2) | query | solve est. @3.1 µs |
|---|---|---|---|---|---|
| football_2023 | 735 | 27 058 | 0.100 | 0.02 s | 0.1 s |
| digits | 14 732 | 1 380 736 | 0.0127 | 1.1 s | 4.3 s |
| **tabula** | **49 368** | **11 375 494** | 0.00934 | **9.5 s** | **35 s** |

The guide predicted K≈1e6 for digits and 1e7 for tabula; measured 1.38e6 and 1.14e7. **The model
is sound.** Straight and curved variants give K within 0.2 % of each other, so §4.3 doubles the
cost and nothing else.

Three consequences the plan acts on:

1. **Tabula is ~35 s of solve plus ~10 s of query ≈ 45 s per layout**, against (31)'s "below
   roughly 30 s there is no case for compiling" — over the line by ~1.5×. But every term here is
   an estimate assembled from per-stage costs, and the Bernstein rejection is 35 % rather than
   the assumed 21 %, which takes the solve to ~23 s. **The call is deferred to commit 4 and an
   end-to-end measurement**; do not escalate on these numbers.
2. ⚠️ **The cost model omits the broad-phase query.** It budgets `O(m log m)` for the *build*
   (measured: 0.0 s) but the query returning 11.4 M pairs takes **9.5 s and 365 MB**. That is 27 %
   of tabula's runtime and the second-largest line item. Not a problem — but it must be in the
   block loop, not treated as free.
3. ⚠️ **Blocking the solve does not cap the 365 MB.** That peak is the bulk query's *return
   array*, which materialises before any block loop begins. Capping it needs **edge-chunked
   queries** — query the tree with a slice of the boxes at a time — which is a different change
   from blocking the downstream passes. Do both, and do not assume the second buys the first.

### Three corrections to the guide — one of them changes the code

The Bernstein numbers and the $Q_4$ reframing are being applied to `CROSSING_GEOMETRY.md`
**now**, so that the reference is not edited mid-implementation. The rest of the corrections land
at commit 9 as planned.

- **§4's "108 ns/pair, rejects 21 %, no false negatives" understates it and overstates the cost**
  — 35 % and 65 ns on real data. Worth updating so nobody "optimises" a filter that is already
  paying for itself sixfold.
- ⚠️ **The Bernstein sign test has a real false negative, and the filter must be fixed.** A pair
  whose only root is exactly at `s=0` or `s=1` is rejected: for `s(s+1)(s²+1)`, i.e.
  `Q = [0,1,1,1,1]`, the coefficients are `β = [0, .25, .667, 1.5, 4]` — no sign change, yet
  there is a root at 0. Verified. On real data this is **23 % of all rejected pairs**, because
  adjacent edges *share a vertex* and their curves meet exactly at an endpoint.

  For adjacent pairs that is harmless — the root is excluded by vertex identity anyway. For a
  **non-adjacent** pair it is a genuine lost crossing: a vertex lying exactly on another edge's
  curve, which is the coincident-vertex crowding case this codebase writes conventions for.

  Since `β₀ = Q(0)` and `β₄ = Q(1)` **exactly** (verified), the fix is inside the filter and
  costs nothing:

  ```
  keep if sign_changes > 0
       or (pair is non-adjacent and (|β₀| < tol or |β₄| < tol))
  ```

  ⚠️ **The non-adjacency gate is load-bearing.** Without it the 23 % of adjacent rejections all
  re-enter the solve and the filter stops paying for itself.

  Interior tangencies are still kept — `(s−½)²(s²+1)` is **kept**, because Bernstein coefficients
  may straddle zero where the polynomial does not. And a no-false-negatives test must use the
  **open** interval `(0,1)`; written with inclusive bounds it reports a spurious 23 % failure.
- **§5's `Q₄` guard is real but its stated trigger is wrong, and the correct one changes the
  design.** "Parallel bulges, common in any layout with orientational coherence" is not what the
  data shows: on a real `football_2023` twin layout **0 %** of 27 058 candidate pairs have
  $|Q_4|/\max|Q| < 10^{-9}$, 0.05 % below $10^{-6}$, 2.96 % below $10^{-3}$ — SG-*t*-SNE places
  edge points independently, so bulges are not parallel.

  ⚠️ **The real trigger is near-straight curves.** $\det(D_A,D_B)$ is small whenever *either*
  bulge is small, and $Q_4$ goes as its square — so a single near-straight edge degrades $Q_4$
  quadratically. That is what the 2.96 % within $10^{-3}$ actually is.

  **Consequence: the $\lVert D\rVert\approx0$ and $|Q_4|\approx0$ thresholds are coupled.** There
  is a band in which $\lVert D_A\rVert$ sits just *above* the straight-edge cutoff — so no
  segment dispatch happens — while $Q_4$ is already small enough to wreck the companion matrix.
  Counting pairs below fixed thresholds cannot find that band. **Measure the joint distribution
  of $\lVert D\rVert$ against the companion matrix's condition number on a real layout, and
  choose $\varepsilon_{\text{lin}}$ so the band is empty.** This is a commit-4 task, not a
  documentation note.

---

## Commit sequence

### Machinery first

1. **`refactor: twin/geometry.py` — the curve geometry moves out of `draw_curved_edges`.**

   γ→Bézier conversion, the power form, and the exact AABB. `draw_curved_edges` calls it and
   keeps its own sampling for rendering; **no behaviour changes**, and that is the point — per
   CLAUDE.md the refactor and the new metric must not share a commit.

   ⚠️ Pin `draw_curved_edges`'s output **before** touching it (a characterisation pin, the
   pattern of steps 2–4): the sampled curve arrays for a few γ, asserted bit-identical after.
   Without it "behaviour-preserving" is an assertion rather than a check.

   Use `shapely.box(xmin, ymin, xmax, ymax)` **vectorised over arrays** — building 49 368 box
   objects in a Python loop is pure overhead.

2. **`metrics: edge_curves` — layer 1.**

   `edge_curves(G, V, E=None, gamma=0.5) -> (m, 3, 2)`, in **`G.edges()` order** (CLAUDE.md;
   `edge_lengths` is the order of record and step 5's per-edge weights must align with it).
   `E is None` or `gamma=0` → `B1` at the chord midpoint, i.e. an exact straight segment, so both
   §4.3 variants go through one code path. No `n_samples`.

3. **`metrics: the broad phase, and print K.`**

   `_candidate_pairs(controls) -> (2, K)` over the exact boxes, plus the layout normalisation to
   a unit box. **Stop here and report K on tabula** before building the solver — the numbers above
   say what to expect, and a large deviation from them means something upstream is wrong.

### Detection

4. **`metrics: crossing_pairs` — layer 2.**

   Quartic build → Bernstein filter → batched companion-matrix solve, as passes over arrays in
   blocks of ~1e6 pairs. Returns `(k, 6)`: `(edge_a, edge_b, t, s, x, y)`.

   - implicitise the curve with the larger `|β|` (larger control triangle), substitute the other;
   - **deflate to the true degree before forming the companion matrix** — `Q4 = −16 det(D_A,D_B)²`;
   - accept a root when real, `s ∈ [0,1]`, **and** recovered `t ∈ [0,1]` — filtering on `s` alone
     admits roots on the parabola's extension (~⅓ of them);
   - exclude roots at a **shared vertex by index**, never by coordinate ((13), (29));
   - `include_adjacent=True` default.

   Degeneracy dispatch per (30): `‖D‖≈0` → segment–segment; `|β|≈0` → segment over the swept
   interval; all `Q_k≈0` → count 1; double root → count 1 and increment `n_near_tangential`.
   The Bernstein filter carries the endpoint fix above, with its non-adjacency gate.

   **Two measurements to take in this commit, both before moving on:**

   - **The coupled `‖D‖` / `Q₄` band.** Joint distribution of $\lVert D\rVert$ against the
     companion matrix's condition number on a real layout; pick $\varepsilon_{\text{lin}}$ so
     no pair escapes the segment dispatch while already ill-conditioned.
   - **Straight and curved timed separately.** K differs by 0.2 % but the *cost* should not: at
     γ=0 every pair takes the segment path and builds no quartic at all. If straight is much
     cheaper then §4.3's "doubling" is really ≈1.2×, and the ~2 h tabula figure below drops
     accordingly. Report the two, not their sum. This is also the end-to-end number that
     decides (31) — see the decisions section.

5. **`tests: the oracle and the constructed degeneracies.`**

   ⚠️ **Do not expand the random-quartic sweep.** No-false-negatives for *transversal* crossings
   is a **theorem**, not an empirical fact: the hull property gives
   $\min\beta \le \min_{[0,1]} Q < 0 < \max_{[0,1]} Q \le \max\beta$, so a sign-changing $Q$
   always straddles zero in its coefficients and is always kept. The 103 040 random quartics
   confirm a theorem and therefore have no power to find anything. **The entire gap is the exact
   interior tangency**, where $Q$ touches zero without changing sign — the constructed
   `(s−½)²(s²+1)` case *is* the test that matters, along with the endpoint case from commit 4.

   - **Independent oracle**: a dense-polyline reference (~1200 segments) built **in the test
     module only** — *not* `plots.count_intersections`, which shares code and is documented to
     disagree on shared-endpoint pairs. ≥1000 random pairs, counts must agree. (600/600 in
     pre-validation.)
   - **Constructed** cases for exactly 0/1/2/3/4 intersections; interior tangency; identical
     curves; adjacent edges sharing a vertex, with and without a genuine second crossing;
     collinear controls; **parallel bulges, the Q4=0 path**; and a **vertex lying exactly on
     another edge's curve** (see the correction above). Constructed, not sampled: none of these
     appear reliably under random generation.
   - Invariance of the count under rotation, translation and uniform scaling.

### The metrics

6. **`metrics: crossing_number`, `crossings_per_edge_pair`, `length_weighted_crossings` — layer 3.**

   `L(e)` is the **chord**, from `metrics.visual.edge_lengths` — do not recompute it, and do not
   use the arc. The reasoning goes in the module docstring: the chord is identical for the
   straight and curved rendering of one layout, so LWCr's weight vector is held fixed and the
   difference between variants isolates the change in crossings. $\bar L$ (eq. 80) is already
   pinned in the golden file.

7. **`metrics: detour_ratio` — arc over chord, closed form.**

   Deferred from step 4 for exactly this moment. The quadratic Bézier arc length is closed-form
   (no sampling, no `n_samples`); ≥1 by construction and exactly 1 at γ=0, which is the test.

8. **`plots: reimplement count_intersections on layer 2.`**

   **Behaviour-preserving, deliberately**: pass `include_adjacent=False` so Figure 3 does not
   move (decision 1 below). Pin the old `(lengths, per_edge, cumulative)` output *before* the
   change and assert the new implementation reproduces it — the figure smoke test only catches a
   raise, not a moved curve. The output shape must be preserved for `paper_figures.py:364-366`.

9. **`docs:` close step 5** — golden-file keys for the new scalars (additive diff), the measured
   K table and the commit-4 timings into `DECISIONS.md`, the remaining guide corrections (the
   Bernstein numbers and the $Q_4$ reframing are already applied), **§0.5's divergence row for
   `count_intersections` moved from "scheduled fix" to "intentional divergence"**, then
   `STATUS.md`.

---

## Traps

- **Two per-edge orderings.** `G.edges()` for everything here; ADR alone uses the CSR upper
  triangle. They agree on karate and football and disagree in 12 377 of 14 732 positions on
  digits, so a test on the small graphs proves nothing.
- **The self-referential-comparison trap.** `twin_adjacency` was untested for three sessions
  because the two things compared shared an implementation. `crossings.py` must not be validated
  against `count_intersections` once the latter is reimplemented on top of it.
- **`plots.py` is still `(d, n)`.** `count_intersections` takes `y` as `(d, n)`; layer 2 works in
  `(n, d)`. Convert at the boundary and do not let the legacy orientation leak into `crossings.py`.
- **Normalise the layout to a unit box first**, or the root-reality and interval tolerances become
  dataset-dependent — the whole reason the method can claim to be tolerance-free on geometry.
- **`count_intersections` has a benign wraparound**: at `base == 0` it reads
  `cumulative_intersections[sorted_by_len[-1]]`, i.e. the shortest edge, which is 0 only because
  the array is freshly zeroed. Do not port that.

## Verification

```bash
venv/bin/python -m pytest -q          # 656 + step 5's; nothing existing moves
venv/bin/python -m pytest -q -m slow  # the tabula figure still renders
git status --short                    # clean
```

Then:

- `draw_curved_edges` is **bit-identical** to its pre-refactor output on the pinned γ set.
- The exact AABB contains every sampled point and equals the sampled extent.
- ≥1000 random pairs agree with the 1200-segment oracle on **counts**, and every reported point
  lies on both curves to <1e-7.
- Every constructed degeneracy returns the documented `cr`, the Q4=0 path included.
- Counts are invariant under rotation, translation and uniform scaling.
- K on tabula is within a factor of ~2 of 11.4 M; if it is not, stop and find out why.
- `metric_golden.json`'s diff is **additive only**.
- All eight figures render, Figure 3 included.

## Two decisions taken at review, so they are not re-litigated

1. **`count_intersections` keeps `include_adjacent=False` — permanently, and by intent.** It
   currently `continue`s on shared vertices, and the reimplementation passes `False` to preserve
   that, so the refactor stays behaviour-preserving and **Figure 3 does not move**. A refactor
   must not become a silent result change.

   ⚠️ **Record this as a standing intentional divergence from eq. (78) for that figure, not as a
   defect awaiting repair.** §0.5's divergence table currently lists `count_intersections` versus
   eq. (78) among the *scheduled code fixes* — "step 5 reimplements it on layer 2 anyway" — which
   after this commit is the wrong reading: the reimplementation happens and the divergence
   deliberately survives it. Commit 9 must move that row from "scheduled" to "intentional, with
   the reason", so a later session does not close it as an outstanding bug. The thesis convention
   lives in `crossing_pairs`' own default and is what §4.3's reporting carries.

2. **The NumPy-versus-compiled call is deferred to commit 4**, when an end-to-end tabula number
   exists rather than an estimate assembled from per-stage costs. The pre-measured ~35–45 s
   assumes 3.1 µs/pair on the survivors and the measured 35 % rejection, neither of which has
   been through the real code path. Take the reading, then apply (31).

   Record wall time for query / build / filter / solve **separately**, on tabula, and **straight
   and curved separately** — at γ=0 no quartic is built at all, so if the straight variant is
   much cheaper then §4.3 costs ≈1.2× rather than 2×.

   ⚠️ **Escalation rung 2 of (31) — parallelism across layouts — is not a step-5 or step-8
   bolt-on.** The grid is embarrassingly parallel, and where that parallelism lives is a
   property of the **runner**, so it belongs in step 7's design. Raise it there; do not
   improvise it later around a slow tabula run. For scale: 3 representations × 5 seeds × 5 λ ×
   2 variants is ~2 h for tabula alone at the current estimate.
