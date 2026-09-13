# Crossing detection for gamma-parameterised curved edges

Reference for `twin/metrics/crossings.py` and `twin/geometry.py` (spec §2.4, step 5).
**Read this before implementing step 5** — spec §2.4 is the summary, this is the specification.
Design record: `DECISIONS.md` (27)–(31); change log: spec §0.5, "Session 5 pre-work".

Proofs and the full derivation are in a companion LaTeX report held by the author (not in the
repo); everything needed to implement is here.

**Revision 2 supersedes Revision 1.** R1 proposed a bounding-triangle filter and a
subdivision fallback. Measurement showed the triangle filter costs 3x what it saves in a
vectorised implementation, and subdivision is unnecessary for quadratics. Both removed,
along with the flatness / recursion-depth analysis that supported them.

## 1. The curve is a quadratic Bézier

`draw_curved_edges` builds `P0 = y_V[u]`, `P1 = y_E[e]`, `P2 = y_V[v]` and evaluates
`C(t) = [1, t, t^2] @ K_gamma @ [P0; P1; P2]`. This is exactly the quadratic Bézier

```
B0 = P0
B1 = 2*gamma*P1 + (1 - 2*gamma)*(P0 + P2)/2
B2 = P2
```

Verified for gamma in {0, 0.25, 0.5, 0.75, 1, 1.5}, max error ~1e-16.

**The control points in `plots.py` are NOT Bézier control points.** Except at gamma = 0.5
they differ from {B0, B1, B2}, and using them as if they were gives the wrong curve.
Every consumer must convert first.

Power form used throughout: `C0 = B0`, `C1 = 2*(B1-B0)`, `C2 = B0 - 2*B1 + B2`.
Bulge vector `D = B1 - (B0+B2)/2 = 2*gamma*(P1 - (P0+P2)/2)`, so `C2 = -2D`.
`D` bounds nothing here — it is only a conditioning quantity (§5).

**Action:** factor the conversion + power form + AABB into `twin/geometry.py`, called by
both `draw_curved_edges` and `edge_curves`, so drawn and measured curves cannot diverge.

## 2. Exact bounding box (closed form)

Each coordinate is a scalar quadratic with one turning point, and **each coordinate is
treated independently**. Fix a coordinate and let `B0, B1, B2` be the three control
*values* in it (scalars, not points); `u = B1 - B0`, `w = B2 - B1`:

```
t*    = u / (u - w)                  # denominator is B0 - 2*B1 + B2, the coord of -2D
value = B0 + u**2 / (u - w)

range = min/max over {B0, B2}, plus `value` if 0 < t* < 1
```

`t*_x` and `t*_y` are **generally different**, and each is kept or discarded on its own,
so the box can be touched at up to four distinct parameters: 0, 1, `t*_x`, `t*_y`.
Example: `B0=(0,0)`, `B1=(4,1)`, `B2=(3,0.5)` gives `t*_x = 0.8`, `t*_y = 2/3`, both
interior; exact box `[0,3.2] x [0,2/3]` vs control-point box `[0,4] x [0,1]` — 53% of the
area. In code this is one `(m,2)` array computation with no branching between coordinates.

Measured over 2e5 random curves: exact (curve never leaves it), **37% smaller in area**
than the box on the control points (median 41% smaller, best 75%), **123 ns per curve** —
per *edge*, not per pair, so 6 ms for all of tabula. Since candidate counts scale with box
area, this also cuts K by roughly a third.

Guard: denominator is zero for a straight edge; then the endpoints are the whole range.

## 3. Intersections: one quartic per pair

Fix a point `(x,y)`; set `p(t) = xA(t) - x`, `q(t) = yA(t) - y`. The point is on curve A
iff `p` and `q` share a root. Eliminating the `t^2` term and the constant term gives two
*linear* equations in the shared root, and with `[i,j] = p_i q_j - p_j q_i`:

```
f(x,y) = [0,1]*[1,2] - [0,2]^2 = 0        # implicit equation of A's parabola
t      = -[0,2] / [1,2]                   # the parameter, free
```

Substituting curve B (quadratic in s) gives a **quartic in s**. At most 4 roots = Bézout,
all returned at once.

### Coefficients

A: `x = a0+a1 t+a2 t^2`, `y = b0+b1 t+b2 t^2`.  B: `x = c0+c1 s+c2 s^2`, `y = d0+d1 s+d2 s^2`.

```python
beta = a1*b2 - a2*b1
k01  = a0*b1 - a1*b0
k02  = a0*b2 - a2*b0

P01 = [k01 - b1*c0 + a1*d0,  -b1*c1 + a1*d1,  -b1*c2 + a1*d2]
P02 = [k02 - b2*c0 + a2*d0,  -b2*c1 + a2*d1,  -b2*c2 + a2*d2]

Q0 = beta*P01[0] - P02[0]**2
Q1 = beta*P01[1] - 2*P02[0]*P02[1]
Q2 = beta*P01[2] - P02[1]**2 - 2*P02[0]*P02[2]
Q3 =             - 2*P02[1]*P02[2]
Q4 =             -   P02[2]**2
```

~30 multiply-adds. Parameter recovery is free: **`t = -P02(s) / beta`** — `P02` is the
quadratic already built.

**Filtering `s in [0,1]` is not sufficient.** `f = 0` is the whole parabola, not the arc.
Recover `t` and require `t in [0,1]` too; ~1/3 of in-range `s` roots lay on A's extension.

### Validation

4000 random pairs, gamma in [0.1, 1.3]: counts {0: 2211, 1: 1165, 2: 548, 3: 49, 4: 27}.
All reported points matched to <1e-7 except 2 near-tangential cases. Against a
1200-segment polyline reference on 200 pairs: 200/200 agreement.

## 4. Pipeline

| Stage | Cost | What |
|---|---|---|
| Broad | `O(m log m)` build | `shapely.STRtree` over the exact boxes of §2; query returns the K box-overlapping pairs without enumerating the rest |
| Filter | 197 + 108 ns/pair | Build the quartic, convert coefficients to the Bernstein basis on [0,1], discard if no sign change |
| Solve | 3582 ns/pair | Companion-matrix eigenvalues, batched; filter `s`, recover `t`, filter `t` |

Every stage is conservative: may pass a non-crossing pair, never discards a crossing one.

### Bernstein filter (replaces the R1 mid phase)

Bernstein weights are non-negative and sum to 1, so the polynomial's values on [0,1] lie
within the range of its coefficients — the same hull property as the Bézier curve, applied
in 1-D. **No sign change among the 5 coefficients proves no root in [0,1].**

```
beta_j = sum_{k<=j} [ C(j,k) / C(4,k) ] * Q_k        # fixed 5x5 triangular map
```

108 ns/pair, rejects 21%, no false negatives.

**Why the triangle filter was dropped:** separating-axis on the two control triangles costs
2339 ns/pair and rejects 22% — it costs 3x the 790 ns it saves. SAT is cheap in scalar code
only because it early-exits on the first separating axis; vectorised execution has no early
exit, so every pair pays all 6 axes. Selectivity was measured on AABB-overlapping random
curves (half genuinely crossing), which is harsh; on real layouts with long thin edges,
boxes overlap far more often than hulls and SAT would reject more. Keep it behind a flag
and measure once — crossover is at ~65% rejection.

### Batching

Run stages as **passes over arrays**, not per pair: compute stage over all candidates ->
boolean mask -> compact -> next stage. Per-pair Python control flow costs ~1 us/pair in
interpreter overhead; the 4.1e5 pairs/s figure exists only because the solve is one call on
a `(P,4,4)` stack. Process the broad-phase output in **blocks of ~1e6 pairs** so memory
stays bounded.

### Which curve to implicitise

`beta = [1,2] = 2 * det(B1-B0, B2-B0)` — four times the control-triangle area (verified to
7.5e-15). It is the denominator in `t` recovery, so:

> **Implicitise the curve with the larger control-triangle area; substitute the other.**

One comparison. `beta = 0` is exactly collinear controls.

### Solving the quartic

**Companion-matrix eigenvalues, batched — not Ferrari's closed form.** Ferrari routes
through a resolvent cubic whose cancellation is worst when two roots are close, which is
exactly near-tangency, exactly where the answer matters. Balanced companion + QR is
backward stable and reports near-tangency legibly as two nearly-coincident roots.
Do not isolate roots to [0,1] (Sturm/Descartes): iterative, and all roots are needed for
multiplicity anyway. The Bernstein test already gives the cheap [0,1] information.

**Trap.** Building the companion matrix divides by `Q4`, and

```
Q4 = -P02[2]**2 = -16 * det(D_A, D_B)**2
```

so **the leading coefficient vanishes exactly when the two curves bulge in parallel
directions** — with shared gamma, whenever two edges bow the same way. Common in any layout
with orientational coherence. **Deflate to the true degree before building the matrix.**

### If this moves to C

A C version is a *different design*, not a translation: with early exit and cheap
branching, the triangle filter is worth having again and per-pair processing becomes
correct. Two algorithms to test, not one in two languages.

Decide on measured `K`. NumPy runtime ≈ `K * 3.1 us`:
`K=1e6` -> 3 s/layout, `K=1e7` -> 31 s, `K=1e8` -> 5 min. Below ~30 s/layout there is no
case for C — the metric runs once per layout. Cheaper escalations first: the exact box (§2,
cuts K ~1/3 for 6 ms), parallelism across layouts (the grid is embarrassingly parallel),
and the length-thresholded variant (shrinks m directly).

## 5. Degeneracies

`D` and `beta` bound nothing in this design; they appear only here.

| Case | Detect | Handle |
|---|---|---|
| Straight edge | `norm(D) < eps * scale` | segment–segment test (also covers gamma=0 and A_V layouts) |
| Collinear controls | `abs(beta) < eps` | segment path over swept interval |
| Parallel bulges | `abs(Q4) < eps` | deflate to cubic or lower before the companion matrix |
| Identical / overlapping | all `Q_k ~ 0` | count as exactly 1 (spec convention); detect before root-finding |
| Adjacent edges | shared vertex | exclude roots at the shared vertex **by vertex identity**, not coordinate; keep genuine second crossings |
| Tangency | double real root | count as 1; report near-tangential pair count alongside Cr |
| Scale | — | normalise layout to unit bounding box first |

## 6. Cost

Naive sampled polyline = `C(m,2) * (s-1)^2` at s=64:

| Dataset | m | naive segment tests | new pipeline |
|---|---|---|---|
| football_2023 | 735 | 1.1e9 | negligible |
| barabasi_albert | 6951 | 9.6e10 | K=1e5 -> 0.3 s |
| digits | 14732 | 4.3e11 | K=1e6 -> 3.1 s |
| tabula | 49368 | 4.8e12 | K=1e7 -> 31 s |

**Output sensitivity is the real limit.** If a layout has X crossings then K >= X, and no
method beats `Omega(m log m + X)`; a bad tabula layout can have `X = Theta(m^2)`. Report Cr
normalised by `C(m,2)`, and prefer the **length-thresholded crossing curve** (crossings
among edges shorter than tau, vs tau) — what claim (c) of the thesis is actually about, and
cheap on the short-edge subset.

## 7. Layer structure (spec §2.4, revised)

- **`twin/geometry.py` (new).** Conversion, power form, exact AABB. Shared by
  `draw_curved_edges` and `edge_curves`.
- **Layer 1** `edge_curves(G, V, E=None, gamma=0.5)` -> `(m,3,2)` **Bernstein control
  points**, not sampled points. `E is None` -> B1 at chord midpoint.
- **Layer 2** `crossing_pairs(controls, edges, include_adjacent=True)` -> broad / Bernstein
  filter / batched solve with deflation, in blocks of ~1e6. Returns `(e1,e2,t,s,x,y)`.
- **Layer 3** `crossing_number`, `crossings_per_edge_pair`, `length_weighted_crossings`.
  Interfaces unchanged. `L(e)` stays the **chord** `norm(P2-P0)` — now positively
  justified: the chord is independent of gamma, the arc length is not, and gamma is a
  drawing parameter with no graph-theoretic meaning.
- `plots.py::count_intersections` reimplemented on layer 2.
- **Test oracle only:** keep the sampled version as `_crossings_reference` in the test
  module, with `N >= sqrt(norm(D) / (2*delta))` — typically 4–16, not the 64 used for
  rendering.

## 8. Tests

1. Conversion vs the kernel, gamma in {0, 0.25, 0.5, 0.75, 1, 1.5}.
2. Exact AABB vs dense sampling: nothing outside, and no smaller box contains the samples.
3. Exact vs 1200-segment reference, >= 1000 random pairs, counts must agree.
4. Constructed: exactly 0/1/2/3/4 intersections; tangency; identical curves; adjacent edges
   sharing a vertex; collinear controls; **parallel bulges (the Q4 = 0 path)**.
5. Count invariant under rotation, translation, uniform scaling.

## 9. Thesis

Appendix: §1, §2, §3 in full (short), the `beta = 2*det(...)` fact, the degeneracy table,
the cost comparison. One paragraph on what the CAD literature does (subdivision / Bézier
clipping) and why quadratics don't need it — implicitising a cubic gives degree 9, a
quintic degree 25; quadratics give degree 4 in closed form. Main text: one sentence in the
metrics section. Separately, where gamma is introduced: the curve's maximum perpendicular
excursion from the straight edge is exactly `gamma * h`, with h the edge point's distance
from the chord — that is what gamma means geometrically.

Prior art: Lane & Riesenfeld (1980); Sederberg, Anderson & Goldman (1984);
Sederberg & Parry (1986); Sederberg & Nishita (1990); Guttman (1984);
Leutenegger et al. (1997); Bentley & Ottmann (1979); Balaban (1995); Shewchuk (1997).
