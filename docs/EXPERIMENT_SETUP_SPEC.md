# TWIN — Experimental setup overhaul

## What this document is

An implementation specification. It describes changes to this repository so that it can
(a) compute a defined suite of graph-layout quality metrics, and (b) run, record and reproduce
comparative layout experiments.

**Scope boundary.** Nothing here changes the Twin Neighbor Embedding method itself.
`twin_adjacency`, the `GraphMatrixMethod` subclasses and the embedding wrappers implement the
method and are treated as given. This work is measurement and orchestration around them.

**Companion input: `thesis.pdf`.** Useful but not required — every formula this spec depends on
is written out inline, with the thesis equation number given as a cross-reference. If you have
the PDF, use it to check formulas; if not, the spec stands alone. Section numbers prefixed
"thesis §" refer to that document; bare "§" refers to this one.

**`[THESIS EDIT]` markers.** A few places specify behaviour that deliberately differs from
`thesis.pdf`. Implement what this spec says. **Do not "correct" the code back to match the
PDF** — the document is being revised separately, and these markers record the known
divergences.

---

## Background — what this project is

Enough context to work from without reading the thesis.

### The problem domain

Graph layout: given a graph $G(V,E)$ with $n$ vertices and $m$ edges, assign each vertex a
position in $\mathbb{R}^d$ (here $d=2$) such that the drawing reveals the graph's structure.
The target graphs are **sparse data networks** — social, biological, citation-like — where
$m = O(n \log n)$.

### The method under study

Conventional layout embeds **vertices only**; edges are then drawn as straight lines between
their endpoints, so edge geometry is a by-product rather than something the algorithm reasons
about.

**Twin Neighbor Embedding (TNE)** is a *pre-embedding transform*. Instead of laying out $G$, it
builds a larger "twin" graph in which **both vertices and edges of $G$ are nodes**, and then
hands that to an ordinary layout algorithm. The layout therefore produces two point sets: one
for the original vertices, one for the original edges ("edge points"). Because the transform
happens before layout, TNE composes with any existing layout method rather than replacing one.

Algebraically, with $B$ the $n \times m$ vertex–edge incidence matrix:

$$\tilde{A} = \begin{pmatrix} 0 & B^{\top} \\ B & 0 \end{pmatrix},
\qquad
A_{\mathrm{tne}}(\alpha, k_{\max}) = \sum_{k=1}^{k_{\max}} \alpha^{k}\tilde{A}^{k}$$

$\tilde A$ is a bipartite vertex–edge incidence structure; its powers couple vertex adjacency,
edge adjacency and vertex–edge incidence. $\alpha \in (0,1)$ damps longer paths, $k_{\max}$ sets
how many steps are accumulated. Defaults in this repo: $\alpha = 0.85$, $k_{\max} = 2$.

Two identities matter because the code relies on them:
$$BB^{\top} = A_V + D, \qquad B^{\top}B = A_E + 2I_m$$
where $A_V$ is the vertex adjacency matrix, $A_E$ the edge–edge adjacency matrix (the adjacency
of the **line graph** $G_\ell$, whose nodes are the edges of $G$), and $D$ the degree matrix.

### The three representations — the core comparison

Every experiment compares three ways of feeding the same graph to the same layout algorithm.
These are the `GraphMatrixMethod` subclasses in `twin/graphmatrix.py`:

| Class | Matrix | Points laid out | Meaning |
|---|---|---|---|
| `VertexMatrix` | $A_V$ | $n$ | conventional vertex layout (the baseline) |
| `EdgeMatrix` | $A_E$ | $m$ | line-graph layout; vertex positions derived from incident edge points |
| `TwinEmbeddingMatrix` | $A_{\mathrm{tne}}$ | $n + m$ | the method under study |

Each exposes `vertex_and_edge_embeddings(X) -> (X_v, X_e, slopes)`, which splits or derives the
vertex and edge point sets. **Comparing these three is the central ablation**: it isolates what
the transform contributes as against vertex-only or edge-only representations.

Ordering note that is easy to get wrong: `twin_adjacency` builds
`[[None, Bᵀ], [B, None]]`, so rows are **[edges ($m$); vertices ($n$)]**. Accordingly
`TwinEmbeddingMatrix.vertex_and_edge_embeddings` takes `X[:, ne:]` as vertices and `X[:, :ne]`
as edge points.

### The layout backend

The primary backend is **SG-$t$-SNE-$\Pi$** (`sgtsnepi`), a graph adaptation of $t$-SNE. It
takes a sparse **affinity** matrix as its target distribution $P$ and produces a low-dimensional
layout by minimising a KL divergence against a Student-$t$ kernel in the layout space.

Its one significant parameter is **λ** (`lambda_par`): SG-$t$-SNE rescales each row of $P$ by
finding a per-row exponent $\gamma_i$ such that $\sum_j p_{ij}^{\gamma_i} = \lambda$. λ therefore
controls how concentrated or spread the target neighbourhood distribution is — it is the graph
analogue of $t$-SNE's per-point perplexity calibration. Typical values are 1–10, and its effect
scales roughly logarithmically.

Other layout methods in `twin/embedding.py` (Fruchterman–Reingold, Kamada–Kawai, Yifan Hu,
neato, ForceAtlas2, spectral) serve as comparison baselines.

### What is being measured, and why

The method claims to reduce three artifacts of layouts produced by $t$-SNE-family methods on
sparse graphs:

- **(a) vertex crowding** — vertices that are close in the graph collapse into a blob, so
  substructure inside the group is unresolvable;
- **(b) erroneous clustering** — vertices are placed together that are not actually related,
  forming groups that suggest structure not present in the data;
- **(c) excessive long-edge crossings** — weakly-linked vertices are pushed far apart and their
  long edges sweep across the drawing.

The metric suite (§2) exists to test these claims quantitatively rather than by eye. Roughly:
crowding metrics for (a), label-based metrics for (b), crossing metrics for (c), plus
neighbourhood-preservation metrics to confirm the transform does not degrade what the baseline
already does well, plus cost measurements.

### Vocabulary

| Term | Meaning |
|---|---|
| **layout** / **embedding** | the array of point positions produced for one graph by one method |
| **vertex points** $X_v$, **edge points** $X_e$ | the two point sets a twin/edge layout produces |
| **representation** | which matrix is fed to the layout method: $A_V$, $A_E$, $A_{\mathrm{tne}}$ |
| **λ** (`lambda_par`) | SG-$t$-SNE neighbourhood-scale parameter |
| **α**, **$k_{\max}$** (`alpha`, `k`) | TNE damping factor and polynomial order |
| **γ** (`gamma`) | curvature of drawn edges: 0 straight, 0.5 quadratic Bézier, 1 interpolating |
| **affinity** | edge weight where larger = more strongly related |
| **distance** | edge weight where larger = further apart |
| **line graph** $G_\ell$ | graph whose nodes are the edges of $G$, adjacent when they share an endpoint |

---

## 0. Current state — read this before planning

### What already exists and works

| Area | File | Notes |
|---|---|---|
| Graph representations $A_V$, $A_E$, $A_{tne}$ | `twin/graphmatrix.py` | `VertexMatrix`, `EdgeMatrix`, `TwinEmbeddingMatrix(alpha, k, normalize)`. Each exposes `vertex_and_edge_embeddings(X) -> (X_v, X_e, slopes)`. **This is already the ablation axis.** |
| TNE matrix | `twin/twin_embedding.py` | `twin_adjacency(B, alpha, n)` — Horner-style accumulation of $\sum_{k=1}^{n}\alpha^k \tilde A^k$ |
| Layout methods | `twin/embedding.py` | Spring (FR), Kamada–Kawai, YifanHu (sfdp), Neato, ForceAtlas2, SGtSNE, Spectral |
| Batch spec engine | `twin/batching.py` | Well-built, domain-agnostic: recursive spec → cases → keys → records. Do **not** rewrite it. |
| Batch runner | `twin/pipelines.py` | `graph_embedding`, `batch_graph_embedding` over `graph × representation × embedding` |
| Curved edge drawing | `twin/plots.py::draw_curved_edges` | Quadratic curve through the embedded edge point, `gamma` controls curvature. **This is the geometry the curved-crossing metric must reuse.** |
| Straight crossing count | `twin/plots.py::count_intersections` | Shapely-based, per-edge, sorted by length |
| Datasets | `twin/dataset.py` | ring/tree of cliques, karate, football_2023, small_world, barabasi_albert, mnist, tabula, polblogs, sbm-er, lfr |
| Labels | `twin/metadata.py` | `football_2023["conference"]`, `football_2023["conference_standing"]`, `mnist["digit"]` only |

### What is missing

Of the metrics defined in thesis §3, **only three partial implementations exist**
(`twin/metrics.py`, 112 lines). Everything else is absent:

| Thesis §3 metric | Eq. | Status |
|---|---|---|
| Adjacency Distance Ratio, ADR | (27), (29) | ✗ — `point_distance_metric` is a rough ancestor, not ADR |
| $k$-NN recall / precision / Jaccard $R^k, P^k, J^k$ | (32)–(35) | ~ — `knn_neighbor_preservation_accuracy` implements these |
| Weighted recall $W^k$ | (36)–(38) | ~ — via `mode="weighted recall"` |
| Scale-normalized stress SNS | (43)–(45) | ✗ |
| Silhouette Sil, and graph-side Sil* | (49)–(55) | ✗ |
| $k$-NN classification accuracy Acc$^k$, and graph-side Acc* | (57)–(64) | ✗ (`knn_classification_accuracy` exists but **is broken**, see §1) |
| Normalized NN distance NND$_k$ | (65)–(67) | ✗ |
| Node resolution NR, vertex-wise VNR | (68), (69) | ✗ |
| Distance distributions | (70), (71) | ✗ |
| Crossings Cr | (78) | ~ — `count_intersections`, straight-line only, **excludes** adjacent pairs (thesis includes them) |
| Length-weighted crossings LWCr | (81) | ✗ |

Missing infrastructure:

- **No persistence whatsoever.** `batches/batch.py` builds a dict in memory and exits. No run
  is reproducible or recoverable.
- **No seed axis.** `graph_embedding` takes a single `seed`; there is no repeated-run machinery
  and no dispersion reporting.
- **No λ sweep or selection.**
- **No runtime / memory instrumentation.**
- **No label registry** — labels are reached ad hoc from `paper_figures.py`.
- **No straight-vs-curved crossing variants.**
- **No metric layer at all** in the batch path: `batch_graph_embedding` returns layouts, and the
  only metric aggregator (`embedding_metrics_all`) is the *old* hardcoded
  vertex/edge/twin triple that bypasses the batch engine.

---

## 0.5 Errata and decisions — these supersede the text below

Corrections following a verification pass against the actual repository and environment. Where
this section conflicts with anything later in the document, **this section wins.**

### Claims in §1 that did not reproduce

| Item | Correction |
|---|---|
| §1.1 | **The code is not broken.** `KNeighborsClassifier.predict(None)` is valid, and `kneighbors(X=None)` excludes each point from its own neighbour set — so the leave-one-out behaviour is already there and accuracy is not trivially 1.0. Rewrite it anyway (no tie control, no chance floor, no $d=0$ guard, transpose sniffing, and it leans on a subtle sklearn convention), but as a **reimplementation for control, not a bug fix**. |
| §1.2 | Worse than described, and unreachable. The adjacency branch **raises** `ValueError` rather than misbehaving quietly, and `embedding_metrics_all` only ever calls it with `adjacency=False`. Deletion is still correct; "silently" is wrong. |
| §1.5 | The import claim is stale — scipy ≥1.9 exposes `scipy.sparse.linalg` lazily, and the code runs as written. The `which="SM"` robustness and disconnected-multiplicity concerns stand. |
| §1.7 | `fa2` 1.1.2 imports and runs in the current environment. Not a blocker. Keep the switch to `fa2_modified` as a **reproducibility** item — an unmaintained package that happens to work today is a rebuild risk — but at low priority. |
| §1.6 | Substance confirmed. Two details: `VertexMatrix` builds from $\sqrt{B_w}$, so entries are $w_{ij}$ rather than $w_{ij}^2$ (the sign inversion is unaffected); and the broken lines in §1.2 are 97–102, not 96–100. |

**Consequence for §7.** Step 1 is titled "they corrupt current output." Only §1.6 does, and it is
scheduled in step 2. Retitle step 1 to something honest — correctness and conventions
groundwork — and keep the ordering.

### §1.8 — new: `sgtsnepi(silent=True)` closes file descriptor 2

Not in the original spec. The default `silent=True` path in the `sgtsnepi` C extension closes
fd 2 and does not restore it. Computation is unaffected, but every subsequent write to stderr
raises `[Errno 9] Bad file descriptor` — and **loguru logs to stderr**, so `batches/batch.py`
(which passes `progress_fn=logger.info`) dies at the first progress line after the first
SG-$t$-SNE case.

This blocks `log.txt`, progress reporting and any resumable overnight batch, so **fix it in the
first session**: save and `dup` fd 2 before the call, restore after. Verify by asserting that
logging still works following a silent embed.

### Facts established about the datasets

- **`mnist` is not MNIST.** It is `sklearn.datasets.load_digits()` — 1797 samples of 8×8 digits,
  giving $n=1797$, $m=14732$ at `n_neighbors=12`. It is neither large nor MNIST. **Rename it
  `digits`** throughout the code, and describe it accurately. The stored side is the **distance**
  side, confirmed: weights range 5.3–37.8, so the affinity form is the derived one.
- **Two different karates exist.** `batch_registries["graph"]["karate_club"]` is
  `nx.karate_club_graph` with weights 1–7 (Zachary interaction strengths); `dataset.karate()`
  reads `karate.mat` and is unweighted. The §3.4 table's "karate → `none`" is true of the
  unregistered one only.
- **football_2023 is effectively unweighted** — 732 of 735 edges weigh 1, three weigh 2. Declare
  it `Unweighted()` on both sides. This closes the corresponding question in §8.
- **Every currently registered dataset is connected**, single component, no self-loops. §4.4's
  policy machinery is inert until the roster changes.
- `polblogs`, `sbm-er-3modes`, `lfr` and `homer` exist in `dataset.py` but are **not** in
  `batch_registries`, so they are unreachable from the batch path. `sbm-er-3modes` carries a
  single constant weight $8.691\times10^{-5}$ on all edges — a case the kind table does not cover;
  treat a constant weight as `Unweighted()`.
- `twin/features.py` is dead cookiecutter scaffolding. Delete it.

### Decisions on the flagged questions

**1. The `affinity → distance` transform — split the kind; drop the clamp.**

An earlier draft of this section proposed clamping,
$d = -\log(w/(w_{\max}(1+\varepsilon)))$. **Do not implement that.** It patches a symptom of a
modelling problem: `affinity` was covering two situations that need different treatment, and the
clamp is only ever right for a third situation that no transform can actually fix.

Each kind must name a model under which the distance form is *derived*, not merely monotone:

| kind | What $w$ is | $d$ | Justification |
|---|---|---|---|
| `probability` | $P(\text{same event / transition})$ | $-\log w$ | surprisal; $d=0$ at $w=1$ is **correct**, not a defect |
| `kernel` | evaluation of a similarity kernel | $-\log(w/w_{\max})$ | recovers the latent metric; $d=0$ means genuinely coincident |
| `strength` | an arbitrary positive magnitude (counts, interaction strengths) | $1/w$ | resistance model: weights compose in series like conductances |
| `distance` | a metric | identity | — |
| `none` | indicator | hop count | — |

`count` becomes an alias of `strength`. `affinity` is **retired as a kind** — it named a property
of the sign of the relationship, not a model, which is exactly why it needed a clamp.

> **Superseded in form, not in substance — see the §3.4/§3.5 rewrite entry below.** This
> vocabulary survives as the `meaning` of a **stored** source; what changed is that it no longer
> doubles as a prescription for deriving the other side, because a dataset with both sides stored
> needs no derivation at all. `adaptive_kernel` was added for the case this table cannot express:
> a kernel with per-point parameters, which is globally non-invertible.

Two consequences worth stating in the code and in the write-up:

- **Zero ideal distance is not automatically a defect.** Under `probability` and `kernel` it is
  the model saying two vertices coincide. Whether a layout method can honour that is a separate
  question, and a poor embedding under those conditions is a *layout* limitation, not a
  modelling error. Keep the two concerns apart.
- **$1/w$ is not a fallback**, it is the resistance/random-walk model, and it has as much claim
  to principle as $-\log$ does. Use it because the model fits, not because it avoids zeros.

**Tabula must be classified before it can be transformed.** 3971 of 49368 edges sitting at
exactly $w_{\max}=1.0$ is the diagnostic: genuine kernel evaluations do not pile up on the
boundary. Find out how those weights were produced.

- If they are kernel evaluations $\Rightarrow$ `kernel`, $-\log(w/w_{\max})$, no clamp, and the
  zero distances are real.
- If they are normalised magnitudes $\Rightarrow$ `strength`, $1/w$, and the problem never
  arises.
- If they are **clipped or saturated** $\Rightarrow$ no transform can repair it. The information
  was destroyed upstream; those 3971 edges are genuinely indistinguishable. Either recover the
  unclipped source or state the limitation explicitly. A clamp here manufactures a distinction
  the data does not contain.

**Before intervening at all, measure whether it matters.** Zero ideal distances damage different
metrics differently: $\mathrm{SNS}$ tolerates them (a zero target is well-defined and the closed
form for $\alpha^{*}$ is unaffected); silhouette is distorted, since zero distances shrink
$a(i)$ and inflate $s(i)$; shortest-path derivations are worst, because zero-weight edges make
whole groups mutually distance-zero and collapse the metric. So report, in the connectivity
report, the fraction of **vertex pairs** at $d=0$ under the chosen transform — not just the
fraction of edges at $w_{\max}$. If that fraction is small the question is academic; if it is
large, it is evidence about the data rather than a reason to reach for a clamp.

**2. Karate — adopt the split.** Register `karate` as `dataset.karate()` (the `.mat`, unweighted,
`Unweighted()` on both sides) and register `nx.karate_club_graph` separately as `karate_weighted`
with a stored affinity side, `Stored(meaning="strength")`. Its 1–7 interaction strengths make it the
genuinely count-weighted small test case that football turned out not to be, which is worth
having now that the `count` path exists.

**3. `disconnected="components"` — do not build it.** Implement `"lcc"` (a no-op on everything
currently registered) and `"mask"`; raise `NotImplementedError` with an explanatory message for
`"components"` until the roster contains a disconnected graph. Untested code for a case that
does not occur is worse than an honest refusal.

**4. §2.0 vs §4.4 disagreed on the default.** §4.4 wins: `"lcc"` is the default everywhere, and
`silhouette_graph` refuses `"mask"`. Ignore §2.0's `"mask"` default.

**5. Derived versus laid-out point sets — label them, do not exclude them.** `EdgeMatrix`
derives vertex positions as a degree-weighted centroid of incident edge points; `VertexMatrix`
derives edge points as a convex combination of endpoints. Scoring those derived sets is correct
— deriving them *is* how $A_E$ produces a graph layout at all — but the comparison is
asymmetric, and that must be visible. Add `point_provenance: Literal["primary","derived"]` per
point set on `LayoutResult`, carry it into the metric rows, and surface it in the reporting
layer.

**6. $k$ = mean degree of $G$**, the input graph — not of the line graph or the twin graph —
for every representation, since $R^k$ is evaluated against $G$'s neighbourhoods regardless.
Round to nearest, floor of 1.

**7. `affinity_matrix_to_distance` must symmetrise.** Row-normalisation makes the matrix
asymmetric and `nx.from_scipy_sparse_array` then silently keeps one of each pair — a bug present
in `KamadaKawaiLayout(mode="similarity")` today. Use symmetric normalisation
$D^{-1/2} M D^{-1/2}$, or symmetrise as $(M+M^{\top})/2$ after normalising, before taking logs.

**8. Crossing normalisation — report three columns**, not one: raw $\mathrm{Cr}$,
$\mathrm{Cr}/m$, and $\mathrm{Cr}/\binom{m}{2}$. For good layouts of sparse graphs crossings
scale closer to $m$ than to $m^2$, so dividing by $\binom{m}{2}$ trends to zero and largely
measures $m$. All three are cheap; which to feature can be decided from the data. Treat
cross-graph crossing comparisons as weak evidence regardless — within-graph is the defensible
claim.

**9. Composite-label scoring needs a guard.** 14 conferences × 3 divisions over 132 teams
produces many tiny classes, where silhouette is 0 by convention and $k$-NN accuracy is ~0 by
construction — so a flat composite score may be class-size collapse rather than absent
substructure. Report the class-count and minimum/median class size alongside the score, and
restrict the headline composite to classes above a stated minimum size.

**10. Commit the spec.** Move `EXPERIMENT_SETUP_SPEC.md` into the repository (`docs/`), since
`CLAUDE.md` references it and it is the design record for these decisions.

**11. Add `pytest` and `pyarrow` to `pyproject.toml`** in the first session — steps 3–6 are all
"+ tests" and step 7 writes parquet.

**12. §1.8 resolved upstream.** `sgtsnepi` 0.7 fixes the descriptor bug; `pyproject.toml` pins
`sgtsnepi>=0.7` and the workaround has been removed. Layouts are bit-identical to 0.5.1, so the
upgrade is numerically neutral. Details in `docs/DECISIONS.md`.

### §3.4 and §3.5 rewritten — the two-source model

Those sections have been **rewritten in place** rather than corrected here, because the change is
structural and a reader should not have to reconstruct the design from a long correction. What
changed, and why:

1. **`Dataset` and `LayoutGraph` are separate layers.** A dataset is a *source of truth plus
   labels plus drawing instructions*; a layout graph is *an affinity form plus a distance form
   plus a way to derive vertex/edge points*. The original §3.4 fused them, which left no answer to
   "what are $A_E$'s labels?" — its nodes are edges, so a length-$n$ label vector does not apply.
   Under the split, labels never reach the layout layer and are applied afterwards, because
   `vertex_and_edge_points` returns points in the **original** indexing. A metric or figure is
   then a function of the embedded points plus a `Dataset`.
2. **`weight_kind` is replaced by declared stored sources.** It was doing two jobs — saying what
   stored weights mean, and prescribing how to derive the other side — which coincide only for a
   dataset that has one weight vector and must guess the other. Tabula has both `A` and `D`, so it
   satisfies the interface directly and no derivation arises. `Dataset` keeps the first job; the
   `vertex_graph` mapping does the second.
3. **The representations are mappings, not owned objects.** `vertex_graph` is
   `Dataset → LayoutGraph`; `edge_graph` and `twin_graph` are `LayoutGraph → LayoutGraph` off the
   vertex graph, which matches the algebra and lets the derivation chain be read off a `parent`
   link instead of maintained as a string. It also builds the incidence matrix once instead of
   three times.
4. **Layout methods ask; the runner does not convert.** Every `LayoutGraph` exposes both forms.
   This is what lets tabula's *measured* distances drive a Kamada–Kawai vertex layout, which the
   old "convert once at the boundary" rule could not express.
5. **`check_weight_consistency` is new**, and is the hazard the change introduces: two
   independently stored sides can disagree about which pairs are close, and the layout optimises
   against one while SNS and Sil\* score against the other.

§3.1, §3.7, §2.5, §4.6, §7 step 2 and the §8 table were updated to match.

### Erratum in §3.3

The code block slices rows: `y0_edge = y0[:ne]`. `init_embedding` returns `(d, n)`, so that
slices the *dimension* axis. The correct form — and what the working `graph_embedding_all`
already does — is **column** slicing:

```python
y0 = E.init_embedding(nv + ne, d)     # shape (d, nv + ne)
y0_edge   = y0[:, :ne]                # A_E
y0_vertex = y0[:, ne:]                # A_V
y0_twin   = y0                        # A_tne
```

Appendix A states the convention correctly. Follow Appendix A.

---

## 1. Bugs to fix first

These are independent of the new work. **Read §0.5 first** — three of the six claims below did
not reproduce against the current environment, and the corrections there take precedence.

### 1.1 `knn_classification_accuracy` is broken — `twin/metrics.py:23`

```python
knn_prediction = knn_classifier.predict(None)
```

`predict(None)` is not valid. Beyond that, the design is wrong even if it ran: fitting on all
points and predicting the same points makes each point its own nearest neighbour at distance 0,
and with `weights="distance"` that is an infinite weight — accuracy would be trivially 1.0.

**Required:** leave-one-out. Query `k+1` neighbours with `kneighbors()`, drop the self column,
then vote over the remaining `k`. Implement from scratch per thesis eqs. (57)–(60) rather than
via `KNeighborsClassifier`; you need the tie rule (§2.3) anyway.

### 1.2 `point_distance_metric` adjacency branch is dead code — `twin/metrics.py:96-100`

```python
distances = adj_point_distances.data
distances = adj_point_distances / all_point_distances.max()   # overwrites, sparse matrix
...
mean_distance = np.nanmean(distances)                          # nanmean of a sparse matrix
```

The first assignment is discarded, the second leaves a sparse matrix, and `np.nanmean` on it
does not do what the code assumes. It is also normalised by `max`, not by the mean pairwise
distance the thesis specifies. **This function is superseded by ADR (§2.1) — delete it** and
migrate the two call sites in `pipelines.py::embedding_metrics_all`.

### 1.3 Transpose sniffing is a latent bug — `twin/metrics.py` throughout

```python
if X.shape[0] != n_samples and X.shape[1] == n_samples:
    X = X.T
```

Silently wrong whenever `n == d` or on any square input. The embedding layer returns `(d, n)`
and the metric layer wants `(n, d)`; guessing at each call site is fragile.

**Required:** one convention, enforced once. Inside `twin/metrics/` **every layout array is
`(n_points, d)`**. Convert exactly once, at the boundary, in the `LayoutResult` adapter (§3.1).
Add `assert X.ndim == 2 and X.shape[0] == n` at metric entry points instead of reshaping.

### 1.4 `YifanHuLayout` and `NeatoLayout` ignore `y0` and `d` — `twin/embedding.py`

Both build a `pos` dict and then never pass it to `graphviz_layout`. Both ignore `self.d`
(graphviz is 2-D). Neither takes a seed that graphviz honours.

**Required:** pass `pos` through where graphviz supports it, and — more important for the review
— mark these methods as **deterministic** in the registry (§4.5) so their single value is not
reported as a zero-variance win against stochastic methods' means.

### 1.5 `SpectralLayout` eigensolver — `twin/embedding.py`

`sp.linalg.eigsh(L, k=d+1, which="SM")` needs an explicit `import scipy.sparse.linalg`, and
`which="SM"` is slow and unreliable on large Laplacians. Use shift-invert
(`sigma=0, which="LM"`) or `which="SA"`, and handle the disconnected case (multiplicity of the
zero eigenvalue > 1) explicitly rather than assuming column 0 is the only trivial vector.

Note also: thesis §2.2 says "the $d$ smallest non-zero eigenvalues" but `thesis/sota.typ`
currently says "the $d$ dominant eigenvectors". The code implements smallest, which is correct;
the inconsistency is in the prose. **[THESIS EDIT]** — no code change needed here.

### 1.6 **Edge weights are being used with the wrong sign on at least one dataset**

This is the most consequential thing in this section — it silently corrupts existing results.

`dataset.mnist()` builds its graph via `construct_knn_graph`, which does:

```python
knn_distance_matrix = knn.kneighbors_graph(mode="distance")
knn_graph = nx.from_numpy_array(knn_distance_matrix)
```

So **mnist edge weights are Euclidean distances** — larger means *less* similar. But
`VertexMatrix.__init__` then builds the weighted adjacency as $B_w B_w^{\top}$ from
`nx.incidence_matrix(graph, weight="weight")`, and `SGtSNELayout.embed` hands that straight to
`sgtsnepi` as the **affinity** matrix. Distant point pairs therefore receive the *strongest*
attraction. Every weighted result on mnist is inverted.

`football_2023` weights are game counts (affinity-like), `tabula` weights come from a
similarity kernel, and karate is effectively unweighted — so the same code path means different
things on different datasets, with nothing in the graph object declaring which.

`KamadaKawaiLayout` and `NeatoLayout` half-acknowledge this with their
`mode="distance"|"similarity"` parameter, but that puts the knowledge in the *layout method*
when it belongs to the *dataset*. Two layout methods know; five do not; the metrics do not.

**Fix: §3.4 and §3.5.** Do not patch this per call site — the weight semantics must become a declared
property of the dataset. Until that lands, treat all weighted mnist results as invalid.

### 1.7 `fa2` is unmaintained

Does not build against current NumPy/Python. Switch to `fa2_modified`. Whatever you use, the
settings must be recorded in the run manifest (§4.6) — in particular whether LinLog mode is on,
since `batches/batch.py` currently sweeps it but nothing records which run had it.

---

## 2. New metric package

Replace the flat `twin/metrics.py` with a package. Roughly 10 metrics with shared machinery
justifies the split.

```
twin/metrics/
    __init__.py        # re-exports + METRIC_REGISTRY
    common.py          # distance helpers, ideal-distance model, validation
    structural.py      # ADR, kNN recovery, weighted recall, SNS
    label.py           # silhouette (+ graph), kNN accuracy (+ graph)
    visual.py          # NND, NR, VNR, distance distributions, edge lengths, detour
    crossings.py       # curve construction, crossing detection, Cr, LWCr
    registry.py        # name -> (callable, kind, requires) metadata
```

Every metric function signature follows one of three shapes:

```python
def metric(X: np.ndarray, G: nx.Graph, **params) -> float | np.ndarray   # kind="structural"
def metric(X: np.ndarray, labels: np.ndarray, **params) -> float          # kind="label"
def metric(X: np.ndarray, **params) -> float                              # kind="visual"
```

with `X` strictly `(n, d)`. Crossing metrics additionally take the edge-point array and the
graph. Each returns a scalar, or an array when a parameter (such as `k`) is swept.

### 2.0 `common.py` — shared pieces

**Ideal distances $d_{ij}$** (thesis §3.1.1, and §2.1.1 for the weight transform). One function:

```python
def ideal_distances(G, weight_transform="identity", weight="weight") -> np.ndarray
```

Transform options per the thesis: `identity` ($f(w)=w$), `inverse` ($1/w$),
`one_minus` ($1-w$), `neg_log` ($-\log w$). For unweighted graphs all edge lengths are 1.
Non-adjacent pairs get the shortest-path distance under those edge lengths.

Disconnected graphs: `disconnected="lcc"` is the default and `"mask"` is available;
`"components"` raises `NotImplementedError` (§0.5 decision 3, decision 4).

Cost warning: this is $O(n^2)$ memory. Guard it — `ideal_distances` should raise unless
`n <= max_n` (default 20000, overridable), so SNS and Sil\* fail loudly rather than exhausting
memory on the large graphs. **The guard lives in this primitive**, so there is exactly one place
it can be forgotten.

**Which graph the ideal distances come from, and who owns the result.** This is not a free choice:

> Ideal distances are computed from **`vertex_graph(dataset).distance`**, once per dataset per
> run, and **shared by every representation of it**.

`vertex_graph`'s distance form is by construction the *least derived* one available — the stored
distance side if there is one, otherwise the single derivation from the stored affinity (§3.4).
Never the generic fallback exposed by `edge_graph` or `twin_graph`.

The reason is the central ablation. SNS and Sil\* compare a layout against the graph's ideal
distances; if those distances were taken from whichever `LayoutGraph` produced the layout, then
$A_V$, $A_E$ and $A_{tne}$ would each be scored against a *different* target, and comparing their
scores would mean nothing. One target per dataset is what makes the comparison legitimate.

Ownership: the **runner** computes it once and holds it. Not `Dataset`, which holds no derived
information (§3.4), and not the metric, which would recompute an $O(n^2)$ expansion per call.

**Pairwise distances.** One cached helper; several metrics need the full condensed pairwise
distance vector and the mean $\bar d$ (eq. 66). Use `scipy.spatial.distance.pdist` and keep the
condensed form — do not materialise the square matrix twice.

### 2.1 `structural.py`

#### ADR — eq. (27), (29)

```python
def adjacency_distance_ratio(X, G, weight="weight") -> float
```

$$\mathrm{ADR}(X;G)=\frac{\frac{1}{W}\sum_{i<j} w_{ij}\lVert x_i-x_j\rVert}
{\frac{2}{n(n-1)}\sum_{i<j}\lVert x_i-x_j\rVert},\qquad W=\sum_{i<j}w_{ij}$$

For unweighted graphs this reduces to eq. (29) automatically with $w_{ij}\in\{0,1\}$ — do not
write a second branch, just set unit weights.

Docstring must state: **comparable across layouts of a fixed graph under a fixed weighting, and
across nothing else.** Different graphs, or the same graph with a re-parameterised weight
($w$ vs. $w^2$), are not on a common scale. The reporting layer must enforce this (§5.3).

#### $k$-NN neighbourhood recovery — eq. (30)–(35)

Keep the current `knn_neighbor_preservation_accuracy` logic — it is correct — but:

- split into four named functions (`knn_recall`, `knn_precision`, `knn_jaccard`,
  `weighted_recall`) sharing a private core, rather than a `mode` string;
- accept `k` as scalar or array-like and vectorise over `k` by querying `max(k)` neighbours once
  and slicing, instead of the current recursive re-fit per `k`;
- keep the existing NaN handling for zero-degree vertices (correct per §3.1.1) and additionally
  **return the excluded count** so it can be reported rather than silently absorbed.

`weighted_recall` is eq. (36)–(38); assert that on an unweighted graph it equals `knn_recall`
(eq. 39) — this is a cheap and meaningful unit test.

#### SNS — eq. (43)–(45)

```python
def scale_normalized_stress(X, G, weight_transform="identity", **kw) -> float
```

$$\alpha^{*}=\frac{\sum_{i<j} d_{ij}\lVert x_i-x_j\rVert}{\sum_{i<j}\lVert x_i-x_j\rVert^{2}},
\qquad
\mathrm{SNS}=\sqrt{\frac{\sum_{i<j}(d_{ij}-\alpha^{*}\lVert x_i-x_j\rVert)^{2}}{\sum_{i<j}d_{ij}^{2}}}$$

Compute $\alpha^*$ in closed form; do not run an optimiser. Unit test: SNS is invariant to
$X \mapsto \beta X$ and to $d \mapsto \beta d$ for $\beta > 0$ (thesis asserts both).

Also provide `raw_stress` and `normalized_stress` (eqs. 40, 41) — not for the results tables,
but so the §3 argument that they are scale-dependent can be *demonstrated* rather than asserted
if wanted.

### 2.2 `visual.py`

#### NND$_k$ — eq. (65)–(67)

```python
def normalized_nn_distance(X, k=10) -> float | np.ndarray
```
Mean over vertices of the mean distance to the $k$ nearest embedded neighbours, divided by the
global mean pairwise distance $\bar d$ (eq. 66). Vectorise over `k` as above.

#### NR and VNR — eq. (68), (69)

```python
def node_resolution(X) -> float                                # eq. 68, global
def node_resolution_vertexwise(X, normalize="max") -> float    # eq. 69
```

**`normalize="mean"` is the default** — per vertex,
$\min_{j\neq i}\lVert x_i-x_j\rVert / \bar d$. Rationale: normalising
per-vertex by $\max_{j\neq i}\lVert x_i-x_j\rVert$ (the thesis as currently written) means that
with a single outlier vertex every denominator is approximately the diameter, so VNR collapses
back toward the global NR it was introduced to fix. Dividing by $\bar d$ removes that
sensitivity and makes NND and VNR one family with one normaliser.

**[THESIS EDIT]** — thesis eq. (69) specifies `max`; implement `mean` as the default anyway.
Keep `normalize="max"` selectable so the two can be compared once on real layouts.

#### Distance distributions — eq. (70), (71)

```python
def pairwise_distance_distribution(X) -> np.ndarray     # all pairs, normalized by mean
def nn_distance_distribution(X) -> np.ndarray           # per-vertex nearest, normalized by mean
```
Return the arrays; do not summarise. §3.1.3 is explicit that these are diagnostics, not scores.
Persist them separately from the scalar metrics (§4.6) — they are large.

#### Edge lengths and detour

```python
def edge_lengths(X, G) -> np.ndarray                       # chord lengths L(e), eq. (79)
def edge_length_cv(X, G) -> float                          # coefficient of variation
def detour_ratio(X, E, G, gamma=0.5, n_samples=64) -> np.ndarray
```

`detour_ratio` is arc length of the drawn curve over chord length, per edge — bounded below by
1, exactly 1 for straight edges. It measures what the curved formulation costs in ink; it is
deliberately **not** folded into LWCr (see §2.4).

### 2.3 `label.py`

#### Silhouette — eq. (49)–(52), and graph-side eq. (53)–(55)

```python
def silhouette(X, labels) -> float
def silhouette_graph(G, labels, **ideal_distance_kw) -> float
```

Singleton clusters score 0 (Rousseeuw's convention, stated in §3.1.2). You may use
`sklearn.metrics.silhouette_score` for the layout version, but the graph version needs the
precomputed $d_{ij}$ from `common.ideal_distances` — `sklearn` accepts
`metric="precomputed"`, which is the simplest correct route for both.

`silhouette_graph` must be labelled in the output schema as a **label-validity check, not a
baseline**. It answers "are these labels reflected in the topology at all?",
and it is on an incomparable scale to the layout score. Enforce this at the schema level (§5.3):
the two must not land in adjacent columns of the same table.

#### $k$-NN classification accuracy — eq. (57)–(60), graph-side (61)–(64)

```python
def knn_accuracy(X, labels, k=10, weighting="uniform", tie_rule="fractional") -> float | np.ndarray
def graph_neighborhood_accuracy(G, labels, weighting="uniform", tie_rule="fractional") -> float
```

Rewrite from scratch (see bug 1.1). Requirements:

- **Leave-one-out**: query `k+1` neighbours, drop self.
- `weighting="uniform"` is eq. (57); `weighting="distance"` is eq. (58) with $f(d)=1/d$. Guard
  $d=0$ (coincident points, which crowding produces) — clamp with a small epsilon and count how
  often it triggers.
- **`tie_rule="fractional"`**: when $p$ labels tie for the argmax, the vertex contributes $1/p$
  if its true label is among the tied set, else 0. This is the expected accuracy under uniform
  random tie-breaking, in closed form — no variance, no seed. Also implement
  `tie_rule="first"` for comparison. Document in the docstring that the result is then an
  **expectation over tie-breakings**, not a literal fraction of vertices.
- `graph_neighborhood_accuracy` uses the 1-hop graph neighbourhood (variable size $\deg(i)$),
  with edge-weight voting for weighted graphs (eq. 62). Isolated vertices excluded, and the
  excluded count returned.
- **Also return the majority-class rate** as a chance floor. Accuracy on imbalanced labels is
  misleading without it (football_2023 has ~14 conferences of very unequal size).

### 2.4 `crossings.py` — the most involved piece

This module is what makes the straight-vs-curved experiment (§4.3) possible. Build it in three
layers.

#### Layer 1 — curve construction

Factor the curve geometry out of `plots.py::draw_curved_edges` into a reusable function; the
plotting code should then call it too, so the drawn picture and the measured crossings cannot
diverge.

```python
def edge_curves(G, V, E=None, gamma=0.5, n_samples=64) -> np.ndarray   # (m, n_samples, d)
```

Reuse the existing quadratic kernel exactly:

```
[[1, 0, 0], [-1-2γ, 4γ, 1-2γ], [2γ, -4γ, 2γ]]
```

with control points $(x_u,\; y_{(u,v)},\; x_v)$. `gamma=0` must produce exact straight
segments — that is how the straight variant is obtained, so both variants go through one code
path. `E=None` also yields straight segments (for baselines that have no edge points).

#### Layer 2 — crossing detection

```python
def crossing_pairs(curves, edges, include_adjacent=True, tol=1e-12) -> list[tuple[int,int,int]]
```
Returning `(edge_a, edge_b, n_crossings)`.

- **`include_adjacent=True` is the thesis convention** (§3.1.3 argues explicitly that curved
  edges incident to a shared vertex may still cross away from it). The existing
  `count_intersections` does the opposite — it `continue`s on shared vertices. Keep the flag so
  both conventions can be reported, but default to the thesis one.
- **Degeneracy convention.** Eq. (74) defines `cr` as the *cardinality* of
  an intersection set, which is infinite for collinear overlap — routine under crowding, and
  certain when two vertices land on the same coordinate. Required behaviour: collinear/segment
  overlap counts as **exactly 1** crossing; coincident endpoints are excluded by **vertex
  identity**, not by coordinate value (eq. 73 excludes the four coordinate values, which
  silently suppresses genuine crossings when two distinct vertices coincide). Count and report
  degenerate incidents separately. **[THESIS EDIT]** — thesis §3.1.3 does not yet state a
  degeneracy convention; implement the one above.
- Performance: the thesis notes $O(s^2m^2)$. Use an STR-tree bounding-box prefilter
  (`shapely.STRtree`) before pairwise intersection; Shapely is already a dependency. For the
  large graphs this is the difference between minutes and hours.

#### Layer 3 — the metrics

```python
def crossing_number(curves, edges, **kw) -> int                 # Cr, eq. (78)
def crossings_per_edge_pair(curves, edges, **kw) -> float       # Cr / C(m,2)
def length_weighted_crossings(curves, edges, V, **kw) -> float  # LWCr, eq. (81)
```

$$\mathrm{LWCr}(X)=\frac{1}{\bar L(X)}\sum_{a<b}\frac{L(e_a)+L(e_b)}{2}\,\mathrm{cr}(e_a,e_b)$$

**$L(e)$ is the chord $\lVert x_i - x_j\rVert$ (eq. 79), never the arc length.** This is
load-bearing, not stylistic: the chord is identical for the straight and curved rendering of the
same layout, as is the normaliser $\bar L(X)$, so the whole weight vector is held fixed and the
difference in LWCr between the two variants isolates the change in crossings alone. Arc-length
weighting would move crossings and weights together and make the comparison uninterpretable.
Put that reasoning in the module docstring so nobody "improves" it later.

`crossings_per_edge_pair` is the cross-graph-comparable form. Do **not** normalise by Purchase's
$c_{\max}=\binom{m}{2}-\frac12\sum_v \deg(v)(\deg(v)-1)$: it assumes at most one crossing per
pair and that adjacent edges never cross, both of which this codebase deliberately discards.

#### Keep `count_intersections` working

`plots.py::count_intersections` feeds the crossings-vs-length-threshold curve (the paper's
Figure 3). Reimplement it on top of Layer 2 rather than leaving two independent intersection
routines in the tree.

### 2.5 Registry

```python
METRIC_REGISTRY = {
    "adr":            MetricSpec(fn=adjacency_distance_ratio, kind="structural", needs=("dataset",)),
    "knn_recall":     MetricSpec(..., sweeps=("k",)),
    ...
    "silhouette":     MetricSpec(..., kind="label",  needs=("dataset", "labels")),
    "lwcr":           MetricSpec(..., kind="crossing", needs=("dataset", "V", "E"), variants=("straight","curved")),
    ...
}
```

`MetricSpec` records: callable, kind, required inputs, swept parameters, whether higher is
better, cost class (`cheap` / `quadratic` / `crossing`), and whether it applies to the vertex
layout, the edge layout, or both. The cost class matters — SNS, Sil\* and the crossing metrics
are the ones that will not run on `tabula` or `mnist`, and the driver needs to know that
statically rather than discovering it at hour three.

---

## 3. Pipeline layer

### 3.1 A result type

Introduce a dataclass so metrics stop receiving loose arrays:

```python
@dataclass
class LayoutResult:
    V: np.ndarray          # (n, d)  vertex layout — canonical orientation
    E: np.ndarray | None   # (m, d)  edge-point layout
    slopes: np.ndarray | None
    dataset: Dataset       # carries the graph, the labels and the style (§3.4)
    representation: str
    embedding: str
    params: dict
    seed: int | None
    point_provenance: dict[str, Literal["primary", "derived"]]   # §0.5 decision 5
    runtime_s: float           # embed only
    representation_s: float    # matrix construction only
    n_points: int              # n for A_V, m for A_E, n+m for A_tne
    peak_mem_mb: float | None

    @property
    def G(self) -> nx.Graph: return self.dataset.graph
```

`graph_embedding` currently returns `(d, n)` arrays. **Transpose once, here**, and never again
(bug 1.3). Add a `from_raw()` constructor that does the transpose and asserts shapes.

It holds the **`Dataset`**, not a bare graph, because labels, style and weight provenance all
have to reach the metric and reporting layers, and a layout of $A_E$ or $A_{tne}$ has no labels of
its own (§3.4). `G` stays available as a property so existing shape assertions read unchanged.

### 3.2 Instrumentation

Wrap the embedding call in `graph_embedding` with `time.perf_counter()` and
`tracemalloc`/`resource.getrusage` for peak memory. This must be inside the run loop, not
around the whole batch, and must cover **only** the embed step — the representation
construction (which for TNE builds the $(n{+}m)$ matrix) should be timed separately, since the
cost of the transform is a first-order reported quantity:

```python
runtime_s          # embed only
representation_s   # matrix construction only
n_points           # n for A_V, m for A_E, n+m for A_tne
```

`n_points` makes the "TNE embeds $n+m$ points" cost visible in every row without further
computation.

### 3.3 Common initialisation across representations

The old `graph_embedding_all` handled this and the new batch path lost it. For the paired
protocol (§4.2) the three representations must start from the *same* random configuration,
which requires slicing one $(n+m)$ initialisation:

```python
y0 = E.init_embedding(nv + ne, d)
y0_edge   = y0[:ne]      # A_E   uses the first ne rows
y0_vertex = y0[ne:]      # A_V   uses the last nv rows
y0_twin   = y0           # A_tne uses all
```

Note the ordering: `twin_adjacency` builds `[[None, Bᵀ], [B, None]]`, so rows are
**[edges (m); vertices (n)]**, and `TwinEmbeddingMatrix.vertex_and_edge_embeddings` splits
`X[:, ne:]` as vertices. Preserve that convention and add a test that asserts it, because it is
easy to invert and produces a plausible-looking but wrong layout.

Add `common_init: bool` to the batch runner and make it a recorded manifest field.

### 3.4 Datasets become unified objects

**This is on the critical path, not a tidy-up.** Three separate requirements all bottom out in
the same missing abstraction: the label metrics (§2.3) need labels, the distance-based metrics
(§2.0, SNS, Sil\*) need to know what edge weights *mean*, and the motivating figure for the
thesis needs to render an arbitrary layout of an arbitrary dataset. All
three are currently blocked by dataset knowledge being scattered across `dataset.py`,
`metadata.py`, `paper_figures.py` and the `mode=` parameters of two layout classes.

#### The layering: `Dataset` and `LayoutGraph`

The abstraction is **not** "a dataset is a graph plus affinities plus distances". Splitting it in
two is what makes $A_E$ and $A_{tne}$ evaluable at all:

```
Dataset                        # stored information only. Owns identity. Never laid out.
    affinity / distance        #   whichever sides are STORED, with what they mean
    label_sets                 #   per ORIGINAL vertex
    style, params, description

        │  vertex_graph(dataset, normalize=False)      ← derivation happens HERE
        ▼
LayoutGraph(A_V)               # computational layer. Laid out. Never labelled.
    affinity -> sp.sparray     #   SG-t-SNE, spectral, FR, ForceAtlas2 consume this
    distance -> nx.Graph       #   Kamada-Kawai, neato, Yifan Hu consume this
    incidence                  #   what the two derived graphs consume
    vertex_and_edge_points(X) -> (V, E, slopes)   ← back into ORIGINAL indexing
    n_points, point_provenance, parent

        │  edge_graph(vertex_graph)        twin_graph(vertex_graph, alpha, k)
        ▼                                  ▼
LayoutGraph(A_E)                           LayoutGraph(A_tne)
```

Named `LayoutGraph` rather than `Graph` only so it does not read as `nx.Graph`, which it contains.

```python
@dataclass
class Dataset:
    name: str
    graph: nx.Graph

    # --- which sides are STORED, and what the numbers mean ---
    affinity: Stored | StoredMatrix | Unweighted | None
    distance: Stored | StoredMatrix | Unweighted | None

    label_sets: dict[str, LabelSet] = field(default_factory=dict)
    style: DatasetStyle | None = None
    params: dict = field(default_factory=dict)   # generator params, for the manifest
    description: str = ""

    def labels(self, name: str) -> np.ndarray: ...
    def is_connected(self) -> bool: ...
```

`largest_component()` was in an earlier draft of this sketch and is **deliberately not built**.
Subsetting a dataset means subsetting the graph, *every* label vector, and *every* stored matrix
(tabula has two, ~50k edges each) while keeping all of them index-consistent — and every currently
registered dataset is connected, so none of that code would ever run. Same reasoning as
`disconnected="components"` (§0.5 decision 3): untested code for a case that does not occur is
worse than an honest absence. Add it when a disconnected dataset arrives, not before, and not
under time pressure at step 3 because SNS turned out to want it.

**`Dataset` holds no derived information.** It declares which sides are *stored*; the
`Dataset → LayoutGraph` mapping that builds the vertex graph is the single place a missing side
is derived. **`edge_graph` and `twin_graph` are `LayoutGraph → LayoutGraph`**, built from the
vertex graph's affinity form — which matches the algebra, since $A_E = B^{\top}B$ and
$A_{tne} = \sum_k \alpha^k \tilde A^k$ are both functions of the affinity incidence matrix. Two
consequences worth having:

- **The derivation chain is structural, not a maintained string.** Walking `parent` yields
  `Dataset → A_V → A_tne → generic distance`, which is what §3.5 requires to be recorded.
- **The incidence matrix is built once.** All three representations construct their own today;
  the derived graphs now reuse the vertex graph's.

**Why this resolves the $A_E$ / $A_{tne}$ label problem.** A derived graph's *nodes* are not the
original's vertices — $A_E$'s are edges, $A_{tne}$'s are edges and vertices — so labels cannot
ride along, and a length-$n$ label vector against a length-$m$ node set is a live bug. They do not
need to ride along: `vertex_and_edge_points` returns points in the **original** indexing, so the
dataset's labels apply after layout. Labels never touch the layout layer. Therefore:

> **A metric or a figure is a function of the embedded vertex/edge points plus a `Dataset`.**

**`Dataset` does not own the representations.** They are the *method*; it is the *data*. Keeping
them mappings keeps two open sets independent: adding a dataset touches no representation, and
adding a representation touches no dataset. Ownership would also force `Dataset` to carry
`alpha`, `k` and `normalize`, which are method knobs, not data properties. Caching an expensive
$A_{tne}$ across a λ sweep belongs in the runner.

#### Stored sources, and what they mean

An earlier draft of this section used a single `weight_kind`. That field was doing two jobs —
describing what the stored weights *mean*, and prescribing how to derive the other side. Those
coincide only when a dataset has one weight vector and must guess the other. `Dataset` keeps the
first job; the mapping does the second.

```python
Stored(weight="weight", meaning=...)   # the graph's own edge weights
StoredMatrix(key, meaning=...)         # a separate stored matrix on the same edge set
Unweighted()                           # combinatorial, both sides
```

| Dataset | affinity | distance |
|---|---|---|
| tabula | `StoredMatrix("A", meaning="adaptive_kernel")` | `StoredMatrix("D")` |
| digits (was `mnist`, see §0.5) | — | `Stored(meaning="distance")` |
| karate_weighted (`nx.karate_club_graph`) | `Stored(meaning="strength")` | — |
| karate (`.mat`), football_2023, sbm-er-3modes, ring/tree of cliques, small_world, barabasi_albert | `Unweighted()` | `Unweighted()` |

Derivation rules, applied by `vertex_graph` to whichever side is absent:

| stored | `meaning` | derives | rule |
|---|---|---|---|
| affinity | `strength` (alias `count`) | distance | $1/w$ — resistance model: weights compose in series like conductances |
| affinity | `kernel` | distance | $-\log(w/w_{\max})$ — recovers the latent metric |
| affinity | `probability` | distance | $-\log w$ — surprisal |
| affinity | `adaptive_kernel` | distance | **raises** — see below |
| distance | — | affinity | $\exp(-d/d_{\text{scale}})$, $d_{\text{scale}} = \mathrm{median}(d)$ over edges |

Both sides absent is a construction error — there is nothing to derive from. `Unweighted` on both
sides is the combinatorial case, not a derivation.

**`adaptive_kernel` is a declared refusal, not a gap.** A per-point-adaptive kernel — UMAP's
$\exp(-(d-\rho_i)/\sigma_i)$ is the case in hand — is **globally non-invertible**: $\rho_i$ and
$\sigma_i$ are per-vertex and are not stored, so no single transform of $w$ recovers $d$. A
dataset declaring `adaptive_kernel` must supply a stored distance side; if it does not, the
mapping raises rather than applying a transform that does not apply. This is what stops tabula
silently falling back to $-\log(w/w_{\max})$ if its `D` matrix is ever dropped. **[THESIS EDIT]** —
thesis §3.1.1 lists $1/w$, $1-w$ and $-\log w$ as interchangeable options; they are all *global*,
and for an adaptively-normalised affinity no global transform exists. That belongs in the prose.

**`none` is combinatorial, and the distance case matters.** Affinity is the 0/1 indicator.
Distance is 1 on existing edges and **infinite on non-edges** — which in implementation means
simply *not present*, so that the shortest-path routine produces hop counts rather than treating
absent pairs as adjacent. Do not materialise an all-ones dense matrix; pass the unweighted graph
and let `nx.shortest_path_length` do the work.

**`strength` → distance is $1/w$ because the resistance model fits**, not because it avoids
zeros. Weights that compose in series like conductances give $1/w$ a derivation, in the same way
surprisal gives $-\log$ one. See §0.5 for why `affinity` was retired as a kind and why zero
ideal distances are not, by themselves, a defect.

*Settled by inspection, not assumption:* football_2023's weights are games played, and 732 of its
735 edges weigh 1. It is effectively unweighted and is declared so. The connectivity report
(below) prints the weight distribution per dataset, which is how this was answered.

#### Both sides stored: the consistency hazard

When the two sides come from independent sources, nothing forces them to agree about which pairs
are close. The failure is silent and severe: the layout optimises against the affinity while SNS
and Sil\* score against the distance, so a method could look good or bad for reasons unconnected
to the method.

```python
def check_weight_consistency(dataset) -> WeightConsistencyReport
```

Runs whenever both sides are stored. Two statistics:

- **Global Spearman** over shared edges — expected strongly negative but *not* $-1$, since
  $\rho_i$ and $\sigma_i$ vary by row.
- **Median per-row Spearman** — the sharp test. Within a row, UMAP membership is a strictly
  decreasing function of distance, so this should be $-1$.

One caveat to handle rather than paper over: the stored `A` is the *fuzzy-union symmetrised*
membership, so `A[i,j]` mixes in row $j$'s $\rho_j,\sigma_j$ and within-row monotonicity is not
guaranteed a priori. **Measure it and pin the observed value**; if it is not exactly $-1$, record
symmetrisation as the cause rather than treating it as a data fault. A materially weaker
correlation (worse than $\approx -0.9$) is the genuine inconsistency signal. Do not hardcode $-1$
into a test that later gets "fixed" by loosening it.

#### Deriving affinity from distance

**Use the exponential $w = \exp(-d/d_{\text{scale}})$**, the exact inverse of
$d = -\log(w/w_{\max})$. Three reasons, in increasing order of importance:

1. *Round-trip consistency.* The two forms on the vertex graph become exact inverses, so the
   affinities SG-*t*-SNE is given and the ideal distances SNS and Sil\* measure against describe
   the **same** geometry. With a Gaussian forward and a log reverse, the round trip returns
   $d^2/\sigma^2$ — the layout method and the metric would be working from different geometries,
   silently.
2. *It is the UMAP membership form.* UMAP uses $\exp(-(d-\rho_i)/\sigma_i)$ — exponential in $d$,
   not $d^2$. The $\exp(-d^2)$ form is the heat-kernel/*t*-SNE shape. Both are defensible; only
   one round-trips.
3. *It composes correctly with $\lambda$, which is the decisive argument.* The reason a Gaussian
   "matches *t*-SNE better" is only half true: *t*-SNE's kernel is $\exp(-d^2/2\sigma_i^2)$ with
   a **per-point, perplexity-calibrated** $\sigma_i$, and the adaptivity is the part that
   matters. A single global $\sigma$ is not the *t*-SNE formulation, it is a cruder relative of
   it. But SG-*t*-SNE already supplies the adaptivity: the $\lambda$-rescaling finds per-row
   exponents $\gamma_i$ with $\sum_j p_{ij}^{\gamma_i} = \lambda$, which is the graph analogue
   of the per-point $\sigma_i$ calibration. And on an exponential affinity,
   $$p^{\gamma_i} = \exp(-d)^{\gamma_i} = \exp(-\gamma_i d),$$
   so $\gamma_i$ **is** a per-vertex inverse bandwidth acting directly on the original
   distances. The composition is exact and interpretable. Supplying a global $\sigma$ on top
   would be choosing a bandwidth twice — once badly, once well — and $\sigma$ would be largely
   redundant with $\gamma_i$ anyway, shifting only where the $\lambda$ search starts.

$d_{\text{scale}}$ is **required**, not optional: raw $k$-NN distances can span an order of
magnitude, and $\exp(-d)$ on unnormalised input underflows to zero for most edges. Use
$d_{\text{scale}} = \mathrm{median}(d)$ over existing edges (robust to the few long links a
$k$-NN graph produces at cluster boundaries), so the median edge lands at affinity
$e^{-1}\approx 0.37$. Record $d_{\text{scale}}$ in the manifest.

Keep $\exp(-d^2/d_{\text{scale}}^2)$ available as a selectable rule (`exp_decay_sq`) and run the
comparison **once** on digits, to confirm the choice empirically rather than leaving it as an
assumption. That is a paragraph in §5, not a research project.

This resolves the "weight transform" open decision from §9 better than picking one global
transform, and better than a single `weight_kind` would: **a stored side needs no transform at
all, and a derived side gets one determined by what the stored weights mean.** "Both stored" is
the ideal; derivation is the degraded fallback for datasets that shipped only one side. Thesis
§3.1.1 currently lists $1/w$, $1-w$, $-\log w$ as interchangeable options without committing to
one. **[THESIS EDIT]** — implement the model above rather than a single global transform. The
reporting layer should emit, per dataset, which sides are stored, the rule used for any derived
side, and the resulting derivation chain.

**Labels.** A `LabelSet` carries the vector plus how it was derived:

```python
@dataclass
class LabelSet:
    name: str
    values: np.ndarray                  # per-vertex, integer-coded
    display_names: dict[int, str]
    derivation: str                     # "ground truth" | "generator" | "derived:<expr>" | "detected:leiden"

    # --- two roles, deliberately separate: see below ---
    scoring_role: Literal["primary", "sub", "auxiliary"] = "primary"
    display_role: Literal["hue", "shade", "none"] = "hue"

    parent: str | None = None           # the label set this one subdivides
    ordered: bool = False               # e.g. West < None < East
    noise_label: int | None = None      # e.g. FBS independents -> -1
```

**Label sets are hierarchical, and the hierarchy drives both scoring and colour — but not always
in the same direction.** football_2023 is the first worked example: `conference` is the primary
scoring label *and* supplies hue, while the East/West division (from the `conference_standing`
strings) subdivides it and supplies shade. That is why the existing figures pick hue from the
conference and vary only *shade* by division.

Tabula **inverts** the arrangement: the fine-grained `cell_ontology_class` (13 classes) is what
the method claims to resolve and is therefore the scoring primary, while the coarse `compartment`
(4 classes) is its parent and supplies hue. A single `role` field cannot express both cases, which
is why scoring and display are separate fields:

```python
# football_2023
"conference": LabelSet(scoring_role="primary", display_role="hue", noise_label=-1)
"division":   LabelSet(scoring_role="sub", display_role="shade",
                       parent="conference", ordered=True)

# tabula
"compartment":         LabelSet(scoring_role="sub", display_role="hue")
"cell_ontology_class": LabelSet(scoring_role="primary", display_role="shade",
                                parent="compartment")
"donor":  LabelSet(scoring_role="auxiliary", display_role="none")   # batch covariate
"method": LabelSet(scoring_role="auxiliary", display_role="none")   # batch covariate
```

**Auxiliary label sets are scored deliberately, and a low score is the good outcome.** `donor`
and `method` are technical covariates; a layout that separates cells by donor rather than by cell
type is displaying an artifact. Reporting their silhouette alongside the primary one is how that
becomes visible instead of assumed away.

`DatasetStyle` then derives colours by one rule everywhere — hue from the `display_role="hue"`
set, shade from the ordered `display_role="shade"` set, noise colour for `noise_label` — instead
of the hand-built `3*l + 1 + s` index arithmetic currently repeated in `football_figure` and
`football_curves`.

For the **metrics**, score on the primary label by default. But also compute the label metrics
on the **composite** `(conference, division)` label, because that is a direct test of a claim
the method makes: the paper asserts that TNE reveals "the sub-conference sub-clusters." A
silhouette or accuracy score that improves on the composite label while the primary-label score
holds steady is evidence *for* that claim; a composite score that does not move says the
sub-structure is not actually resolved. Right now nothing in the evaluation would detect either.
Add `composite_of: tuple[str, ...]` to the label registry so this is declared, not improvised.
Guard it per §0.5 decision 9 — report class count and minimum/median class size alongside the
score, since a flat composite may be class-size collapse rather than absent substructure.

Three sources, all currently handled inconsistently:

- **Ground truth from file** — `football_2023["conference"]`, `digits["digit"]`, and tabula's
  `cell_ontology_class` / `compartment` / `donor` / `method`, which are present in
  `tabula-liver.mat` and currently loaded by a second ad-hoc `loadmat` inside a figure. Move all
  of it out of `metadata.py` and out of `paper_figures.py` into the dataset definition.
- **Known by construction** — ring/tree of cliques, SBM blocks, Barabási–Albert epoch. Have the
  generators *return* labels rather than reconstructing them afterwards; the clique index of a
  vertex in `ring_of_cliques(n, k)` is `i // k` and should not be re-derived by eye.
- **Derived** — the `conference_standing` → East/None/West mapping in `football_figure` is a
  string-slicing expression (`stand[-5:-1]`) buried inside a plotting function. That belongs in
  the dataset definition as a named, tested derivation.

Datasets without labels simply have an empty `label_sets`; the metric driver skips label metrics
without special-casing at the call site.

**Registry.** `DATASET_REGISTRY: dict[str, Callable[..., Dataset]]` replaces the raw graph
factories in `pipelines.batch_registries["graph"]`. Batch specs keep the same syntax —
`("football_2023", {})`, `("small_world", {"n": 150, "k": 12, "p": 0.01})` — but the runner now
receives a `Dataset`, so labels and weight semantics arrive with the graph instead of being
looked up separately.

**Extensibility is a hard requirement, not a nicety.** The dataset roster will change: more
networks will be added, chosen deliberately so that some demonstrate where TNE helps and some
where it does not. Design accordingly:

- Adding a dataset means adding **one file** under `twin/datasets/` and one registry line.
  Nothing in the metric layer, batch layer, plotting layer or reporting layer may branch on a
  dataset name — no `if name == "football"` anywhere.
- Every dataset-specific behaviour must be reachable through the `Dataset` object: stored weight
  sources and their meanings, labels and their hierarchy, style, and generator parameters.
- The registry entry is a **factory**, so parameterised generators
  (`("small_world", {"n": 150, ...})`) and file-backed datasets share one interface, as they do
  today via `resolve_object_choice`.
- **Prefer storing both sides.** A dataset that ships both an affinity and a distance needs no
  transform and no guess; one that ships a single side forces a derivation that is, at best, a
  modelling assumption. Any future point-cloud-derived graph should persist the distance matrix it
  was built from rather than transforming on the fly — tabula already does, and that is why its
  measured distances can drive a layout.
- Keep the digits rationale in the dataset definition itself as a `description` field: it is
  retained as a representative **point-cloud-derived $k$-NN network**, present to test whether
  the twin transform improves on *t*-SNE-style point-cloud embeddings. That sentence belongs in
  the thesis's dataset table, and storing it with the dataset means the table can be generated
  rather than written by hand.

**Connectivity report** (`twin/report.py`, `python -m twin.report`). Prints, for every registered
dataset: $n$, $m$, density, connectivity, component count, largest-component fraction, **mean
degree** (the λ selection criterion in §4.2 is defined at $k$ = mean degree, so it must be
computable from this table), which sides are **stored** and the rule and chain for any derived
side, the **weight distribution** (min / median / max / number of distinct values), available
label sets with both their roles, the `check_weight_consistency` result where both sides are
stored, and the **fraction of vertex pairs at $d = 0$** under the resulting distances (§0.5).

Nothing in it may branch on a dataset name; every column comes off the `Dataset` object. Run it
before anything else — §4.4's disconnected-graph policy cannot be chosen sensibly without knowing
which datasets are actually disconnected — and commit its output, which is itself a reportable
artifact and feeds the thesis's experimental-setup section.

### 3.5 What each layout method consumes

Every `LayoutGraph` exposes **both** forms, so a layout method asks for the side it needs rather
than being handed one by a runner that guessed. This is the other half of the fix for bug §1.6.

```python
def embed(self, layout_graph: LayoutGraph, y0=None) -> np.ndarray:
    A = layout_graph.affinity     # SG-t-SNE, graph t-SNE, spectral, spring (FR), ForceAtlas2
    # or
    G = layout_graph.distance     # Kamada-Kawai, neato, Yifan Hu
```

The `mode="distance"|"similarity"` parameters are deleted from `KamadaKawaiLayout` and
`NeatoLayout`. That decision never belonged to the layout class, and it is not a runner decision
either — it is a property of the graph being laid out.

```python
@dataclass
class EmbeddingSpec:
    cls: type
    expects: Literal["affinity", "distance"]   # declared metadata for the manifest and report
    stochastic: bool
    has_lambda: bool
```

`expects` is retained so the report can state what each method consumes and so the driver can
size a run, but **the runner no longer branches on it** — nothing converts at the boundary.

Because nothing branches on it, nothing keeps it honest either: it can drift from what a method
actually reads and no test would notice. **Assert it.** Pass each method a `LayoutGraph` that
records which of `.affinity` / `.distance` was touched, embed a small graph, and check the
accessed set matches the declaration. One parametrised test over the embedding registry covers
every method, costs almost nothing, and fails the moment a method starts reading the other side.

**Derived graphs are built from affinities, or they are nonsense.** $A_E = B^{\top}B$ and
$A_{tne} = \sum_k \alpha^k \tilde A^k$ are affinity algebra: larger entry means stronger coupling.
The polynomial accumulates multi-step coupling strength, so feeding it distances inverts the
meaning of every term and makes the matrix uninterpretable. `edge_graph` and `twin_graph`
therefore take the **vertex graph's affinity form**, always.

**Which distance form each graph exposes:**

| Graph | `affinity` | `distance` |
|---|---|---|
| `vertex_graph` | $A_V$ | the dataset's distance — stored, or derived once by §3.4's rules |
| `edge_graph` | $A_E$ | generic fallback from $A_E$ |
| `twin_graph` | $A_{tne}$ | generic fallback from $A_{tne}$ |

$A_V$ is a **faithful** view: $B_w^{1/2}(B_w^{1/2})^{\top}$ reproduces $w_{ij}$ on edges, so the
classical vertex layout can use the dataset's own distances — tabula's *measured* `D` drives a
Kamada–Kawai vertex layout, which is the point of storing it. `normalize=True` perturbs $A_V$ and
breaks faithfulness; declare that on the graph rather than inferring it.

$A_E$ and $A_{tne}$ are not in the dataset's units — neither has entries that are "strengths" or
"distances" in the original sense — so a distance-consuming method on them uses a **generic
affinity→distance fallback**: normalise, symmetrise, then $-\log$. `KamadaKawaiLayout(mode=
"similarity")` implements the un-symmetrised version today; lift it out as
`affinity_matrix_to_distance()` and **symmetrise** it. Row normalisation alone leaves the matrix
asymmetric, and `nx.from_scipy_sparse_array` then silently keeps one value of each pair — a live
bug in the current code (§0.5 decision 7).

**The chain is structural, and must be recorded.** In the worst case — a distance-stored dataset
laid out by a distance-consuming method on the twin graph — it runs
$d \rightarrow \text{affinity} \rightarrow A_{tne} \rightarrow \text{derived } d$, three
derivations deep. That is the nature of the method, not a defect, but a reader needs to see it.
Because the graphs form a `parent` chain back to the `Dataset`, the manifest can record the real
chain rather than a `distance_source="generic"` flag someone has to keep in sync.

### 3.6 The plotting refactor

`paper_figures.py` is 1018 lines containing six figure functions that are largely the same
program. `football_figure` and `football_curves` differ only in whether edges are drawn as
segments or curves — and each independently re-runs the embedding, re-derives the conference and
standing labels, and rebuilds the same husl palette with the same brightness adjustments. The
same duplication exists for `tree_of_cliques`/`tree_of_cliques_curves` and
`small_world`/`small_world_curves`. Every one of them hardcodes `graph_embedding_all` with
`SGtSNELayout`, so no figure in the file can show any other layout method.

Decompose along the three axes that are actually varying:

```
Dataset  ×  LayoutResult  ×  Renderer
```

**Renderers.** One function per visual mode, all taking `(ax, dataset, layout, style, **kw)`:

```python
render_vertices(...)        # scatter of V, coloured by a label set
render_edges_straight(...)  # segments between vertex points
render_edges_curved(...)    # curves through edge points, shared geometry with §2.4 Layer 1
render_edge_points(...)     # scatter of E, with optional orientation bars (existing drawsegments)
render_block_adjacency(...) # existing block_adj, generalised off tabula
```

The `*_curves` variants of the figure functions then disappear — they become
`renderer="curved"`.

**Styling.** `DatasetStyle` holds what the figure functions currently rebuild by hand: the
palette for each label set, the brightness variants used for ordered sub-labels (the
East/None/West darkening in `football_figure`), the noise colour (black for FBS independents),
intra- vs inter-cluster edge alpha, marker sizes, and the per-method frame colours. Derive edge
colours from vertex labels through one shared rule — currently that logic is re-implemented in
each figure function with slightly different conditions.

**Generic figure driver.**

```python
def layout_panel(ax, dataset, layout, *, renderers, label_set=None, style=None): ...
def layout_grid(dataset, layouts: dict[str, LayoutResult], *, renderers, ...) -> Figure: ...
```

`layout_grid` reproduces every existing paper figure as a call with different arguments, and —
the point — it renders **any** layout, not just the vertex/edge/twin SG-t-SNE triple. That is
what makes the planned motivating figure — three panels demonstrating artifacts (a), (b), (c)
on baseline layouts — a configuration rather than a seventh bespoke function.

Keep the existing paper figures reproducible: retain thin wrappers with the same names and
signatures that call the new driver, and add a visual-regression check (render at a fixed seed,
compare against committed PNGs) so the refactor can be verified rather than hoped about.

### 3.7 Metric evaluation entry point

```python
def evaluate_layout(result: LayoutResult,
                    dataset: Dataset,
                    metric_spec: dict) -> dict[str, float | np.ndarray]
```

It takes the `Dataset`, not a loose label vector: a metric or a figure is a function of the
embedded vertex/edge points **plus a dataset** (§3.4), and the driver unpacks whichever label set
each metric asks for — primary, composite, or an auxiliary batch covariate. Per-metric signatures
in §2 are unchanged.

Driven by `METRIC_REGISTRY`. Skips metrics whose `needs` are unmet (no labels, no edge points),
records a `skipped` status rather than a NaN (§0.5), and respects cost-class limits.

Then add a batch-level counterpart alongside `batch_graph_embedding`:

```python
def batch_layout_metrics(batch_input, *, metric_spec, seeds, ...) -> pd.DataFrame
```

which runs the embedding batch and evaluates metrics per case per seed, returning a **tidy long
DataFrame**, one row per `(graph, representation, embedding, params…, seed, metric, variant,
k, value)`. Long format, not wide — the sweeps over `k`, over seeds, over λ and over
straight/curved make any wide schema collapse.

`twin/batching.py::results_to_records` and `key_to_flat_record` already produce the key columns.
Use them; do not hand-roll the key flattening.

---

## 4. Experiment protocol

This is the part that gates Section 5 of the thesis. Each item below is a protocol decision that
is cheap now and expensive after the runs.

### 4.1 Seeds and dispersion

- `seed` becomes a first-class batch dimension, swept independently of the spec axes.
- Default 5 seeds; configurable.
- Aggregate to **median and IQR**, not mean and standard deviation. Crossing counts are
  heavy-tailed and a standard deviation will misrepresent them. Report the mean too if wanted,
  but IQR is the headline dispersion.
- **Deterministic methods must be flagged** in the embedding registry (`stochastic: bool`).
  Spectral, Kamada–Kawai, Neato and Yifan Hu are deterministic given the input; running them
  5× produces 5 identical rows, and their zero dispersion must be presented as "deterministic",
  not as a variance win. Run them once and mark them.

### 4.2 The two λ protocols

Implement both; they answer different questions and one protocol cannot serve both.

**Protocol A — paired.** Same λ, same seed, across `A_V`, `A_E`, `A_tne`. Isolates the effect
of the transform. This is the ablation, and it is *not* a claim that TNE beats a tuned
SG-t-SNE. Tag rows `protocol="paired"`.

**Protocol B — per-method best.** Sweep λ independently for each representation, select per
method, compare selected configurations. This is the fair comparison. Tag `protocol="tuned"`.

```python
def sweep_lambda(batch_input, lambdas, *, seeds=(0,), ...) -> pd.DataFrame
def select_lambda(sweep_df, criterion, *, higher_is_better=True) -> dict[key, float]
```

Selection rules, to be enforced in code:

- **One declared criterion, fixed before the runs.** Default `criterion="knn_recall@k=mean_degree"`.
  Deliberately a metric TNE is not expected to win on — if crowding and crossings still improve
  at a λ chosen for neighbourhood recall, that is a far stronger result.
- **Per-metric λ selection must be impossible to express.** `select_lambda` returns one λ per
  `(graph, representation, embedding)` key and nothing finer. If each metric could pick its own
  λ, the reported row would correspond to no drawing that ever existed.
- Do not offer KL divergence as a criterion. $P^{(\lambda)}$ itself changes with λ, so
  $D_{KL}(P^{(\lambda)}\|Q)$ is a different objective at each setting, not a better score.
  Reject it explicitly with a clear error if someone passes it.

**Grid (decided):** $\lambda = 10^{k}$ for $k \in \{-1, -0.5, 0, 0.5, 1\}$, i.e.
$\{0.1,\ 0.316,\ 1,\ 3.16,\ 10\}$. Log-spaced because the effect of λ is observed to scale
logarithmically; the range brackets the 1–10 band in normal use with one point below it on
either side, so a selection at an interior point is evidence the grid is wide enough.

**Boundary check — implement this, do not leave it to inspection.** If the selected λ for any
`(graph, representation, embedding)` key lands on 0.1 or 10, emit a warning and record
`lambda_at_boundary=True` in the manifest. A boundary selection means the grid is too narrow and
the sweep should be extended before the numbers are used.

One seed during the sweep, full seeds only at the selected λ.

**Sweep artifact.** Persist the full sweep, not just the selection. The plot of metric vs λ with
one curve per representation is the single most persuasive figure available for §5, and it comes
free once the sweep is stored.

**Large graphs.** If the sweep is skipped for a graph, the carried-over λ and its provenance
must be recorded in the manifest (`lambda_source: "swept" | "carried:<graph>" | "default"`).
No silent defaults.

### 4.3 Straight and curved crossing variants

Every crossing metric is computed **twice** for representations that produce edge points:

- `variant="straight"` — straight segments between vertex points (comparable with all baselines)
- `variant="curved"` — quadratic curves through edge points (`gamma` configurable, default 0.5)

Baselines without edge points produce only `straight`. `variant` is a column in the long
DataFrame, never a separate table. Together with `detour_ratio` (§2.2) this is what separates
"the transform reduces crossings" from "the drawing style reduces crossings".

**γ is fixed, never optimised.** Default `gamma=0.5` — the classic quadratic Bézier, and what
the existing figures use, so measurement and picture agree.

Do **not** implement a γ optimiser that minimises crossings. Three reasons, and they compound:
it would make the straight-vs-curved comparison meaningless, since "straight vs best-of-curved"
is not the contrast the experiment is for; the reported number would correspond to a drawing
chosen *after* seeing its score; and both LWCr and the curved formulation are the author's own,
so a fitted γ turns a legitimate contribution into exactly the metric-gaming a reviewer will
look for. The λ discipline in §4.2 applies here too — sweep, report the sweep, but select on
something declared in advance, and γ has no such criterion.

The legitimate version is a **γ sweep as a diagnostic**: report crossings and detour against
$\gamma \in \{0, 0.25, 0.5, 0.75, 1\}$ for one or two representative graphs, with $\gamma=0.5$
fixed everywhere in the headline results. That shows how the rendering parameter behaves without
letting it be tuned.

One observation worth acting on. The curve kernel in `draw_curved_edges` is documented as:
$\gamma=0$ straight, $\gamma=0.5$ classic Bézier, and **$\gamma=1$ the interpolating curve with
$P(1/2)=P_1$** — that is, at $\gamma=1$ the drawn curve actually *passes through* the embedded
edge point. Since the thesis's claim is precisely that TNE embeds edges as first-class points
with real positions, $\gamma=1$ is the only setting at which the drawing is faithful to that
claim, and it is therefore a principled choice rather than an aesthetic one. $\gamma=0.5$ is the
better default (smoother, lower detour, matches existing figures), but include $\gamma=1$ in the
sweep and produce it at least once as a figure: it is the rendering in which each curve passes
through its edge's embedded location.

### 4.4 Disconnected-graph policy

Applies **only** to metrics that require finite graph distances $d_{ij}$ — SNS, and the
graph-side silhouette Sil\*. ADR, the kNN family, NND, NR/VNR, Cr, LWCr and detour need no
graph distances and must run on the **full graph** regardless. Do not restrict the whole
experiment to a subgraph because two metrics need one.

Implement all three policies; record which was used per metric per dataset, along with the
fraction of vertices and pairs excluded.

| Policy | Behaviour | Trade-off |
|---|---|---|
| `"lcc"` **(default)** | compute on the largest connected component only | Unambiguous, field-standard, needs no defence, all distances finite, no aggregation choice. Discards data — so the discarded fraction must be reported. The repo already has the precedent: `polblogs-lwcc.mat` is a largest-component extract. |
| `"mask"` | drop infinite pairs from the sums | Keeps every vertex in one coordinate system and measures the whole layout. But the denominator varies with component structure, so values are not comparable across graphs — and for silhouette it fails outright: if the nearest *other* cluster lies in another component then $b(i)=\infty$ and $s(i)\to 1$, so a disconnected layout gets an **inflated** score. Acceptable for SNS, dangerous for Sil\*. |
| `"components"` | compute per component, aggregate weighted by size | All distances finite, no inflation, matches the thesis's stated restriction. But it changes what is measured: a label class spanning two components is effectively split, cross-component label structure becomes invisible, size-weighted aggregation is an arbitrary choice, and components of size 1–2 give degenerate silhouettes. Most importantly it ignores where the layout places components *relative to each other*, which is real layout quality. |

Recommendation: `"lcc"` as the declared default, `"mask"` allowed for SNS only, `"components"`
available but not used in the headline results. Guard it in code — `silhouette_graph` should
refuse `"mask"` with an explanatory error rather than returning an inflated number.

Run the connectivity report from §3.4 first. If it turns out all the datasets in the final
selection are connected, this decision costs nothing; record that fact in the manifest and move
on.

### 4.5 Registry additions

Extend the embedding registry entries with metadata the protocol needs:

```python
"sgtsne": EmbeddingSpec(cls=emb.SGtSNELayout, stochastic=True,  has_lambda=True),
"spectral": EmbeddingSpec(cls=emb.SpectralLayout, stochastic=False, has_lambda=False),
...
```

Also add the new baseline: **graph t-SNE** (Böhm et al.), which is ~10 lines on `openTSNE` —
row-normalise $A$ so each row sums to 1, symmetrise $P=(\tilde A + \tilde A^{\top})/2n$, pass to
`openTSNE.affinity.PrecomputedAffinities(..., normalize=True)`, Laplacian Eigenmaps init,
learning rate $=n$. It is the closest current competitor to TNE's premise and its absence would
be the first thing asked about. Add `openTSNE` to `pyproject.toml`.

### 4.6 Persistence and manifests

Currently nothing is saved. Required:

```
runs/<run_id>/
    manifest.json      # git SHA, package versions, spec, seeds, λ selection + source,
                       # common_init (requested AND whether it took effect), disconnected
                       # policy, per-dataset stored sources + derivation rule + d_scale,
                       # per-case derivation chain, weight-consistency results, gamma,
                       # include_adjacent, VNR normalisation, host, timestamp
    metrics.parquet    # tidy long-format metric rows
    layouts/<key>.npz  # V, E, slopes per case (compressed)
    sweeps/lambda.parquet
    distributions/<key>.npz   # eqs. (70), (71) arrays — large, kept separate
    log.txt
```

- `run_id` = timestamp + short git SHA.
- Cases must be **resumable**: check for an existing output before running a case, so an
  interrupted overnight batch continues rather than restarting.
- Layouts are persisted because recomputing them to add one metric later is the expensive
  mistake this whole structure exists to avoid.
- ⚠️ **`layouts/<key>.npz` stores a dataset *reference*, never the `Dataset` object.**
  `LayoutResult` holds a `Dataset` (§3.1), which holds the graph, every label vector, the style,
  and — for tabula — two stored matrices of ~50k edges each. Serialising it would write all of
  that into every layout file, once per representation, per seed, per λ. Persist the registry
  `name` plus the factory `params` and rehydrate through `DATASET_REGISTRY` on load; the batch key
  already carries exactly that pair, so nothing new has to be recorded. The `npz` payload is
  `V`, `E`, `slopes` and the scalar instrumentation, and nothing else.

### 4.7 Dataset selection, including an adversarial case

The batch spec should be organised into named experiment groups, and one of them must be a case
where TNE is **expected to underperform** — the natural candidate is a denser graph, where $m$
grows and the $n+m$ inflation should hurt both runtime and quality. Add a density-parameterised
generator (e.g. `sbm` or `erdos_renyi` with a density sweep) so this is a deliberate, declared
part of the design rather than something discovered later.

---

## 5. Reporting

### 5.1 Aggregation

```python
def aggregate(df, by=("graph","representation","embedding","metric","variant","k"))
    -> pd.DataFrame   # median, IQR, n_seeds, n_excluded, deterministic flag
```

### 5.2 Typst export

`pipelines.py::metrics_to_typ` currently hardcodes the vertex/edge/twin triple and the exact
table shapes. Rewrite it against the long DataFrame: given a tidy frame plus a column/row spec,
emit a Typst table. Keep the existing `hlmin`/`hlmax` highlighting helpers — they are useful and
already match the thesis styling.

### 5.3 Schema-level guards

Three constraints are easiest to enforce in the exporter rather than to remember later:

1. **ADR must not be aggregated across graphs.** The exporter should refuse to produce a
   cross-graph mean for any metric whose `MetricSpec` is marked `within_graph_only=True`
   (ADR, Cr, LWCr).
2. **Graph-side scores are label-validity checks, not baselines.** `silhouette_graph` and
   `graph_neighborhood_accuracy` must be emitted in a *separate* table from their layout
   counterparts. Column adjacency is itself an invitation to difference two incomparable
   quantities.
3. **No silent caps.** If the exporter drops rows — top-N, skipped metrics, unswept λ — it must
   emit a footnote saying what was dropped and why.

---

## 6. Tests

Add `tests/` (nothing exists). At minimum:

**Analytic cases** — small graphs where the answer is known by hand:
- $K_4$, a path, a star, a single triangle: check ADR, NND, VNR, Cr against values computed
  manually in the test file.
- Two straight segments constructed to cross exactly once → `Cr == 1`.
- Two parallel segments → `Cr == 0`. Two collinear overlapping segments → `Cr == 1` (degeneracy
  convention).
- Two edges sharing a vertex, straight → no crossing; the same two edges with edge points placed
  so the curves bow across each other → `Cr == 1` with `include_adjacent=True`, `0` with
  `include_adjacent=False`.

**Invariance properties** — these are claims the thesis makes and should be tested, not trusted:
- SNS invariant under $X \mapsto \beta X$ and $d \mapsto \beta d$.
- NND, VNR, ADR, LWCr invariant under uniform scaling, rotation, reflection and translation.
- `weighted_recall` on an unweighted graph equals `knn_recall` (eq. 39).
- `edge_curves(gamma=0)` reproduces straight segments to floating-point tolerance.
- `detour_ratio >= 1` always, `== 1` for straight.

**Degenerate inputs:**
- zero-degree vertices → excluded from recall, counted in the exclusion report;
- coincident points → `knn_accuracy(weighting="distance")` does not divide by zero;
- disconnected graph → both `disconnected` policies run and give documented, different answers;
- singleton label class → silhouette contributes 0.

**Regression:** a golden-file test on `karate_club` — full metric suite, fixed seed, values
committed. This is what catches an "improvement" to a metric silently changing §5's numbers
after they are written into the thesis.

---

## 7. Suggested order

1. **Correctness and conventions groundwork** — §1.1–§1.3 and the new §1.8 fd-2 fix, plus the
   `(n, d)` convention and `LayoutResult`. See §0.5: only §1.6 corrupts current output, and it
   is scheduled in step 2. Everything else is built on the array convention.
2. **`Dataset` and `LayoutGraph` (§3.4, §3.5)** — the stored-source declarations and the
   `Dataset → LayoutGraph → LayoutGraph` mappings, which is where bug §1.6 closes; then labels
   (including moving them out of `metadata.py` and out of `paper_figures.py`); then the registry
   swap in `pipelines.batch_registries`. Run the connectivity report and commit its output; it
   settles §4.4 and feeds the thesis's experimental-setup section. Note that `batches/batch.py`
   depends on three things this step changes — the `karate_club` and `mnist` registry names and
   the `mode=` sweep — and will otherwise fail only at run time.
3. `common.py` + `structural.py` + tests — ADR, kNN family, SNS
4. `visual.py` + tests — NND, NR/VNR, distributions, edge lengths
5. `crossings.py` + tests — the curve/detection/metric layers, and `detour_ratio`
6. `label.py` + tests — silhouette, accuracy, graph-side checks (needs step 2)
7. Registry, `evaluate_layout`, `batch_layout_metrics`, instrumentation, persistence
8. λ sweep and selection; seeds; deterministic flags; straight/curved variants
9. Bugs §1.4, §1.5, §1.7 (embedding wrappers, `fa2` → `fa2_modified`), graph *t*-SNE baseline
10. Plotting refactor (§3.6) — renderers, `DatasetStyle`, `layout_grid`, wrappers for the
    existing paper figures, visual-regression check
11. Reporting: aggregation, Typst export, schema guards
12. Dataset groups including the adversarial density case

**Step 2 moved up deliberately.** It was originally step 8-ish as a tidy-up; it is in fact a
prerequisite for the label metrics (step 6), for the ideal-distance model used by SNS and Sil\*
(steps 3 and 6), and for the fix to the weight-sign bug. Doing it late means doing steps 3 and 6
twice.

Steps 1–6 are independently testable and are the real bottleneck. Steps 7–8 are what make the
results defensible. **Do not start producing Section 5 numbers before step 8 is done** — the
entire point of this ordering is that re-running is the expensive part.

Step 10 can slip without blocking the numbers, but it should land before the thesis's motivating
figure is made, or that figure becomes a seventh hardcoded function in `paper_figures.py`.

---

## 8. Settled decisions — implement as stated

Each of these was an open question and has been decided. Implement as written; do not re-open
them in code comments or propose alternatives in the implementation.

| Decision | Resolution | Where |
|---|---|---|
| VNR normalisation | `mean` (divide by $\bar d$). Keep `max` available for one comparison. **Thesis eq. (69) needs updating.** | §2.2 |
| Weight transform for ideal distances | Not a global choice, and not a single `weight_kind` either. Each dataset declares which sides are **stored**; only a *derived* side takes a transform, chosen by what the stored weights mean. **Thesis §3.1.1 needs updating.** | §3.4 |
| Disconnected-graph policy | `"lcc"` default, `"mask"` for SNS only, `"components"` available but not headline. Applies only to SNS and Sil\*. | §4.4 |
| γ for curved edges | Fixed at 0.5 for all headline results. **No optimiser.** Sweep $\{0, 0.25, 0.5, 0.75, 1\}$ as a diagnostic on one or two graphs; show $\gamma=1$ (the interpolating case) at least once. | §4.3 |
| λ grid | $10^k$, $k \in \{-1,-0.5,0,0.5,1\}$. Warn and flag on boundary selection. | §4.2 |
| λ selection criterion | One declared metric, fixed before the runs; default `knn_recall@k=mean_degree`. Per-metric selection must be structurally impossible. KL rejected with an error. | §4.2 |
| Crossing conventions | Adjacent pairs **included** (thesis convention); collinear overlap counts as 1; endpoint exclusion by vertex identity, not coordinate. **Thesis §3.1.3 needs the degeneracy convention added.** | §2.4 |
| LWCr edge length | Chord, never arc. Load-bearing — see the reasoning in §2.4. | §2.4 |
| Dispersion statistic | Median and IQR. Deterministic methods flagged, run once. | §4.1 |

| Label roles | Scoring role and display role are **separate fields**. football scores `conference` and takes hue from it; tabula scores the fine `cell_ontology_class` and takes hue from the coarse `compartment`. One field cannot express both. Also score the composite, guarded per §0.5 decision 9. | §3.4 |
| Auxiliary labels | `donor` and `method` are batch covariates. Score them deliberately — a **low** silhouette is the good outcome, since separating by donor rather than cell type is an artifact. | §3.4 |
| Derived side: distance → affinity | $\exp(-d/d_{\text{scale}})$, $d_{\text{scale}}=\mathrm{median}(d)$ over edges. Exact inverse of $-\log(w/w_{\max})$, and $\gamma_i$ from λ-rescaling then acts as a per-vertex bandwidth. | §3.4 |
| Derived side: `strength` → distance | $1/w$, the resistance model. The log form gives zero ideal distance to maximum-weight edges. | §3.4 |
| Derived side: `adaptive_kernel` | **Raises.** A per-point-adaptive kernel is globally non-invertible ($\rho_i$, $\sigma_i$ unstored), so such a dataset must supply a stored distance side. | §3.4 |
| `Unweighted` | Combinatorial: affinity = 0/1 indicator; distance = 1 on edges, absent otherwise, so shortest paths are hop counts. | §3.4 |
| Both sides stored | No transform is applied in either direction — but run `check_weight_consistency`, since independent sources can disagree silently. | §3.4 |
| digits | Retained, as the representative point-cloud-derived $k$-NN network — present to test whether the transform improves on *t*-SNE-style point-cloud embeddings. | §3.4 |
| Representation input | $A_E$ and $A_{tne}$ are built from the vertex graph's **affinity** form, always. Every `LayoutGraph` exposes both forms and the layout method asks for the side it needs; nothing converts at the runner boundary. | §3.5 |

### Deliberately unresolved — do not pick a value, make it configurable

These are not oversights. Build so that each can be changed without touching more than one file,
and surface the current value in the run manifest.

- **The dataset roster.** Being revised to add networks that show where the method helps *and*
  where it does not. The code must not assume the current list — see the extensibility
  requirements in §3.4. Adding or removing a dataset is a one-file change.
- **$\exp(-d)$ vs. $\exp(-d^2)$ when deriving affinity from distance.** §3.4 argues for the former
  and it is the default; `exp_decay_sq` is selectable, and the comparison is run once on digits so
  the choice rests on a measurement.
- **The median per-row Spearman that `check_weight_consistency` should expect on tabula.** The
  argument says $-1$; fuzzy-union symmetrisation may perturb it. Measure it, pin the observed
  value, and record the reason — do not hardcode $-1$ and then loosen the test.

---

## Appendix A — `CLAUDE.md` (done; that file is now authoritative)

`CLAUDE.md` was created at the repo root from this appendix and is **the** copy. It has since
diverged deliberately, and keeping a second copy here would only drift:

- the array-orientation convention now names the one remaining legacy `(d, n)` surface and the
  step at which it converges (tracked in `docs/DECISIONS.md`);
- the representation convention is stated in `Dataset` / `LayoutGraph` terms per the §3.4/§3.5
  rewrite;
- it records the one-time `git config filter.jupyter.clean` step a fresh clone needs, without
  which no notebook can be committed cleanly;
- the package layout distinguishes what exists from what is planned.

**Read `CLAUDE.md`, then `docs/STATUS.md`** for where the work currently stands. The version
control policy — `master` is off limits, commit granularity, run reproducibility, what is and is
not committed — lives there in full.
