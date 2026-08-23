# Dataset provenance

Where each dataset's graph and edge weights actually come from. This exists because a dataset's
stored sources and their `meaning` (spec §3.4) cannot be declared without knowing how a weight was
produced — the same number means different things depending on the model that generated it, and
the wrong declaration silently picks the wrong derivation for the other side.

Claims here are checked by `tests/test_dataset_provenance.py`, which skips when the data files
are absent.

---

## tabula-liver

`data/networks/real-world/tabula-liver.mat`, exported from an AnnData object
(MATLAB 5.0, created 2024-07-06). Tabula Sapiens liver, **n = 5007 cells**.

### Contents

| Key | What it is |
|---|---|
| `A` | `obsp['connectivities']` — UMAP fuzzy simplicial set. Symmetric, 98736 nnz = **49368 undirected edges**, values in (0, 1] |
| `D` | `obsp['distances']` — **directed** kNN distances, 70098 entries, exactly **14 per row**, range 0.201–6.370 |
| `X_scvi` | 50-d scVI latent representation — **the space the graph was built in** |
| `X_pca` | 50-d PCA representation — *not* the space the graph was built in |
| `X_umap`, `X_scvi_umap` | 2-d reference embeddings |
| `cell_ontology_class` | 13 cell types: hepatocyte 1558, macrophage 1381, monocyte 612, … |
| `compartment` | 4: immune 2643, epithelial 1600, endothelial 688, stromal |
| `donor` | 2: TSP6 2860, TSP14 2147 |
| `method` | 2: 10X 4794, smartseq2 213 |
| `gender`, `anatomical_information` | constant (`male`, `nan`) — carry no information, do not register |

### The construction chain

```
Tabula Sapiens liver (5007 cells)
  -> scVI latent space, 50-d, float32              X_scvi
  -> exact 14-NN, Euclidean, in X_scvi             D  (directed, 70098 entries)
  -> UMAP fuzzy simplicial set + fuzzy-union symmetrisation
                                                   A  (symmetric, 49368 edges)
```

Verified:

- **`D` is Euclidean distance in `X_scvi`.** Max absolute error 3.1e-07 over 300 sampled rows
  (float32 storage precision). `X_pca` gives a median *relative* error of 8.2 — it is a different
  representation and is not what the graph was built in.
- **`D`'s neighbours are exactly the 14-NN in `X_scvi`** for 300/300 sampled rows.
- **`A` and `D` span the same edge set.** 70098 directed entries − 20730 reciprocal pairs =
  **49368**, exactly `A`'s undirected edge count, and the symmetrised supports are identical
  (0 edges in either that are not in the other).

### Why the weights pile up at 1.0

7.05% of `A`'s entries are exactly 1.0, touching **4660 of 5007 vertices (93.1%)**. This is
**neither saturation nor a clip**:

- no value exceeds 1, the minimum is 2.6e-08, and the distribution below 1 is continuous;
- for 4515 of 5007 rows, `A[i, argmin_j D[i,j]]` is **exactly** 1.0.

That is UMAP's local-connectivity term. Membership is
$\mu_{ij} = \exp(-(d_{ij} - \rho_i)/\sigma_i)$ with $\rho_i$ the distance to $i$'s nearest
neighbour, so $\mu = \exp(0) = 1$ there **by construction**, and fuzzy union preserves it. The
pile-up is structural, and no information was destroyed upstream.

### The consequence: `A` is not invertible to distances at all

$\rho_i$ and $\sigma_i$ are per-point and are **not stored**. No global transform of `A`
recovers $d$, because the map from distance to membership is a different function for every
vertex. This is a property of *any* per-point-adaptive kernel, not a defect of this export —
worth a line in thesis §3.1.1, where $1/w$, $1-w$ and $-\log w$ are currently listed as
interchangeable options: they are all global, and a globally-invertible transform does not exist
for an adaptively-normalised affinity.

Applying the `kernel` rule $d = -\log(w/w_{\max})$ anyway would assign ideal distance exactly 0
to one edge per vertex, requiring 93.1% of vertices to coincide.

### Consequence for `Dataset`

**Do not transform `A` in either direction.** Both sides are stored, independently:

```python
affinity = StoredMatrix("A", meaning="adaptive_kernel")
distance = StoredMatrix("D")
```

| Direction | Source | Note |
|---|---|---|
| affinity | `A` | UMAP membership. Per-point adaptive, which is the property we want, and composes correctly with SG-*t*-SNE's λ-rescaling. This direction was never broken for tabula. |
| distance | `D` | **Measured**, in `X_scvi`. Not a transform of `A`. |

`meaning="adaptive_kernel"` has no derivation rule — it **raises**. That is deliberate: if `D` were
ever dropped, tabula must fail loudly rather than fall back to `-log(w/w_max)`, which the
non-invertibility above shows does not apply.

Provenance to record in the manifest: `distance_space="X_scvi"`, `distance_metric="euclidean"`,
`knn_k=14`, `affinity="umap_connectivities"`.

**Sparsity** is declared as `KNearest(k=14, space="X_scvi")` — shipped sparse, so nothing is
computed and the edge set must not be touched — and checked: minimum degree is exactly 14, and
m = 49368 sits inside the implied [35049, 70098]. `X_scvi` is retained as `source_points`, which
makes the cost of that sparsification measurable: rank ρ 0.7876 against the ambient metric,
relative stress 1.489, or **0.2102** after the optimal uniform rescaling. The stored edge weights
agree with ambient distances in `X_scvi` to 3.5e-07, which is the precondition for any of that
meaning anything.

**Both sides stored is the ideal; derivation is the degraded fallback.** Most datasets ship one
weight vector and a guess for the other side, and that guess is a modelling assumption however
well argued. Tabula shipped both, which is why its measured distances can drive a Kamada–Kawai
layout while its adaptive affinities drive SG-*t*-SNE — the same geometry, from the same source,
in both directions. Any future point-cloud-derived graph should persist the distance matrix it was
built from rather than transforming on the fly.

The hazard that comes with it: two independent sources can disagree about which pairs are close,
silently. `check_weight_consistency` is the guard — see `DECISIONS.md`.

**Measured, and they do not disagree.** Over the 49368 shared pairs: global Spearman −0.2364,
median per-row −0.7820, and — restricted to the pairs `D` recorded in one direction only, where
fuzzy union cannot have mixed row $j$'s $\rho_j,\sigma_j$ into row $i$ — **exactly −1.0 in all
3819 scored rows**. So `A[i,j]` is a strictly decreasing function of `D[i,j]` wherever it is
$\mu_{ij}$ unmixed, and the weaker aggregate figures are symmetrisation and per-row $\sigma_i$
variation, not a fault. Pinned in `tests/test_datasets_registry.py`.

### Also in the file, and currently reached ad hoc

- **`X_umap`** — a 2-d reference embedding, used as a layout by `tabula_sapiens_teaser` (it is
  passed through `vertex_and_edge_embeddings` to derive edge points for it). A `Dataset` should
  expose it as a named reference layout rather than have a figure re-open the `.mat`.
- **The Leiden partition** that figure colours by is computed inside the figure via `leidenalg`
  with `resolution_parameter=1.0, seed=10`. That is a `derivation="detected:leiden"` label set and
  belongs in the dataset definition, not in a plotting function.

### Labels

Scoring role and display role are different things, and `LabelSet` must carry both.

| Label set | Scoring role | Display role | Note |
|---|---|---|---|
| `cell_ontology_class` (13) | **primary** | shade | What the method claims to resolve, and what the figures colour by |
| `compartment` (4) | super-label | **hue** | The coarser grouping; hue from compartment, shade from cell type |
| `donor` (2) | auxiliary | — | Batch covariate — **score it deliberately**: separation by donor is an artifact, so a *low* silhouette is the good outcome |
| `method` (2) | auxiliary | — | As `donor`. 10X vs smartseq2 is a technical covariate |
| `gender`, `anatomical_information` | — | — | Constant; do not register |

`compartment` is the parent of `cell_ontology_class` in the hierarchy, the same shape as
football's conference/division — but note the roles are inverted relative to football, where the
coarse label is also the scoring primary. That is why the two roles have to be separate fields
rather than one `role`.

---

## football_2023

Weights are games played: **732 of 735 edges weigh 1, three weigh 2**. Effectively unweighted —
`Unweighted()` on both sides (spec §0.5). The `strength` path is irrelevant for it.

## karate — two different graphs

| Registry name | Source | Weights | Sources |
|---|---|---|---|
| `karate` | `karate.mat` | all 1.0 | `Unweighted()` / `Unweighted()` |
| `karate_weighted` | `nx.karate_club_graph` | Zachary interaction strengths 1–7, 7 distinct | affinity `Stored(meaning="strength")`; distance derived, `1/w` |

The registry previously exposed only the *weighted* one under the name `karate_club` while the
spec's table described the unweighted one. `karate_weighted` is the genuinely strength-weighted
small test case that football turned out not to be.

## digits (formerly `mnist`)

`sklearn.datasets.load_digits()` — 1797 samples of **8×8** digits, not MNIST. At `n_neighbors=12`
(the registry default): n = 1797, m = 14732. Weights are Euclidean distances 5.3–37.8, so the
**distance** side is the stored one — `Stored(meaning="distance")` — and the affinity form is
derived, `exp(-d/d_scale)` with `d_scale = median(d) = 21.8403`. Retained as the representative
point-cloud-derived kNN network (spec §8).

This is the dataset bug §1.6 was about. Before the restructure, its stored distances were handed
to the representations as if they were affinities; the Spearman correlation between the old and
the new edge weights is **exactly −1** over all 14732 edges, i.e. every edge was ranked in
precisely the reverse order. The *layout* consequence is smaller than that sounds — see the
commit that closed it.

Note the contrast with tabula, which is the same kind of object built better: both were derived
from a point cloud, but tabula persisted the distance matrix *and* the affinities, while digits
persists only distances and must derive the rest.

**Sparsity** is declared, `KNearest(k=12, space="raw 8x8 pixels")`, and checked: minimum degree
is exactly 12 and m = 14732 sits inside the implied [10782, 21564]. The raw pixels are retained
as `source_points` so the cost of that sparsification is measurable — rank ρ 0.6955 between
graph geodesic and ambient distance, relative stress 1.784, or **0.2027** after the optimal
uniform rescaling that SNS performs anyway.

## sbm-er-3modes

All 5753 edges carry the single constant weight 8.691e-05. A constant weight carries no
information, so `Unweighted()` on both sides (spec §0.5).
