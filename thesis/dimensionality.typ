#import "commenting.typ":*
#import "macros.typ": *

Dimensionality reduction methods have not primarily been developed in
the context of network layout, but as general methods for embedding
high-dimensional data into lower-dimensional spaces. However their
generality also makes them useful for visualization, and specifically for network layouts.

Multidimensional Scaling (#mds), including the Stress model described in @force-based, 
forms a family of dimensionality reduction methods that has long been central 
to network layout.
More recently, Stochastic Neighbor Embedding methods (#sne @hinton2002sne and #tsne @tsne)
have been used in network visualization @kruiger2017Graphlayouts 
@pitsianis2019SpacelandEmbedding. 
In addition, #sne and #tsne, as well as more recent methods like 
#umap @mcinnes2020umap can generally be interpreted as layout
methods for probabilistic neighborhood graphs derived from point-cloud data.

==== Stochastic Neighbor Embedding (#sne and #tsne)
Stochastic Neighbor Embedding was developed by Geoffrey Hinton and Sam Roweis in 2002 
@hinton2002sne as a method for general dimensionality reduction whose objective 
is to preserve neighborhood relationships among the high-dimensional data points.

Given a high dimensional set of points $Y = [vec(y)_1, vec(y)_2, dots, vec(y)_n]^T$
we attempt to find a lower dimensional layout $X = [vec(x)_1, vec(x)_2, dots, vec(x)_n]^T$
such that the neighborhood structure is preserved.

From $Y$, we construct conditional neighborhood probabilities $p_(j|i)$,
interpreted as the probability that point $i$ chooses point $j$ as a neighbor,
defined as
$ p_(j|i) = exp(-||vec(y)_i - vec(y)_j||^2 slash 2 sigma_i^2) /
  (sum_(k eq.not i) exp(-||vec(y)_i - vec(y)_k||^2 slash 2 sigma_i^2)) $
where the standard deviation $sigma_i$ is chosen for each $i$ using binary search such 
that the Shannon entropy of the distribution $P_i$ measured in bits 
(defined as $H(P_i) = - sum_j p_(j|i) log_2 p_(j|i)$) satisfies the equation $2^(H(P_i)) = k$.
The value $2^(H(P_i))$ is called the perplexity $"Perp"(P_i)$, and represents
the effective number of neighbors $k$ of the distribution.
#footnote[
  Consider that $2^(H(P_i)) = k => H(P_i) = log_2 k = - log_2 (1 slash k) = I(1 slash k)$.
  Then the entropy (or average uncertainty) of $P_i$ is equal to the information 
  in a uniform distribution with $k$ values.
]
The value $k$ is chosen beforehand.

For a given low dimensional layout $X$ we also construct a conditional distribution with 
_induced_ conditional probability $q_(j|i)$ that point $i$ picks point $j$ given by
$ q_(j|i) = exp(-||vec(x)_i - vec(x)_j||^2) / 
  (sum_(k eq.not i) exp(-||vec(x)_i - vec(x)_k||^2)) $

In the above formulas, the self-neighbor probabilities are excluded, 
so $p_(i|i) = q_(i|i) = 0$.

The objective is to make the induced neighborhood distribution in the embedding
$X$ match the target distribution derived from $Y$, formulated as a minimization 
problem of a cost function that measures the discrepancy between the neighborhood 
distributions in the original and embedded spaces.
A natural choice is a sum of Kullback-Leibler divergences of the conditional distributions.
$ cal(E)_"SNE" (X) = sum_i cal(D)_"KL" (P_i || Q_i)
 = sum_i sum_j p_(j|i) log p_(j|i) / q_(j|i) $ <energy-sne>
 

Differentiating @energy-sne w.r.t the vectors $vec(x)_i$ results in gradient
$ (partial cal(E)_"SNE")/(partial vec(x)_i) = 
  2 sum_j (p_(j|i) - q_(j|i) + p_(i|j) - q_(i|j)) (vec(x)_i - vec(x)_j) $
which has the interpretation of forces on $vec(x)_i$, pulling toward 
or pushing away from $vec(x)_j$, depending on whether $j$ is assigned
too much or too little neighborhood probability in the embedding.

The #sne method preserves local neighborhood structure well and can reveal
cluster structure, but often suffers from substantial point crowding.
Additionally, the objective @energy-sne is difficult to optimize effectively,
and the solution tends to become trapped in local minima.

Van der Maaten and Hinton later introduced $t$-Distributed Stochastic
Neighbor Embedding (#tsne) to address these limitations in the context of visualization. 
#tsne differs from classical #sne in two major ways. 

The first is the symmetrization of the neighborhood probabilities.
Consider the probabilities $p_(j|i)$ and $p_(i|j)$. Generally $p_(j|i) eq.not p_(i|j)$
since the standard deviations $sigma_i$ and $sigma_j$ depend on the density of $Y$ near
points $i$ and $j$. 
The symmetric version instead introduces a joint probability distribution $P$,
defined by $p_(i j) = (p_(j|i) + p_(i|j))/(2 n)$, together with a corresponding
low-dimensional joint distribution $Q$ defined in the next paragraph.
This allows for the objective to be written as a single KL divergence
$ cal(E)_(t"-SNE") (X) = cal(D)_"KL" (P || Q) = sum_i sum_j p_(i j) log p_(i j)/q_(i j) $
<energy-tsne>

The definition of the joint probability $Q$ is the second major change,
as it now uses a heavy-tailed Student-$t$ distribution instead of the 
previous Gaussian. The probability $q_(i j)$ between points $i$ and $j$ is given by
$ q_(i j) = (1 + ||vec(x)_i - vec(x)_j||^2)^(-1) / 
  (sum_(k eq.not l) (1 + ||vec(x)_k - vec(x)_l||^2)^(-1)) quad "for" i eq.not j $

For consistency, self-neighbor probabilities are again excluded in the symmetric
formulation, so $p_(i i) = q_(i i) = 0$.

These two changes result in the following gradient
$ (partial cal(E)_(t"-SNE"))/(partial vec(x)_i) = 
  4 sum_j (p_(i j) - q_(i j)) (vec(x)_i - vec(x)_j)/(1 + ||vec(x)_i - vec(x)_j||^2) $
which has a simpler symmetric form than the gradient of classical #sne.

Together, these changes mitigate both of the above limitations of #sne.
The heavy-tailed kernel produces repulsive forces for dissimilar points
that decay more slowly with distance compared to #sne, effectively reducing crowding
in the layout.
In addition, the objective @energy-tsne produces a more stable optimization
landscape in practice using standard optimization methods.

#tsne uses a modified steepest descent algorithm with momentum and an initial
early exaggeration phase to optimize the energy function. 
In practice, this produces substantially more useful visual embeddings for many
point-cloud datasets compared to classical #sne and several earlier nonlinear
dimensionality reduction methods such as Sammon mapping and Isomap.

==== Extension to sparse network visualization (#sgtsne)
As presented above, #sne and #tsne are not intrinsically network layout methods,
but methods for matching a low-dimensional embedding to a target neighborhood
probability model derived from input data. In the standard setting this model is
constructed from pairwise distances in a high-dimensional point cloud.

This perspective extends naturally to network layout by replacing the point-cloud-derived
probability model with one derived directly from the graph structure. The target
distribution $P$ may then be constructed from adjacency, edge weights, random-walk
affinities, geodesic distances, or other graph-based similarity measures, while the
induced distribution $Q$ is still determined by the low-dimensional layout.
This is the basic idea underlying sparse graph #tsne methods such as #sgtsne.
