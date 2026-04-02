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
  2 sum_(j eq.not i) (p_(j|i) - q_(j|i) + p_(i|j) - q_(i|j)) (vec(x)_i - vec(x)_j) $
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
$ cal(E)_(t"-SNE") (X) = cal(D)_"KL" (P || Q) = sum_(i eq.not j) p_(i j) log p_(i j)/q_(i j) $
<energy-tsne>

The definition of the joint probability $Q$ is the second major change,
as it now uses a heavy-tailed Student-$t$ distribution instead of the 
previous Gaussian. The probability $q_(i j)$ between points $i$ and $j$ is given by
$ q_(i j) = (1 + ||vec(x)_i - vec(x)_j||^2)^(-1) / 
  (sum_(k eq.not l) (1 + ||vec(x)_k - vec(x)_l||^2)^(-1)) quad "for" i eq.not j $
  <induced-prob-tsne>

For consistency, self-neighbor probabilities are again excluded in the symmetric
formulation, so $p_(i i) = q_(i i) = 0$.

These two changes result in the following gradient
$ (partial cal(E)_(t"-SNE"))/(partial vec(x)_i) = 
  4 sum_(j eq.not i) (p_(i j) - q_(i j)) 
  (1 + ||vec(x)_i - vec(x)_j||^2)^(-1) (vec(x)_i - vec(x)_j) $ <gradient-tsne>
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

In the exact formulation of #tsne, the joint probability distribution $P$ is dense,
since the Gaussian neighborhood model assigns nonzero probability to essentially every
pair of points. However, when the perplexity is small, the vast majority of these
probabilities are negligible. Consequently, practical implementations of #tsne often
replace $P$ by a sparse approximation obtained by retaining only the strongest local
neighborhood relations, typically through an exact or approximate $k$-nearest-neighbor
search, followed by symmetrization and renormalization. In this sense, practical #tsne
is already well approximated by a sparse stochastic #knn graph, even when the
original method is formally defined on a dense similarity matrix.

To understand the computational effect of sparsifying $P$, it is useful to
rewrite the gradient @gradient-tsne using the definition of $q_(i j)$ in
@induced-prob-tsne as
$ (partial cal(E)_(t"-SNE"))/(partial vec(x)_i) = 
  4 sum_(j eq.not i) (p_(i j) - q_(i j)) q_(i j)/Z (vec(x)_i - vec(x)_j) $ 
with normalization term $1 / Z = sum_(k eq.not l) (1 + ||vec(x)_k - vec(x)_l||^2)^(-1)$.

This allows us to rewrite the gradient as a sum of attractive and repulsive forces
$ (partial cal(E)_(t"-SNE"))/(partial vec(x)_i) = 
  4 / Z underbrace(sum_(j eq.not i) p_(i j) q_(i j) (vec(x)_i - vec(x)_j), F_"attr") - 
  4 / Z underbrace(sum_(j eq.not i) q_(i j)^2 (vec(x)_i - vec(x)_j), F_"rep") $

The sparsification substantially reduces the cost of the attractive part of the
gradient, since only pairs with nonzero $p_(i j)$ contribute. However, the repulsive
part remains dense, because the low-dimensional distribution $Q$ depends on all
pairs of points. For this reason, the dominant computational cost in large-scale
#tsne lies in approximating the repulsive interactions. Barnes-Hut-SNE
@vandermaaten2014 uses a spatial tree to approximate distant repulsive forces and
reduces the overall complexity to approximately $BigO(n log n)$, while also computing
a sparse approximation of $P$ from nearest-neighbor relations.
FIt-SNE @linderman2017fitsne accelerates the repulsive term further by 
interpolating the interaction kernel onto a regular grid and using the 
Fast Fourier Transform to evaluate the resulting convolution efficiently, 
together with approximate nearest-neighbor methods for constructing the sparse 
high-dimensional affinities.

==== Extension to sparse stochastic graph embedding (#sgtsne)
The sparse formulation used in practical #tsne suggests a natural extension to
network layout.  In conventional #tsne, the target distribution $P$ is obtained indirectly 
from high-dimensional feature vectors by first constructing a distance-based #knn graph,
then converting each neighborhood into a conditional Gaussian distribution, and
finally symmetrizing the result. 
Once the target distribution $P$ has been constructed, the subsequent embedding
optimization depends only on $P$ and the induced low-dimensional distribution $Q$,
not directly on the original point-cloud coordinates $Y$.

This suggests that the probability-matching formulation of #tsne need not be tied
specifically to point-cloud data. Instead, one may ask whether the same objective can
be posed directly on a sparse stochastic graph given as input, rather than using such
a graph only as an approximation to a dense similarity model. #sgtsne 
@pitsianis2019SpacelandEmbedding removes this interface restriction and instead 
takes as input a sparse stochastic graph directly. 

The method takes as input a sparse stochastic graph $G = (V, E, P_c)$, 
where $P_c = [p_(j|i)]$ is a row-stochastic conditional probability matrix
supported on the adjacency pattern of $G$. 

In the formulation used in practice, #sgtsne applies a row-wise nonlinear
rescaling to $P_c$, constructing a parameterized family of
stochastic matrices #box[$P_c^((lambda)) = [p_(j|i)^((lambda))]$].
For each vertex $i$, an exponent $gamma_i$ is chosen such that
$ sum_(j ~ i) p_(j|i)^(gamma_i) = lambda $
and the rescaled conditional probabilities are then defined, such that each row 
remains stochastic. They are given by
$ p_(j|i)^((lambda)) = p_(j|i)^(gamma_i) / lambda $
This corresponds to the special case used in the reported experiments, where the
more general monotone transform of the original formulation is taken to be the identity.

This rescaling plays a role analogous to the perplexity-based bandwidth selection
used in #sne and #tsne, but is adapted directly to a sparse stochastic graph.
Rather than deriving neighborhood distributions from distances and then enforcing
a fixed entropy level, it allows the concentration of the existing conditional
probabilities to be varied directly while preserving the sparsity pattern of the graph.

As in #tsne, this rescaled conditional model is symmetrized into a joint target distribution 
$ P^((lambda)) = (P_c^((lambda)) + (P_c^((lambda)))^T)/(2 n) $
while the low-dimensional distribution $Q$ is still defined 
by the Student-$t$ kernel in the layout space, like in @induced-prob-tsne. 
The embedding is then obtained by minimizing the same Kullback-Leibler divergence 
$cal(D)_"KL" (P^((lambda))|| Q)$ as in #tsne.

By varying $lambda$, the method interpolates between more concentrated and more
uniform neighborhood distributions on the fixed sparse support of the graph, thereby
providing a controllable family of target probability models for the embedding.

Although the input graph is sparse, the low-dimensional distribution $Q$ remains
dense, so the repulsive term in the gradient still involves all pairs of vertices.
For this reason, the practical implementation #sgtsnepi uses #tsnepi as its
computational back-end. #tsnepi is a high-performance computational implementation 
for large sparse graph embeddings, accelerating both the sparse attractive term and
the dense repulsive term through permutations and memory-aware access patterns
tailored to modern hierarchical memory architectures.

#alex[
  This needs a paragraph for the limitations of #sgtsne as a network
  layout method. These are described in detail in @related. I don't
  want to repeat myself again.
]

#nikos[Please check that everything here is correct]
