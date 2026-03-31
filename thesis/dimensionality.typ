#import "commenting.typ":*
#import "macros.typ": *

Dimensionality Reduction methods have not traditionally been developed
in the context of Network Layout. They are methods for embedding high
dimensional data into lower dimensional spaces, useful in Machine Learning
methods for transforming general inputs to vector-like objects that 
neural networks can more effectively process. However their generality
allows them to be effectively utilized for visualization, and specifically
in the field of network layout where they have historically become standard.

Multidimensional Scaling (#mds) like the Stress model described in @force-based,
is fundamentally a family of dimensionality reduction methods which has been long
utilized for network layouts. More recently, Stochastic Neighbor Embedding 
(#sne @hinton2002sne and #tsne @tsne) methods have been utilized for Network Layout 
@kruiger2017Graphlayouts @pitsianis2019SpacelandEmbedding. 
In addition, #sne and #tsne, as well as more recent methods like #umap @mcinnes2020umap
can generally be interpreted as layout generation methods for $k$NN networks
of point-cloud data. 

==== Stochastic Neighbor Embedding (#sne and #tsne)
The #sne method was developed by Geoffrey Hinton and Sam Roweis in 2002 @hinton2002sne as 
a method for general dimensionality reduction with the stated objective of preserving
the neighbor relationships of the high-dimensional objects.

Given a high dimensional set of points $Y = [vec(y)_1, vec(y)_2, dots, vec(y)_n]^T$
we attempt to find a lower dimensional layout $X = [vec(x)_1, vec(x)_2, dots, vec(x)_n]^T$
such that the neighbor relationship is preserved.

We construct a gaussian stochastic network from $Y$,
with probabilistic weights $p_(i j)$ given by
$ p_(i j) = exp(-d_(i j)^2 slash 2 sigma_i^2) /
  (sum_(k eq.not i) exp(-d_(i k)^2 slash 2 sigma_i^2)) $
where the distances $d_(i j)^2$ are given by $d_(i j)^2 = ||vec(y)_i - vec(y)_j||^2$
and $sigma_i^2$ is chosen for each $i$ such that it makes the entropy of the
distribution over its neighbors equal to $log k$, where $k$ is the effective 
number of local neighbors, also called the "perplexity".

For a given low dimensional layout $X$ we also construct a gaussian stochastic graph
with _induced_ probability $q_(i j)$ that point $i$ picks point $j$ as its neighbor,
and is given by
$ q_(i j) = exp(-||vec(x)_i - vec(x)_j||^2) / 
  (sum_(k eq.not i) exp(-||vec(x)_i - vec(x)_k||^2)) $

The aim of the embedding is to match these two distributions as well as possible,
formulated as a minimization problem of a cost function which measures information
loss of the embedding compared to the true distribution. A natural choice is a
sum of Kullback-Leibler divergences.
$ cal(E)_"SNE" (X) = sum_i cal(D)_"KL" (P_i || Q_i)
 = sum_i sum_j p_(i j) log p_(i j) / q_(i j) $ <energy-sne>
 

Differentiating @energy-sne w.r.t the vectors $vec(x)_i$ results in
$ (partial cal(E)_"SNE")/(partial vec(x)_i) = 
  2 sum_j (vec(x)_i - vec(x)_j) (p_(i j) - q_(i j) + p_(j i) - q_(j i)) $
which has the interpretation of forces on $vec(x)_i$, pulling toward 
or pushing away from $vec(x)_j$, depending on whether $j$ is a neighbor
more or less probably than desired.

The #sne method produces good embeddings in terms of neighborhood preservation
and cluster separation, but features high degrees of point crowding.
Additionally, as it turns out the stated objective @energy-sne is hard 
to effectively optimize as the solution tends to get trapped in 
local minima.

Van der Maaten and Hinton later introduced $t$-Distributed Stochastic
Neighbor Embedding (#tsne) to adress these limitations in the context of visualization. 
The major change in #tsne is the induced probability distribution of the low-dimensional
layout, which uses a heavy-tailed Student-$t$ Distribution with $1$ degree of freedom
instead of the Gaussian. Then the induced probabilities $q_(i j)$ are given by
$ q_(i j) = (1 + ||vec(x)_i - vec(x)_j||^2)^(-1) / 
  (sum_(k eq.not i) (1 + ||vec(x)_i - vec(x)_k||^2)^(-1)) $

With an energy $cal(E)_(t"-SNE")$ in the same form as @energy-sne, 
but with gradient now given by
$ (partial cal(E)_(t"-SNE"))/(partial vec(x)_i) = 
  2 sum_j (vec(x)_i - vec(x)_j)/(1 + ||vec(x)_i - vec(x)_j||^2) 
  (p_(i j) - q_(i j) + p_(j i) - q_(j i)) $

This change in effect mitigates both of the afore-mentioned limitations of #sne.
The gradient now includes strong repulsive forces between points placed 
too closely in the layout $X$, decreasing the effect of crowding. 
In addition the objective with the $t$-Distributed probabilities is 
much easier to optimize compared to the original objective.

#tsne uses a modified steepest descend algorithm with a momentum term to optimize
the energy function. This produces very good visualization results for
point cloud data compared to other popular visualization techniques like
Sammon mapping and Isomap (Modifications of the Stress model in #mds).

==== Extension to sparse network visualization (#sgtsne)
