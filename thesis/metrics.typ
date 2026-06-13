#import "@preview/run-liners:0.1.0": *
#import "commenting.typ": *
#import "macros.typ": *

As previously described, it is useful to construct numerical
measures aimed at quantitatively evaluating the quality of a graph
layout.

The measures considered here correspond broadly to two categories of desirable
properties in scientific network layout:
#run-in-enum(
  numbering-pattern: "(i)",
  [the degree to which structural features of the graph are preserved in the layout],
  [the degree to which the layout is visually clear and free from artifacts such as
  crowding, clutter, or excessive edge complexity],
).

In what follows, we use criterion to refer to the qualitative property of
interest, such as neighborhood preservation or point crowding, and metric to
refer to the concrete numerical quantity used to evaluate that property.
#footnote[
  This use of the word metric should not be confused with a metric as a distance
  function on a metric space.
]

Not every structural feature of a graph has a canonical geometric equivalent in
a layout. A graph and a set of points in Euclidean space are different
mathematical objects, and a layout algorithm necessarily chooses how graph
relationships are to be represented geometrically. Consequently, every criterion
requires a modeling decision: one must decide which graph-theoretic property is
being evaluated and which geometric property is taken to represent it.

For example, suppose we want the adjacency relation between two vertices
$i,j in V$ to be preserved by their point representations $vec(x)_i,vec(x)_j in RR^d$.
Since geometric points cannot be adjacent in the graph-theoretic sense, adjacency
must be translated into some geometric criterion. One natural interpretation is
proximity: adjacent vertices should be placed close to one another relative to
non-adjacent vertices. This can be measured in several ways. One may compare the
distances of adjacent and non-adjacent vertex pairs, or construct the $k$ nearest
neighbors of each point in the layout and compare them to the graph neighborhood
of the corresponding vertex

Each of these choices captures a different interpretation of the same informal
criterion. There is therefore no canonical metric for a general property such as
"adjacency preservation" or "visual clarity." For this reason, the evaluation in
this thesis adopts a pluralistic approach: for each structural or visual property
of interest, we use one or more representative metrics that capture common and
practically useful interpretations of that property.

=== Feature Preservation Metrics
Consider an (optionally weighted) graph $G = (V,E,w)$ with $n$ vertices
and $m$ edges, and a layout $X in RR^(n times d)$. 
A layout metric is a rule $f$ which assigns
a numerical value to such a pair $(G,X)$. Since the layout dimension depends
on the number of vertices of $G$, we write this schematically as
$f(G, X) in RR$.

For a fixed graph $G$, the metric induces a function
$ f_G : RR^(n times d) -> RR $
defined by $f_G (X) = f(G,X)$. We also write this value as $f(X ; G)$,
emphasizing that the layout $X$ is the object being evaluated while the graph
$G$ provides the structural reference.

Some metrics are first defined locally, for each vertex or edge, and then
aggregated over the graph. In such cases $f(X ; G)$ denotes the final aggregate
score unless otherwise stated.

For weighted graphs we define $w_(i j) = w(e_(i j))$ to be the weight of edge 
$e_(i j) in E$, and $w_(i j) = 0$ when $e_(i j) in.not E$. 
For unweighted graphs, we construct the trivially weighted graph where all 
weights are $1$, such that $w_(i j) = 1$ if $i ~ j$, and $w_(i j) = 0$ otherwise.

==== Adjacency Distance Ratio
We would like to preserve the adjacency relationship between vertices in $G$
in the layout $X$. As previously described, this can be done by measuring
the proximity of adjacent or strongly related vertices relative to the
overall spatial scale of the layout.
There are several possible interpretations of this criterion; following our
pluralistic approach, we choose one representative metric.

We measure proximity-based adjacency preservation by comparing the average
embedded distance of graph-related vertex pairs against the average embedded
distance of all vertex pairs.
For a weighted graph $G = (V,E,w)$, we define
$ "ADR"(X ; G) 
  = (1/W sum_(i < j) w_(i j) ||vec(x)_i - vec(x)_j||)
  / (2/(n(n-1)) sum_(i < j) ||vec(x)_i - vec(x)_j||) $
where 
$ W = sum_(i < j) w_(i j) $

Smaller values indicate that strongly related vertices are placed closer 
together relative to the global spatial scale of the layout. 
Typically, we want $"ADR"(X ; G) < 1$.

For an unweighted graph, this reduces to
$ "ADR"(X ; G) 
  = (1/m sum_(i ~ j) ||vec(x)_i - vec(x)_j||)
  / (2/(n(n-1)) sum_(i < j) ||vec(x)_i - vec(x)_j||) $

This metric is not intended to exhaust all possible interpretations of
adjacency preservation. It captures one simple and practically useful
interpretation: graph-related pairs should be short relative to the global
spatial scale of the layout. Neighborhood-recovery metrics based on
$k$-nearest neighbors, discussed next, capture the complementary rank-based
question of whether graph-neighbors are recovered among spatial neighbors.

==== #knn Neighborhood Recovery
This metric corresponds to a rank-based interpretation of the same criterion:
if adjacency is preserved by spatial proximity, then graph-neighbors of a
vertex should appear among its nearest embedded neighbors.
In contrast to the previous distance-ratio metric, this metric does not compare 
distance values directly, but instead evaluates the ordering induced by embedded distances.

Consider the vertex $i in V$. We define its graph neighborhood as 
$cal(N)_G (i) = {j in V | i ~ j}$ and its degree $deg(i) = |cal(N)_G (i)|$.
For the point $vec(x)_i in X$ we find its #knn neighborhood defined as
$ cal(N)^k_X (i)
  = {j in V | j eq.not i 
    "and" vec(x)_j "is among the" k"-nearest neighbors of" vec(x)_i "in" X} $

The vertices $j in cal(N)_G (i) inter cal(N)^k_X (i)$ are called true positives, since
they denote true neighbors of $i$ recovered from the $k$ nearest neighbors of $vec(x)_i$.
We would like then, to maximize the number of true positives 
$"TP"_i = |cal(N)_G (i) inter cal(N)^k_X (i)|$.
However, the absolute number of true positives is not by itself an appropriate
metric. The maximum possible value of $"TP"_i$ depends on both the degree of
$i$ and the chosen neighborhood size $k$, since
$ "TP"_i <= min{deg(i), k} $

As a result, vertices of large degree may contribute more true positives simply
because they have more graph-neighbors that can be recovered. A global sum of
true positives would therefore be dominated by high-degree vertices, rather than
measuring neighborhood recovery in a vertex-balanced way.
For this reason, we normalize the overlap using standard set-recovery scores.

- The *Recall* of the graph neighborhood of $i$ is defined as 
  $ R^k_i (X ; G)
    = (|cal(N)_G (i) inter cal(N)^k_X (i)|)/(|cal(N)_G (i)|) 
    = "TP"_i / deg(i) $
  and measures the fraction of true graph-neighbors of $i$ that are recovered
  among its $k$ nearest embedded neighbors.

- The *Precision* of the recovered embedded neighborhood is defined as
  $ P^k_i (X ; G)
    = (|cal(N)_G (i) inter cal(N)^k_X (i)|)/(|cal(N)^k_X (i)|) 
    = "TP"_i / k $
  and measures the fraction of the $k$ nearest embedded neighbors of $i$
  that are true graph-neighbors.

- Finally, the *Jaccard index* is defined as
  $ J^k_i (X ; G)
    = (|cal(N)_G (i) inter cal(N)^k_X (i)|)/(|cal(N)_G (i) union cal(N)^k_X (i)|) 
    = "TP"_i / (deg(i) + k - "TP"_i) $
  and  measures the relative overlap between the graph neighborhood
  and the embedded neighborhood, penalizing both graph-neighbors that are not
  recovered and embedded neighbors that are not graph-neighbors.

Then the associated global scores become
$ M^k (X ; G) = mean_(i in V) {M^k_i (X ; G)} $
with $M in {R, P, J}$, possibly excluding zero-degree vertices for Recall,
since that would result in an undefined result.

Since the graph-neighborhood sizes are given by the vertex degrees, the choice
of $k$ should be interpreted relative to the degree distribution of the graph.
When $k$ is fixed globally, the different normalized scores have different
degree-dependent ceilings. Precision can only reach $1$ for vertices with
$deg(i) >= k$, while recall can only reach $1$ for vertices with
$deg(i) <= k$. Therefore, fixed-$k$ precision tends to favor high-degree
vertices, whereas fixed-$k$ recall tends to favor low-degree vertices. The
Jaccard index balances the two effects and is largest when the graph degree
of the vertex is comparable to $k$.
For this reason, it is often useful to evaluate the scores for several values
of $k$, rather than relying on a single neighborhood scale.

==== Weighted #knn Neighborhood Recovery
The previous metric treats neighborhood recovery combinatorially: each graph
neighbor is either recovered or not recovered. For weighted graphs, this may
not capture the intended notion of neighborhood preservation. For example, a
weighted graph may be fully connected from a combinatorial point of view, while
most of the weight of each vertex is concentrated on only a small number of
strong relations. In this case, recovering a weakly related vertex and
recovering a strongly related vertex should not contribute equally.

One possible solution is to sparsify the weighted graph and then apply the
combinatorial metric. For example, for each vertex one could retain the
strongest edges whose cumulative weight exceeds a prescribed threshold.

However, a more direct approach is to measure how much of the vertex's incident weight
is recovered among its nearest embedded neighbors.
$ W^k_i (X ; G) = (sum_(j in cal(N)^k_X (i)) w_(i j)) / (sum_(j eq.not i) w_(i j)) $
This measures the fraction of the total weight of $i$ that is recovered among
the $k$ nearest embedded neighbors of $vec(x)_i$. 

Equivalently, if the normalized weights
$ p_(i j) = w_(i j) / (sum_(j eq.not i) w_(i j)) $
are interpreted as a probability distribution over vertices adjacent or related
to $i$, then $W_i^k(X ; G)$ is the probability mass recovered by the embedded
$k$-nearest-neighbor set.

The associated global metric is then given as
$ W^k (X ; G) = mean_(i in V) {W^k_i (X ; G)} $

For an unweighted graph, with $w_(i j) = 1$ when $i ~ j$ and $w_(i j) = 0$
otherwise, this definition reduces to combinatorial recall:
$ W^k_i (X ; G) 
  &= (sum_(j in cal(N)^k_X (i)) w_(i j)) / (sum_(j eq.not i) w_(i j))
   = (sum_(j in cal(N)^k_X (i) inter cal(N)_G (i)) 1) / (sum_(j in cal(N)_G (i)) 1) \ 
  &= (|cal(N)_G (i) inter cal(N)^k_X (i)|) / (|cal(N)_G (i)|)
   = "TP"_i / deg(i) 
   = R^k_i (X ; G) $
For this reason, we refer to this metric as weighted recall.

==== Scale-Normalized Stress
Another graph level feature we would like preserved is the global distances
between vertices. For a graph $G$ we can derive a set of ideal distances $d_(i j)$
for every vertex pair. 
Similar to the definition in @force-based, we generally define $d_(i j)$ for 
adjacent vertices to be a function of the weight of the edge $f(w_(i j))$, 
and for non-adjacent vertices to be the shortest-path length between 
$i$ and $j$ induced by these edge lengths.
We say then, that a layout $X$ is global distance preserving, if
the geometric distances between points in $X$ are proportional
to the distances in $d_(i j)$.
Since it is generally impossible to exactly realize the graph distances as
Euclidean distances in $RR^d$,
we must construct a metric which measures the degree of distance preservation.

Borrowing from #mds and the Stress model defined in @force-based,
we can use the Raw Stress function as a crude metric for
global distance deviation
$ "Stress"(X ; G) = sqrt(sum_(i < j) (d_(i j) - ||vec(x)_i - vec(x)_j||)^2) $
This function does indeed penalize non-matching distances between $d_(i j)$
and $||vec(x)_i - vec(x)_j||$, however it is too strict in that sense. 
There are two main issues with using Raw Stress as a metric, and
they both stem from its non-invariance to scale:
+ It is sensitive to the overall scale of the distances $d_(i j)$,
  so the magnitude of the score reflects not only distortion but also the
  absolute scale of the chosen ideal-distance model.
+ More importantly, it is not invariant to scaling 
  transformations of the layout $X |-> beta X$. 
  In fact, it assumes that $X$ has the same implicit scale
  as the distances $d_(i j)$ which cannot be guaranteed
  for every layout algorithm. 
  As a result Raw Stress is not an appropriate metric for comparing 
  accross different layout algorithms, which is the explicit
  purpose of a metric.

A common improvement over Raw Stress is to normalize the squared error.
There are several related normalizations in the literature.
In Multidimensional scaling, Kruskal's Stress-1 @kruskal1964MDS 
normalizes the squared error by $sum_(i < j) ||vec(x)_i - vec(x)_j||^2$, while
another common form, often also called normalized stress, normalizes
by $sum_(i < j) d_(i j)^2$. In our notation, the latter is
$ "NStress"(X ; G) 
  = sqrt((sum_(i < j) (d_(i j) - ||vec(x)_i - vec(x)_j||)^2) / (sum_(i < j) d_(i j)^2)) $

This normalization makes the score dimensionless and expresses the squared
error relative to the total squared magnitude of the target distance matrix.
However, it does not fully solve the scale issue. If the layout is uniformly
rescaled as $X |-> beta X$, then
$ "NStress"(beta X ; G) 
  = sqrt((sum_(i < j) (d_(i j) - beta||vec(x)_i - vec(x)_j||)^2) / (sum_(i < j) d_(i j)^2)) $

which is generally not equal to $"NStress"(X ; G)$. 
Thus, two layouts that differ only by a uniform scaling 
transformation may receive different scores.

Kruskal's Stress-1 partly alleviates this issue by normalizing with respect to
the squared layout distances, but it is still not invariant under uniform
rescaling of the layout. Since the global scale of a layout is arbitrary, this
remaining scale dependence is undesirable when comparing different layout
algorithms.

For this reason, we use scale normalized stress @ahmed2024, @smelser2025.
Because the distances $d_(i j)$ define the target graph geometry, we keep the
target-distance normalization, but first rescale the embedded distances by the
optimal scalar factor $alpha > 0$:
$ "SNS"(X ; G) 
  &= min_(alpha > 0) "NStress"(alpha X ; G) \
  &= min_(alpha > 0) sqrt(
    (sum_(i < j) (d_(i j) - alpha||vec(x)_i - vec(x)_j||)^2) / (sum_(i < j) d_(i j)^2)
  ) $

The optimal value for $alpha$ can be found analytically and turns out to be
$ alpha^* = (sum_(i < j) d_(i j) ||vec(x)_i - vec(x)_j||) 
  / (sum_(i < j) ||vec(x)_i - vec(x)_j||^2) $

Therefore, the metric used in the evaluation is
$ "SNS"(X ; G) 
  = sqrt((sum_(i < j) (d_(i j) - alpha^*||vec(x)_i - vec(x)_j||)^2) / (sum_(i < j) d_(i j)^2)) $

This metric measures the discrepancy between graph distances and embedded
distances after accounting for the arbitrary global scale of the layout.
It can be proven that it is invariant to scaling transformations
of both the layout $X$ and the ideal distances $d_(i j)$.
The square root is included so that the stress value is expressed on 
the scale of relative distance error rather than squared relative error; 
it does not affect the optimal scaling factor.
Smaller values of $"SNS"$ indicate better global distance preservation.

==== Silhouette Score
Community structure is a feature of networks which is of critical importance 
to scientific network analysis. In fact, one of the primary goals of 
network visualization often is the visual detection of communities in the embedded graph.
For this reason, it is desirable to preserve the community structure in the layout.

It is hard to formally define what a community is, however, we understand 
communities generally as groups of vertices which are connected more densely with
each other than with other vertices in the graph.
Nevertheless, various approaches exist for detecting communities in networks. 
This is the task of assigning to each vertex $i$ a label $cal(l)_i$, 
such that communities are made up of vertices with a common label:
$ C(cal(l)) = {i in V | cal(l)_i = cal(l)} $
Additionally, real-world networks often include labels for each of the vertices,
which correspond to some feature of the entity which the vertex represents:
e.g. a citation network where each vertex represents a unique academic article 
may provide, for each vertex, information like its primary author,
the publication journal, its research topic etc.
It is not necessary that these external labels are reflected in any way in 
the network structure.
In practice however, communities found in real-world networks often correlate
with meaningful external attributes of the represented entities.

It is a significant methodological concern whether external labels or 
communities detected from the graph should be used as the reference partition 
when evaluating the community-preservation properties of a layout.

- Since all of the layout algorithms discussed have access only to the graph topology, 
  using external labels may result in evaluating layout algorithms according to 
  information which is not necessarily encoded in the graph structure available to them. 

- On the other hand, using a community detection algorithm necessarily imposes 
  the assumptions of the algorithm onto the evaluation of a layout. 
  These assumptions differ significantly between algorithms, 
  even in their basic definition of what a community is. 
  Additionally, some community detection algorithms use similar machinery as 
  some layout algorithms, even so far as to use point-cloud embeddings as a 
  first step in the community detection process.
  Then the evaluation may be biased toward layout algorithms which agree with 
  the assumptions or utilize similar machinery with that of the detection algorithm.

The purpose of the metric is not to measure agreement with a particular community 
detection algorithm, but rather to assess whether semantically meaningful categories 
present in the data are reflected in the visualization.
We therefore use external labels when available for the evaluation of
label/community separation.
Although such labels are not guaranteed to correspond to structural communities 
in the graph, they are independent of both the layout and evaluation procedures.
Consequently, they provide a common reference against which different layout 
algorithms may be compared without introducing assumptions inherited from a 
particular community detection method.
Additionally, we can test whether a given external labeling is
reflected in the graph structure using similar metrics to the ones
used to evaluate the layout. 
This is discussed in more detail after defining the relevant metrics.

As for the layout, in order for communities to be visually distinct,
it is preferred that vertices within a community are placed compactly
in the layout, with good relative separation between communities,
thus creating visually distinguishable point clusters.

One metric to measure this criterion is the commonly used silhouette score.
It is defined for a single vertex as the normalized difference of
distances between the points in its own community and the points in
the closest neighboring community.

Let $cal(L)$ be the set of unique labels such that for every $i in V => cal(l)_i in cal(L)$.
Then, let $cal(C)$ be the set of all labeled communities:
$ cal(C) = {C(cal(l)) | cal(l) in cal(L)} $

We define the community of vertex $i$ as
$ C_i = C(cal(l)_i) = {j in V | cal(l)_j = cal(l)_i}. $

Then we calculate its average distance to points in its own community as
$ a_i = 1/(|C_i| - 1) sum_(j in C_i , i eq.not j) ||vec(x)_j - vec(x)_i|| $
Then the average distance to points in the closest neighboring community is
$ b_i = min_(C in cal(C), C eq.not C_i) 1/(|C|) sum_(j in C) ||vec(x)_j - vec(x)_i|| $
The silhouette score for vertex $i$ is defined as
$ "Sil"_i (X ; cal(l)) = (b_i - a_i) / max{a_i, b_i} $

This value ranges for each vertex from $-1$ to $+1$, where large positive
values indicate a vertex is well matched to its own community, 
large negative values indicate that the vertex is closer, on average,
to another community than to its assigned community,
and values near $0$ indicate that the vertex is close to the
border between the communities.
Notably, this score is not well defined for singleton communities
where $|C_i| = 1$, so we set the score for these vertices as $0$.

The global score is given as the mean of the scores for each point
$ "Sil"(X ; cal(l)) = 1/n sum_(i in V) "Sil"_i (X ; cal(l)) $
A large positive global score indicates compact and well-separated 
labeled communities in the layout.
Inversly, negative scores indicate that vertices are, on averag, 
closer to other labeled groups than to their assigned group.

As mentioned, we can use a similar metric in order to test whether the
external labels reflect the graph topology: given the external labels
we define the graph-theoretic silhouette score given by
$ a^*_i = 1/(|C_i| - 1) sum_(j in C_i , i eq.not j) d_(i j) $
$ b^*_i = min_(C in cal(C), C eq.not C_i) 1/(|C|) sum_(j in C) d_(i j) $
$ "Sil"^*_i (G ; cal(l)) = (b^*_i - a^*_i) / max{a^*_i, b^*_i} $
Where $d_(i j)$ denote the graph-theoretic distances like described previously.
Then the global score $"Sil"^* (G ; cal(l))$ is defined as above.

The graph-theoretic silhouette provides a baseline for how strongly the
external labels are reflected in the graph distances themselves. Layout scores
should therefore be interpreted relative to this baseline. This does not
introduce the same concern as using a community detection algorithm, since the
reference partition is not produced by an algorithm whose assumptions may
overlap with those of the layout method. Instead, we compare the graph and the
layout according to their ability to reflect the same external partitioning
under the same separation criterion.

==== #knn Classification Accuracy
Another metric corresponding to the same criterion is to measure the accuracy 
of using a #knn based classifier in its recovery of the initial labels.
This is based on the principle that for any vertex, a community preserving layout 
should place it near vertices of the same community. 
Therefore, predicting the label of a vertex from the labels of its 
nearest neighbors should be accurate.

A classifier is defined as a procedure which assigns each vertex $i$ to a
predicted label $hat(cal(l))_i$, without prior knowledge of $cal(l)_i$. 
A classifier of the form we discuss, however, has access to the labels 
$cal(l)_j$ with $j eq.not i$.

Specifically, let $cal(N)_X^k (i)$ denote the set of $k$ nearest neighbors of
$x_i$ in the layout $X$. Equivalently, this defines the #knn graph $G_X^k$ of
$X$, whose vertices are the vertices of $G$, and where each vertex $i$ is
connected to its $k$ nearest neighbors in $X$. Notably, this graph is not
necessarily undirected, since there is no guarantee that
$ j in cal(N)^k_X (i) <=> i in cal(N)^k_X (j) $

For each vertex $i$, we construct a score for each possible label $cal(l) in cal(L)$
according to the labels of the embedded neighbors of $i$. The simplest form of
this is a voting scheme, where the score is the number of neighbors that have label $cal(l)$:
$ s_i (cal(l)) = |cal(N)^k_X (i) inter C(cal(l))| $
Alternatively, we may weigh the score according to the distance
of $vec(x)_j$ to $vec(x)_i$, such that closer points contribute more
strongly than those farther away:
$ s_i (cal(l)) = sum_(j in cal(N)^k_X (i) inter C(cal(l))) f(||vec(x_i) - vec(x_j)||) $
Here $f$ is a decreasing function of distance, so that larger distances
correspond to smaller contributions. The choice of $f(d) = 1$ reduces to
the simple voting scheme, while a common distance-weighted alternative is 
$f(d) = 1 slash d$.

In all cases, the classifier is defined by selecting the label with
the highest score:
$ hat(cal(l))_i = argmax_(cal(l) in cal(L)) s_i (cal(l)) $

The #knn classification accuracy is then defines as the 
fraction of vertices whose predicted labels match their
true labels.
$ "Acc"^k (X ; cal(l)) = 1/n sum_(i in V) [hat(cal(l))_i = cal(l)_i] $
Here $[ dot ]$ denotes the Iverson bracket.
A high value of $"Acc"^k (X ; cal(l))$ indicates that the local
neighborhoods in the layout are label-homoegeneous enough to 
recover the original labels from nearby vertices.

Conversely, the equivalent graph metric uses a neighborhood classifier which 
assigns a label to each vertex according to the labels of its 1-hop neighborhood in the graph.
The voting scheme from above becomes:
$ s^*_i (cal(l)) = |cal(N)_G (i) inter C(cal(l))| $
For weighted graphs, we may instead weigh the score according to the edge weights:
$ s^*_i (cal(l)) = sum_(j in cal(N)_G (i) inter C(cal(l))) w_(i j) $
Then as above, the graph-based predicted labels are
$ hat(cal(l))^*_i = argmax_(cal(l)) s^*_i (cal(l)) $
The corresponding graph-neighborhood classification accuracy is
$ "Acc"^* (G ; cal(l)) = 1/n sum_i [hat(cal(l))^*_i = cal(l)_i] $
For isolated vertices, the graph neighborhood contains no information from
which the label can be recovered. We therefore count such vertices as incorrect,
except when they belong to singleton label classes.

This measures how well the external labels are reflected in the graph topology 
according to the same local classification criterion, while also serving as a 
baseline against which the scores of the layout algorithms can be compared.

=== Visual Quality Metrics
