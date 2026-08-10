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
  [the degree to which features of the graph are preserved in the layout],
  [the degree to which the layout is visually clear and free from artifacts such as
  crowding, clutter, or excessive edge complexity],
).
Feature preservation, in turn, covers two related but distinct concerns: the
preservation of the graph's topological structure, such as adjacency and
distance, and the preservation of externally meaningful labels or communities,
which need not be recoverable from the topology alone. We treat each of these
three concerns in its own subsection below: structural feature preservation,
label-based feature preservation, and visual quality.

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
of the corresponding vertex.

Each of these choices captures a different interpretation of the same informal
criterion. There is therefore no canonical metric for a general property such as
"adjacency preservation" or "visual clarity." For this reason, the evaluation in
this thesis adopts a pluralistic approach: for each structural or visual property
of interest, we use one or more representative metrics that capture common and
practically useful interpretations of that property.

=== Structural Feature Preservation
Consider an (optionally weighted) graph $G = (V,E,w)$ with $n$ vertices
and $m$ edges, and a layout $X in RR^(n times d)$. 
A layout metric is a rule $f$ which assigns a numerical value to such a pair $(G,X)$. 

For a fixed graph $G$, we write the corresponding graph-specific metric as $X |-> f(X ; G)$,
where the notation emphasizes that the layout $X$ is the object being evaluated 
while the graph $G$ provides the structural reference.

Some metrics are first defined locally, for each vertex or edge, and then
aggregated over the graph. In such cases $f(X ; G)$ denotes the final aggregate
score unless otherwise stated.

For weighted graphs we define $w_(i j) = w(e_(i j))$ to be the weight of edge 
$e_(i j) in E$, and $w_(i j) = 0$ when $e_(i j) in.not E$. 
For unweighted graphs, we construct the trivially weighted graph where all 
weights are $1$, such that $w_(i j) = 1$ if $i ~ j$, and $w_(i j) = 0$ otherwise.
In this case, the weights are taken to represent a sense of similarity,
or affinity, rather than distance. This means that vertices which are
connected with high edge weights are more closely related, something which
is often desirable to preserve in the layout.

Conversely, we may define an ideal distance $d_(i j)$ between each pair of vertices.
Similar to the definition in @force-based, we generally define $d_(i j)$ for 
adjacent vertices to be a function of the weight of the edge $f(w_(i j))$, 
and for non-adjacent vertices to be the shortest-path length between 
$i$ and $j$ induced by these edge lengths.

Datasets of networks may provide values for the edges corresponding to
either similarities/strengths/affinity scores or to distances/costs. 
Since these interpretations are inversely ordered but not metrically equivalent, 
any conversion from affinity to distance requires an explicit modeling choice. 
At minimum the transformation should be monotone decreasing.
Stronger requirements and the specifics of this conversion are outside the 
scope of this thesis, however, we mention that some common choices used
in practice include $d = 1 slash w$, $d = 1 - w$ and $d = -log(w)$.

When a local score is undefined because its denominator is zero, the corresponding
vertex is excluded from the aggregate unless otherwise stated. For metrics based
on graph-theoretic distances, we evaluate only vertex pairs with finite distance,
or equivalently restrict the metric to connected components according to the
convention specified in the experimental setup.

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
  = (1/m sum_(i ~ j, i < j) ||vec(x)_i - vec(x)_j||)
  / (2/(n(n-1)) sum_(i < j) ||vec(x)_i - vec(x)_j||) $
where, as throughout, $i ~ j$ ranges over unordered adjacent pairs, each
counted once.

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
Comparing the graph neighborhood of a vertex against its embedded #knn set
through set-overlap statistics of this kind is a standard approach for
evaluating graph layouts and node embeddings @kruiger2017Graphlayouts
@ahmed2020gd2 @bohm2025.

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
Another graph-level feature we would like preserved is the global distances between vertices. 
For a graph $G$ we can derive a set of ideal distances $d_(i j)$ for every vertex pair. 
We say that a layout $X$ is global distance preserving, if the geometric
distances between points in $X$ are proportional to the distances in $d_(i j)$.
Since it is generally impossible to exactly realize the graph distances as
Euclidean distances in $RR^d$, we must construct a metric which measures the degree 
of distance preservation.

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
  As a result, Raw Stress is not an appropriate metric for comparing 
  across different layout algorithms.

A common improvement over Raw Stress is to normalize the squared error.
There are several related normalizations in the literature.
In #mds, Kruskal's Stress-1 @kruskal1964MDS 
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
It can be proven @ahmed2024 @smelser2025 that it is invariant to scaling
transformations of both the layout $X$ and the ideal distances $d_(i j)$.
The square root is included so that the stress value is expressed on 
the scale of relative distance error rather than squared relative error; 
it does not affect the optimal scaling factor.
Smaller values of $"SNS"$ indicate better global distance preservation.

=== Label-based Feature Preservation
Many real-world networks include labels or categories associated with their
vertices. These labels may correspond to known communities, classes, topics,
functional groups, or other external attributes of the entities represented by
the graph. When such labels are meaningful, a useful layout should often make
them visually interpretable: vertices with the same label should tend to form
compact and visually distinguishable regions in the drawing.

Community labels are an important special case. Although community structure is
difficult to define uniquely, communities are generally understood as groups of
vertices which are more densely connected internally than externally. Community
detection methods attempt to infer such groups from the graph topology by
assigning to each vertex $i$ a label $cal(l)_i$, such that each community is the
set of vertices with a common label:
$ C(cal(l)) = {i in V | cal(l)_i = cal(l)} $

However, labels used for evaluation need not be produced by a community
detection algorithm. In many datasets, labels are supplied externally and
correspond to attributes of the represented entities. For example, in a citation
network where each vertex represents an academic article, labels may indicate
the article's author, publication venue, research topic, or field. Such labels
are not guaranteed to be reflected in the graph topology, but in many real-world
networks they correlate with meaningful structural organization.

There is therefore a methodological question about which reference partition
should be used when evaluating label or community preservation in a layout.
Using external labels may evaluate layout algorithms according to information
which is not necessarily encoded in the graph structure available to them. On
the other hand, using labels produced by a community detection algorithm imposes
the assumptions of that algorithm onto the evaluation. These assumptions may
differ significantly between algorithms, and may even overlap with the machinery
used by some layout methods. In such cases, the evaluation may favor layouts
which agree with the assumptions of the detection algorithm rather than layouts
which more generally reveal meaningful structure.

For this reason, the purpose of the following metrics is not to measure
agreement with a particular community detection algorithm, but to assess whether
semantically meaningful labels present in the data are reflected in the
visualization. We therefore use external labels when available. Although such
labels are not guaranteed to correspond to structural communities in the graph,
they are independent of both the layout and evaluation procedures, and thus
provide a common reference against which different layout algorithms may be
compared.

Whenever a metric uses external labels, we also compute the corresponding
graph-based score when possible. This allows layout results to be interpreted
relative to the extent to which the same labels are encoded in the graph itself.
Thus, we compare the graph and the layout according to their ability to reflect
the same external partition under the same evaluation criterion.


==== Silhouette Score
For a given layout, in order for labeled groups to be visually distinct,
vertices with the same label should be placed compactly in the layout, 
with good relative separation between groups, thus creating visually distinguishable 
point clusters.

One metric to measure this criterion is the commonly used silhouette score @rousseeuw1987.
It is defined for a single vertex as the normalized difference of
distances between the points in its own label group and the points in
the closest neighboring group.

Let $cal(L)$ be the set of unique labels such that for every $i in V => cal(l)_i in cal(L)$.
Then, let $cal(C)$ be the set of all labeled communities:
$ cal(C) = {C(cal(l)) | cal(l) in cal(L)} $

We define the label group of vertex $i$ as
$ C_i = C(cal(l)_i) = {j in V | cal(l)_j = cal(l)_i}. $

We calculate its average distance to points in its own community as
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
where $|C_i| = 1$, so in our implementation we set the score for 
these vertices as $0$.

The global score is given as the mean of the scores for each point
$ "Sil"(X ; cal(l)) = 1/n sum_(i in V) "Sil"_i (X ; cal(l)) $
A large positive global score indicates compact and well-separated 
labeled communities in the layout.
Conversely, negative scores indicate that vertices are, on average, 
closer to other labeled groups than to their assigned group.

As mentioned, we can use a similar metric in order to test whether the
external labels reflect the graph topology: given the external labels
we define the graph-theoretic silhouette score given by
$ a^*_i = 1/(|C_i| - 1) sum_(j in C_i , i eq.not j) d_(i j) $
$ b^*_i = min_(C in cal(C), C eq.not C_i) 1/(|C|) sum_(j in C) d_(i j) $
$ "Sil"^*_i (G ; cal(l)) = (b^*_i - a^*_i) / max{a^*_i, b^*_i} $
Here $d_(i j)$ denotes the graph-theoretic distances like described previously.
Then the global score $"Sil"^* (G ; cal(l))$ is defined as above.

==== #knn Classification Accuracy
Another metric corresponding to the same criterion is to measure the accuracy
of using a #knn based classifier in its recovery of the initial labels, an
approach used to quantify embedding quality since the original #tsne paper
@tsne and in more recent graph-embedding evaluations @bohm2025.
In contrast to Silhouette score, which is a global score of separation
and compactness, this metric measures instead local label-homogeneity.
This is based on the principle that a label-preserving layout should place each
vertex near other vertices with the same label.
Therefore, predicting the label of a vertex from the labels of its
nearest neighbors should be accurate.

A classifier is defined as a procedure which assigns each vertex $i$ to a
predicted label $hat(cal(l))_i$, without prior knowledge of $cal(l)_i$. 
A classifier of the form we discuss, however, has access to the labels 
$cal(l)_j$ with $j eq.not i$.

Specifically, let $cal(N)_X^k (i)$ denote the set of $k$ nearest neighbors of
$vec(x)_i$ in the layout $X$. Equivalently, this defines the #knn graph $G_X^k$ of
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
$ s_i (cal(l)) = sum_(j in cal(N)^k_X (i) inter C(cal(l))) f(||vec(x)_i - vec(x)_j||) $
Here $f$ is a decreasing function of distance, so that larger distances
correspond to smaller contributions. The choice of $f(d) = 1$ reduces to
the simple voting scheme, while a common distance-weighted alternative is 
$f(d) = 1 slash d$.

In all cases, the classifier is defined by selecting the label with
the highest score:
$ hat(cal(l))_i = argmax_(cal(l) in cal(L)) s_i (cal(l)) $

The #knn classification accuracy is then defined as the 
fraction of vertices whose predicted labels match their
true labels.
$ "Acc"^k (X ; cal(l)) = 1/n sum_(i in V) [hat(cal(l))_i = cal(l)_i] $
Here $[ dot ]$ denotes the Iverson bracket.
A high value of $"Acc"^k (X ; cal(l))$ indicates that the local
neighborhoods in the layout are label-homogeneous enough to 
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
which the label can be recovered, so such vertices are excluded from this
graph-based baseline.

This graph-based score measures how strongly the external labels are reflected
in the one-hop topology of the graph. It therefore provides a baseline for
interpreting the layout-based classification accuracy: a low layout score is less
meaningful when the same labels are weakly recoverable from the graph itself.

=== Evaluation of Visual Quality
For any layout to be useful, it is required for it to be visually interpretable.
Regardless of how well a layout preserves some graph-level feature, if its
features are unclear or obscured, it loses its ability to meaningfully improve
understanding of the underlying network by visual examination. These features
can be obscured in a number of ways that make interpretation confusing. Point
crowding and excessive edge crossings are among the most common artifacts which
decrease the readability of a network layout, and are the artifacts this section
will focus on.

It should be clear that feature preservation and visual clarity are separate,
although not unrelated, qualities. High visual clarity scores do not imply high
feature preservation scores, and high feature preservation scores do not imply
high visual clarity scores. In some cases, they might even be in tension: for
example, point crowding scores penalize vertices placed too close together,
while adjacency and community preservation scores may reward similar behavior in
some cases. A good layout therefore must be at the same time feature-preserving
and visually interpretable in order to be useful for network analysis.

For this reason, visual quality metrics should be interpreted as complementary
to the feature-preservation metrics defined above. They do not measure whether
the layout correctly represents a particular structural property of the graph;
rather, they measure whether the drawing is readable enough for the represented
structure to be visually inspected.

We therefore focus on point crowding and edge crossings as representative
visual quality metrics, intended to measure common readability artifacts rather
than structural preservation directly.

==== Point Crowding
This artifact appears when some subset of vertices is placed at distances
that are very small compared to the overall scale of the layout. It is closely
related to the crowding problem discussed in the #sne and #tsne literature
@tsne, where a disproportionate number of points are forced close together
near the center of the embedding.
In such cases, the vertices may become difficult to distinguish visually,
and the local substructure among them may be obscured.

Again, the absolute scale of the distances does not matter, only their
relative magnitudes, since one may uniformly scale any layout without
changing its quality. Therefore, a point-crowding metric should compare
local point spacing against a global spatial scale of the layout.

A direct way to measure local spacing is to consider, for each vertex,
the distances to its nearest embedded neighbors. Let $cal(N)^k_X (i)$
denote the set of the $k$ nearest neighbors of $vec(x)_i$ in the layout.
We define the average local $k$-nearest-neighbor distance as

$ d_k (X) = 1/n sum_(i in V) 1/k sum_(j in cal(N)^k_X (i)) ||vec(x)_i - vec(x)_j|| $

This quantity measures the typical local spacing around a vertex.
However, it is scale-dependent. To remove the dependence on the
arbitrary scale of the layout, we normalize it by the global average
pairwise distance
$ overline(d) = 2/(n(n - 1)) sum_(i < j) ||vec(x)_i - vec(x)_j|| $

The normalized $k$-nearest neighbor distance is therefore defined as

$ "NND"_k (X) = (d_k (X)) / overline(d)
  = (1/n sum_(i in V) 1/k sum_(j in cal(N)^k_X (i)) ||vec(x)_i - vec(x)_j||) 
  / (2/(n(n - 1)) sum_(i < j) ||vec(x)_i - vec(x)_j||) $

This metric compares the typical local spacing of the layout to its global
spatial scale. Smaller values indicate stronger point crowding, since the
nearest embedded neighbors of a typical vertex are very close relative to
the overall size of the drawing.
Larger values indicate that vertices are better separated locally and
are therefore easier to distinguish visually.

A closely related graph-drawing aesthetic is node resolution @purchase2002,
which measures whether distinct vertices are sufficiently separated in the
drawing. A common global form of this criterion, used for example by
@ahmed2020gd2, compares the smallest pairwise vertex distance against the
largest pairwise vertex distance:

$ "NR" (X) = (min_(i < j) ||vec(x)_i - vec(x)_j||) / (max_(i < j) ||vec(x)_i - vec(x)_j||) $

This metric is scale-invariant and directly penalizes near-overlapping
vertices. However, because it depends on the single closest pair of vertices,
it behaves as a worst-case overlap detector. 
One nearly coincident pair can make the score very small, even if the rest of 
the layout is visually readable.

For this reason, we use a vertex-wise averaged version of node resolution.
For each vertex, we compare its distance to the nearest other vertex with its
distance to the farthest other vertex:

$ "VNR" (X) = 1/n sum_(i in V) 
  (min_(j eq.not i) ||vec(x)_i - vec(x)_j||) / (max_(j eq.not i) ||vec(x)_i - vec(x)_j||) $

This metric measures the typical local separation of a vertex relative to its
own visual range in the layout. Smaller values indicate that, for many vertices,
the nearest visible neighbor is very close compared with the scale over which
the rest of the layout extends. Larger values indicate better vertex separation.

Unlike the global node-resolution score, $"VNR"$ is not determined by a single
closest pair. It therefore remains sensitive to point crowding while better
reflecting the typical readability of the layout. Like the global version, 
it is invariant under uniform scaling of the layout, since both the numerator
and denominator of each vertex-wise ratio scale by the same factor.

The two metrics capture related but distinct aspects of point crowding.
$"NND"_k$ measures the average spacing inside small local neighborhoods,
whereas $"VNR"$ measures how close the nearest point to a typical vertex
is relative to that vertex's full visual range. We use both as representative
point-crowding metrics: $"NND"_k$ for local neighborhood crowding and $"VNR"$
for node-resolution readability.

In addition to these scalar metrics, we also examine the distribution of 
normalized pairwise vertex distances in the layout. 

We can use either all pairwise distances,
$ {1/overline(d) ||vec(x)_i - vec(x)_j|| : i < j} $
or local spacing values, such as nearest-neighbor distances,
$ {1/overline(d) min_(j eq.not i) ||vec(x)_i - vec(x)_j|| : i in V} $

This distribution can be visualized using histograms, box plots, or 
one-dimensional scatter plots.
These plots are not treated as formal evaluation metrics, since they do not
produce a single scalar score by which layouts can be ordered.
However, they provide useful diagnostic information about the structure
of point spacing in the drawing.
In particular, they make it possible to compare whether one layout contains
a larger lower tail of very small distances, whether its distances are
more evenly distributed, or whether the scalar crowding metrics are being
influenced by a small number of near-overlapping vertices.

==== Edge Crossing
When visualizing graphs, the vertices are usually drawn as points, while the
edges are drawn as curves connecting the corresponding vertex positions. In
many graph layouts, an edge $i ~ j$ is drawn as a straight line segment from
$vec(x)_i$ to $vec(x)_j$. However, this is not always the case, as we will see
in @tne. We therefore consider edges more generally as curve segments in the
layout space, beginning at $vec(x)_i$ and ending at $vec(x)_j$. When two such
edge curves intersect away from their common endpoints, we call this an edge
crossing.

Edge crossings are among the most studied readability artifacts in graph
drawing, and reducing them is a classical aesthetic criterion @purchase2002.
In practical graph layout, however, edge crossings are often unavoidable. First, a
non-planar graph cannot be drawn in two-dimensional Euclidean space without
edge crossings. For planar graphs, there exist algorithms that produce planar
drawings, such as the Tutte embedding @tutte1963. However, such methods are not
sufficient for scientific network visualization in general, since planarity is
only one possible aesthetic criterion and does not guarantee the preservation
of other relevant structural features. A crossing-free drawing may still
produce other visual artifacts, such as point crowding, or may fail to reflect
important graph-level properties.

Moreover, many real-world networks are not planar, and even when a graph is
planar, the layout algorithms considered in this thesis are not designed to
guarantee a planar drawing. For this reason, edge crossings are not treated as
a defect that can always be eliminated, but as a visual artifact whose
frequency should be measured and compared across layouts.

Furthermore, not every edge crossing has the same effect on the visual quality
of a layout. Cognitive studies of graph readability support the general
principle that crossings are not uniformly disruptive: in path-tracing tasks,
Ware et al. @ware2002 found that edge crossings degrade performance more
strongly when they occur along long paths traced through the drawing, in
addition to the separate and dominant effect of path continuity. Motivated by
this context-dependence, we make an analogous distinction based on edge length
rather than path length: crossings between short edges in locally dense
regions may have a limited effect on the interpretation of the global
structure, since their visual impact is confined to a small part of the
drawing. By contrast, we expect that crossings between longer edges are more
disruptive, since they create visual clutter across larger parts of the
layout and make it harder to follow the connectivity between separated
regions.

For this reason we introduce two global metrics for measuring edge crossing
as a negative artifact: the first is the total number of crossings $"Cr"(X)$,
the other is a length-weighted version $"LWCr"(X)$.

Formally, we consider the curve representing edge $e_(i j)$ which joins
vertices $i$ and $j$ as the set of values of a parametric function
$vec(p)_(i j): [0,1] -> RR^d$ with $vec(p)_(i j)(0) = vec(x)_i$ and 
$vec(p)_(i j)(1) = vec(x)_j$ such that $ cal(P)_(i j) = {vec(p)_(i j)(t), t in [0,1]} $

Then for two such edges $e_(i j)$ and $e_(k l)$ and their curves
$cal(P)_(i j)$ and $cal(P)_(k l)$, we say that $e_(i j)$ and $e_(k l)$ cross
iff there exists some $vec(p) in RR^d$ with $vec(p) eq.not vec(x)_i$ and 
$vec(p) eq.not vec(x)_j$, such that $vec(p) in cal(P)_(i j)$ and $vec(p) in cal(P)_(k l)$.
Equivalently, $e_(i j)$ and $e_(k l)$ cross when
$ (cal(P)_(i j) inter cal(P)_(k l) backslash {vec(x)_i, vec(x)_j}) eq.not emptyset $

The number of crossings between $e_(i j)$ and $e_(k l)$ is given as 
$ "cr"(e_(i j), e_(k l)) = |cal(P)_(i j) inter cal(P)_(k l) backslash {vec(x)_i, vec(x)_j}| $
which might be more than $1$ since we are working with curves and not straight lines.

Since the sets $cal(P)_(i j)$ are uncountably infinite, it is useful to work instead with 
the index formulation: the edges $e_(i j)$ and $e_(k l)$ cross iff there exists 
$psi, xi in (0, 1)$ such that
$ vec(p)_(i j)(psi) = vec(p)_(k l)(xi) $
Where root-finding methods can be used to find the intersections.
In practice a bounding-polygon scheme can be used to efficiently check
for possible intersections, then followed by the more expensive root-finding
refinement step.

In the special case where the two curve segments are linear, given by
$ vec(p)_(i j)(psi) = vec(p)_i + vec(v)_(i j) psi
\ vec(p)_(k l)(xi) = vec(p)_k + vec(v)_(k l) xi $
finding the roots is equivalent to solving a 2 dimensional linear system:
$ mat(vec(v)_(i j), -vec(v)_(k l)) mat(psi ; xi) = vec(x)_k - vec(x)_i $
then checking that $psi$ and $xi$ are both in $(0, 1)$.

Following from this fact, an approximate method to find the crossings of two edges 
is often to split the curve segments into "polylines" wherein each curve is represented
as a set of line segments that begin and end at sampled points on the original curve,
and then check for intersections between each pair of line segments in the two polylines.
This is not exactly accurate with regards to the previous problem, but especially as the
number of segments used for each polyline increases, this method becomes a sufficiently good
approximation for the purposes of our metrics.
Note that for a uniform augmentation of $m$ edges into polylines, each with $s$ line segments,
the time complexity to detect all intersections becomes $BigO(s^2 n^2)$.

Having established how individual crossings are detected, we can now define the
two global metrics introduced at the start of this subsection. Let
$e_1, dots, e_m$ denote an arbitrary enumeration of the edges of $G$. The total
number of crossings in the layout $X$ is defined as the sum of $"cr"(dot, dot)$
over all pairs of distinct edges:
$ "Cr"(X) = sum_(1 <= a < b <= m) "cr"(e_a, e_b) $
This counts every crossing between every pair of edges in the drawing, including
the rare case of curved edges crossing more than once. Unlike the classical
graph-drawing convention, which excludes pairs of edges sharing a common
vertex from the crossing count since straight-line edges cannot cross away
from a shared endpoint, we do not exclude such pairs here, since curved edges
incident to the same vertex may still cross one another away from it. Larger
values of $"Cr"(X)$ indicate a more visually cluttered layout, and
$"Cr"(X) = 0$ corresponds to a crossing-free drawing.

As discussed above, however, not all crossings are equally disruptive:
crossings between edges that each span a large portion of the layout tend to be
more disruptive to visual interpretation than crossings confined to a small,
locally dense region. To capture this asymmetry, we define a length-weighted
crossing metric that accounts for the length of the two edges involved in each
crossing.

We take the length of an edge to be the Euclidean distance between its
endpoints,
$ L(e_(i j)) = ||vec(x)_i - vec(x)_j|| $
regardless of the shape of the curve used to draw it, since it is the distance
spanned across the layout, rather than the path taken, that determines how much
of the drawing a crossing on that edge can disrupt. As with the other metrics in
this section, we compare this to the average edge length of the layout,
$ overline(L)(X) = 1/m sum_(e in E) L(e) $
so that the contribution of a crossing is judged relative to the typical edge
length of the drawing rather than to an arbitrary absolute scale.

The length-weighted crossing number is then defined as
$ "LWCr"(X) = 1/overline(L)(X) sum_(1 <= a < b <= m) (L(e_a) + L(e_b))/2 "cr"(e_a, e_b) $
Each crossing is weighted by the average length of the two edges involved,
relative to the average edge length of the layout. When all edges have equal
length, this weight is $1$ for every crossing and $"LWCr"(X)$ reduces exactly
to $"Cr"(X)$. In general, crossings between long edges contribute more to
$"LWCr"(X)$ than crossings between short edges, in agreement with the
observation that the former tend to be more disruptive to visual
interpretation.

Together, $"Cr"(X)$ and $"LWCr"(X)$ provide complementary global views of the
crossing artifact. $"Cr"(X)$ weighs every crossing equally and thus quantifies
the overall visual density due to edge overlap, while $"LWCr"(X)$ emphasizes
crossings that are more likely to obscure the large-scale connectivity of the
graph.

This concludes the set of evaluation metrics used throughout this thesis.
Taken together, the metrics defined in this section span three complementary
views of layout quality. Structural feature preservation ---
$"ADR"(X ; G)$, the neighborhood-recovery scores $R^k$, $P^k$, and $J^k$,
weighted recall $W^k$, and scale-normalized stress $"SNS"(X ; G)$ --- measures
the extent to which the graph's topological and distance information is
reflected in the geometry of the layout. Label-based feature preservation ---
the silhouette score $"Sil"(X ; cal(l))$ and #knn classification accuracy
$"Acc"^k (X ; cal(l))$ --- measures the extent to which externally meaningful
groupings are visually recoverable. Visual quality --- the point-crowding
scores $"NND"_k (X)$ and $"VNR"(X)$, together with the crossing metrics
$"Cr"(X)$ and $"LWCr"(X)$ --- measures the readability of the drawing
independent of any structural correspondence to the graph.

No single metric among these is sufficient on its own, nor is any intended as
a canonical or exhaustive measure of its corresponding criterion. Rather, this
pluralistic set is meant to be applied jointly: a layout method is judged not
by any individual score, but by the pattern of trade-offs it exhibits across
all three categories. This is the basis on which the Twin Neighbor Embedding
method introduced in @tne, and the layout methods it is compared against, are
evaluated in @results.
