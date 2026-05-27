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
$i,j in V$ to be preserved by their point representations $x_i,x_j in RR^d$.
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
Consider an (optionally weighted) graph $G = (V,E,w)$ with $n$ vertices and a layout
$X in RR^(n times d)$. A layout metric is a rule $f$ which assigns
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

=== Visual Quality Metrics
