#import "commenting.typ": *
#import "macros.typ": *

Various software packages exist for the practical embedding of networks.

We briefly present a set of practical network layout algorithms based on the
principles described in @conventional. The selected methods were chosen to
represent a broad range of layout paradigms in terms of mathematical
formulation, layout objective, and computational scope.

These algorithms will later be used in the experimental comparison with our
method #twin.

==== Fruchterman-Reingold
The classical Fruchterman-Reingold (FR) method @fruchterman1991 is a
force-directed spring-electrical layout algorithm described in @conventional.
It seeks layouts with approximately uniform edge lengths while maintaining
global vertex separation through repulsive forces. We use the _NetworkX_
@hagberg2008networkx implementation via the `spring_layout` method.

==== Yifan Hu
The Yifan Hu method @yifanhu2006layout is a scalable force-directed layout
algorithm that combines a multilevel strategy with Barnes-Hut approximation of
repulsive forces. It belongs to the spring-electrical family, but is designed
specifically for efficient layout of large graphs. We use the _GraphViz_
@gansner2003graphviz implementation `sfdp`.

==== ForceAtlas2
ForceAtlas2 @jacomy2014fa2 is a force-directed layout method with a force model
that differs from classical FR and is designed for interactive exploratory
network visualization. It is the default layout algorithm in the widely used
network visualization software _Gephi_ @bastian2009gephi, and often produces
visually effective general-purpose layouts across a broad range of networks.
Its optional "LinLog" mode is inspired by Noack's LinLog model and tends to
improve cluster separation, although it is not a direct implementation of the
original LinLog energy. We use the implementation in the `fa2` Python package.

==== Kamada-Kawai
The classical Kamada-Kawai method @kamadakawai1989 is a stress-based layout
method that aims to preserve graph-theoretic distances between vertex pairs.
It is generally best suited to small and medium-sized networks, where the
all-pairs distance model remains computationally practical. We use the
_NetworkX_ @hagberg2008networkx implementation via the `kamada_kawai_layout`
method.

==== Stress Majorization (MDS-style)
The _GraphViz_ @gansner2003graphviz engine `neato` implements a stress-based
layout approach closely related to multidimensional scaling, using stress
majorization @gansner2004 for more stable optimization of the stress objective.
Compared to classical Kamada-Kawai, this typically yields more robust
convergence behavior and better practical layouts for larger graphs.

==== Spectral Layout
Spectral layout, as described in @conventional, uses eigenvectors of the graph
Laplacian to embed the graph in Euclidean space. It emphasizes adjacency
preservation and tends to produce smooth, globally structured layouts, but does
not explicitly optimize a force or stress objective. We use a custom
Python implementation which computes the $d$ dominant eigenvectors of
the graph Laplacian.

==== #sgtsnepi
This method is described in detail in @conventional. It is a stochastic graph
embedding method based on the #tsne probability-matching formulation, using a
sparse graph-derived target distribution together with Student-$t$
low-dimensional affinities. We use the `sgtsnepi` Python package, which wraps
the reference #sgtsnepi implementation.

#alex[
  I will create a table summary for each method comparing
  Formulation,
  Primary objective,
  Local neighborhood preservation,
  Global structure preservation,
  Cluster separation tendency,
  Scalability,
  and Implementation used
]
