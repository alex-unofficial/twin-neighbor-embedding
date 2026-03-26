#import "commenting.typ": *
#import "@preview/run-liners:0.1.0": *

A simple undirected graph $G$ is the tuple $(V,E)$ consisting of a set of 
vertices (nodes) $V$, and a set of edges (links) $E subset.eq V times V$. 
Denote by $|V|$ the number of vertices and by $|E|$ the number of edges. 
If two vertices $v, u in V$ form an edge $(v,u) in E$ then we call $v$ and $u$ 
neighboring or adjacent, and we write $v ~ u$.
A vertex $v$ is incident to an edge $e$ if it is part of $e$.
Throughout this work we shall use the terms network and graph interchangeably.

Network visualization conventionally is the task of assigning each vertex
$v in V$ a position in some surface $Sigma$. 
Then the edges may be drawn as lines or curves (not strictly in $Sigma$) 
connecting their incident vertices. 
As a post-processing step, additional network properties and features may 
be overlaid on top of the layout as functions defined on the vertices or edges.

Consider the graph $G = (V,E)$.
Each vertex $v in V$ is mapped to a unique point $x(v) in Sigma$.
The set $X = {x(v) | v in V}$ is called the vertex layout or embedding
#footnote[
  In topological graph theory an embedding strictly is a drawing of the vertices 
  and edges in a way such that the edges may intersect only at their ends in 
  the target embedding space. 
  In this thesis embedding and layout are used interchangeably, in all cases meaning a layout.
]
and is determined by a network layout algorithm.


The surface $Sigma$ is called the embedding space and is commonly $RR^2$,
but sometimes $RR^3$ or even restricted to $SS^1$ or $SS^2$
@WS @miller2023 @lu2019DoublyStochastic.
For the purposes of visualization $Sigma$ is low-dimensional.

Existing software packages for scientific network visualization differ
primarily in three aspects:
#run-in-enum(
  numbering-pattern: "(i)",
  [the choice of embedding space $Sigma$],
  [the choice of embedding criteria and layout algorithm],
  [post-processing utilities and overlays]
).
The analysis in this thesis is primarily concerned with (ii), and from this
point forward it will be assumed that $Sigma eq.triple RR^d$ (normally $d=2$) 
and overlays will be used only to highlight structural aspects of the network 
useful in the visual understanding and qualitative assessment of the layout.

Constructing such embeddings is inherently challenging. 
A low-dimensional layout must represent a complex network topology while preserving 
structural relationships that are important for visual interpretation. 
Because many structural constraints cannot be simultaneously satisfied in a 
two-dimensional embedding, layout algorithms inevitably introduce distortions 
whose nature depends on the chosen embedding objective.
