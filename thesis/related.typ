#import "macros.typ": *
#import "commenting.typ": *

Various layout algorithms exist with differing criteria for determining the spatial
configuration $X$. Some of these methods and criteria will be discussed in more detail
in @conventional and @sota.

A common criterion is to preserve the pairwise geodesic distances of vertices on $G$
proportionally in the layout $X$. Yet it is impractical to rigidly impose all 
geodesic distance pairs be preserved in equal measures, considering the confined 
low-dimensional embedding space does not allow for the variety, irregularity, and
uncertainty in sparse data networks. 
We prioritize that adjacent vertices in network $G$ are placed in close proximity
in the layout $X$. This is the Stochastic Neighbor Embedding (#sne) @hinton2002sne 
principle applied to network topology.

However #{sne}-based layouts suffer from undesirable artifacts that obscure
structural relationships and hinder accurate interpretation of the data. We identify
three recurring issues:
#enum(numbering: "(a)")[
  vertex crowding: vertices with small graph-theoretic distance or strong edge weights
  are often placed excessively close in the layout, leading to visual crowding 
  and partial collapse of local substructure.
][
  erroneous clustering: certain vertices or groups of vertices are erroneously placed 
  close together, forming hallucinatory clusters that suggest relationships not present 
  in the data.
][
  excessive long-edge crossings: conventional layout methods do not explicitly encode 
  geometric edge placement, which often results in excessive crossings of long edges.
] 
#tsne @tsne was largely successful in alleviating the crowding issue
compared to classical #sne for dimensionality reduction of point-cloud data.
#sgtsne @pitsianis2019SpacelandEmbedding, which adapts #tsne for sparse networks,
has achieved good performance, while still exhibiting some of the limitations
of #sne methods.

In an effort to address the remaining issues of #sgtsne for sparse
networks we developed the Twin Neighbor Embedding (#tne) method and the #twin
software package as part of yet unpublished research with
Dimitris Floros, Nikos Pitsianis, and Xiaobai Sun @twinPaperUnpublished.
