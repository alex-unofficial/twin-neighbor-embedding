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
in the layout $X$. Neighbor preservation is the principal embedding criterion in 
#sgtsne @pitsianis2019SpacelandEmbedding, which adapts the stochastic neighbor embedding 
framework to graph-structured data.
#alex(inline: true)[
  is this statement correct? would it be more appropriate to say
  that it is the counterpart to #sne and mention #sgtsne later?
]

However #{sne}-based vertex layout suffers from undesirable artifacts that obscure
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

