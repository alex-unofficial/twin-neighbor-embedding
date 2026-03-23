#import "@preview/run-liners:0.1.0": *

This thesis aims to address challenges in existing methods for the visual
analysis of networks by introducing the Twin Neighbor Embedding method, 
a simple pre-embedding transform designed to improve the quality
of existing network layout algorithms. 
Network representations play a central role in scientific data analysis, as many 
real-world systems and relationships are naturally modeled as networks.
Network visualization is therefore an integral component of network analysis, 
as spatial layouts can reveal structural properties of the underlying 
network in an intuitive and interpretable manner.
However, existing network layout methods exhibit key limitations that can
obscure structural relationships and hinder accurate interpretation of the data.
We identify three recurring issues which the Twin Neighbor Embedding method aims to 
address:
#run-in-enum(
  numbering-pattern: "(a)",
  [vertices with small graph-theoretic distance or strong edge weights
  are often placed excessively close in the layout, leading to visual crowding 
  and partial collapse of local substructure],
  [certain vertices or groups of vertices are erroneously placed close together, 
  forming hallucinatory clusters that suggest relationships not present in the data],
  [conventional layout methods do not explicitly encode geometric edge placement, 
  which often results in excessive crossings of long edges],
).
The Twin Neighbor Embedding method improves on these limitations by encoding
vertex adjacency, edge adjacency, and vertex--edge incidence relationships 
in a unified augmented matrix used as a pre-layout transform. This representation
is then supplied to standard layout algorithms in place of the traditional
adjacency matrix.
Despite its conceptual simplicity, the proposed transform improves layout quality
across a variety of real-world and synthetic networks, as demonstrated through
visual analysis and quantitative evaluation of neighborhood preservation,
classification accuracy, vertex crowding and edge crossings.
