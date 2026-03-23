#import "commenting.typ": *

Network diagrams have been used for centuries to analyze complex relationships, 
appearing as early as the 14th century @knuth2013combinatorics,
long before the formal development of graph theory.

Early work in graph theory was closely related to properties of drawings of
graphs, such as planarity --- the ability to represent a graph in the Euclidean
plane without intersecting edges. Investigations of polyhedral graphs by Leonhard
Euler, particularly his formula relating vertices, edges, and faces of convex
polyhedra, are often cited among the earliest results connected to graph theory.

Prior to the development of computer graphics, graph drawings in the form of 
node--link diagrams were created by manually placing each vertex and connecting 
them with lines or arrows (depending on the type of graph). 
In most cases the effective "layout algorithm" was simply 
the intuition of the artist drawing the diagram, placing related vertices 
closer together, or arranging them in a visually meaningful manner. 
However, more sophisticated layout methods were occasionally used such as
circular layouts appearing in the combinatorial diagrams of Ramon Llull 
or the top-down arrangement of medieval family-tree drawings.

#alex[Too much historical context?]

While the history of graph drawing spans several centuries, 
the modern field of network visualization largely emerged alongside 
the development of computer graphics in the 20th century.

The need to represent larger, more complex networks created the demand for 
automatic layout generation methods. Such methods vary widely in both
methodology and objectives, depending on the type of input graph, the 
specific use case, and the scientific intent of the user.

We restrict our scope to scientific network visualization of general
undirected sparse networks. Our aim is the analysis of networks
for exploratory study and visual confirmation of community structures
and relationships in data. Consequently we restrict our attention
to unconstrained euclidean layouts of general networks.
Specialized layouts designed for specific application domains
and particular graph classes ---such as grid layouts for VLSI and PCB design,
tree layouts for hierarchical data networks, or spatially constrained
layouts like circular layouts and arc diagrams---
are outside the scope of this work.
