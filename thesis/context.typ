#import "commenting.typ": *

Network diagrams have been used for centuries in analysing complex relationships, 
even as early as the 14th century @knuth2013combinatronics, 
before the scientific treatment of graph theory was even developed.

Some of the earliest work in graph theory was related to proving properties of 
graphs related to drawing, such as planarity, the ability to construct a drawing
of a graph in the euclidean plane without intersecting edges. This property
was studied by Leonhard Euler (often credited as originating graph theory as 
a scientific topic of study) in the context of convex polyhedra.

Previous to the development of computer graphics, graph drawings in the form of 
node--link diagrams were created by manually placing each vertex and connecting 
them with lines or arrows (depending on the type of graph). 
In most cases the "layout algorithm" used in these drawings would be using 
the intuition of the artist in drawing related vertices closer together, 
or arranging them in a visually appealing or intuitive manner. 
However more sophisticated layout methods were occasionally used such as
the circular layouts of Ramon Llull or the top-down arrangement of medieval
family-tree drawings.

#alex[Too much historical context?]

Since the 20th century, the development of computer graphics as well as the
need to represent larger, more complex networks created the demand for 
automatic layout generation methods. Such methods are vastly varied in
methodology and objective, depending on the type of input graph, the 
specific use-case, and the scientific intent of the user.

We focus our scope to scientific network visualization of general
undirected sparse networks. Our aim is towards analysis of networks
for exploratory study and visual confirmation of community structures
and relationships in data. Consequently we will not consider common
and important layout methods such as grid-layout methods (useful in 
arranging electronic components for VLSI and PCB layouts) or tree layout
methods (useful in the specific case of tree-structured networks).
Additionally we will not consider artificially confined layouts such as
arc-diagrams or circular layouts.
