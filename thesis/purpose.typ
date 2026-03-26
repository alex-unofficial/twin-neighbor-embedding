#import "macros.typ": *

The aim of this thesis is to present and analyze the #tne method and
the #twin software and compare it to existing network layout methods in terms of 
visual quality and quantitative evaluation criteria. Additional contributions
are presented in @contributions.

#tne is not a layout algorithm in the traditional sense, 
but instead a pre-layout transform of $G$ which constructs a
supra-network consisting of $G$, its line graph $G_cal(l)$, and additional connections
linking vertices of $G$ to corresponding incident edges represented in $G_cal(l)$. 
This is represented algebraically via a unified augmented matrix which encodes 
vertex and edge adjacency as well as vertex--edge incidence. 
This representation is then used in place of the traditional adjacency 
matrix to extend and improve pre-existing layout methods, primarily
---but not limited to--- #sgtsne. The method is described in detail in @tne.

Despite its conceptual and mathematical simplicity, #tne proves effective in 
mitigating vertex crowding and erroneous clustering across various real-world and 
synthetic networks. Additionally, due to its explicit geometric encoding of edge 
placement it significantly alleviates the problem of excessive edge crossing. 
Various visualization results, as well as quantitative evaluations of the method are
presented in @results.

Our software package #twin allows the user to interact with the Twin Neighbor Embedding
method and compare to traditional vertex embedding in an interactive GUI with 
various tuning parameters. The package is open source and the source code can be
found at #box(link("https://github.com/alex-unofficial/twin-neighbor-embedding")).
The #twin software package is discussed in @software.

Finally, we discuss the limitations of our method as well as potential future
improvements in @conclusion.
