#import "config.typ": draft
#import "commenting.typ": *
#import "styling.typ"

#set document(title: [
  Twin Neighbor Embedding: A Pre-Embedding Transform for
  Improved Network Visualization.
])

#let style = if draft {styling.draft} else {styling.thesis}
#show: style.with(
  author: (
    name: [Alexandros Athanasiadis],
    affil: [
      Department of Electrical and Computer Engineering \
      Aristotle University of Thessaloniki
    ],
    supervisor: [Nikos Pitsianis]
  ),
  abstract: include "abstract.typ",
)

= Introduction <intro>
#include "introduction.typ"

= Background <background>
#include "background.typ"

= Layout evaluation and metrics <metrics>
#alex[Description of visual examination of layouts and metrics for layout performance]

= The Twin Neighbor Embedding method <tne>
#alex[The method, how and why it works.]

= Results <results>
#alex[Including visualization results as well as computed metrics]

= The TWIN software package and GUI <software>
#alex[Brief description of the Software, the parameters and the GUI]

= Conclusion and Limitations <conclusion>
#alex[What the method improves on, where it doesn't, and how it may be improved in the future.]
