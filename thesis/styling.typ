#let styling(
  author: (
    name: [],
    affil: [],
    supervisor: [],
  ),
  abstract: [],
  doc,
) = {
  set page(
    paper: "a4",
    margin: (x: 3cm, y: 3cm),
  )
  set par(
    justify: true,
    first-line-indent: 1cm,
    leading: 1em,
    spacing: 1em,
  )
  set text(
    font: "New Computer Modern",
    size: 12pt,
  )

  show heading.where(level: 1): set text(size: 22pt)
  show heading.where(level: 1): set block(below: 3em)
  show heading.where(level: 1): it => pagebreak(weak: true) + it

  show heading.where(level: 2): set text(size: 15pt)
  show heading.where(level: 2): set block(
    above: 2em,
    below: 1em,
  )


  set page(numbering: "i")

  show title: set text(size: 22pt, weight: "bold")
  show title: set align(center)
  show title: set block(below: 1.5em)

  page(numbering: none)[
    #align(center+top)[
      #image(
        "pictures/auth-logo.png",
        height: 5cm,
      )

      #text(size: 17pt)[Aristotle University of Thessaloniki]

      #v(2cm)

      #title()

      #text(size: 17pt)[#author.name]

      #text(size: 15pt)[#author.affil]
    ]
    
    #align(center+horizon)[
      #text(size: 15pt)[
        *Faculty Supervisor:* \
        #author.supervisor
      ]
    ]

    #align(center+bottom)[
      #text(size: 15pt)[
        2026
      ]
    ]
  ]

  [
    = Abstract

    #abstract
  ]

  show outline.entry.where(level: 1): set block(above: 2em)
  show outline.entry.where(level: 1): strong

  [
    #outline()
  ]

  set page(numbering: "1")
  counter(page).update(1)

  set heading(numbering: "1.")

  doc
}
