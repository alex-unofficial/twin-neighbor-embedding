#import "config.typ": font, images-dir, bib-file

#let thesis(
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
    font: font,
    size: 12pt,
  )

  show heading.where(level: 1): set text(size: 22pt)
  show heading.where(level: 1): set block(below: 2.5em)
  show heading.where(level: 1): it => pagebreak(weak: true) + it

  show heading.where(level: 2): set text(size: 15pt)
  show heading.where(level: 2): set block(
    above: 2em,
    below: 1em,
  )

  set page(numbering: "i")

  show title: set par(justify: false)
  show title: set text(size: 22pt, weight: "bold")
  show title: set align(center)
  show title: set block(below: 1.5em)

  page(numbering: none)[
    #align(center+top)[
      #image(
        images-dir + "auth-logo.png",
        height: 5cm,
      )

      #v(1cm)

      #title()

      #text(size: 17pt)[#author.name]
      
      #text(size: 15pt)[
        #set par(leading: 0.5em)
        _Faculty Supervisor_ \
        #author.supervisor 
      ]

      #v(1em)

      #text(size: 15pt)[#author.affil]
    ]
    

    #align(center+bottom)[
      #text(size: 15pt)[
        2026
      ]
    ]
  ]

  [
    = Abstract <abstract>
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
  set math.equation(numbering: "(1)")

  show link: underline

  doc

  bibliography(
    bib-file,
    style: "ieee",
  )
}

#let draft(
  author: (
    name: [],
    affil: [],
    supervisor: [],
  ),
  abstract: [],
  doc,
)= {
  set page(
    paper: "a4",
    margin: (x: 2cm, y: 2cm),
    numbering: "1",
    header: text(
      size: 9pt,
      align(horizon)[#context document.title #h(1fr) *DRAFT*]
    ),
    footer: text(
      size: 9pt,
      align(horizon)[*DRAFT* #h(1fr) #context counter(page).display()]
    ),
  )
  set par(
    justify: true,
    leading: 0.6em,
    spacing: 1.2em,
  )
  set text(
    font: font,
    size: 12pt,
  )

  show title: set par(justify: false)
  show title: set text(size: 20pt)
  show title: set align(center)
  show title: set block(below: 1.2em)

  [
    #align(center)[
      #title()

      #text(size: 15pt)[
        #grid(
          columns: (1fr, 1fr),
          gutter: 0.5em,
          [#author.name], [#author.supervisor],
          [], [_Faculty Supervisor_],
        )

        #author.affil
      ]
    ]

    #block(
      inset: (x: 1cm)
    )[
      #set text(size: 10pt)
      #set par(leading: 0.5em)
      #show heading: set text(size: 11pt)

      = Abstract <abstract>
      #abstract

    ]
  ]
  set heading(numbering: "1.")
  set math.equation(numbering: "(1)")

  show link: underline

  doc

  pagebreak(weak: true)
  bibliography(
    bib-file,
    style: "ieee",
  )
}
