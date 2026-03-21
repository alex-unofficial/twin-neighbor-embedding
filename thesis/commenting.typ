#import "config.typ": draft

#let comment(
  initials: "CMT", 
  color: black, 
  numbering: "1",
  body,
) = if draft{
  box[
    #counter(initials).step()
    #text(fill: color)[
      #strong(initials + context counter(initials).display(numbering) + ":") #body
    ]
  ]
} else {none}

#let alex = comment.with(initials: "AA", color: blue)
#let nikos = comment.with(initials: "NP", color: red)
