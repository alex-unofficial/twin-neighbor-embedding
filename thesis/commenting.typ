#import "config.typ": draft

#let comment(
  initials: "CMT", 
  color: black, 
  numbering: "1",
  inline: false,
  body,
) = if draft{

  counter(initials).step()

  let content = [
    #text(fill: color)[
      #strong(initials + context counter(initials).display(numbering) + ":") #body
    ]
  ]
  
  if inline {content} else {block(content)}

} else {none}

#let alex = comment.with(initials: "AA", color: blue)
#let nikos = comment.with(initials: "NP", color: red)
