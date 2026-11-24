// Page style for the libtt report: a plain single-column memorandum in the
// manner of troff -ms. Times-like text, Courier-like program text, numbered
// headings at body size, ruled tables, page numbers at the head of the page.

#let body-font = ("TeX Gyre Termes", "Libertinus Serif")
#let math-font = ("TeX Gyre Termes Math", "New Computer Modern Math")
#let mono-font = ("TeX Gyre Cursor", "DejaVu Sans Mono")

// Names that must not break across lines.
#let cpp = box[C++]

// A ruled table in the style of tbl: a rule above, one under the heading
// row, one below. `head` is the number of cells in the heading row.
#let ruled(columns: 2, align: left, head: none, ..cells) = {
  let n = if head == none {
    if type(columns) == int { columns } else { columns.len() }
  } else { head }
  let all = cells.pos()
  table(
    columns: columns,
    align: align,
    table.hline(stroke: 0.8pt),
    table.header(..all.slice(0, n)),
    table.hline(stroke: 0.5pt),
    ..all.slice(n),
    table.hline(stroke: 0.8pt),
  )
}

#let memo(title: none, author: none, date: none, abstract: none, keywords: (), body) = {
  set document(title: title, author: author, keywords: keywords, date: date)
  set page(
    paper: "us-letter",
    margin: (x: 1.25in, top: 1.15in, bottom: 1.1in),
    header: context {
      let n = counter(page).get().first()
      if n > 1 { align(center, text(size: 10pt)[\- #n \-]) }
    },
  )
  set text(font: body-font, size: 11pt, lang: "en", hyphenate: true)
  set par(justify: true, leading: 0.56em, spacing: 0.95em, first-line-indent: 1.6em)
  show math.equation: set text(font: math-font)
  show raw: set text(font: mono-font, size: 1.12em, ligatures: false, features: (liga: 0, clig: 0, dlig: 0))
  show raw.where(block: true): it => block(
    inset: (left: 1.6em), above: 1em, below: 1em, text(size: 0.88em, it),
  )
  set heading(numbering: "1.1.")
  show heading: set text(size: 11pt, weight: "bold")
  show heading: set block(above: 1.5em, below: 0.85em)
  show heading.where(level: 3): set text(style: "italic", weight: "regular")
  set list(indent: 1.6em, body-indent: 0.6em, marker: [•])
  set enum(indent: 1.6em, body-indent: 0.6em)
  set figure(gap: 0.9em)
  show figure: set block(above: 1.5em, below: 1.5em)
  show figure.caption: it => {
    set text(size: 10pt)
    set par(justify: false)
    [*#it.supplement #context it.counter.display(it.numbering).* #it.body]
  }
  show figure.where(kind: table): set figure.caption(position: top)
  set table(stroke: none, inset: (x: 0.7em, y: 0.36em))
  show table: set text(size: 10pt)
  show table: set par(justify: false)
  set footnote.entry(separator: line(length: 2in, stroke: 0.5pt))
  set bibliography(title: "References", style: "ieee")
  show bibliography: set text(size: 10pt)
  show bibliography: set par(justify: false)

  align(center)[
    #v(0.3in)
    #text(size: 13pt, weight: "bold", title)
    #v(1.1em)
    #text(style: "italic", author)
    #v(2.2em)
    #text(style: "normal")[ABSTRACT]
    #v(0.2em)
  ]
  pad(x: 0.45in)[
    #set par(first-line-indent: 0pt)
    #abstract

    #v(0.6em)
    #date.display("[month repr:long] [day padding:none], [year]")
  ]
  v(1.2em)
  body
}
