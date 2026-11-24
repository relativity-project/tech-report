#!/bin/sh
# Build report.pdf from report.typ. Needs Typst 0.15 or newer on PATH.
# The fonts in fonts/ are embedded in the PDF (as subsets).
set -e
cd "$(dirname "$0")"
typst compile --font-path fonts --pdf-standard a-2b report.typ report.pdf
