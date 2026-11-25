# Libtt technical report

`report.typ` is the report, in [Typst](https://typst.app/). `report.pdf` is
the compiled result.

    report.typ        title, abstract, and the list of sections
    style.typ         page style (a plain memorandum, after troff -ms)
    sec/*.typ         sections and subsections; tables.typ and figures.typ
    refs.bib          references
    fonts/            TeX Gyre Termes, Termes Math and Cursor (GUST Font License)
    scripts/          fig_sft.py and fig_rl.py: the training figures, regenerated from the runs' logs
    results/          training-validation.md: the training sections' experiments, results and logs

## Building

    ./build.sh

or `typst compile --font-path fonts report.typ`. The figures use the
`cetz` package, which Typst downloads on first use.

On typst.app: create an empty project, upload this folder's contents keeping
the `sec/` and `fonts/` directories, and open `report.typ`. Fonts uploaded to
a project are found automatically.

## arXiv

arXiv does not take Typst source, so the submission is the PDF alone
("PDF-only" on the upload page). `build.sh` writes PDF/A-2b with every font
embedded as an outline font; there is no Type 3 font, bitmap, or JavaScript in
it. Margins are at least one inch and the body is 11 point, which is what
arXiv asks of a PDF.

## Sources

The implementation descriptions include libtt through commit dc73ad4356b454df389d9425e5f8f4250c6ef740 (November 13, 2026). Benchmark, test-count and source-size snapshots retain their stated dates and versions. Numbers
with a pull-request number come from that pull request's description in the
git history. The serving benchmark tables are from https://relativity.cc/models/
as read on November 23, 2026 (libtt 0.2.2 for both SGLang-JAX and
SGLang-TorchTPU). The training sections' fine-tuning and reinforcement-learning results (November 24–25, 2026; libtt 0.2.2, the release build main 2c68912a for the second pass, and 0.2.1 for the original RL run) are documented with their logs, scripts and checks in results/training-validation.md. Hardware facts are from Tenstorrent's documentation.


## License

The report (text, figures, tables and `report.pdf`) is licensed under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/);
see `LICENSE`. The fonts in `fonts/` are under the GUST Font License; see
`fonts/GUST-FONT-LICENSE.txt`.
