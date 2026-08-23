# Manual conventions
`status: decided`

- This is the **public** trioron manual: `manual/` in the repo, MkDocs Material, KaTeX math, tutorial chapters as **executed Jupyter notebooks** (downloadable = the reader's scratchpad). Serve: `mkdocs serve` from the repo root.
- Textbook voice: each chapter builds one idea; every step carries its formula and its *why*; failures are shown and explained, not edited out.
- Every page opens with `status:` — `final` / `draft` / `skeleton`.
- Notebook pages are committed **executed** (outputs in). Re-execute after any related substrate change: `python3 -m jupyter nbconvert --to notebook --execute --inplace manual/book/*.ipynb`.
- API reference is generated from docstrings (mkdocstrings). Improve the docstring in code, never hand-edit `manual/reference/` beyond the `:::` directives.
- Internal docs (spec, HANDOFF, TRIORON_MANUAL) are **not** part of this site and must not be linked from it.
- The Aldarof game manual is separate (`~/aldarof/wiki/`) and owned by the game side.
