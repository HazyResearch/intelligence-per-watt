# ICP offline study artifact

This artifact recomputes the supplied, author-validated ICP paper values in an
isolated output directory. All analysis inputs are bundled. It does not call
inference providers, download benchmark tasks, or rerun hardware measurements.

## Run

From the repository root:

```sh
uv sync --project ipj-research --frozen --extra reproduce
uv run --project ipj-research --no-sync \
  python ipj-research/studies/icp/fetch_data.py --verify
uv run --project ipj-research --no-sync \
  python ipj-research/studies/icp/reproduce.py --check --figures \
  --output-dir /tmp/icp-reproduction
```

Use a new output path each time. The command refuses to overwrite an existing
directory or write inside this study directory. It copies the pipeline to the
output path before running it. Omit `--figures` for numerical definitions only.
Comparison with the paper baseline is always performed; `--check` makes that
intent explicit. A mismatch, altered input, missing figure, or failed generator
returns a nonzero exit code.

`report.json` records the input verification, macro comparison, figure label
comparison, and coverage. Generator output is retained in `logs/`, numerical
definitions in `pipeline/paper/`, and figures in
`pipeline/paper/figures_and_tables/`.

## Coverage

- All **1,085 effective zero-argument TeX macro definitions** are compared
  exactly after removing comments and normalizing whitespace. This includes
  unused definitions; the macro count is not a count of independent claims.
- Six figure generators are included: overview, policy families, cost/energy
  scatter, task flow, historical cascade frontier, and Apple silicon energy.
  Figure PDF text is compared after Unicode and whitespace normalization.
  Visual layout is reviewed separately; matching text does not establish
  identical rendering or independent measurement validity.
- `fig1_ai_stack.png` is retained conceptual artwork.
- `fig2_ipj_longitudinal.png` is a retained figure from the published IPW study,
  cited by the ICP paper. Its original measurement campaign and generator are
  outside this artifact.
- [provenance/paper-items.json](provenance/paper-items.json) maps all eight
  figures and fourteen tables to generators, inputs, paper macro dependencies,
  reference assets, and literal source text. Inline literals and externally
  cited results use the author's validation; they are not presented as new
  independent computations.

## Provenance and interpretation

The author confirmed on 2026-10-24 that the paper's numbers had already been
validated and authorized their use as the completed release baseline.
`expected/ipj-numbers.tex` freezes that baseline. The retained pipeline's older
`target` filenames and internal status fields document its development history;
they have not been relabeled as raw trials. Some values are retained inputs or
legacy definitions, while others are computed from those inputs. Successful
recomputation establishes consistency of this bundle, not independent
replication of the source campaigns.

`manifest.json` pins every bundled analysis input, expected definition file,
reference image, and paper-item mapping with SHA-256. The exported source ledger
is in [provenance/source-files.json](provenance/source-files.json). Source code
uses the repository's Apache-2.0 license. Retained paper artwork and results are
attributed to the ICP authors and the original IPW study where indicated.

The cascade inputs preserve their historical per-task cost reconstruction and
mean-cost caveats. The Apple input preserves the distinction between joules
per query and TerminalBench joules per task. Electricity rates, prices, task
counts, and uncertainty conventions remain frozen to the supplied study.
No current tariff lookup or replacement benchmark data is performed.

The reference images remain unchanged. Portability adaptations only relocate
input paths, use a noninteractive plotting backend, and remove an operational
machine address from a metadata annotation. They do not change numerical inputs.
