# The Intelligence Control Plane

**The Intelligence Control Plane: Optimizing Intelligence per Joule and per
Dollar for Agentic Tasks** studies the efficiency of systems composed from
models, middleware, and machines, including local-to-cloud escalation.

The code lives alongside the existing IPW tooling in this repository. The
research package retains the `ipj` namespace; IPW users keep the `ipw` CLI and
its existing installation. Project information is available on the
[Intelligence Per Watt website](https://www.intelligence-per-watt.ai/).

## Offline artifact

From the repository root, using Python 3.13 and uv:

```sh
uv sync --project ipj-research --frozen --extra reproduce
uv run --project ipj-research --no-sync \
  python ipj-research/studies/icp/fetch_data.py --verify
uv run --project ipj-research --no-sync \
  python ipj-research/studies/icp/reproduce.py --check --figures \
  --output-dir /tmp/icp-reproduction
```

Choose a new output directory for each run. Inputs are bundled, so the verify
step downloads nothing. Dependency installation needs package-index access;
the subsequent analysis uses no provider accounts or private repositories.

The command verifies input hashes, recomputes the 1,085 effective TeX macro
definitions, and compares them with the author-validated paper baseline. It
also regenerates six figures and checks their labels against the supplied
PDFs. A conceptual diagram and a longitudinal figure from the original IPW
study are retained as reference images. The output contains `report.json`,
logs, generated numerical definitions, and figure PDFs.

The artifact includes the source map for all eight figures and fourteen
tables. Literal values and externally cited results retain the author's
validation and source attribution. These checks establish reproducibility
from the retained inputs; they do not execute new benchmark trials.

See the [study README](https://github.com/HazyResearch/intelligence-per-watt/blob/main/ipj-research/studies/icp/README.md)
for coverage and provenance, and the
[package README](https://github.com/HazyResearch/intelligence-per-watt/blob/main/ipj-research/README.md)
for development and optional dependencies.

## Code and validation

`ipj-research/src/ipj/` provides energy integration, accounting, experiment
result ingestion, routing and escalation, orchestration, and statistical
analysis. The shared `ipw` additions provide explicit unknown-price handling
and validated public trace exports. The research environment has its own
lockfile and resolves `ipw` from the sibling source directory.

```sh
uv sync --project ipj-research --frozen --extra dev --extra router --extra reproduce
uv run --project ipj-research --no-sync ipj --help
uv run --project ipj-research --no-sync pytest ipj-research/tests
uv run --project ipj-research --no-sync python ipj-research/examples/synthetic_energy.py
```

The example integrates synthetic power samples and reports 220 joules. Real
measurements additionally need a supported monitor and complete device and
time coverage. Real benchmark runs need the appropriate harness, tasks,
inference endpoints, and optional dependencies described in the package guide.

## Citation

Use this citation for the ICP research. The repository README preserves the
original IPW citation for the measurement toolkit. An ICP arXiv identifier
will be added once assigned.

```bibtex
@misc{saadfalcon2026intelligencecontrolplane,
  title={The Intelligence Control Plane: Optimizing Intelligence per Joule and per Dollar for Agentic Tasks},
  author={Jon Saad-Falcon and Avanika Narayan and Catherine Deng and Ravenor Davion and Elliot Slusky and Amirreza Zeinali and Matthew Hart and James Noraky and Sheng Li and John Hennessy and Azalia Mirhoseini and Christopher Ré},
  year={2026},
  url={https://www.intelligence-per-watt.ai/}
}
```
