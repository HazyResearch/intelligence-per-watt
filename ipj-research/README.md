# Intelligence Control Plane research

This directory contains the ICP measurement, orchestration, and analysis code
inside the Intelligence Per Watt repository. It preserves the `ipj` Python
namespace and CLI while depending explicitly on the sibling `ipw` package.

The package contains measurement and accounting, Harbor result analysis,
orchestration, routing, statistics, and optional training modules. The
[ICP study artifact](studies/icp/README.md) provides an offline reproduction
entry point with bundled inputs and checksums.

From the repository root:

```sh
uv sync --project ipj-research --frozen --extra dev --extra router --extra reproduce
uv run --project ipj-research --no-sync ipj --help
uv run --project ipj-research --no-sync pytest ipj-research/tests
uv run --project ipj-research --no-sync python ipj-research/examples/synthetic_energy.py
```

The `orchestrate` extra enables optional provider/terminal-bench integrations;
`train` enables the optional training code. Installing the default package or
reproducing stored numerical inputs does not require model weights, API keys,
Docker, or an accelerator. Actual benchmark execution additionally requires a
compatible external Harbor installation and the selected inference endpoints.

Existing IPW installation and commands are unchanged. Research dependencies
have their own lock and do not modify the existing IPW environment.
