# GDPval × Qwen3.6-27B × OpenHands — smoke-test debugging log

Date: 2026-05-19/20
Hardware: 2 × NVIDIA H200 (lambda box, only GPU 0 used)
Local inference: vLLM 0.21.0, preset `qwen36-27b`
Judge: `gpt-5-nano-2025-08-07` (OpenAI), rubric LLM-as-judge over `rubric_json`
Goal: end-to-end smoke-test the new `gdpval` dataset on Qwen3.6-27B with a real
agent harness, with energy/IPJ/IPW measurement.

## TL;DR

Final 2-task smoke-test: **2 / 2 = 100 % rubric pass** in 42 min, 11.9 kJ GPU
energy, IPJ 8.4e-5, IPW 0.212.

Getting there took eight iterations because four independent bugs were stacked
and the first three were masking the fourth. All four fixes are now in
`intelligence-per-watt/src/ipw/agents/openhands.py`, `gdpval.py`, and the
`openhands` extras list. See "Fixes that survived" at the bottom.

## Setup

```bash
ipw run \
  --agent openhands \
  --preset qwen36-27b \
  --dataset gdpval \
  --dataset-kwargs '{"max_samples": 2}' \
  --max-queries 2 \
  --client-base-url http://localhost:8000 \
  --query-timeout 2400 \
  --estimate-flops \
  --agent-kwargs '{"max_turns": 40}'
```

Tools given to the agent: `terminal` (subprocess mode), `file_editor`. Built-in
SDK tools `finish`/`think` are always available. No `task_tracker`,
no `web_search`, no MCP servers.

## Run-by-run history

| # | Setup | Result | What went wrong |
|---|---|---|---|
| 1 | Terminus + Qwen3.5-9B | 0/2 dud | `tmux: error connecting to /tmp/tmux-0/default` — harness flake, never reached LLM |
| 2 | OpenHands + Qwen3.5-9B (no `openhands-tools` package) | 0/5 | Only `finish`/`think` tools registered. Model said "I need to read the Excel" then hit the only tool it had — finish |
| 3 | + `openhands-tools` installed, terminal + file_editor + task_tracker | 0/5 live | Model used 16 tool calls / task and wrote real scripts (q0000 produced a 150-line `audit_analysis.py`). Three independent failure modes appeared: TmuxPanePool crashes, vLLM `qwen3_coder` tool-call parser mismatch, and premature `finish` |
| 4 | file-io agent + Qwen3.6-27B (5 tasks) | 1/4 graded = 25 % | Detour: prototyped a minimal HTTP-direct agent with `read_file` + `write_file` + `finish` (no shell). q0001 passed with `Fall_Music_Tour_PnL_Report.csv`; the other 3 wrote Python scripts the agent had no way to execute. Abandoned in favour of OpenHands+shell; agent code not shipped |
| 5 | OpenHands + Qwen3.6-27B, 5 tasks, `max_turns=40` | 0/5 live → 1/5 (20 %) via post-hoc regrade | Real fix needed: `gdpval_outputs_dir` was never wired from agent → judge, so the live judge saw zero deliverables. Cwd-carryover surfaced — q0001-q0004 deliverables all landed in q0000/workspace |
| 6 | + `tmux kill-server` between tasks | killed mid-run | Nuking the tmux server *broke the executor*; model spent turns testing terminal then fell back to `file_editor view` and got disoriented by q0000's files |
| 7 | + `Tool(name="terminal", params={"terminal_type": "subprocess"})` | killed | Switching off the tmux pool didn't fix it either — q0001's `Fall_Music_Tour_P&L_Report.xlsx` still landed in q0000/workspace |
| 8 | + rebuild `Agent` per task in `_create_conversation()`, 2 tasks | **2/2 = 100 %**, 42 min, 11.9 kJ | Root cause finally addressed: `AgentBase._initialize()` builds `self._tools` exactly once and refuses to re-init. Fresh Agent ⇒ fresh tools ⇒ fresh `TerminalExecutor(working_dir=conv_state.workspace.working_dir)` |

## Bugs found and fixes that survived

These are the changes still on the `feat/gdpval-dataset` branch and rsync'd to
the lambda box. They are required for the harness to work; rolling any of them
back will reproduce one of the failure modes above.

### 1. HuggingFace URI parser (`ipw/datasets/gdpval.py`)

The `reference_file_hf_uris` field in `openai/gdpval` looks like:

```
hf://datasets/openai/gdpval@main/reference_files/<hash>/Population%20v2.xlsx
```

The first parser passed `openai/gdpval@main` as repo id; `huggingface_hub`
rejects the `@`. The path is also URL-encoded. Fix: split `@<revision>` off
the second path segment, URL-decode the rest.

### 2. `openhands-tools` is a separate package

`[openhands] = ["openhands-sdk"]` was not enough — without
`openhands-tools` (pinned to the *exact same version* as the SDK to avoid the
`openhands.sdk.utils.path` ImportError between 1.17 and 1.22), the Agent only
gets the SDK's built-in `finish`/`think` tools. Added to the `[openhands]`
and `[all]` extras in `pyproject.toml`.

### 3. `gdpval_outputs_dir` wired from agent → judge

`GdpvalHandler.evaluate()` reads deliverable files via
`metadata["gdpval_outputs_dir"]`. The OpenHands agent's
`set_task_metadata()` needed to surface its workspace path:

```python
if metadata is not None and self._workspace and not metadata.get("session"):
    metadata.setdefault("gdpval_outputs_dir", self._workspace)
```

Without this the rubric judge sees zero deliverable files and every task
scores 0.

### 4. `terminal_type="subprocess"` to bypass the tmux pane pool

`openhands-tools`' default `terminal` backend is a tmux pane pool keyed by
process. Even if each conversation has its own `TerminalExecutor`, the
underlying tmux server is shared and panes carry state across conversations.
Subprocess mode is stateless across calls — each `terminal` command spawns a
fresh `subprocess.Popen(..., cwd=working_dir)`.

```python
Tool(name="terminal", params={"terminal_type": "subprocess"})
```

### 5. Fresh `Agent` instance per `_create_conversation()`  ← the actual root cause

`openhands.sdk.AgentBase._initialize()` builds the `self._tools` dict on the
first conversation and short-circuits all subsequent calls
(`if self._initialized: return`). The `TerminalExecutor`'s `working_dir` is
baked in at that moment and never refreshed. Result: tasks 2-N have an
executor whose cwd is task-1's workspace, so every `python build.py` writes
into the wrong dir.

Patch in `openhands.py`:

```python
def _create_conversation(self) -> Any:
    self.agent = self._Agent(**self._agent_kwargs)  # rebuild per task
    return self._LocalConversation(
        agent=self.agent,
        workspace=self._workspace,
        ...
    )
```

This is the single change that turned 0/5 into 2/2 — every other fix was
necessary but not sufficient without this one.

### 6. Prompt simplifications (`ipw/datasets/gdpval.py`)

The original verbose 6-step workflow + anti-Finish lectures was making small
models spend tool calls on terminal diagnostics instead of doing the work.
The current prompt is two paragraphs: tools you have, what to produce, where
it should land, run `ls` before `finish`.

### 7. Routing-aware regrade (post-hoc only)

When the cwd-carryover bug was still active in run 5, all deliverables pooled
in q0000/workspace. A standalone `regrade_all.py` script routes files to
per-task regrade dirs by filename keyword (`audit/sample/population` →
q0000, `pnl/tour/music` → q0001, etc.) and reruns the judge. This is no
longer needed for new runs but kept under `lambda-stanford/amir/` for
re-grading old runs.

## Reproduced metrics (final run, OH run 9)

```
Task q0000:  OK in 1616.3 s   resolved=TRUE   Sample.xlsx
Task q0001:  OK in  902.3 s   resolved=TRUE   Fall_Music_Tour_PNL_Report_2024.xlsx

TerminalExecutor inits   = 2     (one per task — confirms Agent rebuild)
Total queries completed  = 2 / 2
Resolved at threshold    = 2 / 2 (100 %)
Total GPU energy         = 11,899.81 J
Avg GPU power            = 4.72 W       (wall-clock-averaged, includes idle)
IPJ                       = 8.4 × 10⁻⁵
IPW                       = 0.212
```

`Sample.xlsx` content: full 1516-row population with audit columns (variance
%, sample flag), separate `Sample Size Calculation` tab with z=1.645, p=0.5,
e=0.10, FPC-adjusted n=65.

`Fall_Music_Tour_PNL_Report_2024.xlsx` content: per-country gross revenue,
withholding-tax columns (UK 20 %, France 15 %, Spain 24 %, Germany
15.825 %), three-column expense breakdown (Tour Manager / Production Co. /
Total), net income $120,422.94.

## Known limitations / open follow-ups

- **Per-LLM-call token tracking is broken**: `num_turns` in the trace shows 1
  for every task because OpenHands fires `lm_inference_start/end` once per
  conversation rather than per LLM call. Real tool-call count is in
  `total_tool_calls`. Doesn't affect rubric scoring but does break
  per-turn energy attribution.
- **Judge non-determinism**: `gpt-5-nano` at `temperature=0` gives ±10 % swings
  on the same files when graded twice. The `judge_audit_run4.json` from this
  experiment has every per-criterion verdict for spot-checking.
- **OCR & web search not in toolkit**: GDPval reference materials we saw
  were native digital PDFs, so OCR wasn't needed; no task required web
  search. Stirrup (Artificial Analysis) gives Qwen3.6-27B both in their
  benchmark — worth wiring up if we extend to harder GDPval categories.
- **5-task run still needed**: this experiment was scoped down from 5 → 2
  tasks once the harness bugs were piling up. With the fixes in place, the
  next step is the full 5-task (and eventually 220-task) Qwen3.6-27B
  baseline.

## Files of record

- `intelligence-per-watt/src/ipw/datasets/gdpval.py` — dataset provider
- `intelligence-per-watt/src/ipw/evaluation/gdpval.py` — rubric judge
- `intelligence-per-watt/src/ipw/agents/openhands.py` — fixes 3, 4, 5
- `intelligence-per-watt/src/ipw/cli/model_presets.py` — `qwen36-27b` preset
- `intelligence-per-watt/pyproject.toml` — `openhands-tools` extra
- `lambda-stanford/amir/runs/gdpval_openhands/` (remote) — final 2/2 run
- `lambda-stanford/amir/judge_audit_run4.json` (remote) — per-criterion verdicts for the buggy 5-task run, useful for judge quality spot-checks
