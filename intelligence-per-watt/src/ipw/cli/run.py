"""Run agentic benchmarks against a dataset."""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from pathlib import Path

import click

from ._console import error, info, success, warning
from ._display import (
    compute_trace_metrics,
    print_banner,
    print_config_summary,
    print_efficiency_panel,
    print_metrics_table,
    print_output_path,
)

_LITELLM_PREFIXES = (
    "openai/", "ollama/", "anthropic/", "gemini/", "google/",
    "azure/", "bedrock/", "vertex_ai/",
)

_ALWAYS_CLOUD_PROVIDER_PREFIXES = (
    "anthropic/", "gemini/", "azure/", "bedrock/", "vertex_ai/",
)

_BARE_CLOUD_PREFIXES = ("claude-", "gemini-", "gpt-5", "gpt-4", "o1", "o3", "o4")
_DEFAULT_LLM_TIMEOUT_SECONDS = 120
_TOOL_AGENT_IDS = ("react", "openhands", "dspy-rlm", "forgecode")
_DEEPRESEARCH_TOOL_DATASETS = {
    "gaia",
    "frames",
    "deepresearchbench",
    "liveresearchbench",
    "browsecomp",
}
_CODING_TOOL_DATASETS = {"swebench", "livecodebench", "swefficiency"}
_TERMINAL_TOOL_DATASETS = {"terminalbench-native"}
_DEEPRESEARCH_MCP_TOOLS = [
    "think",
    "web_search",
    "file_read",
    "code_interpreter",
    "calculator",
    "bash",
]
_CODING_MCP_TOOLS = [
    "think",
    "file_read",
    "file_write",
    "code_interpreter",
    "bash",
]
_TERMINAL_MCP_TOOLS = ["think", "bash", "file_read", "file_write", "code_interpreter"]


def _default_mcp_tool_spec_for_dataset(dataset_id: str) -> list[str] | None:
    """Return the plan's default MCP tools for tool-using datasets."""
    if dataset_id in _DEEPRESEARCH_TOOL_DATASETS:
        return list(_DEEPRESEARCH_MCP_TOOLS)
    if dataset_id in _CODING_TOOL_DATASETS:
        return list(_CODING_MCP_TOOLS)
    if dataset_id in _TERMINAL_TOOL_DATASETS:
        return list(_TERMINAL_MCP_TOOLS)
    return None


def _is_cloud_model(model: str, base_url: str) -> bool:
    """Return True when the model targets a cloud API provider.

    Cloud models are detected by their litellm provider prefix.  ``openai/``
    is treated as local when the base_url points to a local server
    (localhost / 127.0.0.1 / 0.0.0.0), and cloud otherwise.
    """
    _local_hosts = ("localhost", "127.0.0.1", "0.0.0.0")
    local_base_url = any(h in base_url for h in _local_hosts)
    if model.startswith(_BARE_CLOUD_PREFIXES) and not model.startswith("gpt-oss"):
        return True
    if model.startswith(_ALWAYS_CLOUD_PROVIDER_PREFIXES):
        return True
    if model.startswith("google/"):
        return not local_base_url
    if model.startswith("openai/"):
        return not local_base_url
    return False


def _add_litellm_prefix(model: str, base_url: str) -> str:
    """Add LiteLLM provider prefix if not already present."""
    if model.startswith(_LITELLM_PREFIXES):
        if (
            not _is_cloud_model(model, base_url)
            and not model.startswith(("openai/", "ollama/"))
        ):
            return f"openai/{model}"
        return model
    if model.startswith("claude-"):
        return f"anthropic/{model}"
    if model.startswith("gemini-"):
        return f"gemini/{model}"
    if model.startswith(("gpt-", "o1", "o3", "o4")) and not model.startswith("gpt-oss"):
        return f"openai/{model}"
    is_ollama = "11434" in base_url or "ollama" in base_url.lower()
    prefix = "ollama" if is_ollama else "openai"
    return f"{prefix}/{model}"


def _create_model_for_agent(agent_id: str, model: str, base_url: str, api_key: str):
    """Create the framework-specific model object for the given agent type."""
    import os

    cloud = _is_cloud_model(model, base_url)

    # Normalize api_key: treat "EMPTY" as None so litellm reads env vars
    effective_key = None if api_key == "EMPTY" else api_key

    if agent_id == "react":
        from agno.models.litellm import LiteLLM

        react_timeout = int(
            os.getenv("IPW_REACT_LLM_TIMEOUT")
            or os.getenv("IPW_OPENAI_COMPAT_LLM_TIMEOUT")
            or str(_DEFAULT_LLM_TIMEOUT_SECONDS)
        )
        request_params = {"timeout": react_timeout}
        if cloud:
            return LiteLLM(
                id=_add_litellm_prefix(model, base_url),
                api_key=effective_key,
                request_params=request_params,
            )
        clean_base = base_url.rstrip("/")
        if not clean_base.endswith("/v1"):
            clean_base = f"{clean_base}/v1"
        return LiteLLM(
            id=_add_litellm_prefix(model, base_url),
            api_key=api_key,
            api_base=clean_base,
            request_params=request_params,
        )
    elif agent_id == "openhands":
        from openhands.sdk import LLM

        litellm_model = _add_litellm_prefix(model, base_url)
        llm_kwargs = {
            "timeout": int(
                os.getenv("IPW_OPENHANDS_LLM_TIMEOUT", str(_DEFAULT_LLM_TIMEOUT_SECONDS))
            )
        }
        if os.getenv("IPW_OPENHANDS_MAX_OUTPUT_TOKENS"):
            llm_kwargs["max_output_tokens"] = int(
                os.environ["IPW_OPENHANDS_MAX_OUTPUT_TOKENS"]
            )
        if os.getenv("IPW_OPENHANDS_REASONING_EFFORT"):
            llm_kwargs["reasoning_effort"] = os.environ[
                "IPW_OPENHANDS_REASONING_EFFORT"
            ]
        if os.getenv("IPW_OPENHANDS_EXTENDED_THINKING_BUDGET"):
            value = os.environ["IPW_OPENHANDS_EXTENDED_THINKING_BUDGET"]
            llm_kwargs["extended_thinking_budget"] = (
                None if value.lower() == "none" else int(value)
            )
        if os.getenv("IPW_OPENHANDS_LITELLM_EXTRA_BODY_JSON"):
            try:
                llm_kwargs["litellm_extra_body"] = json.loads(
                    os.environ["IPW_OPENHANDS_LITELLM_EXTRA_BODY_JSON"]
                )
            except json.JSONDecodeError as exc:
                raise click.ClickException(
                    f"Invalid IPW_OPENHANDS_LITELLM_EXTRA_BODY_JSON: {exc}"
                ) from exc
        if os.getenv("IPW_OPENHANDS_NUM_RETRIES"):
            llm_kwargs["num_retries"] = int(
                os.environ["IPW_OPENHANDS_NUM_RETRIES"]
            )
        if os.getenv("IPW_OPENHANDS_CONNECTION_CLOSE"):
            llm_kwargs["extra_headers"] = {"Connection": "close"}
        if cloud:
            return LLM(model=litellm_model, api_key=effective_key, **llm_kwargs)
        # Ollama native API doesn't use /v1; OpenAI-compatible servers do
        is_ollama = "11434" in base_url or "ollama" in base_url.lower()
        if is_ollama:
            llm_base_url = base_url
        elif base_url.rstrip("/").endswith("/v1"):
            llm_base_url = base_url
        else:
            llm_base_url = f"{base_url}/v1"
        return LLM(
            model=litellm_model,
            api_key=api_key,
            base_url=llm_base_url,
            **llm_kwargs,
        )
    elif agent_id in ("terminus-tb", "terminus"):
        # terminus_tb.py already prepends "openai/" — pass raw model_id
        return model
    elif agent_id in ("dspy-rlm", "forgecode"):
        clean_base = base_url.rstrip("/")
        if not cloud and not clean_base.endswith("/v1"):
            clean_base = f"{clean_base}/v1"
        return {
            "model": model,
            "base_url": None if cloud else clean_base,
            "api_key": effective_key or api_key,
            "cloud": cloud,
        }
    elif agent_id == "react-native":
        from ipw.clients.openai_chat_adapter import OpenAIChatAdapter

        if cloud:
            return OpenAIChatAdapter(model=model, api_key=effective_key)
        return OpenAIChatAdapter(
            model=model, api_key=api_key, base_url=f"{base_url}/v1",
        )
    else:
        return model  # Fallback: pass string


def _resolve_openhands_mcp_tools(spec):
    """Create OpenHands MCP tool instances from JSON-friendly CLI config."""
    if isinstance(spec, list):
        tool_configs = {}
        for item in spec:
            if isinstance(item, str):
                tool_configs[item] = {}
            elif isinstance(item, dict):
                name = item.get("type") or item.get("name")
                if not isinstance(name, str) or not name:
                    raise click.ClickException(
                        "mcp_tools list entries must be strings or objects with a 'type' field"
                    )
                config = {k: v for k, v in item.items() if k not in ("type", "name")}
                tool_configs[name] = config
            else:
                raise click.ClickException(
                    "mcp_tools list entries must be strings or objects"
                )
    elif isinstance(spec, dict):
        tool_configs = {
            name: ({} if config in (True, "auto", None) else config)
            for name, config in spec.items()
        }
    else:
        raise click.ClickException(
            "--agent-kwargs mcp_tools must be a list of names or object of configs"
        )

    from ipw.agents.mcp.tool_server import (
        CalculatorServer,
        CodeInterpreterServer,
        FileReadServer,
        FileWriteServer,
        ShellServer,
        ThinkServer,
        WebSearchServer,
    )

    resolved = {}
    for name, config in tool_configs.items():
        if not isinstance(config, dict):
            raise click.ClickException(
                f"mcp_tools.{name} must be an object, true, null, or 'auto'"
            )
        if name == "think":
            resolved[name] = ThinkServer()
        elif name == "calculator":
            resolved[name] = CalculatorServer()
        elif name == "code_interpreter":
            kwargs = {"isolation": config.get("isolation", "auto")}
            if "timeout" in config:
                kwargs["timeout"] = config["timeout"]
            resolved[name] = CodeInterpreterServer(**kwargs)
        elif name == "file_read":
            kwargs = {"allowed_dirs": config.get("allowed_dirs")}
            if "max_chars" in inspect.signature(FileReadServer.__init__).parameters:
                kwargs["max_chars"] = config.get("max_chars", 20000)
            resolved[name] = FileReadServer(**kwargs)
        elif name == "file_write":
            resolved[name] = FileWriteServer(allowed_dirs=config.get("allowed_dirs"))
        elif name in ("bash", "shell"):
            kwargs = {
                "max_output_length": config.get("max_output_length", 20000),
                "allowed_dirs": config.get("allowed_dirs"),
            }
            if "timeout" in config:
                kwargs["timeout"] = config["timeout"]
            resolved[name] = ShellServer(**kwargs)
        elif name == "web_search":
            resolved[name] = WebSearchServer(
                max_results=config.get("max_results", 5),
                search_depth=config.get("search_depth", "basic"),
                include_answer=config.get("include_answer", True),
                max_content_chars=config.get("max_content_chars"),
                max_total_chars=config.get("max_total_chars"),
            )
        else:
            raise click.ClickException(f"Unsupported OpenHands MCP tool: {name}")
    return resolved


@click.command(help="Run an agentic benchmark against a dataset.")
@click.option("--agent", "agent_id", required=True, help="Agent identifier")
@click.option("--model", default=None, help="Model name for the agent (or use --preset)")
@click.option("--preset", "preset_name", default=None, help="Model preset name (e.g. glm-4.7-flash)")
@click.option("--dataset", "dataset_id", required=True, help="Dataset identifier")
@click.option(
    "--client-base-url",
    default="http://localhost:8000",
    show_default=True,
    help="Inference server base URL",
)
@click.option(
    "--api-key",
    default="EMPTY",
    show_default=True,
    help="API key for the inference server",
)
@click.option(
    "--max-queries",
    type=int,
    default=None,
    help="Max queries to run (default: all)",
)
@click.option(
    "--start-index",
    type=int,
    default=0,
    show_default=True,
    help="Dataset record index to start from; useful for continuing a partial run",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./runs/",
    show_default=True,
    help="Output directory",
)
@click.option(
    "--export-format",
    default="jsonl,hf",
    show_default=True,
    help="Comma-separated export formats (jsonl, hf)",
)
@click.option(
    "--estimate-flops",
    is_flag=True,
    default=False,
    help="Enable FLOPs estimation",
)
@click.option(
    "--agent-kwargs",
    default=None,
    help="JSON string of extra agent keyword arguments",
)
@click.option(
    "--dataset-kwargs",
    default=None,
    help="JSON string of extra dataset keyword arguments",
)
@click.option(
    "--concurrency",
    type=int,
    default=1,
    show_default=True,
    help="Number of tasks to run in parallel",
)
@click.option(
    "--query-timeout",
    type=float,
    default=None,
    help="Wall-clock timeout in seconds per query (default: no limit)",
)
@click.option(
    "--max-retries",
    type=int,
    default=3,
    show_default=True,
    help="Per-turn retry budget on transient errors (0 disables retry).",
)
@click.option(
    "--require-dedicated-hardware",
    is_flag=True,
    default=False,
    help="Abort if preflight detects competing GPU/CPU workloads (precision mode).",
)
@click.option(
    "--eval-client",
    default="openai",
    show_default=True,
    help="Client for evaluation",
)
@click.option(
    "--eval-base-url",
    default=None,
    help="Base URL for the evaluation client",
)
@click.option(
    "--eval-model",
    default="gpt-5-nano-2025-08-07",
    show_default=True,
    help="Model for evaluation",
)
def run_cmd(
    agent_id: str,
    model: str | None,
    preset_name: str | None,
    dataset_id: str,
    client_base_url: str,
    api_key: str,
    max_queries: int | None,
    start_index: int,
    output_dir: str,
    export_format: str,
    estimate_flops: bool,
    agent_kwargs: str | None,
    dataset_kwargs: str | None,
    concurrency: int,
    query_timeout: float | None,
    max_retries: int,
    require_dedicated_hardware: bool,
    eval_client: str,
    eval_base_url: str | None,
    eval_model: str,
) -> None:
    """Execute an agentic benchmark run."""
    from .model_presets import resolve_preset

    # Resolve model from --preset if provided
    if preset_name and model:
        raise click.ClickException("Specify either --model or --preset, not both.")
    if preset_name:
        try:
            preset = resolve_preset(preset_name)
        except KeyError as exc:
            raise click.ClickException(str(exc)) from exc
        model = preset["model_id"]
        info(f"Preset: {preset_name} → {model}")
    if not model:
        raise click.ClickException("Either --model or --preset is required.")

    import ipw.clients
    import ipw.datasets

    ipw.clients.ensure_registered()
    ipw.datasets.ensure_registered()

    # Ensure agent modules are imported for registry population
    from ipw.agents import dspy_rlm as _dspy_rlm  # noqa: F401
    from ipw.agents import forgecode as _forgecode  # noqa: F401
    from ipw.agents import react as _react  # noqa: F401
    try:
        from ipw.agents import react_native as _react_native  # noqa: F401
    except ImportError:
        pass
    try:
        from ipw.agents import openhands as _openhands  # noqa: F401
    except ImportError:
        pass
    try:
        from ipw.agents import terminus, terminus_tb  # noqa: F401
    except ImportError:
        pass

    # Trigger tool registrations so ToolRegistry is populated for ToolUsingAgent
    # construction.  Each import is guarded so missing optional deps don't abort
    # the CLI — the tool is simply absent from the registry.
    for _tool_mod in (
        "shell_exec", "git_tool", "apply_patch", "http_request",
        "repl", "code_interpreter_docker",
        "browser", "browser_axtree", "pdf_tool",
        "image_tool", "audio_tool", "docker_shell_exec",
    ):
        try:
            __import__(f"ipw.tools.{_tool_mod}")
        except Exception:
            pass

    from ipw.agents.base import ToolUsingAgent
    from ipw.core.registry import AgentRegistry, DatasetRegistry
    from ipw.execution.agentic_runner import AgenticRunner
    from ipw.execution.exporters import export_artifacts_manifest, export_hf_dataset, export_jsonl, export_summary_json
    from ipw.execution.telemetry_session import TelemetrySession
    from ipw.telemetry import EnergyMonitorCollector
    from ipw.telemetry.events import EventRecorder

    # Parse agent kwargs
    extra_kwargs: dict = {}
    if agent_kwargs:
        try:
            extra_kwargs = json.loads(agent_kwargs)
        except json.JSONDecodeError as exc:
            raise click.ClickException(
                f"Invalid JSON for --agent-kwargs: {exc}"
            ) from exc

    if agent_id in _TOOL_AGENT_IDS:
        if "mcp_tools" in extra_kwargs:
            extra_kwargs["mcp_tools"] = _resolve_openhands_mcp_tools(
                extra_kwargs["mcp_tools"]
            )
        else:
            default_mcp_tools = _default_mcp_tool_spec_for_dataset(dataset_id)
            if default_mcp_tools is not None:
                extra_kwargs["mcp_tools"] = _resolve_openhands_mcp_tools(
                    default_mcp_tools
                )

    # Parse dataset kwargs
    extra_dataset_kwargs: dict = {}
    if dataset_kwargs:
        try:
            extra_dataset_kwargs = json.loads(dataset_kwargs)
        except json.JSONDecodeError as exc:
            raise click.ClickException(
                f"Invalid JSON for --dataset-kwargs: {exc}"
            ) from exc

    # Resolve agent
    try:
        agent_cls = AgentRegistry.get(agent_id)
    except KeyError:
        available = [k for k, _ in AgentRegistry.items()]
        raise click.ClickException(
            f"Unknown agent '{agent_id}'. Available: {', '.join(available) or 'none'}"
        )

    # Resolve dataset
    try:
        dataset_cls = DatasetRegistry.get(dataset_id)
    except KeyError:
        available = [k for k, _ in DatasetRegistry.items()]
        raise click.ClickException(
            f"Unknown dataset '{dataset_id}'. Available: {', '.join(available) or 'none'}"
        )

    # Preflight: dataset requirements
    try:
        dataset_instance = dataset_cls(**extra_dataset_kwargs)
        dataset_instance.eval_client = eval_client
        if eval_base_url:
            dataset_instance.eval_base_url = eval_base_url
        if eval_model:
            dataset_instance.eval_model = eval_model
        issues = dataset_instance.verify_requirements()
        if issues:
            raise click.ClickException(
                "Dataset requirements not satisfied:\n- " + "\n- ".join(issues)
            )
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f"Failed to initialize dataset: {exc}") from exc

    # Create event recorder and agent
    event_recorder = EventRecorder()
    resolved_model = _create_model_for_agent(agent_id, model, client_base_url, api_key)

    # Terminus-based agents need api_base so LiteLLM can reach the local server,
    # but only for locally-served models — cloud models route via litellm.
    if agent_id in ("terminus", "terminus-tb") and "api_base" not in extra_kwargs:
        if not _is_cloud_model(model, client_base_url):
            base = client_base_url.rstrip("/")
            extra_kwargs["api_base"] = base if base.endswith("/v1") else f"{base}/v1"

    def _build_agent(a_cls, a_resolved_model, a_event_recorder, a_extra_kwargs):
        """Construct an agent instance, handling ToolUsingAgent specially.

        ToolUsingAgent subclasses (e.g. NativeReact / react-native) require
        a bare model string, the LM adapter as ``llm``, and an instantiated
        list of ``tools`` bound to the event bus.  Generic agents receive just
        ``model`` and ``event_recorder``.
        """
        try:
            _is_tool_using = issubclass(a_cls, ToolUsingAgent)
        except TypeError:
            # a_cls is not a real class (e.g. a mock in tests) — use generic path.
            _is_tool_using = False
        if _is_tool_using:
            from ipw.tools.registry import ToolRegistry
            # NativeReact wants the bare model id (no provider prefix).
            bare = model.split("/", 1)[1] if "/" in model else model
            # Build all currently-registered tools bound to this recorder's bus.
            tools: list = []
            for _tid, _tcls in ToolRegistry.items():
                try:
                    tools.append(_tcls(bus=a_event_recorder.bus))
                except Exception:
                    pass
            return a_cls(
                model=bare,
                llm=a_resolved_model,
                tools=tools,
                event_recorder=a_event_recorder,
                **a_extra_kwargs,
            )
        return a_cls(
            model=a_resolved_model,
            event_recorder=a_event_recorder,
            **a_extra_kwargs,
        )

    try:
        agent_instance = _build_agent(agent_cls, resolved_model, event_recorder, extra_kwargs)
    except TypeError as exc:
        raise click.ClickException(
            f"Failed to initialize agent '{agent_id}': {exc}"
        ) from exc

    # Prepare output directory
    model_slug = "".join(c if c.isalnum() else "_" for c in model).strip("_") or "model"
    run_dir = Path(output_dir) / f"run_{agent_id}_{model_slug}_{dataset_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Convert CLI retries count to runner attempt count (retries + 1 = attempts).
    max_attempts = max_retries + 1

    # Build config for summary
    run_config = {
        "agent": agent_id,
        "model": model,
        "dataset": dataset_id,
        "client_base_url": client_base_url,
        "max_queries": max_queries,
        "start_index": start_index,
        "concurrency": concurrency,
        "query_timeout": query_timeout,
        "export_format": export_format,
        "estimate_flops": estimate_flops,
        "eval_client": eval_client,
        "eval_base_url": eval_base_url,
        "eval_model": eval_model,
        "max_retries": max_retries,
        "require_dedicated_hardware": require_dedicated_hardware,
    }

    print_banner()
    run_display_config: dict[str, object] = {
        "Agent": agent_id,
        "Model": model,
        "Dataset": dataset_id,
        "Base URL": client_base_url,
        "Max Queries": max_queries or "(all)",
    }
    if start_index:
        run_display_config["Start Index"] = start_index
    if concurrency > 1:
        run_display_config["Concurrency"] = concurrency
    if query_timeout:
        run_display_config["Query Timeout"] = f"{query_timeout:.0f}s"
    run_display_config["Output"] = str(run_dir)
    print_config_summary(config=run_display_config)

    # Build an agent factory for concurrent execution so each thread gets
    # its own agent instance with independent state.
    _model_ref = resolved_model
    _agent_cls_ref = agent_cls
    _extra_kwargs_ref = dict(extra_kwargs)

    def _make_agent(recorder: EventRecorder) -> "BaseAgent":  # noqa: F821
        return _build_agent(_agent_cls_ref, _model_ref, recorder, _extra_kwargs_ref)

    # Run the agentic benchmark.
    # Cloud models don't use the local GPU — skip energy telemetry so traces
    # report None instead of misleading idle-GPU power draw.
    cloud = _is_cloud_model(model, client_base_url)

    def _run_with_telemetry(telemetry):
        runner = AgenticRunner(
            agent=agent_instance,
            dataset=dataset_instance,
            telemetry_session=telemetry,
            config=run_config,
            event_recorder=event_recorder,
            run_dir=run_dir,
            concurrency=concurrency,
            agent_factory=_make_agent if concurrency > 1 else None,
            query_timeout=query_timeout,
            max_attempts=max_attempts,
            require_dedicated_hardware=require_dedicated_hardware,
        )
        try:
            run_kwargs = {"max_queries": max_queries}
            if start_index:
                run_kwargs["start_index"] = start_index
            return asyncio.run(runner.run(**run_kwargs))
        except Exception as exc:
            error(f"Run failed: {exc}")
            sys.exit(1)

    if cloud:
        traces = _run_with_telemetry(None)
    else:
        collector = EnergyMonitorCollector(timeout=30.0)
        with TelemetrySession(collector) as telemetry:
            traces = _run_with_telemetry(telemetry)

    if not traces:
        warning("No traces collected.")
        return

    # FLOPs estimation
    if estimate_flops:
        try:
            from ipw.compute.flops import estimate_flops as do_estimate_flops

            for trace in traces:
                total_flops, flops_per_token = do_estimate_flops(
                    model,
                    trace.total_input_tokens,
                    trace.total_output_tokens,
                )
                if total_flops > 0:
                    info(
                        f"  {trace.query_id}: {total_flops:.2e} FLOPs "
                        f"({flops_per_token:.2e} per token)"
                    )
        except Exception as exc:
            warning(f"FLOPs estimation failed: {exc}")

    # Export results
    formats = [f.strip().lower() for f in export_format.split(",") if f.strip()]

    if "jsonl" in formats:
        jsonl_path = export_jsonl(traces, run_dir / "traces.jsonl")
        info(f"Exported JSONL: {jsonl_path}")

    if "hf" in formats:
        hf_path = export_hf_dataset(traces, run_dir / "hf_dataset")
        info(f"Exported HF dataset: {hf_path}")

    summary_path = export_summary_json(traces, run_config, run_dir / "summary.json")
    info(f"Exported summary: {summary_path}")

    manifest_path = export_artifacts_manifest(run_dir)
    if manifest_path:
        info(f"Exported artifacts manifest: {manifest_path}")

    # Rich summary display
    total_completed = sum(1 for t in traces if t.completed)
    timed_out = [t for t in traces if t.timed_out]
    success(f"Run complete: {total_completed}/{len(traces)} queries completed")

    if timed_out:
        warning(f"{len(timed_out)} queries timed out:")
        for t in timed_out:
            warning(f"  {t.query_id}: {t.total_wall_clock_s:.0f}s")

    metric_rows = compute_trace_metrics(traces)
    print_metrics_table(rows=metric_rows, title="Run Metrics")

    # Compute accuracy from is_resolved traces
    resolved_traces = [t for t in traces if t.is_resolved is not None]
    accuracy = None
    if resolved_traces:
        accuracy = sum(1 for t in resolved_traces if t.is_resolved) / len(resolved_traces)
    print_efficiency_panel(traces=traces, accuracy=accuracy)
    print_output_path(path=run_dir)


__all__ = ["run_cmd"]
