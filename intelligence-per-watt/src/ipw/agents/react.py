"""React agent implementation using the Agno framework."""

from __future__ import annotations

import functools
import json
import os
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from ipw.agents.base import BaseAgent
from ipw.core.registry import AgentRegistry
from ipw.core.types import AgentRunResult
from ipw.cost.pricing import calculate_cost

if TYPE_CHECKING:
    from ipw.agents.mcp.base import BaseMCPServer
    from ipw.telemetry.events import EventRecorder


@AgentRegistry.register("react")
class React(BaseAgent):
    """React agent that uses the Agno Agent framework for tool-augmented reasoning."""

    MAX_TOOL_RESULT_CHARS = 6_000

    DEFAULT_INSTRUCTIONS = (
        "You are a helpful assistant that can answer questions "
        "and use the tools provided to you if necessary."
    )

    def __init__(
        self,
        model: Any,
        tools: List[Callable] | None = None,
        mcp_tools: dict[str, "BaseMCPServer"] | None = None,
        instructions: str | None = None,
        event_recorder: Optional["EventRecorder"] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the React agent.

        Args:
            model: The Agno Model instance to use.
            tools: Optional list of callable tools/functions for the agent to use.
            mcp_tools: Optional IPW MCP tools to expose as Agno callables.
            instructions: Optional custom instructions for the agent.
            event_recorder: Optional EventRecorder for per-action energy telemetry.
            **kwargs: Additional keyword arguments passed to the Agent constructor.
        """
        super().__init__(mcp_tools=mcp_tools, event_recorder=event_recorder)

        # Lazy import: agno is optional
        try:
            from agno.agent import Agent
        except ImportError:
            raise ImportError(
                "agno package is required for React agent. "
                "Install with: pip install agno"
            )

        self.model = model
        if tools is None and self.mcp_tools:
            tools = self._mcp_tools_to_functions(self.mcp_tools)
        self._original_tools = tools or []
        self.instructions = instructions or self.DEFAULT_INSTRUCTIONS
        max_turns = kwargs.pop("max_turns", None)

        # Instrument tools if event_recorder is provided
        if event_recorder is not None and self._original_tools:
            self.tools = self._instrument_tools(self._original_tools)
        else:
            self.tools = self._original_tools

        agent_kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": self.instructions,
            "compress_tool_results": False,
            **kwargs,
        }
        if self.tools:
            tool_call_limit = os.getenv("IPW_REACT_TOOL_CALL_LIMIT")
            if max_turns is not None:
                tool_call_limit = str(max(1, int(max_turns)))
            if tool_call_limit is not None:
                agent_kwargs.setdefault("tool_call_limit", max(1, int(tool_call_limit)))
            agent_kwargs["tools"] = self.tools
            agent_kwargs["tool_choice"] = "auto"

        self.agent = Agent(**agent_kwargs)

    def set_workspace(self, workspace_path: str) -> None:
        """Set the workspace directory for MCP-backed tools."""
        for server in self.mcp_tools.values():
            if hasattr(server, "working_dir"):
                server.working_dir = workspace_path
            if getattr(server, "_ipw_dynamic_allowed_dirs", False) and hasattr(
                server, "allowed_dirs"
            ):
                server.allowed_dirs = [workspace_path]

    def _mcp_tools_to_functions(
        self, mcp_tools: dict[str, "BaseMCPServer"]
    ) -> List[Callable]:
        """Expose IPW MCP tools through Agno's function-tool interface."""

        functions: list[Callable] = []

        for name, server in mcp_tools.items():
            if name == "web_search":
                def web_search(
                    query: str | None = None,
                    keyword: str | None = None,
                    search_query: str | None = None,
                    thought: str | None = None,
                    parameter: str | None = None,
                    queries: str | list[str] | None = None,
                    __server=server,
                ) -> str:
                    search_queries: list[str] = []
                    if query:
                        search_queries.append(query)
                    if keyword:
                        search_queries.append(keyword)
                    if search_query:
                        search_queries.append(search_query)
                    if parameter and parameter not in {
                        "query",
                        "queries",
                        "keyword",
                        "search_query",
                    }:
                        search_queries.append(parameter)
                    if queries:
                        parsed_queries: Any = queries
                        if isinstance(queries, str):
                            try:
                                parsed_queries = json.loads(queries)
                            except json.JSONDecodeError:
                                parsed_queries = queries
                        if isinstance(parsed_queries, list):
                            search_queries.extend(str(item) for item in parsed_queries)
                        else:
                            search_queries.append(str(parsed_queries))

                    if not search_queries:
                        return "Error: web_search requires query or queries."
                    results = [
                        self._truncate_tool_result(__server.execute(search_query).content)
                        for search_query in search_queries[:3]
                    ]
                    return self._truncate_tool_result("\n\n".join(results))

                functions.append(web_search)
            elif name == "calculator":
                def calculator(expression: str, __server=server) -> str:
                    return self._truncate_tool_result(__server.execute(expression).content)

                functions.append(calculator)
            elif name == "file_read":
                def file_read(path: str, __server=server) -> str:
                    return self._truncate_tool_result(__server.execute(path).content)

                functions.append(file_read)
            elif name == "file_write":
                def file_write(
                    path: str,
                    content: str,
                    mode: str = "w",
                    __server=server,
                ) -> str:
                    return self._truncate_tool_result(
                        __server.execute(path, content=content, mode=mode).content
                    )

                functions.append(file_write)
            elif name == "code_interpreter":
                def code_interpreter(code: str, __server=server) -> str:
                    return self._truncate_tool_result(__server.execute(code).content)

                functions.append(code_interpreter)
            elif name in ("bash", "shell"):
                def bash(command: str, __server=server) -> str:
                    if self._terminal_session() is not None:
                        return self._truncate_tool_result(
                            self._execute_terminal_session_command(command)
                        )
                    return self._truncate_tool_result(__server.execute(command).content)

                functions.append(bash)
            elif name == "think":
                def think(thought: str, __server=server) -> str:
                    return self._truncate_tool_result(__server.execute(thought).content)

                functions.append(think)

        return functions

    def _truncate_tool_result(self, result: Any) -> str:
        text = str(result)
        if len(text) <= self.MAX_TOOL_RESULT_CHARS:
            return text
        return (
            text[: self.MAX_TOOL_RESULT_CHARS]
            + "\n\n[tool result truncated to fit the model context window]"
        )

    def _instrument_tools(self, tools: List[Callable]) -> List[Callable]:
        """Wrap tools to emit start/end events for energy tracking."""
        instrumented = []
        for tool in tools:
            tool_name = getattr(tool, "__name__", str(tool))

            @functools.wraps(tool)
            def wrapper(
                *args: Any,
                __tool: Callable = tool,
                __name: str = tool_name,
                **kwargs: Any,
            ) -> Any:
                self._record_event("tool_call_start", tool=__name)
                try:
                    return __tool(*args, **kwargs)
                finally:
                    self._record_event("tool_call_end", tool=__name)

            instrumented.append(wrapper)
        return instrumented

    def run(self, input: str, **kwargs: Any) -> AgentRunResult:
        """Run the React agent.

        Args:
            input: The input message or prompt for the agent.
            **kwargs: Additional keyword arguments passed to agent.run().

        Returns:
            AgentRunResult with the agent's response.
        """
        result = None
        usage_calls: list[dict[str, Any]] = []
        lm_started = False

        def _coerce_int(value: Any) -> int:
            try:
                return int(value or 0)
            except Exception:
                return 0

        def _cost_provider_model(model_name: Any) -> tuple[str, str]:
            text = str(model_name or "")
            if text.startswith("openai/"):
                return "openai", text.split("/", 1)[1]
            if text.startswith("anthropic/"):
                return "anthropic", text.split("/", 1)[1]
            if text.startswith(("gemini/", "google/")):
                return "gemini", text.split("/", 1)[1]
            if text.startswith(("gpt-", "o1", "o3", "o4")):
                return "openai", text
            if text.startswith("claude-"):
                return "anthropic", text
            if text.startswith("gemini-"):
                return "gemini", text
            return "", text

        def _response_model(response: Any, request_model: Any) -> Any:
            if isinstance(response, dict):
                return response.get("model") or request_model
            return getattr(response, "model", None) or request_model

        def _extract_usage(response: Any, request_model: Any) -> dict[str, Any]:
            usage = None
            if isinstance(response, dict):
                usage = response.get("usage")
            else:
                usage = getattr(response, "usage", None)
            if usage is None:
                return {}
            if isinstance(usage, dict):
                prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
                completion = usage.get("completion_tokens") or usage.get("output_tokens")
                total = usage.get("total_tokens")
            else:
                prompt = getattr(usage, "prompt_tokens", None) or getattr(
                    usage, "input_tokens", None
                )
                completion = getattr(usage, "completion_tokens", None) or getattr(
                    usage, "output_tokens", None
                )
                total = getattr(usage, "total_tokens", None)
            prompt_i = _coerce_int(prompt)
            completion_i = _coerce_int(completion)
            total_i = _coerce_int(total) or prompt_i + completion_i
            if prompt_i == 0 and completion_i == 0:
                return {}
            provider, priced_model = _cost_provider_model(
                _response_model(response, request_model)
            )
            cost_usd = (
                calculate_cost(provider, priced_model, prompt_i, completion_i)
                if provider
                else 0.0
            )
            return {
                "prompt_tokens": prompt_i,
                "completion_tokens": completion_i,
                "total_tokens": total_i,
                "cost_usd": cost_usd,
            }

        def _preserve_empty_reasoning_response(response: Any) -> Any:
            if isinstance(response, dict):
                choices = response.get("choices") or []
                if not choices:
                    return response
                message = choices[0].get("message") or {}
                if not message.get("content") and message.get("reasoning_content"):
                    message["content"] = message["reasoning_content"]
                return response

            try:
                message = response.choices[0].message
            except Exception:
                return response
            if not getattr(message, "content", None) and getattr(
                message,
                "reasoning_content",
                None,
            ):
                try:
                    message.content = message.reasoning_content
                except Exception:
                    pass
            return response

        def _run_with_litellm_usage_capture() -> Any:
            try:
                import litellm
            except Exception:
                return self.agent.run(input, **kwargs)

            original_completion = litellm.completion

            def _completion_with_usage(*c_args: Any, **c_kwargs: Any) -> Any:
                nonlocal lm_started
                lm_started = True
                self._record_event("lm_inference_start", model=str(self.model))
                response = original_completion(*c_args, **c_kwargs)
                usage = _extract_usage(response, c_kwargs.get("model"))
                if usage:
                    usage_calls.append(usage)
                self._record_event(
                    "lm_inference_end",
                    model=str(self.model),
                    prompt_tokens=usage.get("prompt_tokens") if usage else None,
                    completion_tokens=usage.get("completion_tokens") if usage else None,
                    total_tokens=usage.get("total_tokens") if usage else None,
                    cost_usd=usage.get("cost_usd"),
                    token_source="litellm_response_usage" if usage else "missing",
                )
                return _preserve_empty_reasoning_response(response)

            litellm.completion = _completion_with_usage
            try:
                return self.agent.run(input, **kwargs)
            finally:
                litellm.completion = original_completion

        try:
            result = _run_with_litellm_usage_capture()
            # Extract token metrics from the result if available
            end_metadata: dict[str, Any] = {"model": str(self.model)}
            input_tokens: int | None = None
            output_tokens: int | None = None

            # Extract content from result
            content = ""
            raw_result_text = ""
            if result is not None:
                raw_result_text = str(result)
                if hasattr(result, "content"):
                    content = str(result.content)
                else:
                    content = raw_result_text
                if (not content or content == "None") and hasattr(result, "messages"):
                    for message in reversed(getattr(result, "messages") or []):
                        if getattr(message, "role", None) != "assistant":
                            continue
                        if getattr(message, "tool_calls", None):
                            continue
                        if hasattr(message, "get_content_string"):
                            candidate = message.get_content_string()
                        else:
                            candidate = str(getattr(message, "content", "") or "")
                        if candidate.strip():
                            content = candidate
                            break

            token_source = "missing"
            usage_metadata: dict[str, Any] = {}
            if usage_calls:
                input_tokens = sum(call.get("prompt_tokens", 0) for call in usage_calls)
                output_tokens = sum(call.get("completion_tokens", 0) for call in usage_calls)
                total_tokens = sum(call.get("total_tokens", 0) for call in usage_calls)
                total_cost_usd = sum(call.get("cost_usd", 0.0) for call in usage_calls)
                if total_tokens == 0:
                    total_tokens = input_tokens + output_tokens
                token_source = "litellm_response_usage"
                usage_metadata = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "num_turns": len(usage_calls),
                    "token_source": token_source,
                    "missing_usage_responses": 0,
                    "cost_usd": total_cost_usd,
                }
            elif (
                result is not None
                and hasattr(result, "metrics")
                and result.metrics is not None
            ):
                metrics = result.metrics
                input_tokens = _coerce_int(getattr(metrics, "input_tokens", 0))
                output_tokens = _coerce_int(getattr(metrics, "output_tokens", 0))
                if input_tokens > 0 or output_tokens > 0:
                    token_source = "agno_metrics"
                else:
                    input_tokens = None
                    output_tokens = None

            end_metadata["prompt_tokens"] = input_tokens
            end_metadata["completion_tokens"] = output_tokens
            end_metadata["total_tokens"] = (
                input_tokens + output_tokens
                if input_tokens is not None and output_tokens is not None
                else None
            )
            end_metadata["token_source"] = token_source
            if not usage_calls:
                self._record_event("lm_inference_start", model=str(self.model))
                self._record_event("lm_inference_end", **end_metadata)

            metadata = {"token_source": token_source}
            if usage_metadata:
                metadata["usage"] = usage_metadata

            return AgentRunResult(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=usage_metadata.get("cost_usd"),
                num_turns=len(usage_calls),
                metadata=metadata,
            )
        except Exception:
            if not lm_started:
                self._record_event("lm_inference_start", model=str(self.model))
            self._record_event("lm_inference_end", model=str(self.model))
            raise
