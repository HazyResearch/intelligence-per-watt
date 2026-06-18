"""DSPy-RLM style lightweight reasoning harness."""

from __future__ import annotations

from typing import Any, Optional

from ipw.agents.openai_compat import OpenAICompatibleHarness
from ipw.core.registry import AgentRegistry
from ipw.core.types import AgentRunResult


@AgentRegistry.register("dspy-rlm")
class DSPyRLM(OpenAICompatibleHarness):
    """A small plan-act-answer harness for reasoning and research tasks.

    This is intentionally dependency-light: it mimics the DSPy retrieve/LM
    loop shape while using IPW MCP tools and the existing trace event contract.
    """

    def _pre_tool_observation(
        self,
        *,
        tool_name: str,
        tool_input: str,
        turn_index: int,
        tools_attempted: int,
    ) -> Optional[str]:
        return None

    def _post_tool_observation(
        self,
        observation: str,
        *,
        tool_name: str,
        tool_input: str,
        turn_index: int,
        tools_attempted: int,
    ) -> str:
        return observation

    def _turn_limit_final_prompt(self) -> Optional[str]:
        return None

    def run(self, input: str, **kwargs: Any) -> AgentRunResult:
        tool_names = ", ".join(sorted(self.mcp_tools)) or "none"
        system = (
            f"{self.instructions}\n\n"
            "Use a compact reasoning loop. If a tool is needed, respond with:\n"
            "Action: <tool name>\nAction Input: <tool input>\n\n"
            "When done, respond with:\nFinal: <answer>\n\n"
            f"Available tools: {tool_names}."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": input},
        ]

        total_input = 0
        total_output = 0
        total_cost = 0.0
        missing_usage_responses = 0
        missing_cost = False
        token_sources: list[str] = []
        tools_attempted = 0
        tools_succeeded = 0
        tools_used: list[str] = []
        final_answer: Optional[str] = None
        last_content = ""

        for _turn in range(self.max_turns):
            result = self._chat(messages)
            if result.input_tokens is None or result.output_tokens is None:
                missing_usage_responses += 1
            else:
                total_input += result.input_tokens
                total_output += result.output_tokens
            if result.cost_usd is None:
                missing_cost = True
            else:
                total_cost += result.cost_usd
            token_sources.append(result.token_source)
            last_content = result.content

            action = self._parse_action(last_content)
            if action is None:
                final_answer = self._extract_final(last_content)
                if final_answer is not None:
                    break
                final_answer = last_content
                break

            tool_name, tool_input = action
            if tool_name not in self.mcp_tools:
                messages.append({"role": "assistant", "content": last_content})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Observation: tool '{tool_name}' is not available. Use one of: {tool_names}.",
                    }
                )
                continue

            tools_attempted += 1
            tools_used.append(tool_name)
            observation = self._pre_tool_observation(
                tool_name=tool_name,
                tool_input=tool_input,
                turn_index=_turn,
                tools_attempted=tools_attempted,
            )
            if observation is None:
                self._record_event("tool_call_start", tool=tool_name)
                try:
                    if tool_name in {"bash", "shell"} and self._terminal_session() is not None:
                        observation = self._execute_terminal_session_command(tool_input)
                    else:
                        tool_result = self.mcp_tools[tool_name].execute(tool_input)
                        observation = getattr(tool_result, "content", str(tool_result))
                    tools_succeeded += 1
                except Exception as exc:
                    observation = f"Tool error: {exc}"
                finally:
                    self._record_event("tool_call_end", tool=tool_name)
            observation = self._post_tool_observation(
                observation,
                tool_name=tool_name,
                tool_input=tool_input,
                turn_index=_turn,
                tools_attempted=tools_attempted,
            )

            messages.append({"role": "assistant", "content": last_content})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        if final_answer is None:
            final_prompt = self._turn_limit_final_prompt()
            if final_prompt:
                messages.append({"role": "user", "content": final_prompt})
                result = self._chat(messages)
                if result.input_tokens is None or result.output_tokens is None:
                    missing_usage_responses += 1
                else:
                    total_input += result.input_tokens
                    total_output += result.output_tokens
                if result.cost_usd is None:
                    missing_cost = True
                else:
                    total_cost += result.cost_usd
                token_sources.append(result.token_source)
                last_content = result.content
                final_answer = self._extract_final(last_content)
                if final_answer is None:
                    final_answer = last_content
                messages.append({"role": "assistant", "content": last_content})

        token_source = "missing"
        if token_sources:
            unique_sources = sorted(set(token_sources))
            token_source = unique_sources[0] if len(unique_sources) == 1 else ",".join(unique_sources)

        return AgentRunResult(
            content=final_answer if final_answer is not None else last_content,
            tool_calls_attempted=tools_attempted,
            tool_calls_succeeded=tools_succeeded,
            tool_names_used=tools_used,
            num_turns=max(1, len([m for m in messages if m["role"] == "assistant"])),
            input_tokens=None if missing_usage_responses else total_input,
            output_tokens=None if missing_usage_responses else total_output,
            cost_usd=None if missing_cost else total_cost,
            metadata={
                "token_source": token_source,
                "missing_usage_responses": missing_usage_responses
                + sum(1 for source in token_sources if source == "missing"),
            },
        )


__all__ = ["DSPyRLM"]
