from __future__ import annotations

from ipj.orchestrate.harness.clients import AnthropicChatClient, OpenAIChatClient


class FakeAnthropicMsg:
    def __init__(self):
        self.captured = None

    def create(self, **kwargs):
        self.captured = kwargs

        class R:
            content = [type("Blk", (), {"text": "hello", "type": "text"})()]
            usage = type("U", (), {"input_tokens": 100, "output_tokens": 20})()

        return R()


class FakeAnthropicSDK:
    def __init__(self):
        self.messages = FakeAnthropicMsg()


class FakeOpenAISDK:
    class chat:  # noqa: N801
        class completions:  # noqa: N801
            captured = {}

            @classmethod
            def create(cls, **kwargs):
                cls.captured = kwargs

                class R:
                    choices = [type("C", (), {"message": type("M", (), {"content": "hi"})()})()]
                    usage = type("U", (), {"prompt_tokens": 30, "completion_tokens": 7})()

                return R()


def test_anthropic_sets_cache_control_and_cost():
    sdk = FakeAnthropicSDK()
    client = AnthropicChatClient("claude-opus-4-8", sdk=sdk)
    resp = client.complete([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    assert resp.content == "hello"
    assert resp.tokens_in == 100 and resp.tokens_out == 20
    # the system block must carry cache_control (prompt caching enabled)
    system_field = sdk.messages.captured["system"]
    assert any("cache_control" in blk for blk in system_field)


def test_openai_client_maps_response():
    client = OpenAIChatClient("gpt-5.5-mini", sdk=FakeOpenAISDK())
    resp = client.complete([{"role": "user", "content": "hi"}])
    assert resp.content == "hi"
    assert resp.tokens_in == 30 and resp.tokens_out == 7
    assert resp.dollars is None


def test_openai_rewrites_tool_role_to_user():
    # OpenAI chat.completions rejects role:"tool" without a tool_call_id; the client must remap it.
    sdk = FakeOpenAISDK()
    client = OpenAIChatClient("gpt-5.5-mini", sdk=sdk)
    client.complete(
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
            {"role": "tool", "content": "worker said 42"},
        ]
    )
    sent = sdk.chat.completions.captured["messages"]
    assert all(m["role"] != "tool" for m in sent)  # no role:"tool" reaches the SDK
    # the tool turn survived as a user turn (content preserved)
    assert any(m["role"] == "user" and m["content"] == "worker said 42" for m in sent)


def test_anthropic_alternates_and_starts_with_user():
    sdk = FakeAnthropicSDK()
    client = AnthropicChatClient("claude-opus-4-8", sdk=sdk)
    # two consecutive assistant turns (a `think` step with no call_response) -> must be coalesced,
    # and the first non-system turn is assistant -> a leading user turn must be prepended.
    client.complete(
        [
            {"role": "system", "content": "S"},
            {"role": "assistant", "content": "A1"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "U"},
        ]
    )
    sent = sdk.messages.captured["messages"]
    roles = [m["role"] for m in sent]
    assert roles[0] == "user"  # leading user turn required by Anthropic
    assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))  # strictly alternating
    # the two assistant turns were coalesced into one
    assert any(m["role"] == "assistant" and "A1" in m["content"] and "A2" in m["content"] for m in sent)
