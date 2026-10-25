from __future__ import annotations

import pytest

from ipj.orchestrate.harness.actions import ACTION_TYPES, Action, ActionParseError, parse_action


def test_parse_plain_json_call_worker():
    a = parse_action('{"action":"call_worker","worker":"local-fast","sub_prompt":"sort this","context":"c"}')
    assert a.type == "call_worker"
    assert a.target == "local-fast"
    assert a.args == {"sub_prompt": "sort this", "context": "c"}
    assert a.raw.startswith("{")


def test_parse_fenced_json_call_tool():
    a = parse_action('reasoning...\n```json\n{"action":"call_tool","tool":"web_search","args":{"query":"x"}}\n```')
    assert a.type == "call_tool"
    assert a.target == "web_search"
    assert a.args == {"query": "x"}


def test_parse_multiple_objects_takes_first():
    # A model that batches several action objects in one turn -> execute the FIRST one.
    multi = '{"action":"bash","command":"cd /testbed && grep -rn x"}\n{"action":"bash","command":"ls"}'
    a = parse_action(multi)
    assert a.type == "bash"
    assert a.text == "cd /testbed && grep -rn x"
    assert a.raw.count('"action"') == 1  # raw = the single executed action, not the whole blob


def test_parse_prose_then_json():
    a = parse_action('Let me look at the code first.\n{"action":"bash","command":"ls /testbed"}')
    assert a.type == "bash" and a.text == "ls /testbed"


def test_parse_nested_args_not_truncated():
    a = parse_action('{"action":"call_tool","tool":"t","args":{"a":{"b":1},"c":[1,2]}}')
    assert a.args == {"a": {"b": 1}, "c": [1, 2]}


def test_parse_tool_call_tag_finish():
    a = parse_action('<tool_call>{"action":"finish","answer":"42"}</tool_call>')
    assert a.type == "finish"
    assert a.text == "42"


def test_parse_bash_and_think():
    assert parse_action('{"action":"bash","command":"ls -la"}').text == "ls -la"
    assert parse_action('{"action":"think","text":"plan"}').type == "think"


def test_unknown_action_raises():
    with pytest.raises(ActionParseError):
        parse_action('{"action":"teleport"}')


def test_no_json_raises():
    with pytest.raises(ActionParseError):
        parse_action("I have no idea what to do.")


@pytest.mark.parametrize(
    "emission",
    [
        '{"action":"call_tool","args":{"q":"x"}}',  # typed but no "tool"
        '{"action":"call_worker","sub_prompt":"do it"}',  # typed but no "worker"
        '{"action":"delegate_task"}',  # typed but no "worker"
        '{"action":"call_tool","tool":""}',  # present but empty
    ],
)
def test_missing_required_key_raises_parse_error(emission):
    # A well-typed-but-incomplete action must surface as ActionParseError (the brain's catchable
    # fallback), never a raw KeyError that escapes parse_action and crashes the run loop.
    with pytest.raises(ActionParseError):
        parse_action(emission)


def test_action_types_are_six():
    assert ACTION_TYPES == {"think", "bash", "call_tool", "call_worker", "delegate_task", "finish"}
    a = Action(type="think")
    assert a.type == "think"
    assert a.target is None
    assert a.args == {}
