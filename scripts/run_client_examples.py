#!/usr/bin/env python3
"""
Run a handful of example generations against a TrafficBench client.

This helper is intended for quick manual smoke testing. It exercises the
configured client using ``stream_chat_completion`` so that time-to-first-token
and token accounting behave the same way as in the profiling harness.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Iterable

from trafficbench.core.registry import ClientRegistry

DEFAULT_PROMPTS: list[str] = [
    "Summarize the benefits of offline inference with streamed decoding.",
    "Provide three creative uses for a traffic analytics benchmark suite.",
    "Explain, in one paragraph, how sampling temperature affects diversity.",
]


def _parse_key_value(pairs: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        try:
            parsed: Any = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        result[key] = parsed
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sample prompts against a TrafficBench inference client."
    )
    parser.add_argument(
        "--client",
        default="vllm",
        help="Registered client identifier (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier to pass to the client.",
    )
    parser.add_argument(
        "--base-url",
        dest="base_url",
        default=None,
        help="Optional client-specific base URL override.",
    )
    parser.add_argument(
        "--client-config",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="Extra keyword arguments for the client constructor (JSON parsed).",
    )
    parser.add_argument(
        "--param",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="Additional request parameters for stream_chat_completion (JSON parsed).",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Run client.prepare(model) before issuing prompts.",
    )
    parser.add_argument(
        "prompts",
        nargs="*",
        help="Prompts to send. Defaults to a small curated set if omitted.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    client_kwargs = _parse_key_value(args.client_config)

    import trafficbench.clients

    trafficbench.clients.ensure_registered()

    try:
        client = ClientRegistry.create(args.client, args.base_url, **client_kwargs)
    except Exception as exc:  # pragma: no cover - convenience script
        parser.error(f"failed to create client '{args.client}': {exc}")

    requests = args.prompts or DEFAULT_PROMPTS
    request_params = _parse_key_value(args.param)

    try:
        if args.prepare:
            client.prepare(args.model)  # type: ignore[attr-defined]

        for idx, prompt in enumerate(requests, 1):
            header = f"\n=== Prompt {idx} / {len(requests)} ==="
            print(header)
            print(prompt)
            print("-" * len(header))

            try:
                response = client.stream_chat_completion(args.model, prompt, **request_params)
            except Exception as exc:  # pragma: no cover - interactive use
                print(f"Request failed: {exc}", file=sys.stderr)
                continue

            usage = response.usage
            print("\nResponse:\n")
            print(response.content)
            print(
                "\nStats: "
                f"prompt_tokens={usage.prompt_tokens} "
                f"completion_tokens={usage.completion_tokens} "
                f"total_tokens={usage.total_tokens} "
                f"ttft_ms={response.time_to_first_token_ms:.1f}"
            )
    finally:
        try:
            client.close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())

