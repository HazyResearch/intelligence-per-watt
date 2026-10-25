"""Stateful per-turn router for v2: an OpenAI-compatible proxy that routes each
terminus-2 request to a local vLLM or the cloud upstream by a per-task turn budget.

stdlib + requests only — no ipw import, so this module loads in an ipw-free env.
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests


def task_key(messages: list[dict]) -> str:
    """Stable per-task key from the first message. terminus interpolates the task
    instruction into messages[0] and preserves it across summarization."""
    first = messages[0] if messages else {}
    raw = json.dumps(first, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class RoutingState:
    """Per-task turn counter + escalation latch + per-tier request/token tally.
    A forced tier (set by the pre-task router) takes precedence over the turn budget."""

    def __init__(self, local_turn_budget: int = 15, defer_every_k: int = 0) -> None:
        self.budget = local_turn_budget
        # defer_every_k>0 enables the self-assessment deferral: every k local turns, ask the
        # model itself CONTINUE/DEFER and latch to cloud on DEFER (smarter than the fixed budget).
        self.defer_every_k = defer_every_k
        self._task: dict[str, dict] = {}
        self._forced: dict[str, str] = {}
        self.tokens = {"local": {"in": 0, "out": 0}, "cloud": {"in": 0, "out": 0}}
        self.requests = {"local": 0, "cloud": 0}
        self.classifier_tokens: int | None = 0
        self.classifier_dollars: float | None = 0.0
        self.deferrals: list[tuple[str, bool, str]] = []  # (task_key, deferred?, raw) per probe
        self._lock = threading.Lock()

    def force_tier(
        self,
        key: str,
        tier: str,
        classifier_tokens: int | None = 0,
        classifier_dollars: float | None = 0.0,
    ) -> None:
        with self._lock:
            self._forced[key] = tier
            self._record_classifier(classifier_tokens, classifier_dollars)

    def record_classifier(
        self,
        classifier_tokens: int | None,
        classifier_dollars: float | None,
    ) -> None:
        with self._lock:
            self._record_classifier(classifier_tokens, classifier_dollars)

    def _record_classifier(
        self,
        classifier_tokens: int | None,
        classifier_dollars: float | None,
    ) -> None:
        if self.classifier_tokens is None or classifier_tokens is None:
            self.classifier_tokens = None
        else:
            self.classifier_tokens += classifier_tokens
        if self.classifier_dollars is None or classifier_dollars is None:
            self.classifier_dollars = None
        else:
            self.classifier_dollars += classifier_dollars

    def is_routed(self, key: str) -> bool:
        with self._lock:
            return key in self._forced

    def route(self, key: str) -> str:
        with self._lock:
            s = self._task.setdefault(key, {"count": 0, "escalated": False})
            s["count"] += 1
            if key in self._forced:
                return self._forced[key]
            if s["escalated"] or s["count"] > self.budget:
                s["escalated"] = True
                return "cloud"
            return "local"

    def should_check_deferral(self, key: str) -> bool:
        """True on a still-local task whose turn count just hit a multiple of defer_every_k."""
        with self._lock:
            if self.defer_every_k <= 0 or key in self._forced:
                return False
            s = self._task.get(key)
            if s is None or s["escalated"]:
                return False
            return s["count"] % self.defer_every_k == 0

    def record_deferral(self, key: str, deferred: bool, tokens_in: int, tokens_out: int, raw: str) -> None:
        """Log a self-assessment probe; its tokens count as classifier (routing) overhead."""
        with self._lock:
            self.deferrals.append((key, deferred, raw))
            if self.classifier_tokens is not None:
                self.classifier_tokens += tokens_in + tokens_out

    @property
    def n_deferred(self) -> int:
        with self._lock:
            return sum(1 for _, d, _ in self.deferrals if d)

    def record(self, tier: str, tokens_in: int | None, tokens_out: int | None) -> None:
        """Record one routed request and its optional usage.

        Request counts do not depend on providers returning ``usage``. Once a token
        direction is unknown for a tier, its aggregate stays unknown rather than
        treating the missing observation as zero.
        """
        values = {
            "in": tokens_in if _valid_token_count(tokens_in) else None,
            "out": tokens_out if _valid_token_count(tokens_out) else None,
        }
        with self._lock:
            self.requests[tier] += 1
            for direction, value in values.items():
                current = self.tokens[tier][direction]
                self.tokens[tier][direction] = (
                    None if current is None or value is None else current + value
                )

    @property
    def n_escalated(self) -> int:
        with self._lock:
            return sum(1 for s in self._task.values() if s["escalated"])

    @property
    def n_local(self) -> int:
        with self._lock:
            return sum(1 for t in self._forced.values() if t == "local")

    @property
    def n_cloud(self) -> int:
        with self._lock:
            return sum(1 for t in self._forced.values() if t == "cloud")


def _valid_token_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _normalize_cloud_req(req: dict, model: str) -> dict:
    """Adapt a raw terminus-2 request body to provider quirks before forwarding to cloud.

    OpenAI's gpt-5.x models reject ``max_tokens`` (require ``max_completion_tokens``) and only
    accept the default ``temperature``. The proxy forwards terminus-2's body verbatim, so without
    this an escalated turn to gpt-5.x 400s. Gemini's OpenAI-compatible endpoint accepts the raw
    body, so we gate on the model name and leave non-OpenAI tiers untouched.
    """
    if not model.lower().startswith(("gpt-5", "openai/gpt-5", "o1", "o3", "o4")):
        return req
    out = dict(req)
    if "max_tokens" in out and "max_completion_tokens" not in out:
        out["max_completion_tokens"] = out.pop("max_tokens")
    if out.get("temperature") not in (None, 1, 1.0):
        out.pop("temperature", None)  # gpt-5.x only supports the default temperature
    return out


def handle_chat(body: dict, state: RoutingState, upstreams: dict, forward_fn, router=None, deferral=None) -> dict:
    """Route one /chat/completions request to local or cloud, forward it, tally tokens.
    If `router` is set, classify the task once (on its first request) to force the tier.
    If `deferral` is set, every defer_every_k local turns ask the model CONTINUE/DEFER and
    latch the task to cloud on DEFER (`deferral(messages) -> DeferralDecision`)."""
    messages = body.get("messages", [])
    key = task_key(messages)
    # Once-per-task: safe because terminus-2 issues a task's turns sequentially (no overlap per task_key).
    if router is not None and not state.is_routed(key):
        from . import skill_router  # lazy: keeps proxy import-light + monkeypatchable
        task_text = messages[0].get("content", "") if messages else ""
        try:
            d = skill_router.route_task(
                task_text, base_url=router["base_url"], api_key=router["api_key"],
                model=router["model"], margin=router["margin"],
                price_in=router["price_in"], price_out=router["price_out"],
            )
            state.force_tier(key, d.tier, d.classifier_tokens, d.classifier_dollars)
        except Exception:  # classifier failure -> fall back to the turn-budget path
            state.record_classifier(None, None)
    tier = state.route(key)
    # Self-assessment deferral: on a still-local task, every k turns ask the model itself
    # whether to keep going or hand off, and latch to cloud on DEFER.
    if deferral is not None and tier == "local" and state.should_check_deferral(key):
        d = deferral(messages)
        state.record_deferral(key, d.defer, d.tokens_in, d.tokens_out, d.raw)
        if d.defer:
            state.force_tier(key, "cloud")
            tier = "cloud"
    if tier == "cloud":
        url, key_, model = upstreams["cloud"]
        req = _normalize_cloud_req({**body, "model": model}, model)
    else:
        url, key_ = upstreams["local"]
        req = body
    headers = {"Authorization": f"Bearer {key_}", "Content-Type": "application/json"}
    try:
        resp = forward_fn(url.rstrip("/") + "/chat/completions", headers, req)
        usage = resp.get("usage")
    except Exception:
        # The upstream was selected and called, so its request/cost cannot disappear
        # merely because no parseable usage response came back.
        state.record(tier, None, None)
        raise
    if isinstance(usage, dict):
        tokens_in = usage.get("prompt_tokens")
        tokens_out = usage.get("completion_tokens")
    else:
        tokens_in = tokens_out = None
    state.record(tier, tokens_in, tokens_out)
    return resp


def _default_forward(url: str, headers: dict, body: dict) -> dict:
    r = requests.post(url, json=body, headers=headers, timeout=600)
    r.raise_for_status()
    return r.json()


class ProxyServer:
    """OpenAI-compatible proxy: set as terminus-2's api_base; routes each request per RoutingState."""

    def __init__(self, state: RoutingState, upstreams: dict, forward_fn=_default_forward,
                 router=None, deferral=None) -> None:
        self.state = state
        self.upstreams = upstreams
        self.forward_fn = forward_fn
        self.router = router
        self.deferral = deferral
        self._httpd = None
        self._thread = None

    def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        state, upstreams, forward_fn, router, deferral = (
            self.state, self.upstreams, self.forward_fn, self.router, self.deferral)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output pristine
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                try:
                    resp = handle_chat(body, state, upstreams, forward_fn, router, deferral)
                    payload, code = json.dumps(resp).encode(), 200
                except Exception as exc:  # surface upstream failures to the agent as 502
                    payload, code = json.dumps({"error": str(exc)}).encode(), 502
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self._httpd.server_address[1]

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
