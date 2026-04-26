# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "uvicorn[standard]",
#   "httpx",
#   "tenacity>=8.0",
#   "pyyaml",
# ]
# ///
"""
resilient-llm-bridge — a local HTTP proxy that adds resilience between
OpenAI-compatible clients (Codex, opencode, hermes, OpenAI Agents SDK,
custom integrations) and OpenAI-compatible upstreams (LiteLLM proxy,
vLLM, llama.cpp, Together, Fireworks, Groq, NaN Builders, etc.).

What "resilience" means concretely:

* Per-client **profiles** select which transformations apply. Codex needs
  heavy request-body massaging for its OpenAI-native quirks; opencode and
  hermes are well-behaved chat/completions clients that need almost
  nothing. Same proxy, different paths, different behavior.

* **Retry policy** for transient upstream failures (5xx, 429, network
  errors) with exponential backoff. Configurable per profile.

* **Recovery** for the empty-content failure modes that plague
  thinking-mode models without `--reasoning-parser`: detects the
  thinking-overflow pattern, runs a two-tier `continue_final_message`
  fallback per Qwen's official guidance, and stitches the recovered
  answer back into the original response shape.

* **Rate limiting** per upstream — token bucket for RPM and a semaphore
  for max concurrent requests. Queues, doesn't reject. Different upstreams
  can have different limits.

* **vLLM bug workarounds** for clients that hit `/v1/responses` (Codex,
  OpenAI Agents SDK): rewrites the malformed parallel-tool-call SSE that
  vLLM bug #39426 produces, until vLLM PR #39600 ships.

* **Operational fixes** that bite any proxy chain talking to
  Cloudflare-fronted upstreams (gzip mismatch → forced
  `Accept-Encoding: identity`).

* **Live observability**: `/usage/stream` SSE feed for token counters,
  recent-request history, dashboard at `/` for at-a-glance status.

Endpoints:
  GET  /                        HTML dashboard (status + activity)
  GET  /health                  liveness check
  GET  /usage/stream            SSE feed of per-completion usage
  GET  /activity/stream         SSE feed of recent request metadata
  POST /{profile}/v1/responses  profile-routed Responses-API
  POST /{profile}/v1/chat/completions   profile-routed chat-completions
  *    /{profile}/v1/{path}     profile-routed catch-all (embeddings, audio…)
  *    /v1/{path}               default-profile catch-all (backward compat)

Configuration:
  See `BRIDGE_CONFIG_PATH` env var (defaults to
  ~/.config/resilient-llm-bridge/config.yaml).
  If the file is absent, sensible defaults are used (NaN Builders as the
  upstream, profiles for codex / opencode / hermes / default).
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PORT = int(os.environ.get("PORT", "4242"))
DEFAULT_HOST = os.environ.get("HOST", "127.0.0.1")
DEFAULT_CONFIG_PATH = Path(
    os.environ.get(
        "BRIDGE_CONFIG_PATH",
        os.path.expanduser("~/.config/resilient-llm-bridge/config.yaml"),
    )
)


@dataclass
class UpstreamConfig:
    """A single upstream LLM provider — URL + rate-limit/retry policy."""
    name: str
    url: str
    rate_limit_rpm: int = 0           # 0 disables RPM limiting
    rate_limit_concurrent: int = 0    # 0 disables concurrency limiting
    retry_max_attempts: int = 3
    retry_initial_wait: float = 1.0
    retry_max_wait: float = 20.0


# Feature flags. Each profile's `features` list selects which to apply.
# Order in the list does not matter; transforms run in a fixed pipeline.
ALL_FEATURES = {
    # Request transforms (responses-API + chat/completions)
    "qwen_sampling_defaults",       # inject Qwen3 sampling (temp/top_p/top_k/min_p/presence_penalty)
    "drop_oai_only_fields",         # drop OpenAI-only fields the upstream rejects
    "effort_to_thinking_budget",    # translate reasoning.effort → thinking_token_budget
    # Request transforms (responses-API only)
    "normalize_responses_input",    # reshape input items for strict validators
    "drop_namespace_tools",         # drop namespace/MCP/web_search/image_gen tools
    "force_serial_tool_calls",      # parallel_tool_calls=false override
    # Stream rewriters (responses-API only)
    "parallel_tool_sse_fix",        # vLLM #39426 workaround for parallel tool SSE
    # Recovery triggers (responses-API; chat/completions has its own)
    "thinking_overflow_recovery",   # incomplete + max_output_tokens → 2-tier recovery
    "silent_completion_recovery",   # completed + no message item → 2-tier recovery
    "truncated_content_recovery",   # length + content cut mid-thought → continue
    "empty_with_stop_retry",        # empty + stop → one cheap retry
}


@dataclass
class ProfileConfig:
    """A named profile that selects an upstream and a set of features."""
    name: str
    upstream: str
    features: set[str] = field(default_factory=set)

    def has(self, feature: str) -> bool:
        return feature in self.features


@dataclass
class BridgeConfig:
    upstreams: dict[str, UpstreamConfig]
    profiles: dict[str, ProfileConfig]
    default_profile: str

    def profile(self, name: str | None) -> ProfileConfig:
        """Resolve a profile by name, falling back to the default profile."""
        if name and name in self.profiles:
            return self.profiles[name]
        return self.profiles[self.default_profile]


def _builtin_defaults() -> BridgeConfig:
    """Sensible defaults when no config file is present.

    Single upstream pointing at NaN Builders with their advertised limits
    (100 RPM, 5 concurrent), and profiles for every client we ship support
    for: codex (full transform stack), hermes / opencode (light), and
    default (catch-all with mild defenses).
    """
    nan = UpstreamConfig(
        name="nan",
        url="https://api.nan.builders/v1",
        rate_limit_rpm=100,
        rate_limit_concurrent=5,
        retry_max_attempts=3,
    )
    return BridgeConfig(
        upstreams={"nan": nan},
        profiles={
            "default": ProfileConfig(
                name="default",
                upstream="nan",
                features={
                    "qwen_sampling_defaults",
                    "drop_oai_only_fields",
                    "effort_to_thinking_budget",
                    "thinking_overflow_recovery",
                    "silent_completion_recovery",
                    "truncated_content_recovery",
                    "empty_with_stop_retry",
                },
            ),
            "codex": ProfileConfig(
                name="codex",
                upstream="nan",
                features={
                    "qwen_sampling_defaults",
                    "drop_oai_only_fields",
                    "effort_to_thinking_budget",
                    "normalize_responses_input",
                    "drop_namespace_tools",
                    "force_serial_tool_calls",
                    "parallel_tool_sse_fix",
                    "thinking_overflow_recovery",
                    "silent_completion_recovery",
                    "truncated_content_recovery",
                    "empty_with_stop_retry",
                },
            ),
            "hermes": ProfileConfig(
                name="hermes",
                upstream="nan",
                features={
                    "qwen_sampling_defaults",
                    "drop_oai_only_fields",
                    "thinking_overflow_recovery",
                    "silent_completion_recovery",
                    "truncated_content_recovery",
                    "empty_with_stop_retry",
                },
            ),
            "opencode": ProfileConfig(
                name="opencode",
                upstream="nan",
                features={
                    "qwen_sampling_defaults",
                    "drop_oai_only_fields",
                    "thinking_overflow_recovery",
                    "silent_completion_recovery",
                    "truncated_content_recovery",
                    "empty_with_stop_retry",
                },
            ),
        },
        default_profile="default",
    )


def _load_config() -> BridgeConfig:
    """Load configuration from YAML, falling back to bundled defaults."""
    if not DEFAULT_CONFIG_PATH.exists():
        return _builtin_defaults()
    try:
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        print(
            f"[config] failed to load {DEFAULT_CONFIG_PATH}: {exc}; using defaults",
            flush=True,
        )
        return _builtin_defaults()

    upstreams: dict[str, UpstreamConfig] = {}
    for name, body in (raw.get("upstreams") or {}).items():
        upstreams[name] = UpstreamConfig(
            name=name,
            url=str(body.get("url") or "").rstrip("/"),
            rate_limit_rpm=int(body.get("rate_limit_rpm") or 0),
            rate_limit_concurrent=int(body.get("rate_limit_concurrent") or 0),
            retry_max_attempts=int(body.get("retry_max_attempts") or 3),
            retry_initial_wait=float(body.get("retry_initial_wait") or 1.0),
            retry_max_wait=float(body.get("retry_max_wait") or 20.0),
        )
    if not upstreams:
        upstreams = _builtin_defaults().upstreams

    profiles: dict[str, ProfileConfig] = {}
    for name, body in (raw.get("profiles") or {}).items():
        feats = set(body.get("features") or [])
        invalid = feats - ALL_FEATURES
        if invalid:
            print(
                f"[config] profile {name!r} references unknown features: {invalid}",
                flush=True,
            )
            feats -= invalid
        profiles[name] = ProfileConfig(
            name=name,
            upstream=str(body.get("upstream") or ""),
            features=feats,
        )
    if not profiles:
        profiles = _builtin_defaults().profiles

    default_profile = str(raw.get("default_profile") or "default")
    if default_profile not in profiles:
        default_profile = next(iter(profiles))

    return BridgeConfig(
        upstreams=upstreams, profiles=profiles, default_profile=default_profile
    )


CONFIG = _load_config()


# =============================================================================
# Constants used by request transforms
# =============================================================================

_SYSTEM_LIKE_ROLES = {"system", "developer"}
_SYSTEM_NOTE_PREFIX = "[system note]\n"

# Maps reasoning.effort → thinking_token_budget (vLLM extra_body).
# Numbers track the Qwen3-30B-A3B-Thinking-2507 model card's anti-verbose
# defaults. Currently a no-op against deployments missing
# `--reasoning-parser qwen3` on vLLM (the param is silently dropped).
_EFFORT_TO_THINKING_BUDGET = {"low": 1024, "medium": 4096, "high": 8192, "xhigh": 16384}
_DEFAULT_THINKING_BUDGET = 8192

# Qwen3 thinking-mode sampling. From generation_config.json shipped with
# Qwen3-30B-A3B-Thinking-2507. presence_penalty is left at 0 (Qwen warns
# that high values cause language mixing on coding tasks).
_QWEN_TOP_LEVEL_DEFAULTS = {"temperature": 0.6, "top_p": 0.95, "presence_penalty": 0.0}
_QWEN_EXTRA_BODY_DEFAULTS = {"top_k": 20, "min_p": 0}

# OpenAI-only fields that LiteLLM/vLLM upstreams typically reject with 400.
_DROP_FIELDS = {
    "client_metadata",
    "include",
    "text",
    "prompt_cache_key",
    "store",
    "service_tier",
    "user",
    "metadata",
}

# Qwen-official force-close phrase, recommended in their quickstart for
# the recovery prefill. Don't change — it's the literal string the
# model was trained against.
_QWEN_FORCE_CLOSE_THINK = (
    "Considering the limited time by the user, I have to give the "
    "solution based on the thinking directly now."
)
_RECOVERY_MAX_TOKENS = 4096

# Sentinel for content that ends mid-thought (no terminal punctuation),
# used by truncated_content_recovery detection.
_TERMINAL_PUNCT = (".", "?", "!", "。", "?", "!", "”", "\"", "'", "”")

# Patterns for detecting "fake invocation" artifacts. When happy injects
# `CHANGE_TITLE_INSTRUCTION` into the first turn, Qwen sometimes "calls"
# the pseudo-tool by writing the invocation as PLAIN TEXT in the
# assistant message — e.g. `happy__change_title(title="Initial Greeting")`
# — instead of producing a real function_call item or a real answer to
# the user. The bare invocation passes the "non-empty content" check
# but is useless to the user, so silent-completion recovery has to flag
# it as if no message had been emitted.
_FAKE_INVOCATION_LINE_RE = re.compile(
    r"^[a-zA-Z_][\w]*\s*\([^()]*(?:\([^()]*\)[^()]*)*\)\s*;?$"
)
_HAPPY_PSEUDO_TOOL_RE = re.compile(r"\bhappy__\w+\s*\(")


def _is_fake_invocation_message_item(item: Any) -> bool:
    """A responses-API output item that is a `message` whose entire text
    content is a fake-invocation artifact. Used to strip such items
    from the final output when recovery synthesizes a real answer."""
    if not isinstance(item, dict) or item.get("type") != "message":
        return False
    text = ""
    for part in item.get("content") or []:
        if isinstance(part, dict) and part.get("type") == "output_text":
            text += part.get("text") or ""
    return _looks_like_fake_invocation(text)


def _looks_like_fake_invocation(text: str) -> bool:
    """The model emitted a function-call invocation as plain text.

    Returns True when the entire emitted message is just a function call
    expression (no surrounding prose) or contains a `happy__*(...)`
    pseudo-tool invocation. Both shapes mean the model didn't actually
    answer the user.
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    # Any happy__ pseudo-tool invocation in the text — never a real answer.
    if _HAPPY_PSEUDO_TOOL_RE.search(stripped):
        return True
    # Whole message is a single function-call-looking expression.
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if not lines:
        return False
    return all(_FAKE_INVOCATION_LINE_RE.match(ln) for ln in lines)


# =============================================================================
# Rate limiting (token bucket + concurrency semaphore per upstream)
# =============================================================================


class _TokenBucket:
    """Simple async token bucket. Refills at `rate` tokens per second."""

    def __init__(self, capacity: int, rpm: int) -> None:
        self.capacity = max(1, capacity)
        self.rate = max(0.0, rpm / 60.0)
        self._tokens = float(self.capacity)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, n: int = 1) -> None:
        if self.rate <= 0:
            return
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._updated) * self.rate
                )
                self._updated = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
                missing = n - self._tokens
                wait = missing / self.rate
            await asyncio.sleep(min(wait, 5.0))

    def snapshot(self) -> tuple[float, int]:
        """Return (current_token_count, capacity) for the dashboard."""
        now = time.monotonic()
        cur = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
        return cur, self.capacity


class _UpstreamGate:
    """Combined rate limiter + concurrency limiter for a single upstream."""

    def __init__(self, cfg: UpstreamConfig) -> None:
        self.cfg = cfg
        self.bucket = _TokenBucket(cfg.rate_limit_rpm or 1, cfg.rate_limit_rpm or 0)
        self.sema = asyncio.Semaphore(
            cfg.rate_limit_concurrent if cfg.rate_limit_concurrent > 0 else 1024
        )
        # Track in-flight count for the dashboard.
        self._in_flight = 0
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "_UpstreamGate":
        await self.bucket.acquire()
        await self.sema.acquire()
        async with self._lock:
            self._in_flight += 1
        return self

    async def __aexit__(self, *exc: Any) -> None:
        async with self._lock:
            self._in_flight -= 1
        self.sema.release()

    def snapshot(self) -> dict[str, Any]:
        cur, cap = self.bucket.snapshot()
        return {
            "name": self.cfg.name,
            "url": self.cfg.url,
            "rpm_capacity": self.cfg.rate_limit_rpm,
            "rpm_remaining": round(cur, 1),
            "concurrent_limit": self.cfg.rate_limit_concurrent,
            "concurrent_in_flight": self._in_flight,
        }


_UPSTREAM_GATES: dict[str, _UpstreamGate] = {
    name: _UpstreamGate(cfg) for name, cfg in CONFIG.upstreams.items()
}


def _gate_for(profile: ProfileConfig) -> _UpstreamGate:
    return _UPSTREAM_GATES[profile.upstream]


def _upstream_url(profile: ProfileConfig) -> str:
    return CONFIG.upstreams[profile.upstream].url


# =============================================================================
# Retry policy (tenacity-based, per-upstream config)
# =============================================================================


_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class _UpstreamHTTPError(Exception):
    """Raised for upstream non-2xx responses we want to retry."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"upstream HTTP {status}: {body[:200]}")


def _retry_policy(cfg: UpstreamConfig) -> AsyncRetrying:
    """Build a per-upstream tenacity retry policy."""
    return AsyncRetrying(
        stop=stop_after_attempt(cfg.retry_max_attempts),
        wait=wait_exponential_jitter(
            initial=cfg.retry_initial_wait, max=cfg.retry_max_wait
        ),
        retry=retry_if_exception_type(
            (
                _UpstreamHTTPError,
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
                httpx.PoolTimeout,
            )
        ),
        reraise=True,
    )


# =============================================================================
# Activity tracking + usage broadcast (for dashboard)
# =============================================================================

USAGE_RING_SIZE = int(os.environ.get("BRIDGE_USAGE_RING_SIZE", "5000"))
ACTIVITY_RING_SIZE = int(os.environ.get("BRIDGE_ACTIVITY_RING_SIZE", "5000"))

_usage_history: deque[dict] = deque(maxlen=USAGE_RING_SIZE)
_usage_subscribers: set[asyncio.Queue[dict]] = set()
_activity_history: deque[dict] = deque(maxlen=ACTIVITY_RING_SIZE)
_activity_subscribers: set[asyncio.Queue[dict]] = set()
_started_at = time.time()

# Lifetime counters for recovery firings + retries. These are cheap so we
# track them since process start; for time-windowed views the dashboard
# aggregates from `_activity_history` / `_usage_history`.
_recovery_counts: dict[str, int] = {
    "thinking_overflow": 0,
    "silent_completion": 0,
    "fake_invocation": 0,
    "truncated_content": 0,
    "empty_with_stop_retry": 0,
}
_retry_counts: dict[str, int] = {"retried": 0, "gave_up": 0}


def _record_recovery(kind: str) -> None:
    if kind in _recovery_counts:
        _recovery_counts[kind] += 1


def _broadcast_usage(record: dict) -> None:
    _usage_history.append(record)
    print(
        f"usage profile={record.get('profile')} model={record.get('model')}"
        f" in={record.get('input_tokens')} out={record.get('output_tokens')}",
        flush=True,
    )
    for queue in list(_usage_subscribers):
        try:
            queue.put_nowait(record)
        except asyncio.QueueFull:
            pass


def _broadcast_activity(record: dict) -> None:
    _activity_history.append(record)
    for queue in list(_activity_subscribers):
        try:
            queue.put_nowait(record)
        except asyncio.QueueFull:
            pass


def _extract_usage(profile_name: str, model_hint: str | None, payload: dict) -> dict | None:
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return None
    record = {
        "ts": time.time(),
        "profile": profile_name,
        "model": model_hint or payload.get("model"),
        "input_tokens": int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }
    if record["total_tokens"] == 0:
        record["total_tokens"] = record["input_tokens"] + record["output_tokens"]
    return record


# =============================================================================
# Request body transforms
# =============================================================================


def _content_to_string(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n\n".join(parts)
    return str(content or "")


def _normalize_responses_input(
    input_value: Any, *, fold_leading_into_instructions: bool
) -> tuple[list, str]:
    """Reshape `/v1/responses` input array for strict upstream validators.

    See the original module docstring's bug list for what each rule fixes.
    Returns (new_input_list, folded_instructions_text).
    """
    if not isinstance(input_value, list) or not input_value:
        return input_value, ""
    result: list = []
    folded_chunks: list[str] = []
    seen_non_system = False
    for item in input_value:
        if not isinstance(item, dict):
            result.append(item)
            seen_non_system = True
            continue
        item_type = item.get("type", "message")
        if item_type == "reasoning":
            continue
        if item_type in ("function_call", "function_call_output"):
            seen_non_system = True
            result.append(item)
            continue
        if item_type != "message":
            seen_non_system = True
            result.append(item)
            continue
        role = item.get("role")
        is_system_like = role in _SYSTEM_LIKE_ROLES
        if is_system_like and not seen_non_system:
            text = _content_to_string(item.get("content"))
            if fold_leading_into_instructions:
                if text:
                    folded_chunks.append(text)
                continue
            if text:
                result.append({"role": "system", "content": text})
            continue
        if is_system_like:
            text = _content_to_string(item.get("content"))
            if text:
                result.append({"role": "user", "content": _SYSTEM_NOTE_PREFIX + text})
            continue
        seen_non_system = True
        text = _content_to_string(item.get("content"))
        if role == "assistant" and not text:
            continue
        result.append({"role": role or "user", "content": text})
    return result, "\n\n".join(folded_chunks)


def _apply_qwen_sampling_defaults(body: dict) -> None:
    if body.get("temperature") in (None, 0, 0.0):
        body["temperature"] = _QWEN_TOP_LEVEL_DEFAULTS["temperature"]
    if body.get("top_p") is None:
        body["top_p"] = _QWEN_TOP_LEVEL_DEFAULTS["top_p"]
    if body.get("presence_penalty") in (None, 0, 0.0):
        body["presence_penalty"] = _QWEN_TOP_LEVEL_DEFAULTS["presence_penalty"]
    extra = body.get("extra_body")
    if not isinstance(extra, dict):
        extra = {}
    for key, value in _QWEN_EXTRA_BODY_DEFAULTS.items():
        extra.setdefault(key, value)
    body["extra_body"] = extra


def _apply_effort_budget(body: dict) -> None:
    effort: str | None = None
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and isinstance(reasoning.get("effort"), str):
        effort = reasoning["effort"]
    if effort is None and isinstance(body.get("reasoning_effort"), str):
        effort = body["reasoning_effort"]
    if effort:
        budget = _EFFORT_TO_THINKING_BUDGET.get(effort, _DEFAULT_THINKING_BUDGET)
    else:
        budget = _DEFAULT_THINKING_BUDGET
    extra = body.get("extra_body")
    if not isinstance(extra, dict):
        extra = {}
    chat_kwargs = extra.get("chat_template_kwargs")
    if not isinstance(chat_kwargs, dict):
        chat_kwargs = {}
    chat_kwargs.setdefault("enable_thinking", True)
    chat_kwargs.setdefault("thinking_token_budget", budget)
    extra["chat_template_kwargs"] = chat_kwargs
    body["extra_body"] = extra


def _drop_oai_only_fields(body: dict) -> None:
    for field_name in _DROP_FIELDS:
        body.pop(field_name, None)


def _filter_tools_to_function_only(body: dict) -> None:
    tools = body.get("tools")
    if not isinstance(tools, list):
        return
    kept = [t for t in tools if isinstance(t, dict) and t.get("type") == "function"]
    if kept:
        body["tools"] = kept
    else:
        body.pop("tools", None)
        body.pop("tool_choice", None)


def _force_serial_tool_calls(body: dict) -> None:
    body["parallel_tool_calls"] = False


def _apply_request_transforms(body: dict, profile: ProfileConfig, kind: str) -> dict:
    """Apply every feature in the profile that's relevant to this request kind.

    `kind` is "responses" or "chat_completions" — some transforms are
    responses-API-specific (input reshape, parallel_tool_calls override).
    """
    if profile.has("normalize_responses_input") and kind == "responses" and "input" in body:
        instructions = body.get("instructions")
        has_instructions = isinstance(instructions, str) and instructions.strip()
        new_input, folded = _normalize_responses_input(
            body["input"], fold_leading_into_instructions=bool(has_instructions)
        )
        body["input"] = new_input
        if folded:
            body["instructions"] = (instructions or "").rstrip() + "\n\n" + folded
    if profile.has("drop_namespace_tools"):
        _filter_tools_to_function_only(body)
    if profile.has("qwen_sampling_defaults"):
        _apply_qwen_sampling_defaults(body)
    if profile.has("effort_to_thinking_budget"):
        _apply_effort_budget(body)
    if profile.has("force_serial_tool_calls") and kind == "responses":
        _force_serial_tool_calls(body)
    if profile.has("drop_oai_only_fields"):
        _drop_oai_only_fields(body)
    return body


# =============================================================================
# Recovery framework
# =============================================================================


def _input_to_chat_messages(input_value: Any, instructions: Any) -> list[dict]:
    """Convert /v1/responses `input` to chat/completions `messages` for recovery."""
    messages: list[dict] = []
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions.strip()})
    if not isinstance(input_value, list):
        return messages
    for item in input_value:
        if not isinstance(item, dict) or item.get("type", "message") != "message":
            continue
        role = item.get("role")
        if not isinstance(role, str):
            continue
        text = _content_to_string(item.get("content"))
        if text:
            messages.append({"role": role, "content": text})
    return messages


async def _post_chat_for_text(
    body: dict, headers: dict[str, str], profile: ProfileConfig
) -> str | None:
    """Run a non-streaming chat/completions request and extract content."""
    url = f"{_upstream_url(profile)}/chat/completions"
    cfg = CONFIG.upstreams[profile.upstream]
    try:
        async for attempt in _retry_policy(cfg):
            with attempt:
                async with _gate_for(profile):
                    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                        r = await client.post(url, json=body, headers=headers)
                if r.status_code in _RETRYABLE_STATUS:
                    raise _UpstreamHTTPError(r.status_code, r.text)
                if r.status_code >= 400:
                    return None
                payload = r.json()
                choices = payload.get("choices") if isinstance(payload, dict) else None
                if not choices:
                    return None
                message = (choices[0] or {}).get("message") or {}
                content = (message.get("content") or "").strip()
                return content or None
    except (RetryError, httpx.HTTPError, ValueError):
        return None
    return None


async def _recover_thinking_overflow(
    original_body: dict,
    partial_reasoning: str,
    headers: dict[str, str],
    profile: ProfileConfig,
) -> str | None:
    """Two-tier recovery for /v1/responses thinking-overflow.

    Tier 2: continue_final_message with reasoning prefill (Qwen-official).
    Tier 3: fresh request with enable_thinking=false, no prefill.
    """
    messages = _input_to_chat_messages(
        original_body.get("input"), original_body.get("instructions")
    )
    if not messages:
        return None
    # Tier 2
    prefill = (
        "<think>\n"
        + partial_reasoning.strip()
        + "\n"
        + _QWEN_FORCE_CLOSE_THINK
        + "\n</think>\n\n"
    )
    tier2_body: dict = {
        "model": original_body.get("model"),
        "stream": False,
        "messages": messages + [{"role": "assistant", "content": prefill}],
        "max_tokens": _RECOVERY_MAX_TOKENS,
        "extra_body": {
            "continue_final_message": True,
            "add_generation_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    if profile.has("qwen_sampling_defaults"):
        _apply_qwen_sampling_defaults(tier2_body)
    text = await _post_chat_for_text(tier2_body, headers, profile)
    if text:
        return text
    # Tier 3
    tier3_body: dict = {
        "model": original_body.get("model"),
        "stream": False,
        "messages": messages,
        "max_tokens": _RECOVERY_MAX_TOKENS,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if profile.has("qwen_sampling_defaults"):
        _apply_qwen_sampling_defaults(tier3_body)
    return await _post_chat_for_text(tier3_body, headers, profile)


async def _recover_truncated_content(
    original_body: dict,
    partial_content: str,
    headers: dict[str, str],
    profile: ProfileConfig,
) -> str | None:
    """Resume a content cut mid-thought via continue_final_message.

    The model already produced *some* answer; we just want the rest. Feed
    the partial content as a prefilled assistant turn and ask vLLM to
    continue. No reasoning involved; assumes the upstream already moved
    past `</think>`.
    """
    messages = _input_to_chat_messages(
        original_body.get("input"), original_body.get("instructions")
    )
    if not messages:
        return None
    cont_body: dict = {
        "model": original_body.get("model"),
        "stream": False,
        "messages": messages + [{"role": "assistant", "content": partial_content}],
        "max_tokens": _RECOVERY_MAX_TOKENS,
        "extra_body": {
            "continue_final_message": True,
            "add_generation_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    if profile.has("qwen_sampling_defaults"):
        _apply_qwen_sampling_defaults(cont_body)
    extra = await _post_chat_for_text(cont_body, headers, profile)
    if not extra:
        return None
    return partial_content + extra


def _is_responses_overflow(response_obj: dict, message_emitted: bool) -> bool:
    if not isinstance(response_obj, dict):
        return False
    if response_obj.get("status") != "incomplete":
        return False
    reason = (response_obj.get("incomplete_details") or {}).get("reason")
    return reason == "max_output_tokens" and not message_emitted


def _is_responses_silent_completion(
    response_obj: dict, message_emitted: bool, emitted_text: str = ""
) -> bool:
    """Detect a `status: completed` response that didn't actually answer.

    Two sub-cases:

    1. **No message item.** Happy injects `CHANGE_TITLE_INSTRUCTION` into
       the first user message of every session. Qwen3 reasons about
       whether to call the title tool, decides "no real task here",
       emits *only* the reasoning, and finishes — the user sees a
       successful but silent completion.

    2. **Fake-invocation artifact.** Same scenario, except the model
       *does* open a message envelope and writes the pseudo-tool
       invocation (e.g. `happy__change_title(title="Initial Greeting")`)
       as plain text. `message_emitted` is True but the content isn't
       an answer.

    Recovery (same as overflow): `continue_final_message` with the
    reasoning prefilled, asking the model to actually answer the user.
    """
    if not isinstance(response_obj, dict):
        return False
    if response_obj.get("status") != "completed":
        return False
    if not message_emitted:
        return True
    return _looks_like_fake_invocation(emitted_text)


def _detect_truncated_message(message_text: str, finish_reason: str | None) -> bool:
    """Heuristic for "answer was cut mid-thought".

    Looks for finish_reason="length" (or equivalent in responses-API)
    plus content that doesn't end on a terminal punctuation mark and is
    long enough to count as a real attempt.
    """
    if finish_reason != "length":
        return False
    text = (message_text or "").rstrip()
    if len(text) < 50:
        return False
    return not text.endswith(_TERMINAL_PUNCT)


# =============================================================================
# SSE helpers
# =============================================================================


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")


def _build_outgoing_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {
        "content-type": "application/json",
        "accept": "text/event-stream",
        # Force `identity` so upstream doesn't gzip the SSE — re-compressing
        # along the way is a frequent source of "stream closed before
        # response.completed" failures.
        "accept-encoding": "identity",
    }
    auth = request.headers.get("authorization")
    if auth:
        headers["authorization"] = auth
    return headers


# =============================================================================
# /v1/responses streaming with profile-aware rewrites and recovery
# =============================================================================


async def _stream_responses(
    body: dict,
    headers: dict[str, str],
    profile: ProfileConfig,
) -> AsyncIterator[bytes]:
    model_hint = body.get("model")
    cfg = CONFIG.upstreams[profile.upstream]
    upstream_url = f"{_upstream_url(profile)}/responses"

    # State for vLLM #39426 SSE rewrite (fc_state) and overflow recovery.
    fc_state: dict[int, dict] = {}
    reasoning_accum = ""
    message_text_accum = ""
    message_emitted = False
    completed_payload: dict | None = None
    do_parallel_fix = profile.has("parallel_tool_sse_fix")

    async def _open_stream():
        # tenacity on the entire stream open is fine — we don't replay
        # mid-stream, we only retry the connect/headers phase.
        async for attempt in _retry_policy(cfg):
            with attempt:
                async with _gate_for(profile):
                    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, read=None))
                    response = await client.send(
                        client.build_request(
                            "POST", upstream_url, json=body, headers=headers
                        ),
                        stream=True,
                    )
                    if response.status_code in _RETRYABLE_STATUS:
                        body_bytes = await response.aread()
                        await response.aclose()
                        await client.aclose()
                        raise _UpstreamHTTPError(
                            response.status_code, body_bytes.decode("utf-8", errors="ignore")
                        )
                    return client, response
        raise RuntimeError("unreachable")  # pragma: no cover

    try:
        client, response = await _open_stream()
    except (RetryError, _UpstreamHTTPError) as exc:
        status = getattr(exc, "status", 502)
        body_text = getattr(exc, "body", str(exc))
        yield _sse({"type": "error", "status": status, "body": body_text})
        return

    try:
        if response.status_code >= 400:
            error_text = (await response.aread()).decode("utf-8", errors="ignore")
            yield _sse({"type": "error", "status": response.status_code, "body": error_text})
            return

        buffer = b""
        async for chunk in response.aiter_bytes():
            if not chunk:
                continue
            buffer += chunk
            while b"\n\n" in buffer:
                event, buffer = buffer.split(b"\n\n", 1)
                payload: dict | None = None
                for line in event.splitlines():
                    if not line.startswith(b"data: "):
                        continue
                    try:
                        parsed = json.loads(line[6:])
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if isinstance(parsed, dict):
                        payload = parsed
                        break
                if payload is None:
                    yield event + b"\n\n"
                    continue

                etype = payload.get("type")

                # Track reasoning text accumulator + whether actual
                # answer text was emitted. Use `output_text.delta` (real
                # tokens being emitted into a message) as the signal —
                # `output_item.added` for a message can fire with empty
                # content when the model opens a message envelope but
                # then emits a function_call instead, leaving the user
                # with no answer.
                if etype == "response.reasoning_text.delta":
                    delta = payload.get("delta")
                    if isinstance(delta, str):
                        reasoning_accum += delta
                if etype == "response.output_text.delta":
                    delta = payload.get("delta")
                    if isinstance(delta, str) and delta:
                        message_emitted = True
                        message_text_accum += delta

                # vLLM #39426 SSE rewrite (only when feature enabled)
                if do_parallel_fix:
                    if etype == "response.output_item.added":
                        item = payload.get("item") or {}
                        if item.get("type") == "function_call":
                            idx = payload.get("output_index")
                            fc_state[idx] = {
                                "id": item.get("id"),
                                "call_id": item.get("call_id"),
                                "name": item.get("name"),
                                "args": "",
                            }
                        yield _sse(payload)
                        continue
                    if etype == "response.function_call_arguments.delta":
                        idx = payload.get("output_index")
                        st = fc_state.get(idx)
                        if st is not None:
                            st["args"] += payload.get("delta") or ""
                        yield _sse(payload)
                        continue
                    if etype == "response.function_call_arguments.done":
                        # Drop vLLM's bogus concatenated `done`.
                        continue
                    if etype == "response.output_item.done":
                        idx = payload.get("output_index")
                        st = fc_state.get(idx)
                        if st is not None:
                            seq = payload.get("sequence_number")
                            yield _sse(
                                {
                                    "type": "response.function_call_arguments.done",
                                    "item_id": st["id"],
                                    "output_index": idx,
                                    "name": st["name"],
                                    "arguments": st["args"],
                                    "sequence_number": seq,
                                    "model": payload.get("model"),
                                }
                            )
                            fixed = dict(payload)
                            fixed["item"] = {
                                "type": "function_call",
                                "id": st["id"],
                                "call_id": st["call_id"],
                                "name": st["name"],
                                "arguments": st["args"],
                                "status": "completed",
                            }
                            yield _sse(fixed)
                            fc_state.pop(idx, None)
                            continue

                if etype == "response.completed":
                    # Buffer for potential post-stream recovery.
                    completed_payload = payload
                    continue

                yield _sse(payload)
    finally:
        try:
            await response.aclose()
        finally:
            await client.aclose()

    if completed_payload is None:
        return

    response_obj = completed_payload.get("response") or {}

    # Try recoveries in order. Each is gated by the profile feature flag.
    # Both overflow and silent-completion use the same continue_final_message
    # recovery — the only difference is the trigger.
    recovered_text: str | None = None
    overflow = profile.has("thinking_overflow_recovery") and _is_responses_overflow(
        response_obj, message_emitted
    )
    silent = profile.has("silent_completion_recovery") and _is_responses_silent_completion(
        response_obj, message_emitted, message_text_accum
    )
    fake_invocation_kicker = silent and message_emitted  # message text was the artifact
    if (overflow or silent) and reasoning_accum.strip():
        recovered_text = await _recover_thinking_overflow(
            body, reasoning_accum, headers, profile
        )
        if recovered_text:
            if overflow:
                _record_recovery("thinking_overflow")
            elif fake_invocation_kicker:
                _record_recovery("fake_invocation")
            else:
                _record_recovery("silent_completion")

    if recovered_text:
        # Synthesize the responses-API events for the recovered message
        # and emit a patched response.completed with status=completed.
        fake_id = f"msg_recovery_{int(time.time() * 1000)}"
        fake_idx = len(response_obj.get("output", []) or [])
        seq = (completed_payload.get("sequence_number") or 0) + 1
        model_field = response_obj.get("model")
        yield _sse(
            {
                "type": "response.output_item.added",
                "output_index": fake_idx,
                "item": {
                    "id": fake_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "status": "in_progress",
                },
                "sequence_number": seq,
                "model": model_field,
            }
        )
        seq += 1
        yield _sse(
            {
                "type": "response.content_part.added",
                "item_id": fake_id,
                "output_index": fake_idx,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
                "sequence_number": seq,
                "model": model_field,
            }
        )
        seq += 1
        yield _sse(
            {
                "type": "response.output_text.delta",
                "item_id": fake_id,
                "output_index": fake_idx,
                "content_index": 0,
                "delta": recovered_text,
                "sequence_number": seq,
                "model": model_field,
            }
        )
        seq += 1
        yield _sse(
            {
                "type": "response.output_text.done",
                "item_id": fake_id,
                "output_index": fake_idx,
                "content_index": 0,
                "text": recovered_text,
                "sequence_number": seq,
                "model": model_field,
            }
        )
        seq += 1
        done_message_item = {
            "id": fake_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": recovered_text, "annotations": []}
            ],
        }
        yield _sse(
            {
                "type": "response.content_part.done",
                "item_id": fake_id,
                "output_index": fake_idx,
                "content_index": 0,
                "part": done_message_item["content"][0],
                "sequence_number": seq,
                "model": model_field,
            }
        )
        seq += 1
        yield _sse(
            {
                "type": "response.output_item.done",
                "output_index": fake_idx,
                "item": done_message_item,
                "sequence_number": seq,
                "model": model_field,
            }
        )
        seq += 1
        fixed_response = dict(response_obj)
        fixed_response["status"] = "completed"
        fixed_response["incomplete_details"] = None
        cleaned_output = [
            item
            for item in (response_obj.get("output") or [])
            if not _is_fake_invocation_message_item(item)
        ]
        fixed_response["output"] = cleaned_output + [done_message_item]
        fixed_completed = dict(completed_payload)
        fixed_completed["response"] = fixed_response
        fixed_completed["sequence_number"] = seq
        record = _extract_usage(profile.name, model_hint, fixed_response)
        if record:
            _broadcast_usage(record)
        yield _sse(fixed_completed)
        return

    # No recovery applied — emit the original completed event.
    record = _extract_usage(profile.name, model_hint, response_obj)
    if record:
        _broadcast_usage(record)
    yield _sse(completed_payload)


# =============================================================================
# /v1/chat/completions streaming
# =============================================================================


async def _stream_chat_completions(
    body: dict,
    headers: dict[str, str],
    profile: ProfileConfig,
) -> AsyncIterator[bytes]:
    """Forward chat/completions SSE byte-for-byte, sniffing usage."""
    model_hint = body.get("model")
    cfg = CONFIG.upstreams[profile.upstream]
    upstream_url = f"{_upstream_url(profile)}/chat/completions"

    async def _open_stream():
        async for attempt in _retry_policy(cfg):
            with attempt:
                async with _gate_for(profile):
                    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, read=None))
                    response = await client.send(
                        client.build_request(
                            "POST", upstream_url, json=body, headers=headers
                        ),
                        stream=True,
                    )
                    if response.status_code in _RETRYABLE_STATUS:
                        body_bytes = await response.aread()
                        await response.aclose()
                        await client.aclose()
                        raise _UpstreamHTTPError(
                            response.status_code, body_bytes.decode("utf-8", errors="ignore")
                        )
                    return client, response
        raise RuntimeError("unreachable")  # pragma: no cover

    try:
        client, response = await _open_stream()
    except (RetryError, _UpstreamHTTPError) as exc:
        status = getattr(exc, "status", 502)
        body_text = getattr(exc, "body", str(exc))
        yield _sse({"type": "error", "status": status, "body": body_text})
        return

    try:
        if response.status_code >= 400:
            error_text = (await response.aread()).decode("utf-8", errors="ignore")
            yield _sse({"type": "error", "status": response.status_code, "body": error_text})
            return
        buffer = b""
        async for chunk in response.aiter_bytes():
            if not chunk:
                continue
            yield chunk
            buffer += chunk
            while b"\n\n" in buffer:
                event, buffer = buffer.split(b"\n\n", 1)
                for line in event.splitlines():
                    if not line.startswith(b"data: "):
                        continue
                    raw = line[6:].strip()
                    if raw == b"[DONE]":
                        continue
                    try:
                        payload = json.loads(raw)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if not isinstance(payload, dict):
                        continue
                    if isinstance(payload.get("usage"), dict):
                        record = _extract_usage(profile.name, model_hint, payload)
                        if record:
                            _broadcast_usage(record)
    finally:
        try:
            await response.aclose()
        finally:
            await client.aclose()


# =============================================================================
# Non-stream POST helpers (apply recovery on the response object directly)
# =============================================================================


async def _post_responses_nonstream(
    body: dict, headers: dict[str, str], profile: ProfileConfig
) -> tuple[int, dict]:
    cfg = CONFIG.upstreams[profile.upstream]
    upstream_url = f"{_upstream_url(profile)}/responses"
    last_status = 502
    last_body = ""
    try:
        async for attempt in _retry_policy(cfg):
            with attempt:
                async with _gate_for(profile):
                    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
                        r = await client.post(upstream_url, json=body, headers=headers)
                last_status = r.status_code
                last_body = r.text
                if r.status_code in _RETRYABLE_STATUS:
                    raise _UpstreamHTTPError(r.status_code, r.text)
                payload = (
                    r.json()
                    if r.headers.get("content-type", "").startswith("application/json")
                    else {}
                )
                return r.status_code, payload
    except (RetryError, _UpstreamHTTPError):
        return last_status, {"error": {"message": last_body}}
    except httpx.HTTPError as e:
        return 502, {"error": {"message": f"upstream error: {e}"}}
    return 502, {"error": {"message": "unreachable"}}


# =============================================================================
# FastAPI endpoints
# =============================================================================


app = FastAPI(title="resilient-llm-bridge")


def _redact_body(body: Any, max_bytes: int = 4096) -> dict:
    """Return a copy of the request body with bulky / sensitive fields
    summarised so the dashboard can render it without leaking prompt
    content. Keeps every top-level field so you can see the full shape
    of what each client sends; replaces user/assistant/system content
    with length placeholders.
    """
    if not isinstance(body, dict):
        return {"_": "<non-dict body>"}

    def _summary_str(s: str) -> str:
        return f"<str {len(s)} chars>"

    def _summary_content(c: Any) -> Any:
        if isinstance(c, str):
            return _summary_str(c)
        if isinstance(c, list):
            return f"<list[{len(c)}] (text/parts redacted)>"
        return f"<{type(c).__name__}>"

    def _redact_message(m: Any) -> Any:
        if not isinstance(m, dict):
            return m
        out = dict(m)
        if "content" in out:
            out["content"] = _summary_content(out["content"])
        return out

    def _redact_input_item(item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        out = dict(item)
        if "content" in out:
            out["content"] = _summary_content(out["content"])
        if "input" in out:
            out["input"] = _summary_content(out["input"])
        return out

    # Deep-copy: callers will mutate the original body via
    # `_apply_request_transforms`. Without this we'd see post-transform
    # values when the redacted body was supposed to capture the
    # client-sent shape.
    redacted = copy.deepcopy(body)
    if isinstance(redacted.get("messages"), list):
        redacted["messages"] = [_redact_message(m) for m in redacted["messages"]]
    if isinstance(redacted.get("input"), list):
        redacted["input"] = [_redact_input_item(i) for i in redacted["input"]]
    elif isinstance(redacted.get("input"), str):
        redacted["input"] = _summary_str(redacted["input"])
    if isinstance(redacted.get("instructions"), str) and len(redacted["instructions"]) > 200:
        redacted["instructions"] = _summary_str(redacted["instructions"])
    if isinstance(redacted.get("system"), str) and len(redacted["system"]) > 200:
        redacted["system"] = _summary_str(redacted["system"])
    # Tools: keep names + count, drop big JSON schemas.
    if isinstance(redacted.get("tools"), list):
        names = []
        for t in redacted["tools"]:
            if isinstance(t, dict):
                fn = t.get("function") or {}
                names.append(t.get("name") or fn.get("name") or t.get("type") or "?")
        redacted["tools"] = {"_count": len(redacted["tools"]), "_names": names[:20]}

    encoded = json.dumps(redacted, ensure_ascii=False, default=str)
    if len(encoded) > max_bytes:
        # Body still too big after redaction — common for huge tool blobs
        # or input arrays. Truncate the JSON string but keep it parseable
        # by appending a sentinel.
        return {"_truncated_to": max_bytes, "_preview": encoded[:max_bytes]}
    return redacted


def _inspect_thinking_params(body: Any, kind: str) -> dict:
    """Snapshot the thinking-related fields a client sent on a request.

    Used by the dashboard so we can see at a glance how Codex / opencode
    / hermes wire up the budget — and immediately spot wrong placements
    (e.g. `thinking_token_budget` under `chat_template_kwargs`, where
    vLLM silently ignores it).

    The returned shape is intentionally compact for the activity feed.
    Only the fields that exist are included.
    """
    if not isinstance(body, dict):
        return {}
    out: dict = {}
    if model := body.get("model"):
        out["model"] = model
    if (mt := body.get("max_tokens")) is not None:
        out["max_tokens"] = mt
    if (mot := body.get("max_output_tokens")) is not None:
        out["max_output_tokens"] = mot
    # /v1/responses-style reasoning hint (Codex sends this).
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        if (eff := reasoning.get("effort")) is not None:
            out["reasoning.effort"] = eff
        if (mrt := reasoning.get("max_tokens")) is not None:
            out["reasoning.max_tokens"] = mrt
    # chat/completions: top-level reasoning_effort.
    if (eff := body.get("reasoning_effort")) is not None:
        out["reasoning_effort"] = eff
    extra = body.get("extra_body")
    if isinstance(extra, dict):
        # Top-level extra_body.thinking_token_budget — the placement that
        # actually works on vLLM.
        if (b := extra.get("thinking_token_budget")) is not None:
            out["extra_body.thinking_token_budget"] = b
        ctk = extra.get("chat_template_kwargs")
        if isinstance(ctk, dict):
            if (et := ctk.get("enable_thinking")) is not None:
                out["chat_template_kwargs.enable_thinking"] = et
            # The WRONG placements — flag them so the dashboard makes
            # the bug obvious.
            for wrong in ("thinking_token_budget", "thinking_budget"):
                if (b := ctk.get(wrong)) is not None:
                    out[f"chat_template_kwargs.{wrong} (DROPPED)"] = b
    return out


def _record_activity(
    profile: ProfileConfig,
    path: str,
    method: str,
    status: int,
    duration_ms: float,
    params: dict | None = None,
    body: dict | None = None,
) -> None:
    record: dict = {
        "ts": time.time(),
        "profile": profile.name,
        "upstream": profile.upstream,
        "path": path,
        "method": method,
        "status": status,
        "duration_ms": round(duration_ms, 1),
    }
    if params:
        record["params"] = params
    if body is not None:
        record["body"] = body
    _broadcast_activity(record)


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "uptime_s": round(time.time() - _started_at, 1),
        "profiles": list(CONFIG.profiles.keys()),
        "upstreams": [g.snapshot() for g in _UPSTREAM_GATES.values()],
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _aggregate_window(activity_rows: list[dict], usage_rows: list[dict]) -> dict:
    """Aggregate a slice of activity + usage records into a stats blob."""
    durations = [r["duration_ms"] for r in activity_rows if "duration_ms" in r]
    statuses = [r.get("status", 0) for r in activity_rows]
    by_profile: dict[str, dict] = {}
    by_model: dict[str, dict] = {}

    def _bucket(d: dict, key: str) -> dict:
        if key not in d:
            d[key] = {
                "requests": 0,
                "errors": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "duration_ms": [],
            }
        return d[key]

    for row in activity_rows:
        b = _bucket(by_profile, row.get("profile") or "?")
        b["requests"] += 1
        if (row.get("status") or 0) >= 400:
            b["errors"] += 1
        if "duration_ms" in row:
            b["duration_ms"].append(row["duration_ms"])
    for row in usage_rows:
        b_p = _bucket(by_profile, row.get("profile") or "?")
        b_m = _bucket(by_model, row.get("model") or "?")
        for b in (b_p, b_m):
            b["tokens_in"] += int(row.get("input_tokens") or 0)
            b["tokens_out"] += int(row.get("output_tokens") or 0)

    # Finalize percentiles per bucket.
    def _finalize(d: dict) -> dict:
        for key, b in d.items():
            ds = b.pop("duration_ms")
            b["p50_ms"] = round(_percentile(ds, 0.50), 1)
            b["p95_ms"] = round(_percentile(ds, 0.95), 1)
        return d

    return {
        "requests": len(activity_rows),
        "errors": sum(1 for s in statuses if s >= 400),
        "errors_4xx": sum(1 for s in statuses if 400 <= s < 500),
        "errors_5xx": sum(1 for s in statuses if s >= 500),
        "tokens_in": sum(int(r.get("input_tokens") or 0) for r in usage_rows),
        "tokens_out": sum(int(r.get("output_tokens") or 0) for r in usage_rows),
        "p50_ms": round(_percentile(durations, 0.50), 1),
        "p95_ms": round(_percentile(durations, 0.95), 1),
        "p99_ms": round(_percentile(durations, 0.99), 1),
        "by_profile": _finalize(by_profile),
        "by_model": _finalize(by_model),
    }


@app.get("/stats")
async def stats() -> dict:
    now = time.time()
    windows = {"1m": 60.0, "5m": 300.0, "15m": 900.0, "1h": 3600.0}
    activity = list(_activity_history)
    usage = list(_usage_history)
    out: dict = {
        "now": now,
        "uptime_s": round(now - _started_at, 1),
        "lifetime": {
            "requests": len(activity),
            "tokens_in": sum(int(r.get("input_tokens") or 0) for r in usage),
            "tokens_out": sum(int(r.get("output_tokens") or 0) for r in usage),
            "errors": sum(1 for r in activity if (r.get("status") or 0) >= 400),
            "recoveries": dict(_recovery_counts),
        },
        "windows": {},
        "upstreams": [g.snapshot() for g in _UPSTREAM_GATES.values()],
        "profiles": [
            {"name": p.name, "upstream": p.upstream, "features": sorted(p.features)}
            for p in CONFIG.profiles.values()
        ],
        "history": {
            "activity": activity[-200:],
            "usage": usage[-200:],
        },
    }
    for label, span in windows.items():
        cutoff = now - span
        a = [r for r in activity if r.get("ts", 0) >= cutoff]
        u = [r for r in usage if r.get("ts", 0) >= cutoff]
        out["windows"][label] = _aggregate_window(a, u)
    return out


@app.get("/usage/stream")
async def usage_stream() -> StreamingResponse:
    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=128)
    _usage_subscribers.add(queue)

    async def gen():
        try:
            if _usage_history:
                yield _sse(_usage_history[-1])
            while True:
                try:
                    record = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield b": ping\n\n"
                    continue
                yield _sse(record)
        finally:
            _usage_subscribers.discard(queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/activity/stream")
async def activity_stream() -> StreamingResponse:
    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=128)
    _activity_subscribers.add(queue)

    async def gen():
        try:
            for record in list(_activity_history):
                yield _sse(record)
            while True:
                try:
                    record = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield b": ping\n\n"
                    continue
                yield _sse(record)
        finally:
            _activity_subscribers.discard(queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    return HTMLResponse(content=_dashboard_html(), status_code=200)


# Profile-routed endpoints. Order matters in FastAPI route resolution: the
# specific paths must come before the catch-all.

async def _handle_responses(
    request: Request, profile: ProfileConfig
) -> StreamingResponse | JSONResponse:
    started = time.monotonic()
    body = await request.json()
    inspected = _inspect_thinking_params(body, "responses")
    redacted = _redact_body(body)
    body = _apply_request_transforms(body, profile, kind="responses")
    want_stream = bool(body.get("stream", False))
    headers = _build_outgoing_headers(request)
    if want_stream:
        async def _gen():
            async for chunk in _stream_responses(body, headers, profile):
                yield chunk
            _record_activity(
                profile,
                f"/{profile.name}/v1/responses",
                "POST",
                200,
                (time.monotonic() - started) * 1000,
                params=inspected,
                body=redacted,
            )
        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    status, payload = await _post_responses_nonstream(body, headers, profile)
    if status < 400 and isinstance(payload, dict):
        # Apply recovery to the non-streaming response object. A "message
        # item" only counts if it actually carries non-empty text content
        # — a bare `{type:"message", content:[]}` happens when the model
        # opens a message envelope but emits only a function_call, and
        # we want recovery to fire in that case too.
        output_items = payload.get("output") or []
        message_emitted = False
        message_text_accum = ""
        for item in output_items:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content") or []
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "output_text":
                    text = part.get("text") or ""
                    if text.strip():
                        message_emitted = True
                        message_text_accum += text
        overflow_ns = profile.has("thinking_overflow_recovery") and _is_responses_overflow(
            payload, message_emitted
        )
        silent_ns = profile.has("silent_completion_recovery") and _is_responses_silent_completion(
            payload, message_emitted, message_text_accum
        )
        if overflow_ns or silent_ns:
            partial_reasoning = ""
            for item in output_items:
                if isinstance(item, dict) and item.get("type") == "reasoning":
                    for c in item.get("content") or []:
                        if isinstance(c, dict):
                            partial_reasoning += c.get("text") or ""
            if partial_reasoning.strip():
                recovered = await _recover_thinking_overflow(
                    body, partial_reasoning, headers, profile
                )
                if recovered:
                    if overflow_ns:
                        _record_recovery("thinking_overflow")
                    elif message_emitted:
                        _record_recovery("fake_invocation")
                    else:
                        _record_recovery("silent_completion")
                    # Drop any fake-invocation message item from the
                    # original output so the client doesn't render
                    # `happy__change_title(...)` alongside the real
                    # answer. Reasoning/function_call items stay.
                    cleaned_items = [
                        item
                        for item in output_items
                        if not _is_fake_invocation_message_item(item)
                    ]
                    payload["status"] = "completed"
                    payload["incomplete_details"] = None
                    payload["output"] = cleaned_items + [
                        {
                            "id": f"msg_recovery_{int(time.time() * 1000)}",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": recovered,
                                    "annotations": [],
                                }
                            ],
                        }
                    ]
        record = _extract_usage(profile.name, body.get("model"), payload)
        if record:
            _broadcast_usage(record)
    _record_activity(
        profile,
        f"/{profile.name}/v1/responses",
        "POST",
        status,
        (time.monotonic() - started) * 1000,
        params=inspected,
        body=redacted,
    )
    return JSONResponse(content=payload, status_code=status)


async def _handle_chat_completions(
    request: Request, profile: ProfileConfig
) -> StreamingResponse | JSONResponse:
    started = time.monotonic()
    body = await request.json()
    inspected = _inspect_thinking_params(body, "chat_completions")
    redacted = _redact_body(body)
    body = _apply_request_transforms(body, profile, kind="chat_completions")
    want_stream = bool(body.get("stream", False))
    headers = _build_outgoing_headers(request)
    if want_stream:
        async def _gen():
            async for chunk in _stream_chat_completions(body, headers, profile):
                yield chunk
            _record_activity(
                profile,
                f"/{profile.name}/v1/chat/completions",
                "POST",
                200,
                (time.monotonic() - started) * 1000,
                params=inspected,
                body=redacted,
            )
        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    cfg = CONFIG.upstreams[profile.upstream]
    upstream_url = f"{_upstream_url(profile)}/chat/completions"
    last_status = 502
    last_body = ""
    payload: dict = {"error": {"message": "unreachable"}}
    try:
        async for attempt in _retry_policy(cfg):
            with attempt:
                async with _gate_for(profile):
                    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
                        r = await client.post(upstream_url, json=body, headers=headers)
                last_status = r.status_code
                last_body = r.text
                if r.status_code in _RETRYABLE_STATUS:
                    raise _UpstreamHTTPError(r.status_code, r.text)
                payload = (
                    r.json()
                    if r.headers.get("content-type", "").startswith("application/json")
                    else {}
                )
                break
    except (RetryError, _UpstreamHTTPError):
        payload = {"error": {"message": last_body}}
    except httpx.HTTPError as e:
        last_status = 502
        payload = {"error": {"message": f"upstream error: {e}"}}
    if last_status < 400 and isinstance(payload, dict):
        choice = (payload.get("choices") or [{}])[0]
        finish = choice.get("finish_reason")
        message = choice.get("message") or {}
        content = message.get("content") or ""
        # Truncated content recovery for chat/completions: model produced
        # *some* answer but was cut mid-thought. Resume via continue_final_message.
        if profile.has("truncated_content_recovery") and _detect_truncated_message(
            content, finish
        ):
            extra = await _recover_truncated_content(body, content, headers, profile)
            if extra:
                message["content"] = extra
                choice["finish_reason"] = "stop"
                _record_recovery("truncated_content")
        # Empty + stop retry: provider hiccup or stream parsing miss.
        # One cheap retry — if it returns the same empty result we keep
        # the original (don't loop). Tool-call responses are skipped
        # because empty content there is intentional.
        elif (
            profile.has("empty_with_stop_retry")
            and finish == "stop"
            and not content.strip()
            and not message.get("tool_calls")
        ):
            try:
                async with _gate_for(profile):
                    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                        retry_r = await client.post(upstream_url, json=body, headers=headers)
                if retry_r.status_code < 400:
                    retry_payload = retry_r.json()
                    retry_choice = (retry_payload.get("choices") or [{}])[0]
                    retry_message = retry_choice.get("message") or {}
                    retry_content = retry_message.get("content") or ""
                    if retry_content.strip():
                        payload = retry_payload
                        _record_recovery("empty_with_stop_retry")
            except (httpx.HTTPError, ValueError):
                pass  # keep original empty response
        record = _extract_usage(profile.name, body.get("model"), payload)
        if record:
            _broadcast_usage(record)
    _record_activity(
        profile,
        f"/{profile.name}/v1/chat/completions",
        "POST",
        last_status,
        (time.monotonic() - started) * 1000,
        params=inspected,
        body=redacted,
    )
    return JSONResponse(content=payload, status_code=last_status)


async def _handle_passthrough(request: Request, profile: ProfileConfig, path: str):
    started = time.monotonic()
    upstream_url = f"{_upstream_url(profile)}/{path}"
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length"}
    }
    body_bytes = await request.body()
    cfg = CONFIG.upstreams[profile.upstream]
    last_status = 502
    response_content = b""
    response_headers: dict[str, str] = {}
    response_media_type: str | None = None
    try:
        async for attempt in _retry_policy(cfg):
            with attempt:
                async with _gate_for(profile):
                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(600.0, read=None)
                    ) as client:
                        r = await client.request(
                            request.method,
                            upstream_url,
                            content=body_bytes,
                            headers=headers,
                            params=dict(request.query_params),
                        )
                last_status = r.status_code
                response_content = r.content
                response_headers = {
                    k: v for k, v in r.headers.items()
                    if k.lower() not in {"content-encoding", "content-length", "transfer-encoding"}
                }
                response_media_type = r.headers.get("content-type")
                if r.status_code in _RETRYABLE_STATUS:
                    raise _UpstreamHTTPError(r.status_code, r.text)
                break
    except (RetryError, _UpstreamHTTPError):
        pass
    except httpx.HTTPError as e:
        last_status = 502
        response_content = json.dumps(
            {"error": {"message": f"upstream error: {e}"}}
        ).encode("utf-8")
        response_media_type = "application/json"
    _record_activity(
        profile,
        f"/{profile.name}/v1/{path}",
        request.method,
        last_status,
        (time.monotonic() - started) * 1000,
    )
    return Response(
        content=response_content,
        status_code=last_status,
        headers=response_headers,
        media_type=response_media_type,
    )


@app.post("/{profile_name}/v1/responses")
async def profile_responses(profile_name: str, request: Request):
    profile = CONFIG.profile(profile_name)
    return await _handle_responses(request, profile)


@app.post("/{profile_name}/v1/chat/completions")
async def profile_chat_completions(profile_name: str, request: Request):
    profile = CONFIG.profile(profile_name)
    return await _handle_chat_completions(request, profile)


@app.api_route(
    "/{profile_name}/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    include_in_schema=False,
)
async def profile_passthrough(profile_name: str, path: str, request: Request):
    profile = CONFIG.profile(profile_name)
    return await _handle_passthrough(request, profile, path)


# Backward-compat: requests without a profile prefix go to the default profile.

@app.post("/v1/responses")
async def default_responses(request: Request):
    return await _handle_responses(request, CONFIG.profile(None))


@app.post("/v1/chat/completions")
async def default_chat_completions(request: Request):
    return await _handle_chat_completions(request, CONFIG.profile(None))


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    include_in_schema=False,
)
async def default_passthrough(path: str, request: Request):
    return await _handle_passthrough(request, CONFIG.profile(None), path)


# =============================================================================
# Dashboard (HTML at /)
# =============================================================================


def _dashboard_html() -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>resilient-llm-bridge :: {DEFAULT_PORT}</title>
  <style>
    :root {{
      --bg: #0a0a0a; --panel: #111; --panel2: #161616;
      --fg: #d4d4d4; --dim: #6e7681; --line: #1f1f1f;
      --accent: #79c0ff; --accent2: #d2a8ff;
      --good: #56d364; --warn: #f0883e; --bad: #ff7b72;
    }}
    * {{ box-sizing: border-box; }}
    body {{ background: var(--bg); color: var(--fg);
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      padding: 1.2rem; max-width: 1500px; margin: 0 auto;
      font-size: 13px; line-height: 1.45; }}
    h1 {{ font-size: 1.5rem; margin: 0; color: var(--accent); letter-spacing: -0.5px; }}
    h2 {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.12em;
      color: var(--dim); margin: 1.6rem 0 0.5rem 0;
      border-bottom: 1px solid var(--line); padding-bottom: 0.4rem; font-weight: 500; }}
    .header {{ display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap; }}
    .port {{ color: var(--good); font-size: 1.1rem; font-weight: 500; }}
    .uptime {{ color: var(--dim); font-size: 0.9rem; }}
    .pulse {{ display: inline-block; width: 8px; height: 8px;
      background: var(--good); border-radius: 50%; margin-right: 0.4rem;
      animation: pulse 1.6s ease-in-out infinite; }}
    @keyframes pulse {{ 0%, 100% {{ opacity: 0.4; }} 50% {{ opacity: 1; }} }}
    .uptime-bar {{ flex: 1; }}

    /* KPI grid */
    .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
      gap: 0.6rem; margin-top: 0.5rem; }}
    .kpi {{ background: var(--panel); border: 1px solid var(--line);
      border-radius: 6px; padding: 0.75rem 0.9rem; position: relative; overflow: hidden; }}
    .kpi-label {{ color: var(--dim); font-size: 0.7rem; text-transform: uppercase;
      letter-spacing: 0.08em; }}
    .kpi-value {{ font-size: 1.55rem; color: var(--accent); font-weight: 500;
      margin-top: 0.2rem; line-height: 1.1; }}
    .kpi-sub {{ color: var(--dim); font-size: 0.78rem; margin-top: 0.3rem; }}
    .kpi.good .kpi-value {{ color: var(--good); }}
    .kpi.warn .kpi-value {{ color: var(--warn); }}
    .kpi.bad .kpi-value {{ color: var(--bad); }}
    .kpi.accent2 .kpi-value {{ color: var(--accent2); }}
    .kpi-spark {{ position: absolute; bottom: 0; left: 0; right: 0;
      opacity: 0.5; height: 30px; }}

    /* Window selector */
    .windows {{ display: inline-flex; gap: 0.3rem; margin-left: auto; }}
    .windows button {{ background: transparent; border: 1px solid var(--line);
      color: var(--dim); padding: 0.2rem 0.6rem; border-radius: 4px;
      cursor: pointer; font: inherit; font-size: 0.75rem; }}
    .windows button.active {{ color: var(--accent); border-color: var(--accent); }}
    .windows button:hover {{ color: var(--fg); }}

    /* Two-column section */
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; margin-top: 0.5rem; }}
    @media (max-width: 900px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
    .card {{ background: var(--panel); border: 1px solid var(--line);
      border-radius: 6px; padding: 0.9rem 1rem; }}
    .card h3 {{ margin: 0 0 0.5rem 0; font-size: 0.78rem;
      color: var(--dim); text-transform: uppercase; letter-spacing: 0.08em;
      font-weight: 500; }}

    /* Tables */
    table {{ width: 100%; border-collapse: collapse; }}
    td, th {{ text-align: left; padding: 0.35rem 0.5rem; border-bottom:
      1px solid var(--panel2); vertical-align: top; }}
    th {{ color: var(--dim); font-weight: normal; font-size: 0.78rem; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    code {{ background: var(--panel2); padding: 0.08rem 0.32rem; border-radius: 2px;
      font-size: 0.86em; }}

    /* Status colors */
    .status-2xx {{ color: var(--good); }}
    .status-4xx {{ color: var(--warn); }}
    .status-5xx {{ color: var(--bad); }}
    .lat-fast {{ color: var(--good); }}
    .lat-mid {{ color: var(--warn); }}
    .lat-slow {{ color: var(--bad); }}

    /* Param chips */
    .params {{ font-size: 0.85em; color: var(--dim); }}
    .params .pk {{ color: var(--accent); }}
    .params .pv {{ color: var(--fg); }}
    .params .dropped {{ color: var(--bad); font-weight: 500; }}
    .params .pair {{ display: inline-block; margin-right: 0.6rem;
      background: var(--panel2); padding: 0.05rem 0.35rem; border-radius: 3px; }}
    .activity-row {{ cursor: pointer; }}
    .activity-row:hover {{ background: rgba(121, 192, 255, 0.04); }}
    .activity-row.expanded {{ background: rgba(121, 192, 255, 0.06); }}
    .body-row {{ display: none; }}
    .body-row.expanded {{ display: table-row; }}
    .body-pre {{ background: var(--panel2); border-radius: 4px;
      padding: 0.6rem 0.8rem; font-size: 0.82em; line-height: 1.4;
      white-space: pre-wrap; word-break: break-word; max-height: 500px;
      overflow-y: auto; color: var(--fg); }}
    .body-pre .jk {{ color: var(--accent); }}
    .body-pre .js {{ color: var(--good); }}
    .body-pre .jn {{ color: var(--accent2); }}
    .body-pre .jbool {{ color: var(--warn); }}
    .body-pre .jnull {{ color: var(--dim); }}

    /* Sparkline */
    .spark {{ display: block; width: 100%; height: 40px; }}

    /* Recovery card */
    .recoveries {{ display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 0.4rem; }}
    .rec {{ background: var(--panel2); border-radius: 4px; padding: 0.5rem 0.6rem; }}
    .rec-name {{ color: var(--dim); font-size: 0.7rem; text-transform: uppercase;
      letter-spacing: 0.05em; }}
    .rec-count {{ color: var(--accent2); font-size: 1.1rem; margin-top: 0.1rem; }}

    /* Upstream rate bars */
    .rate-bar {{ background: var(--panel2); height: 6px; border-radius: 3px;
      overflow: hidden; margin-top: 0.3rem; }}
    .rate-bar-fill {{ background: var(--accent); height: 100%;
      transition: width 0.3s ease; }}
    .rate-bar-fill.busy {{ background: var(--warn); }}
    .rate-bar-fill.full {{ background: var(--bad); }}

    /* Live ticker */
    .ticker {{ display: flex; align-items: center; gap: 0.5rem; }}
    .ticker .v {{ font-size: 1.7rem; color: var(--accent); font-weight: 500; }}
    .ticker .v.small {{ font-size: 1.05rem; color: var(--accent2); }}
    .ticker .lbl {{ color: var(--dim); font-size: 0.78rem; }}
    .flash {{ animation: flash 0.8s ease; }}
    @keyframes flash {{ 0% {{ background: rgba(86, 211, 100, 0.15); }} 100% {{ background: transparent; }} }}

    .footer {{ margin-top: 2rem; color: var(--dim); font-size: 0.8em;
      border-top: 1px solid var(--line); padding-top: 0.8rem; }}
    .footer code {{ font-size: 0.95em; }}
    .empty {{ color: var(--dim); padding: 1rem; text-align: center; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>resilient-llm-bridge</h1>
    <span class="port">:{DEFAULT_PORT}</span>
    <span class="uptime"><span class="pulse"></span><span id="uptime">uptime —</span></span>
    <div class="uptime-bar"></div>
    <div class="windows">
      <button data-win="1m">1m</button>
      <button data-win="5m" class="active">5m</button>
      <button data-win="15m">15m</button>
      <button data-win="1h">1h</button>
      <button data-win="lifetime">all</button>
    </div>
  </div>

  <h2>Overview</h2>
  <div class="kpis">
    <div class="kpi"><div class="kpi-label">requests</div>
      <div class="kpi-value" id="kpi-req">—</div>
      <div class="kpi-sub" id="kpi-rps">— rps</div>
      <svg class="kpi-spark" id="spark-req" preserveAspectRatio="none" viewBox="0 0 100 30"></svg>
    </div>
    <div class="kpi accent2"><div class="kpi-label">tokens out</div>
      <div class="kpi-value" id="kpi-tok-out">—</div>
      <div class="kpi-sub" id="kpi-tok-rate">— tok/s</div>
      <svg class="kpi-spark" id="spark-tok" preserveAspectRatio="none" viewBox="0 0 100 30"></svg>
    </div>
    <div class="kpi"><div class="kpi-label">tokens in</div>
      <div class="kpi-value" id="kpi-tok-in">—</div>
      <div class="kpi-sub" id="kpi-tok-ratio">— ratio</div>
    </div>
    <div class="kpi"><div class="kpi-label">latency p50</div>
      <div class="kpi-value" id="kpi-p50">—</div>
      <div class="kpi-sub" id="kpi-p95">p95 —</div>
      <svg class="kpi-spark" id="spark-lat" preserveAspectRatio="none" viewBox="0 0 100 30"></svg>
    </div>
    <div class="kpi good"><div class="kpi-label">success rate</div>
      <div class="kpi-value" id="kpi-success">—</div>
      <div class="kpi-sub" id="kpi-errors">— errors</div>
    </div>
    <div class="kpi accent2"><div class="kpi-label">recoveries fired</div>
      <div class="kpi-value" id="kpi-rec">—</div>
      <div class="kpi-sub" id="kpi-rec-sub">since start</div>
    </div>
  </div>

  <h2>Live</h2>
  <div class="grid2">
    <div class="card">
      <h3>Last completion</h3>
      <div class="ticker">
        <div class="v" id="live-tokens">—</div>
        <div>
          <div class="lbl" id="live-meta">waiting for first completion</div>
          <div class="lbl" id="live-rate">— tok/s</div>
        </div>
      </div>
    </div>
    <div class="card">
      <h3>Recoveries (lifetime)</h3>
      <div class="recoveries" id="rec-grid"></div>
    </div>
  </div>

  <h2>Per-profile</h2>
  <div class="card">
    <table>
      <thead><tr>
        <th>profile</th><th>upstream</th><th>features</th>
        <th class="num">req</th><th class="num">errors</th>
        <th class="num">tok in</th><th class="num">tok out</th>
        <th class="num">p50</th><th class="num">p95</th>
      </tr></thead>
      <tbody id="profiles-body"><tr><td colspan="9" class="empty">loading…</td></tr></tbody>
    </table>
  </div>

  <h2>Per-model</h2>
  <div class="card">
    <table>
      <thead><tr>
        <th>model</th>
        <th class="num">req</th><th class="num">errors</th>
        <th class="num">tok in</th><th class="num">tok out</th>
        <th class="num">p50</th><th class="num">p95</th>
      </tr></thead>
      <tbody id="models-body"><tr><td colspan="7" class="empty">no completions yet</td></tr></tbody>
    </table>
  </div>

  <h2>Upstreams</h2>
  <div class="card">
    <table>
      <thead><tr>
        <th>name</th><th>url</th>
        <th>rpm cap</th><th>concurrent</th><th>in flight</th><th>utilization</th>
      </tr></thead>
      <tbody id="upstreams-body"><tr><td colspan="6" class="empty">loading…</td></tr></tbody>
    </table>
  </div>

  <h2>Recent activity</h2>
  <div class="card">
    <table>
      <thead><tr>
        <th>time</th><th>profile</th><th>method</th>
        <th>path</th><th>status</th><th class="num">ms</th>
        <th>thinking params</th>
      </tr></thead>
      <tbody id="activity"><tr><td colspan="7" class="empty">no activity yet</td></tr></tbody>
    </table>
  </div>

  <div class="footer">
    Endpoints: <code>/{{profile}}/v1/responses</code>,
    <code>/{{profile}}/v1/chat/completions</code>,
    <code>/v1/...</code> (default profile).
    Config: <code>~/.config/resilient-llm-bridge/config.yaml</code>.
    Live feeds: <a href="/usage/stream" style="color:var(--accent)">/usage/stream</a>,
    <a href="/activity/stream" style="color:var(--accent)">/activity/stream</a>,
    <a href="/stats" style="color:var(--accent)">/stats</a>.
  </div>

  <script>
    const $ = id => document.getElementById(id);
    const fmt = (n) => Number(n || 0).toLocaleString();
    const fmtMs = (ms) => ms < 1000 ? `${{Math.round(ms)}}ms` : `${{(ms/1000).toFixed(2)}}s`;
    const fmtTime = (ts) => new Date(ts * 1000).toLocaleTimeString();

    function escapeHtml(s) {{
      return String(s).replace(/[&<>"']/g, c => ({{
        '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
      }})[c]);
    }}
    function statusClass(s) {{
      if (s >= 500) return 'status-5xx';
      if (s >= 400) return 'status-4xx';
      return 'status-2xx';
    }}
    function latClass(ms) {{
      if (ms < 1000) return 'lat-fast';
      if (ms < 5000) return 'lat-mid';
      return 'lat-slow';
    }}
    function renderParams(p) {{
      if (!p || typeof p !== 'object') return '<span class="params">—</span>';
      const keys = Object.keys(p);
      if (keys.length === 0) return '<span class="params">—</span>';
      return '<span class="params">' + keys.map(k => {{
        const cls = k.includes('DROPPED') ? 'pk dropped' : 'pk';
        return `<span class="pair"><span class="${{cls}}">${{escapeHtml(k)}}</span>=<span class="pv">${{escapeHtml(p[k])}}</span></span>`;
      }}).join('') + '</span>';
    }}

    // Sparkline renderer (SVG path from points 0..1)
    function spark(svgId, values, color) {{
      const svg = $(svgId);
      if (!svg) return;
      if (!values || values.length === 0) {{ svg.innerHTML = ''; return; }}
      const max = Math.max(...values, 1);
      const w = 100, h = 30;
      const step = values.length > 1 ? w / (values.length - 1) : 0;
      const pts = values.map((v, i) => `${{(i * step).toFixed(2)}},${{(h - (v / max) * h * 0.95 - 1).toFixed(2)}}`);
      const line = 'M ' + pts.join(' L ');
      const fill = `${{line}} L ${{w}},${{h}} L 0,${{h}} Z`;
      svg.innerHTML =
        `<path d="${{fill}}" fill="${{color}}" opacity="0.15"/>` +
        `<path d="${{line}}" fill="none" stroke="${{color}}" stroke-width="1.2"/>`;
    }}

    // Bin events into N time buckets covering `windowSec` seconds.
    function bin(events, windowSec, buckets, valueFn) {{
      const now = Date.now() / 1000;
      const start = now - windowSec;
      const out = new Array(buckets).fill(0);
      const span = windowSec / buckets;
      for (const ev of events) {{
        const t = ev.ts;
        if (t < start) continue;
        const idx = Math.min(buckets - 1, Math.max(0, Math.floor((t - start) / span)));
        out[idx] += valueFn(ev);
      }}
      return out;
    }}

    let activeWindow = '5m';
    let lastStats = null;
    let activityRows = [];
    let usageRows = [];

    document.querySelectorAll('.windows button').forEach(btn => {{
      btn.addEventListener('click', () => {{
        document.querySelectorAll('.windows button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeWindow = btn.dataset.win;
        if (lastStats) renderStats(lastStats);
      }});
    }});

    function pickWindow(stats) {{
      if (activeWindow === 'lifetime') {{
        const lt = stats.lifetime;
        return {{
          requests: lt.requests, errors: lt.errors,
          tokens_in: lt.tokens_in, tokens_out: lt.tokens_out,
          p50_ms: 0, p95_ms: 0, p99_ms: 0,
          by_profile: {{}}, by_model: {{}},
        }};
      }}
      return stats.windows[activeWindow] || {{}};
    }}

    function windowSeconds() {{
      return {{ '1m': 60, '5m': 300, '15m': 900, '1h': 3600, 'lifetime': 3600 }}[activeWindow] || 300;
    }}

    function renderStats(stats) {{
      const w = pickWindow(stats);
      const span = windowSeconds();
      $('kpi-req').textContent = fmt(w.requests);
      $('kpi-rps').textContent = `${{(w.requests / span).toFixed(2)}} rps · ${{w.errors_4xx || 0}}×4xx · ${{w.errors_5xx || 0}}×5xx`;
      $('kpi-tok-out').textContent = fmt(w.tokens_out);
      $('kpi-tok-rate').textContent = `${{Math.round(w.tokens_out / span)}} tok/s avg`;
      $('kpi-tok-in').textContent = fmt(w.tokens_in);
      const ratio = w.tokens_in > 0 ? (w.tokens_out / w.tokens_in).toFixed(2) : '—';
      $('kpi-tok-ratio').textContent = `out/in ${{ratio}}`;
      $('kpi-p50').textContent = fmtMs(w.p50_ms || 0);
      $('kpi-p95').textContent = `p95 ${{fmtMs(w.p95_ms || 0)}} · p99 ${{fmtMs(w.p99_ms || 0)}}`;
      const successRate = w.requests > 0
        ? (((w.requests - w.errors) / w.requests) * 100).toFixed(1) + '%'
        : '—';
      $('kpi-success').textContent = successRate;
      $('kpi-errors').textContent = `${{w.errors || 0}} errors / ${{w.requests || 0}} req`;
      const totalRec = Object.values(stats.lifetime.recoveries).reduce((a, b) => a + b, 0);
      $('kpi-rec').textContent = fmt(totalRec);
      const recParts = Object.entries(stats.lifetime.recoveries)
        .filter(([_, v]) => v > 0)
        .map(([k, v]) => `${{k.replace(/_/g, ' ')}}: ${{v}}`);
      $('kpi-rec-sub').textContent = recParts.length ? recParts.join(' · ') : 'none yet';

      // Sparklines from history
      const sec = activeWindow === 'lifetime' ? 3600 : span;
      const reqBins = bin(activityRows, sec, 24, _ => 1);
      const tokBins = bin(usageRows, sec, 24, e => (e.output_tokens || 0));
      const latBins = bin(activityRows, sec, 24, e => (e.duration_ms || 0));
      // Convert sums to averages for latency.
      const latCounts = bin(activityRows, sec, 24, _ => 1);
      const latAvg = latBins.map((s, i) => latCounts[i] ? s / latCounts[i] : 0);
      spark('spark-req', reqBins, '#79c0ff');
      spark('spark-tok', tokBins, '#d2a8ff');
      spark('spark-lat', latAvg, '#f0883e');

      // Recoveries grid
      const rec = stats.lifetime.recoveries || {{}};
      $('rec-grid').innerHTML = Object.entries(rec).map(([k, v]) =>
        `<div class="rec"><div class="rec-name">${{escapeHtml(k.replace(/_/g, ' '))}}</div>` +
        `<div class="rec-count">${{fmt(v)}}</div></div>`
      ).join('');

      // Per-profile table — merge profile config with current-window stats.
      const byProf = w.by_profile || {{}};
      const profRows = stats.profiles.map(p => {{
        const b = byProf[p.name] || {{requests:0, errors:0, tokens_in:0, tokens_out:0, p50_ms:0, p95_ms:0}};
        const feats = p.features.length ? p.features.join(', ') : '—';
        return `<tr><td>${{escapeHtml(p.name)}}</td><td>${{escapeHtml(p.upstream)}}</td>` +
          `<td><code>${{escapeHtml(feats)}}</code></td>` +
          `<td class="num">${{fmt(b.requests)}}</td>` +
          `<td class="num ${{b.errors>0?'status-4xx':''}}">${{fmt(b.errors)}}</td>` +
          `<td class="num">${{fmt(b.tokens_in)}}</td>` +
          `<td class="num">${{fmt(b.tokens_out)}}</td>` +
          `<td class="num ${{latClass(b.p50_ms)}}">${{fmtMs(b.p50_ms)}}</td>` +
          `<td class="num ${{latClass(b.p95_ms)}}">${{fmtMs(b.p95_ms)}}</td></tr>`;
      }}).join('');
      $('profiles-body').innerHTML = profRows ||
        `<tr><td colspan="9" class="empty">no profiles</td></tr>`;

      // Per-model table.
      const byModel = w.by_model || {{}};
      const modelRows = Object.entries(byModel)
        .sort((a, b) => (b[1].tokens_out || 0) - (a[1].tokens_out || 0))
        .map(([name, b]) => `<tr><td>${{escapeHtml(name)}}</td>` +
          `<td class="num">${{fmt(b.requests)}}</td>` +
          `<td class="num ${{(b.errors||0)>0?'status-4xx':''}}">${{fmt(b.errors)}}</td>` +
          `<td class="num">${{fmt(b.tokens_in)}}</td>` +
          `<td class="num">${{fmt(b.tokens_out)}}</td>` +
          `<td class="num ${{latClass(b.p50_ms)}}">${{fmtMs(b.p50_ms)}}</td>` +
          `<td class="num ${{latClass(b.p95_ms)}}">${{fmtMs(b.p95_ms)}}</td></tr>`).join('');
      $('models-body').innerHTML = modelRows ||
        `<tr><td colspan="7" class="empty">no completions in window</td></tr>`;

      // Upstreams.
      const upRows = (stats.upstreams || []).map(u => {{
        const inFlight = u.concurrent_in_flight || 0;
        const cap = u.concurrent_limit || 1;
        const pct = Math.min(100, (inFlight / cap) * 100);
        const cls = pct >= 90 ? 'full' : pct >= 50 ? 'busy' : '';
        return `<tr><td>${{escapeHtml(u.name)}}</td>` +
          `<td><code>${{escapeHtml(u.url)}}</code></td>` +
          `<td class="num">${{u.rpm_remaining}}/${{u.rpm_capacity}}</td>` +
          `<td class="num">${{cap}}</td>` +
          `<td class="num">${{inFlight}}</td>` +
          `<td><div class="rate-bar"><div class="rate-bar-fill ${{cls}}" style="width:${{pct.toFixed(1)}}%"></div></div></td></tr>`;
      }}).join('');
      $('upstreams-body').innerHTML = upRows ||
        `<tr><td colspan="6" class="empty">no upstreams</td></tr>`;

      // Activity table from rolling buffer. Each main row has a hidden
      // sibling row with the full redacted JSON body — expanded on click.
      const recent = activityRows.slice(-30).reverse();
      $('activity').innerHTML = recent.length === 0
        ? `<tr><td colspan="7" class="empty">no activity yet</td></tr>`
        : recent.map((r, i) => {{
            const rowId = `row-${{i}}-${{(r.ts*1000)|0}}`;
            const main = `<tr class="activity-row" data-target="${{rowId}}">` +
              `<td>${{fmtTime(r.ts)}}</td><td>${{escapeHtml(r.profile)}}</td>` +
              `<td>${{escapeHtml(r.method)}}</td><td><code>${{escapeHtml(r.path)}}</code></td>` +
              `<td class="${{statusClass(r.status)}}">${{r.status}}</td>` +
              `<td class="num ${{latClass(r.duration_ms)}}">${{fmtMs(r.duration_ms)}}</td>` +
              `<td>${{renderParams(r.params)}}</td></tr>`;
            const bodyJson = r.body ? jsonHighlight(r.body) :
              '<span class="jnull">no body captured (passthrough or non-JSON request)</span>';
            const expand = `<tr class="body-row" id="${{rowId}}">` +
              `<td colspan="7"><div class="body-pre">${{bodyJson}}</div></td></tr>`;
            return main + expand;
          }}).join('');
      // Wire click handlers for expandable rows.
      document.querySelectorAll('#activity .activity-row').forEach(row => {{
        row.addEventListener('click', () => {{
          const target = $(row.dataset.target);
          if (!target) return;
          row.classList.toggle('expanded');
          target.classList.toggle('expanded');
        }});
      }});
    }}

    // Tiny JSON syntax highlighter.
    function jsonHighlight(obj) {{
      const json = JSON.stringify(obj, null, 2);
      const escaped = escapeHtml(json);
      return escaped
        .replace(/&quot;([^&]+?)&quot;:/g, '<span class="jk">"$1"</span>:')
        .replace(/: &quot;(.*?)&quot;([,\\n}}\\]])/g, ': <span class="js">"$1"</span>$2')
        .replace(/: (-?\\d+\\.?\\d*)([,\\n}}\\]])/g, ': <span class="jn">$1</span>$2')
        .replace(/: (true|false)([,\\n}}\\]])/g, ': <span class="jbool">$1</span>$2')
        .replace(/: null([,\\n}}\\]])/g, ': <span class="jnull">null</span>$1');
    }}

    async function refreshStats() {{
      try {{
        const r = await fetch('/stats');
        const d = await r.json();
        if (d.uptime_s !== undefined) {{
          const s = Math.round(d.uptime_s);
          const h = Math.floor(s / 3600);
          const m = Math.floor((s % 3600) / 60);
          $('uptime').textContent = h > 0 ? `uptime ${{h}}h${{m}}m` : `uptime ${{m}}m ${{s % 60}}s`;
        }}
        // Adopt server-side history once on each refresh.
        if (Array.isArray(d.history?.activity)) activityRows = d.history.activity;
        if (Array.isArray(d.history?.usage)) usageRows = d.history.usage;
        lastStats = d;
        renderStats(d);
      }} catch (e) {{ console.error(e); }}
    }}

    // SSE feeds keep things live in between /stats refreshes.
    const usage = new EventSource('/usage/stream');
    usage.onmessage = (e) => {{
      try {{
        const d = JSON.parse(e.data);
        usageRows.push(d);
        if (usageRows.length > 1000) usageRows.shift();
        const tot = d.total_tokens || ((d.input_tokens||0)+(d.output_tokens||0));
        const live = $('live-tokens');
        live.textContent = fmt(tot);
        live.classList.remove('flash'); void live.offsetWidth; live.classList.add('flash');
        $('live-meta').textContent = `${{d.profile}} · ${{d.model || '?'}} · in ${{fmt(d.input_tokens)}} / out ${{fmt(d.output_tokens)}}`;
        $('live-rate').textContent = `${{new Date(d.ts*1000).toLocaleTimeString()}}`;
        if (lastStats) renderStats(lastStats);
      }} catch {{}}
    }};
    const activity = new EventSource('/activity/stream');
    activity.onmessage = (e) => {{
      try {{
        const d = JSON.parse(e.data);
        activityRows.push(d);
        if (activityRows.length > 1000) activityRows.shift();
        if (lastStats) renderStats(lastStats);
      }} catch {{}}
    }};

    refreshStats();
    setInterval(refreshStats, 5000);
  </script>
</body>
</html>
"""


# =============================================================================
# Entrypoint
# =============================================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
