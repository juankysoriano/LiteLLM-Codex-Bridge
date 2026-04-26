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
import json
import os
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

USAGE_RING_SIZE = int(os.environ.get("BRIDGE_USAGE_RING_SIZE", "32"))
ACTIVITY_RING_SIZE = int(os.environ.get("BRIDGE_ACTIVITY_RING_SIZE", "20"))

_usage_history: deque[dict] = deque(maxlen=USAGE_RING_SIZE)
_usage_subscribers: set[asyncio.Queue[dict]] = set()
_activity_history: deque[dict] = deque(maxlen=ACTIVITY_RING_SIZE)
_activity_subscribers: set[asyncio.Queue[dict]] = set()
_started_at = time.time()


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


def _is_responses_silent_completion(response_obj: dict, message_emitted: bool) -> bool:
    """Detect a `status: completed` response that has no message item.

    Pattern observed with Codex sessions: happy injects the
    `CHANGE_TITLE_INSTRUCTION` into the first user message of every
    session, which makes Qwen3 reason about whether to call the title
    tool. If the model decides the user just said "hello" with no real
    task, it emits *only* the reasoning explaining its decision and
    finishes the turn without producing an answer — the user sees a
    successful but silent completion.

    Trigger: status=completed + no message item in output + we have
    reasoning text to feed the recovery from. Same recovery as overflow:
    `continue_final_message` with the reasoning prefilled, asking the
    model to actually answer the user.
    """
    if not isinstance(response_obj, dict):
        return False
    if response_obj.get("status") != "completed":
        return False
    return not message_emitted


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
        response_obj, message_emitted
    )
    if (overflow or silent) and reasoning_accum.strip():
        recovered_text = await _recover_thinking_overflow(
            body, reasoning_accum, headers, profile
        )

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
        fixed_response["output"] = list(response_obj.get("output") or []) + [
            done_message_item
        ]
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


def _record_activity(
    profile: ProfileConfig, path: str, method: str, status: int, duration_ms: float
) -> None:
    _broadcast_activity(
        {
            "ts": time.time(),
            "profile": profile.name,
            "upstream": profile.upstream,
            "path": path,
            "method": method,
            "status": status,
            "duration_ms": round(duration_ms, 1),
        }
    )


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "uptime_s": round(time.time() - _started_at, 1),
        "profiles": list(CONFIG.profiles.keys()),
        "upstreams": [g.snapshot() for g in _UPSTREAM_GATES.values()],
    }


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
        for item in output_items:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content") or []
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "output_text" and (part.get("text") or "").strip():
                    message_emitted = True
                    break
            if message_emitted:
                break
        overflow_ns = profile.has("thinking_overflow_recovery") and _is_responses_overflow(
            payload, message_emitted
        )
        silent_ns = profile.has("silent_completion_recovery") and _is_responses_silent_completion(
            payload, message_emitted
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
                    payload["status"] = "completed"
                    payload["incomplete_details"] = None
                    payload["output"] = list(output_items) + [
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
    )
    return JSONResponse(content=payload, status_code=status)


async def _handle_chat_completions(
    request: Request, profile: ProfileConfig
) -> StreamingResponse | JSONResponse:
    started = time.monotonic()
    body = await request.json()
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
    profiles_rows = "\n".join(
        f"""<tr><td>{p.name}</td><td>{p.upstream}</td>"""
        f"""<td><code>{', '.join(sorted(p.features)) or '—'}</code></td></tr>"""
        for p in CONFIG.profiles.values()
    )
    upstreams_rows = "\n".join(
        f"""<tr id="up-{u.cfg.name}">
            <td>{u.cfg.name}</td>
            <td><code>{u.cfg.url}</code></td>
            <td><span class="rpm">{u.cfg.rate_limit_rpm}</span> rpm</td>
            <td><span class="conc">{u.cfg.rate_limit_concurrent}</span> max</td>
            <td><span class="inflight">0</span> in flight</td>
        </tr>"""
        for u in _UPSTREAM_GATES.values()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>resilient-llm-bridge :: {DEFAULT_PORT}</title>
  <style>
    :root {{
      --bg: #0a0a0a; --fg: #d4d4d4; --dim: #6e7681;
      --accent: #79c0ff; --good: #56d364; --warn: #f0883e;
    }}
    body {{ background: var(--bg); color: var(--fg); font-family:
      ui-monospace, "SF Mono", Menlo, Consolas, monospace; padding: 1.5rem;
      max-width: 1100px; margin: 0 auto; font-size: 13px; line-height: 1.5; }}
    h1 {{ font-size: 1.4rem; margin: 0; color: var(--accent); }}
    h2 {{ font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.1em;
      color: var(--dim); margin-top: 2rem; border-bottom: 1px solid #1f1f1f;
      padding-bottom: 0.4rem; }}
    .header {{ display: flex; align-items: baseline; gap: 1rem; }}
    .port {{ color: var(--good); font-size: 1.1rem; }}
    .uptime {{ color: var(--dim); font-size: 0.9rem; margin-left: auto; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 0.5rem; }}
    td, th {{ text-align: left; padding: 0.4rem 0.6rem; border-bottom:
      1px solid #161616; vertical-align: top; }}
    th {{ color: var(--dim); font-weight: normal; }}
    code {{ background: #141414; padding: 0.1rem 0.3rem; border-radius: 2px;
      font-size: 0.85em; }}
    .activity-status-2xx {{ color: var(--good); }}
    .activity-status-4xx {{ color: var(--warn); }}
    .activity-status-5xx {{ color: #ff7b72; }}
    .footer {{ margin-top: 3rem; color: var(--dim); font-size: 0.85em; }}
    .live-counter {{ font-size: 1.6rem; color: var(--accent); }}
    .live-counter small {{ color: var(--dim); font-size: 0.55em; margin-left: 0.6rem; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>resilient-llm-bridge</h1>
    <span class="port">:{DEFAULT_PORT}</span>
    <span class="uptime" id="uptime">uptime —</span>
  </div>

  <h2>Live token counter</h2>
  <div class="live-counter" id="counter">— <small>waiting for first completion</small></div>

  <h2>Upstreams</h2>
  <table>
    <thead><tr><th>name</th><th>url</th><th>rpm</th><th>concurrent</th><th>now</th></tr></thead>
    <tbody>{upstreams_rows}</tbody>
  </table>

  <h2>Profiles</h2>
  <table>
    <thead><tr><th>profile</th><th>upstream</th><th>features</th></tr></thead>
    <tbody>{profiles_rows}</tbody>
  </table>

  <h2>Recent activity</h2>
  <table>
    <thead><tr><th>time</th><th>profile</th><th>method</th><th>path</th><th>status</th><th>ms</th></tr></thead>
    <tbody id="activity"><tr><td colspan="6" style="color: var(--dim);">no activity yet</td></tr></tbody>
  </table>

  <div class="footer">
    Endpoints: <code>/{{profile}}/v1/responses</code>,
    <code>/{{profile}}/v1/chat/completions</code>,
    <code>/v1/...</code> (default profile).
    Upstreams + profiles defined in <code>~/.config/resilient-llm-bridge/config.yaml</code>.
  </div>

  <script>
    const counter = document.getElementById('counter');
    const uptimeEl = document.getElementById('uptime');
    const activityEl = document.getElementById('activity');
    let activityRows = [];

    function fmtTime(ts) {{
      return new Date(ts * 1000).toLocaleTimeString();
    }}
    function statusClass(s) {{
      if (s >= 500) return 'activity-status-5xx';
      if (s >= 400) return 'activity-status-4xx';
      return 'activity-status-2xx';
    }}

    async function refreshHealth() {{
      try {{
        const r = await fetch('/health');
        const d = await r.json();
        if (d.uptime_s !== undefined) {{
          const s = Math.round(d.uptime_s);
          const h = Math.floor(s / 3600);
          const m = Math.floor((s % 3600) / 60);
          uptimeEl.textContent = h > 0 ? `uptime ${{h}}h${{m}}m` : `uptime ${{m}}m`;
        }}
        for (const u of (d.upstreams || [])) {{
          const row = document.getElementById('up-' + u.name);
          if (row) {{
            row.querySelector('.inflight').textContent = u.concurrent_in_flight;
          }}
        }}
      }} catch {{}}
    }}

    const usage = new EventSource('/usage/stream');
    usage.onmessage = (e) => {{
      try {{
        const d = JSON.parse(e.data);
        counter.innerHTML = `${{d.total_tokens.toLocaleString()}} <small>${{d.profile}} · ${{d.model || '?'}} · in ${{d.input_tokens.toLocaleString()}} / out ${{d.output_tokens.toLocaleString()}}</small>`;
      }} catch {{}}
    }};

    const activity = new EventSource('/activity/stream');
    activity.onmessage = (e) => {{
      try {{
        const d = JSON.parse(e.data);
        activityRows.unshift(d);
        activityRows = activityRows.slice(0, 20);
        activityEl.innerHTML = activityRows.map(r =>
          `<tr><td>${{fmtTime(r.ts)}}</td><td>${{r.profile}}</td>` +
          `<td>${{r.method}}</td><td><code>${{r.path}}</code></td>` +
          `<td class="${{statusClass(r.status)}}">${{r.status}}</td>` +
          `<td>${{r.duration_ms}}</td></tr>`
        ).join('');
      }} catch {{}}
    }};

    refreshHealth();
    setInterval(refreshHealth, 5000);
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
