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
import heapq
import itertools
import json
import os
import re
import time
from collections import deque
from contextlib import asynccontextmanager
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
    # Queue: how long an inbound request will wait for a slot before
    # we 503 it. Without this, a saturated semaphore + a hung upstream
    # held all queued requests forever. 120s gives normal queueing
    # plenty of room while bounding worst case.
    queue_timeout_s: float = 120.0
    # Watchdog: log a warning when a slot has been held this long.
    stuck_warn_s: float = 300.0
    # Model context window — used to cap the bumped `max_tokens` when
    # the bridge injects a thinking budget, so we don't push the
    # request past the model's limit (manifested as a 400
    # ContextWindowExceeded from the upstream). 262144 = Qwen3 with
    # YaRN/long-context enabled (NaN's current deployment); the stock
    # Qwen3 native window is 131072. Tune per upstream if you point at
    # other models.
    context_window: int = 262144
    # Safety margin between the estimated prompt size and the cap we
    # set on `max_tokens`. The estimator (chars/3.0) is already biased
    # pessimistic, but tool-result inflation is a moving target: an
    # agent that pastes a 90k-token JSON blob into the next turn can
    # blow past any tight ceiling. 8192 of slack covers the residual
    # estimation error AND a generous chunk of mid-turn growth before
    # the upstream rejects the request — which some clients (e.g.
    # opencode) silently hang on instead of treating as overflow.
    context_safety_margin: int = 8192


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
    "tool_call_args_retry",         # tool_call args missing required fields → retry with thinking off
}


@dataclass
class ProfileConfig:
    """A named profile that selects an upstream and a set of features."""
    name: str
    upstream: str
    features: set[str] = field(default_factory=set)
    model_aliases: dict[str, str] = field(default_factory=dict)
    # Authoritative thinking budget:
    #   None         → profile is silent; the bridge respects whatever
    #                  the client sent and translates `effort` →
    #                  budget if the client gave a hint without an
    #                  explicit budget.
    #   <positive>   → profile FORCES this budget; overrides any client
    #                  effort/budget signal.
    default_thinking_budget: int | None = None
    # Tristate authoritative thinking switch:
    #   None  → profile is silent on thinking; the bridge respects
    #           whatever the client sent (or didn't send).
    #   True  → profile FORCES thinking on, regardless of what the
    #           client sent.
    #   False → profile FORCES thinking off, regardless of what the
    #           client sent.
    # The `default` profile is the only built-in that sets this.
    thinking_enabled: bool | None = None
    # Queue priority for the upstream gate — HIGHER = jumps ahead of
    # lower-priority numbers. Interactive coding agents (codex,
    # opencode) get 10; long-running daemons (hermes, default) get 0.
    # Within the same priority bucket it's first-come-first-served.
    queue_priority: int = 0

    def has(self, feature: str) -> bool:
        return feature in self.features

    def resolve_model(self, model: str | None) -> str | None:
        """Translate a client-facing model id to the upstream id.

        Used to bypass client-side model-id heuristics. Concrete case:
        opencode hard-codes a skip list that drops `reasoning_effort`
        for any model id containing "qwen". Configuring opencode with
        a neutral id like `nan-thinking` and aliasing it to `qwen3.6`
        in the bridge restores the reasoning hint while still routing
        to the right model upstream.
        """
        if not isinstance(model, str):
            return model
        return self.model_aliases.get(model, model)


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
                # `default` is the only built-in profile that opts into
                # forced thinking — anonymous clients can't tell us
                # their preference so we apply a sensible default
                # (medium). Named profiles below leave thinking_enabled
                # at the dataclass default (False) and pass through.
                thinking_enabled=True,
                default_thinking_budget=4096,
                queue_priority=0,  # background / unknown
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
                # No thinking overrides — Codex always sends
                # `reasoning.effort`; the bridge translates that to a
                # budget via the effort map, which implicitly activates
                # thinking on the upstream.
                queue_priority=10,  # interactive — jumps ahead of background
            ),
            "hermes": ProfileConfig(
                name="hermes",
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
                # No thinking overrides. Hermes can't send a reasoning
                # hint to a non-whitelisted host (its supports_reasoning
                # gate is host-based) so it sends nothing — and we
                # honor that. No thinking unless Hermes ever starts
                # forwarding hints.
                queue_priority=1,  # background daemon, but ahead of `default`
            ),
            "opencode": ProfileConfig(
                name="opencode",
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
                # No thinking overrides — opencode now sends its own
                # `reasoning_effort` (configured via
                # `models.<id>.options.reasoningEffort` in
                # opencode.json), which the bridge translates.
                queue_priority=10,  # interactive — jumps ahead of background
            ),
        },
        default_profile="default",
    )


_SAMPLE_CONFIG_YAML = """\
# resilient-llm-bridge config — every key is optional; whatever you
# leave out falls back to the dataclass default. Reload requires
# `systemctl --user restart resilient-llm-bridge.service`.
#
# Schema:
#   upstreams: <name>: { url, rate_limit_rpm, rate_limit_concurrent,
#                        queue_timeout_s, stuck_warn_s,
#                        retry_max_attempts, retry_initial_wait,
#                        retry_max_wait }
#   profiles:  <name>: { upstream, features:[...], queue_priority,
#                        default_thinking_budget, model_aliases:{...} }
#   default_profile: <name>      # which profile catches /v1/... (no prefix)
#
# Available `features` (toggle on/off per profile):
#   qwen_sampling_defaults      inject Qwen3 sampling (top_k, min_p, …)
#   drop_oai_only_fields        strip OpenAI-only fields the upstream rejects
#   effort_to_thinking_budget   reasoning.effort → thinking_token_budget
#   normalize_responses_input   reshape input items for strict validators
#   drop_namespace_tools        drop namespace/MCP/web_search/image tools
#   force_serial_tool_calls     parallel_tool_calls=false override
#   parallel_tool_sse_fix       vLLM #39426 workaround
#   thinking_overflow_recovery  incomplete + max_output_tokens → recover
#   silent_completion_recovery  completed + no message item → recover
#   truncated_content_recovery  length + cut mid-thought → continue
#   empty_with_stop_retry       empty + stop → one cheap retry
#
# `queue_priority`: HIGHER number = jumps ahead in the upstream queue
# when slots are saturated. Same priority resolves FIFO. Suggested:
#   10  interactive coding agents (codex, opencode)
#    1  background agents (hermes)
#    0  default / unknown clients

upstreams:
  nan:
    url: "https://api.nan.builders/v1"
    rate_limit_rpm: 100
    rate_limit_concurrent: 5
    queue_timeout_s: 120.0    # 503 if a request waits longer than this
    stuck_warn_s: 300.0       # watchdog logs slots held longer than this

profiles:
  default:
    upstream: nan
    queue_priority: 0
    default_thinking_budget: 4096   # medium — for unknown clients
    features:
      - qwen_sampling_defaults
      - drop_oai_only_fields
      - effort_to_thinking_budget
      - thinking_overflow_recovery
      - silent_completion_recovery
      - truncated_content_recovery
      - empty_with_stop_retry

  codex:
    upstream: nan
    queue_priority: 10
    default_thinking_budget: 8192
    features:
      - qwen_sampling_defaults
      - drop_oai_only_fields
      - effort_to_thinking_budget
      - normalize_responses_input
      - drop_namespace_tools
      - force_serial_tool_calls
      - parallel_tool_sse_fix
      - thinking_overflow_recovery
      - silent_completion_recovery
      - truncated_content_recovery
      - empty_with_stop_retry

  hermes:
    upstream: nan
    queue_priority: 1
    default_thinking_budget: 4096
    features:
      - qwen_sampling_defaults
      - drop_oai_only_fields
      - effort_to_thinking_budget
      - thinking_overflow_recovery
      - silent_completion_recovery
      - truncated_content_recovery
      - empty_with_stop_retry

  opencode:
    upstream: nan
    queue_priority: 10
    default_thinking_budget: 8192
    # Optional model id rewrites (client-id -> upstream-id), in case
    # you want to dodge a client-side model heuristic. Example:
    # model_aliases:
    #   nan-thinking: qwen3.6
    features:
      - qwen_sampling_defaults
      - drop_oai_only_fields
      - effort_to_thinking_budget
      - thinking_overflow_recovery
      - silent_completion_recovery
      - truncated_content_recovery
      - empty_with_stop_retry

default_profile: default
"""


def _coerce(value: Any, kind: type, default: Any) -> Any:
    """Best-effort cast a YAML scalar to `kind`, falling back to
    `default` on failure. Logs a warning so misconfigs surface fast."""
    if value is None:
        return default
    try:
        return kind(value)
    except (TypeError, ValueError):
        print(
            f"[config] expected {kind.__name__} for value {value!r}, "
            f"using default {default!r}",
            flush=True,
        )
        return default


def _ensure_sample_config() -> None:
    """Write a documented template YAML on first run so the user has
    something to edit. No-op if the file already exists."""
    if DEFAULT_CONFIG_PATH.exists():
        return
    try:
        DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_CONFIG_PATH.write_text(_SAMPLE_CONFIG_YAML, encoding="utf-8")
        print(
            f"[config] wrote sample config to {DEFAULT_CONFIG_PATH} — "
            f"edit and restart to apply.",
            flush=True,
        )
    except OSError as exc:
        print(f"[config] could not write sample config: {exc}", flush=True)


def _load_config() -> BridgeConfig:
    """Load configuration from YAML, falling back to bundled defaults.

    Every field is optional — missing values fall back to whatever the
    dataclass defines as default. Unknown fields are ignored with a
    warning. Unknown features are dropped per profile (also warned).
    """
    _ensure_sample_config()
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
        if not isinstance(body, dict):
            print(f"[config] upstream {name!r} must be a mapping; skipped", flush=True)
            continue
        upstreams[name] = UpstreamConfig(
            name=name,
            url=str(body.get("url") or "").rstrip("/"),
            rate_limit_rpm=_coerce(body.get("rate_limit_rpm"), int, 0),
            rate_limit_concurrent=_coerce(body.get("rate_limit_concurrent"), int, 0),
            retry_max_attempts=_coerce(body.get("retry_max_attempts"), int, 3),
            retry_initial_wait=_coerce(body.get("retry_initial_wait"), float, 1.0),
            retry_max_wait=_coerce(body.get("retry_max_wait"), float, 20.0),
            queue_timeout_s=_coerce(body.get("queue_timeout_s"), float, 120.0),
            stuck_warn_s=_coerce(body.get("stuck_warn_s"), float, 300.0),
        )
    if not upstreams:
        upstreams = _builtin_defaults().upstreams

    profiles: dict[str, ProfileConfig] = {}
    for name, body in (raw.get("profiles") or {}).items():
        if not isinstance(body, dict):
            print(f"[config] profile {name!r} must be a mapping; skipped", flush=True)
            continue
        upstream_name = str(body.get("upstream") or "")
        if upstream_name and upstream_name not in upstreams:
            print(
                f"[config] profile {name!r} references unknown upstream "
                f"{upstream_name!r}; skipped",
                flush=True,
            )
            continue
        feats_raw = body.get("features") or []
        if not isinstance(feats_raw, list):
            print(f"[config] profile {name!r} features must be a list; ignored", flush=True)
            feats_raw = []
        feats = set(str(f) for f in feats_raw)
        invalid = feats - ALL_FEATURES
        if invalid:
            print(
                f"[config] profile {name!r} references unknown features: {sorted(invalid)}",
                flush=True,
            )
            feats -= invalid
        aliases_raw = body.get("model_aliases") or {}
        if not isinstance(aliases_raw, dict):
            print(
                f"[config] profile {name!r} model_aliases must be a mapping; ignored",
                flush=True,
            )
            aliases_raw = {}
        model_aliases = {str(k): str(v) for k, v in aliases_raw.items()}
        # default_thinking_budget is now Optional[int]: missing/null
        # leaves it as None (no budget injected). Numeric values still
        # parse; non-numeric/non-null falls back to None.
        raw_budget = body.get("default_thinking_budget")
        if raw_budget is None:
            budget_val: int | None = None
        else:
            try:
                budget_val = int(raw_budget)
            except (TypeError, ValueError):
                budget_val = None
        # thinking_enabled is tristate. Absent in YAML → None (profile
        # silent, bridge respects client). Boolean in YAML → profile
        # is authoritative and overrides client.
        if "thinking_enabled" not in body:
            thinking_enabled_val: bool | None = None
        else:
            raw_enabled = body.get("thinking_enabled")
            if raw_enabled is None:
                thinking_enabled_val = None
            else:
                thinking_enabled_val = bool(raw_enabled)
        profiles[name] = ProfileConfig(
            name=name,
            upstream=upstream_name,
            features=feats,
            model_aliases=model_aliases,
            default_thinking_budget=budget_val,
            thinking_enabled=thinking_enabled_val,
            queue_priority=_coerce(body.get("queue_priority"), int, 0),
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
# Sized for Qwen3-30B-A3B-Thinking-2507 on a 256k context: low/medium
# cover casual reasoning, high (8k) is the model card's coding sweet
# spot, xhigh (16k) reaches the model card's "highly challenging
# reasoning" target. Even xhigh leaves 16k of the 32k max_output for
# the actual answer. Currently a no-op against deployments missing
# `--reasoning-parser qwen3` on vLLM (the param is silently dropped).
_EFFORT_TO_THINKING_BUDGET = {"low": 2048, "medium": 4096, "high": 8192, "xhigh": 16384}
_DEFAULT_THINKING_BUDGET = 8192  # aligned to "high" in the effort map above

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


class _QueueTimeout(Exception):
    """Raised when a request waits longer than `queue_timeout_s` for a
    slot. The handler converts this to a 503 with a Retry-After header
    so the client backs off instead of looping. Not retryable — by the
    time we hit it, the upstream is sick or the queue is overwhelmed
    and immediately retrying just makes things worse."""

    def __init__(self, upstream: str, waited_s: float) -> None:
        self.upstream = upstream
        self.waited_s = waited_s
        super().__init__(
            f"queue timeout: waited {waited_s:.1f}s for an upstream={upstream} slot"
        )


@dataclass
class _Waiter:
    """One task in the priority queue.

    HIGHER priority pops first (heapq min-heap, but `__lt__` is
    inverted on priority). Ties resolve FIFO via `seq` — a monotonic
    counter taken at enqueue time.
    """
    priority: int
    seq: int
    future: asyncio.Future
    profile: str = "?"
    path: str = "?"

    def __lt__(self, other: "_Waiter") -> bool:
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.seq < other.seq


class _UpstreamGate:
    """Per-upstream rate-limit + priority-queue + per-slot tracking.

    Three independent constraints:
      * Token bucket: RPM cap. Once consumed, tokens regenerate at
        `rate_limit_rpm/60` per second.
      * Concurrency cap: max `rate_limit_concurrent` slots in flight.
      * Priority heap: when slots are saturated, waiters queue in
        priority order. Higher number wins (`codex` and `opencode`
        use priority=10; `hermes` and `default` use priority=0).

    Per-slot state is a dict so the dashboard can show oldest-age and
    the watchdog can spot stuck slots. The whole thing is acquired
    via the `_gated()` async context manager so cancellation always
    frees the slot via the `finally` clause.
    """

    def __init__(self, cfg: UpstreamConfig) -> None:
        self.cfg = cfg
        self.bucket = _TokenBucket(cfg.rate_limit_rpm or 1, cfg.rate_limit_rpm or 0)
        self.concurrent_limit = (
            cfg.rate_limit_concurrent if cfg.rate_limit_concurrent > 0 else 1024
        )
        self._in_flight: dict[int, dict] = {}
        self._waiters: list[_Waiter] = []
        self._lock = asyncio.Lock()
        self._slot_id_counter = itertools.count(1)
        self._seq_counter = itertools.count()
        # Slots that have been logged once as stuck — don't re-spam.
        self._stuck_logged: set[int] = set()

    async def acquire(
        self,
        *,
        profile_name: str,
        path: str,
        priority: int,
        queue_timeout: float,
    ) -> int:
        """Acquire a slot, returning a slot id used for `release()`.

        Order of operations:
          1. Wait for a rate-limit token (bucket). Cheap when not
             saturated; consumed token can't be returned but the
             bucket regenerates anyway.
          2. Take a slot, jumping the priority queue if needed.
        """
        await self.bucket.acquire()

        loop = asyncio.get_event_loop()
        future: asyncio.Future[int] = loop.create_future()
        waiter = _Waiter(
            priority=priority,
            seq=next(self._seq_counter),
            future=future,
            profile=profile_name,
            path=path,
        )

        # Either grant immediately if a slot is free, or push onto
        # the heap. Both must happen under the lock to avoid
        # double-grants when slots free at the same time.
        async with self._lock:
            if len(self._in_flight) < self.concurrent_limit:
                slot_id = next(self._slot_id_counter)
                self._in_flight[slot_id] = {
                    "started": time.monotonic(),
                    "profile": profile_name,
                    "path": path,
                }
                future.set_result(slot_id)
            else:
                heapq.heappush(self._waiters, waiter)

        started_wait = time.monotonic()
        try:
            return await asyncio.wait_for(future, timeout=queue_timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            waited = time.monotonic() - started_wait
            async with self._lock:
                # Remove from heap if still queued.
                try:
                    self._waiters.remove(waiter)
                    heapq.heapify(self._waiters)
                except ValueError:
                    pass
                # Edge case: slot was granted between timeout check
                # and our exception handler — release it so the next
                # waiter gets a turn.
                if future.done() and not future.cancelled():
                    try:
                        slot_id = future.result()
                    except Exception:
                        slot_id = None
                    if slot_id is not None:
                        self._in_flight.pop(slot_id, None)
                        self._dispatch_next_locked()
            if isinstance(e, asyncio.TimeoutError):
                raise _QueueTimeout(self.cfg.name, waited) from None
            raise

    async def release(self, slot_id: int) -> None:
        """Release a slot and wake the next waiter, if any."""
        async with self._lock:
            self._in_flight.pop(slot_id, None)
            self._stuck_logged.discard(slot_id)
            self._dispatch_next_locked()

    def _dispatch_next_locked(self) -> None:
        """Hand a slot to the highest-priority queued waiter. Caller
        must hold `self._lock`."""
        while self._waiters:
            nxt = heapq.heappop(self._waiters)
            if nxt.future.cancelled():
                continue
            slot_id = next(self._slot_id_counter)
            self._in_flight[slot_id] = {
                "started": time.monotonic(),
                "profile": nxt.profile,
                "path": nxt.path,
            }
            try:
                nxt.future.set_result(slot_id)
                return
            except asyncio.InvalidStateError:
                # Future was cancelled between the check and set_result —
                # roll back the slot and try the next waiter.
                self._in_flight.pop(slot_id, None)
                continue

    def snapshot(self) -> dict[str, Any]:
        cur, _ = self.bucket.snapshot()
        now = time.monotonic()
        ages = [round(now - s["started"], 1) for s in self._in_flight.values()]
        return {
            "name": self.cfg.name,
            "url": self.cfg.url,
            "rpm_capacity": self.cfg.rate_limit_rpm,
            "rpm_remaining": round(cur, 1),
            "concurrent_limit": self.concurrent_limit,
            "concurrent_in_flight": len(self._in_flight),
            "queue_waiting": len(self._waiters),
            "oldest_in_flight_s": max(ages) if ages else 0.0,
            "queue_timeout_s": self.cfg.queue_timeout_s,
            "stuck_warn_s": self.cfg.stuck_warn_s,
        }


_UPSTREAM_GATES: dict[str, _UpstreamGate] = {
    name: _UpstreamGate(cfg) for name, cfg in CONFIG.upstreams.items()
}


def _gate_for(profile: ProfileConfig) -> _UpstreamGate:
    return _UPSTREAM_GATES[profile.upstream]


def _upstream_url(profile: ProfileConfig) -> str:
    return CONFIG.upstreams[profile.upstream].url


@asynccontextmanager
async def _gated(profile: ProfileConfig, *, path: str = "?"):
    """Acquire a slot from the profile's upstream gate, with
    priority-aware queueing and a hard queue timeout.

    Always releases on exit and on cancellation. Raises `_QueueTimeout`
    if the gate's `queue_timeout_s` is exceeded — handlers convert
    that to a 503 with a `Retry-After` header.
    """
    gate = _gate_for(profile)
    slot_id = await gate.acquire(
        profile_name=profile.name,
        path=path,
        priority=profile.queue_priority,
        queue_timeout=gate.cfg.queue_timeout_s,
    )
    try:
        yield gate
    finally:
        await gate.release(slot_id)


async def _gate_watchdog_loop(interval_s: float = 30.0) -> None:
    """Background task: periodically log slots that have been held
    longer than the per-upstream `stuck_warn_s` threshold. Doesn't
    auto-cancel them — that's a stronger semantic and we'd rather
    surface the issue than silently kill a long-running tool call."""
    while True:
        try:
            await asyncio.sleep(interval_s)
            now = time.monotonic()
            for gate in _UPSTREAM_GATES.values():
                stuck = []
                async with gate._lock:
                    for slot_id, info in list(gate._in_flight.items()):
                        age = now - info["started"]
                        if age > gate.cfg.stuck_warn_s and slot_id not in gate._stuck_logged:
                            gate._stuck_logged.add(slot_id)
                            stuck.append((slot_id, age, info))
                for slot_id, age, info in stuck:
                    print(
                        f"[gate-watchdog] upstream={gate.cfg.name} slot={slot_id} "
                        f"held {age:.0f}s (>{gate.cfg.stuck_warn_s:.0f}s) "
                        f"profile={info['profile']} path={info['path']}",
                        flush=True,
                    )
        except asyncio.CancelledError:
            return
        except Exception as exc:  # never let the watchdog die
            print(f"[gate-watchdog] error: {exc!r}", flush=True)


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
LOG_MAX_FILES = int(os.environ.get("BRIDGE_LOG_MAX_FILES", "7"))
LOG_MAX_BYTES = int(os.environ.get("BRIDGE_LOG_MAX_BYTES", str(512 * 1024 * 1024)))

_usage_history: deque[dict] = deque(maxlen=USAGE_RING_SIZE)
_usage_subscribers: set[asyncio.Queue[dict]] = set()
_activity_history: deque[dict] = deque(maxlen=ACTIVITY_RING_SIZE)
_activity_subscribers: set[asyncio.Queue[dict]] = set()
_started_at = time.time()

# ---------------------------------------------------------------------------
# Compaction-event detection
#
# opencode's full-summarization compaction (overflow-driven) is invisible
# from the ACP wire and produces no log line in opencode's own log file —
# the only place we can spot it is HERE, where the bridge sees the chat
# completion request. Its system prompt contains the SUMMARY_TEMPLATE
# verbatim, so a substring match is enough.
#
# We map the inbound TCP source port → opencode pid so the panel can
# filter to only the compactions for ITS happy session (otherwise a
# multi-session user would see every panel light up on every other
# session's compaction).
# ---------------------------------------------------------------------------

_COMPACTION_SIGNATURE = (
    "Output exactly the Markdown structure shown inside <template>"
)
_compaction_history: deque[dict] = deque(maxlen=200)

# Disk log data dir (same location as config)
_CONFIG_DIR = Path(
    os.environ.get(
        "BRIDGE_CONFIG_PATH",
        os.path.expanduser("~/.config/resilient-llm-bridge/config.yaml"),
    )
).parent
LOG_DIR = Path(os.environ.get("BRIDGE_LOG_DIR", str(_CONFIG_DIR / "logs")))

# Current log file handles (opened per-session, rotated by size)
_activity_log_file = None
_usage_log_file = None
_activity_log_size = 0
_usage_log_size = 0
_log_lock = asyncio.Lock() if False else None  # lock is not needed — single-threaded ASGI


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _rotate_log(filename: str, current_size: int) -> int:
    """Rotate JSONL log files. Keep at most LOG_MAX_FILES files, total <= LOG_MAX_BYTES."""
    _ensure_log_dir()
    base = LOG_DIR / filename
    # Remove oldest if we already have LOG_MAX_FILES files
    existing = sorted(LOG_DIR.glob(filename.replace("*", ".*") + ".*"))
    while len(existing) >= LOG_MAX_FILES:
        oldest = existing.pop(0)
        try:
            oldest.unlink()
        except OSError:
            pass
    # Rename current to numbered
    if base.exists():
        ts = time.strftime("%Y%m%d-%H%M%S")
        rotated = LOG_DIR / f"{filename}.{ts}"
        # Avoid collision
        counter = 0
        while (rotated := LOG_DIR / f"{filename}.{ts}-{counter}").exists():
            counter += 1
        try:
            base.rename(rotated)
        except OSError:
            pass
    return 0


def _write_log(filename: str, record: dict) -> None:
    """Append a single JSON record to the named log file, rotating if needed."""
    _ensure_log_dir()
    global _activity_log_file, _usage_log_file, _activity_log_size, _usage_log_size

    target = LOG_DIR / filename
    fh = None
    try:
        if filename == "activity.jsonl":
            fh = _activity_log_file
            if fh is None or fh.closed:
                fh = open(target, "a", encoding="utf-8")
                _activity_log_file = fh
        else:
            fh = _usage_log_file
            if fh is None or fh.closed:
                fh = open(target, "a", encoding="utf-8")
                _usage_log_file = fh

        line = json.dumps(record, default=str, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))

        # Rotate if this write would exceed the per-file budget
        budget = max(1, LOG_MAX_BYTES // LOG_MAX_FILES)
        if (filename == "activity.jsonl" and _activity_log_size + line_bytes > budget) or \
           (filename == "usage.jsonl" and _usage_log_size + line_bytes > budget):
            if fh and not fh.closed:
                try:
                    fh.flush()
                    fh.close()
                except OSError:
                    pass
            _rotate_log(filename, 0)
            if filename == "activity.jsonl":
                _activity_log_size = 0
            else:
                _usage_log_size = 0
            fh = open(target, "a", encoding="utf-8")
            if filename == "activity.jsonl":
                _activity_log_file = fh
            else:
                _usage_log_file = fh

        fh.write(line)
        fh.flush()

        if filename == "activity.jsonl":
            _activity_log_size += line_bytes
        else:
            _usage_log_size += line_bytes
    except OSError:
        pass


def _cleanup_old_logs() -> None:
    """Remove any orphaned log files that exceed the cap."""
    _ensure_log_dir()
    all_logs = sorted(LOG_DIR.glob("activity.jsonl.*")) + sorted(LOG_DIR.glob("usage.jsonl.*"))
    total = sum(f.stat().st_size for f in all_logs if f.exists())
    while len(all_logs) > LOG_MAX_FILES and total > LOG_MAX_BYTES:
        oldest = all_logs.pop(0)
        try:
            total -= oldest.stat().st_size
            oldest.unlink()
        except OSError:
            pass

# Lifetime counters for recovery firings + retries. These are cheap so we
# track them since process start; for time-windowed views the dashboard
# aggregates from `_activity_history` / `_usage_history`.
_recovery_counts: dict[str, int] = {
    "thinking_overflow": 0,
    "silent_completion": 0,
    "fake_invocation": 0,
    "truncated_content": 0,
    "empty_with_stop_retry": 0,
    "tool_call_args_retry": 0,
}
_retry_counts: dict[str, int] = {"retried": 0, "gave_up": 0}


def _record_recovery(kind: str) -> None:
    if kind in _recovery_counts:
        _recovery_counts[kind] += 1


def _broadcast_usage(record: dict) -> None:
    _usage_history.append(record)
    print(
        f"usage profile={record.get('profile')} model={record.get('model')}"
        f" in={record.get('input_tokens')} out={record.get('output_tokens')}"
        f" think={record.get('thinking_tokens')}"
        f" total_out={record.get('total_output_tokens')}",
        flush=True,
    )
    _write_log("usage.jsonl", record)
    for queue in list(_usage_subscribers):
        try:
            queue.put_nowait(record)
        except asyncio.QueueFull:
            pass


def _broadcast_activity(record: dict) -> None:
    _activity_history.append(record)
    _write_log("activity.jsonl", record)
    for queue in list(_activity_subscribers):
        try:
            queue.put_nowait(record)
        except asyncio.QueueFull:
            pass


# ---------------------------------------------------------------------------
# /proc/net/tcp parsing — map TCP source port → owning pid
#
# Used only when we detect a SUMMARY_TEMPLATE prompt and want to know
# which opencode process sent it. Linux-only by design; on non-Linux
# the lookup returns None and the panel filter degrades gracefully
# (events with unknown pid are dropped, which is the right thing).
# ---------------------------------------------------------------------------


def _resolve_source_pid(host: str, port: int) -> int | None:
    if not host or not isinstance(port, int) or port <= 0 or port > 65535:
        return None
    inode = _find_socket_inode(host, port)
    if inode is None:
        return None
    return _find_pid_owning_inode(inode)


def _find_socket_inode(host: str, port: int) -> str | None:
    """Look up the local inode for an established TCP connection
    whose remote (peer) end is `host:port`."""
    candidates = ("/proc/net/tcp", "/proc/net/tcp6")
    target_port_hex = f"{port:04X}"
    for path in candidates:
        try:
            with open(path, "r", encoding="ascii") as fh:
                fh.readline()  # header
                for line in fh:
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    # parts[1] = local_address  (HOST:PORT in hex)
                    # parts[2] = remote_address — for inbound conns this
                    #           is the peer (= the client). FastAPI shows
                    #           that peer to us as request.client.{host,port}.
                    local = parts[1]
                    inode = parts[9]
                    # We're the SERVER receiving a connection. The peer's
                    # ephemeral port is the LOCAL port from the peer's
                    # perspective, but in our /proc/net/tcp it's the
                    # LOCAL port of OUR socket only when scanning bridge
                    # process — the peer's pid owns the OTHER side.
                    # Easiest: scan all rows and match on parts[1] split
                    # by ':' — port half — equal to target.
                    if ":" not in local:
                        continue
                    _, port_hex = local.rsplit(":", 1)
                    if port_hex.upper() == target_port_hex:
                        return inode
        except OSError:
            continue
    return None


def _find_pid_owning_inode(inode: str) -> int | None:
    """Walk /proc/*/fd looking for a symlink target like
    `socket:[<inode>]`. Returns the first matching pid (there can only
    be one for a given socket fd in practice)."""
    needle = f"socket:[{inode}]"
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    for entry in entries:
        if not entry.isdigit():
            continue
        fd_dir = f"/proc/{entry}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(f"{fd_dir}/{fd}")
            except OSError:
                continue
            if target == needle:
                return int(entry)
    return None


def _record_compaction(
    *,
    profile_name: str,
    model: str | None,
    source_host: str | None,
    source_port: int | None,
    body: dict,
) -> None:
    pid = (
        _resolve_source_pid(source_host, source_port)
        if source_host and source_port
        else None
    )
    record = {
        "ts": time.time(),
        "profile": profile_name,
        "model": model,
        "source_pid": pid,
        "source_host": source_host,
        "source_port": source_port,
        "input_chars": _estimate_input_chars(body),
    }
    _compaction_history.append(record)


def _estimate_input_chars(body: dict) -> int:
    total = 0
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        c = msg.get("content")
        if isinstance(c, str):
            total += len(c)
    return total


def _looks_like_summary_request(body: dict) -> bool:
    """Match opencode's SUMMARY_TEMPLATE in the system prompt. The
    string is lifted verbatim from sst/opencode's compaction.ts and is
    distinctive enough that a substring match has effectively zero
    false positives — no real user prompt asks for "the Markdown
    structure shown inside <template>"."""
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") not in ("system", "user"):
            continue
        content = msg.get("content")
        if isinstance(content, str) and _COMPACTION_SIGNATURE in content:
            return True
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and _COMPACTION_SIGNATURE in text:
                        return True
    return False


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
        "thinking_tokens": int(usage.get("thinking_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }
    if record["total_tokens"] == 0:
        record["total_tokens"] = record["input_tokens"] + record["output_tokens"]
    record["total_output_tokens"] = record["thinking_tokens"] + record["output_tokens"]
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


_CLIENT_DISABLE_EFFORTS = {"none", "off", "disabled", "false", "no"}


def _apply_effort_budget(body: dict, profile: ProfileConfig) -> None:
    """Resolve the thinking-related fields based on profile policy.

    Two-layer precedence: **profile wins when set, otherwise client wins**.

    For each of `enable_thinking` and `thinking_token_budget`:

      * If the profile config defines the field
        (`thinking_enabled` is not None, or `default_thinking_budget`
        is a positive int), the bridge OVERRIDES whatever the client
        sent — the profile's value goes upstream regardless.
      * If the profile config is silent on the field, the bridge
        respects what the client sent. For budget specifically, when
        the client gave an `effort` hint (e.g. Codex's
        `reasoning.effort=high`) without an explicit
        `extra_body.thinking_token_budget`, the bridge translates the
        effort via `_EFFORT_TO_THINKING_BUDGET` — translation, not
        override. Effort values in `_CLIENT_DISABLE_EFFORTS` are
        translated to `enable_thinking=false`.

    Placement:
      - `thinking_token_budget` → top-level of `extra_body` (vLLM
        SamplingParams reads it there)
      - `enable_thinking` → `extra_body.chat_template_kwargs`
        (Qwen3 chat template reads it there)
    """
    extra = body.get("extra_body")
    if not isinstance(extra, dict):
        extra = {}
    chat_kwargs = extra.get("chat_template_kwargs")
    if not isinstance(chat_kwargs, dict):
        chat_kwargs = {}

    # ----- enable_thinking decision -----
    if profile.thinking_enabled is not None:
        # Profile is authoritative — override.
        chat_kwargs["enable_thinking"] = profile.thinking_enabled
    # else: leave whatever the client sent (or didn't send) intact.

    # ----- thinking_token_budget decision -----
    if isinstance(profile.default_thinking_budget, int) and profile.default_thinking_budget > 0:
        # Profile is authoritative — override any client effort/budget.
        extra["thinking_token_budget"] = profile.default_thinking_budget
    else:
        # Profile silent — translate client signals. Skip if the client
        # already put an explicit budget in extra_body (their value
        # wins).
        if "thinking_token_budget" not in extra:
            effort: str | None = None
            reasoning = body.get("reasoning")
            if isinstance(reasoning, dict) and isinstance(reasoning.get("effort"), str):
                effort = reasoning["effort"].strip().lower()
            if effort is None and isinstance(body.get("reasoning_effort"), str):
                effort = body["reasoning_effort"].strip().lower()
            if effort and effort in _EFFORT_TO_THINKING_BUDGET:
                extra["thinking_token_budget"] = _EFFORT_TO_THINKING_BUDGET[effort]
            elif (
                effort
                and effort in _CLIENT_DISABLE_EFFORTS
                and profile.thinking_enabled is None
            ):
                # Client said no thinking via `effort=none`; surface as
                # explicit `enable_thinking=false`. Skip if the profile
                # already overrode (profile-set wins).
                chat_kwargs["enable_thinking"] = False

    if chat_kwargs:
        extra["chat_template_kwargs"] = chat_kwargs
    body["extra_body"] = extra
    # No more max_tokens inflation: `_ensure_room_for_injected_thinking`
    # is gone. The thinking budget enforces upstream now (verified
    # 2026-04-30), so there's no runaway reasoning to "make room" for.


def _estimate_prompt_tokens(body: dict) -> int:
    """Char-based estimate of how many tokens the prompt will consume.
    We don't ship a tokenizer — the bridge runs on a different process
    and can't import the model's BPE files — so we estimate from char
    count using a deliberately pessimistic 3.0 chars/token ratio.

    Why 3.0 instead of the textbook 3.5: tool results in agent
    workloads are dominated by JSON, code, log dumps, and other
    structurally-dense content, which tokenizes at ~2.5–3.0 chars per
    token (every `{`, `,`, `:` is its own token; short keys like `id`
    burn a token apiece). The textbook 3.5 ratio assumes English
    prose; under-counting on a JSON-heavy prompt drops us below the
    real input size and the cap-clamp lets the request through into
    a 400 ContextWindowExceeded — which some clients (e.g. opencode)
    don't recognise as overflow and silently hang on.

    Over-counting is cheap (slightly tighter `max_tokens` cap);
    under-counting is expensive (silent stuck session). Bias to
    over-count.

    Walks the obvious string-bearing fields: messages.content,
    instructions, input items, tools schemas, system prompts.
    """
    chars = 0
    if isinstance(body.get("instructions"), str):
        chars += len(body["instructions"])
    if isinstance(body.get("system"), str):
        chars += len(body["system"])
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        c = msg.get("content")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    chars += len(str(part.get("text") or ""))
    for item in body.get("input") or []:
        if not isinstance(item, dict):
            continue
        c = item.get("content")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    chars += len(str(part.get("text") or ""))
    # Tools schemas count too — they're inlined in the prompt.
    tools = body.get("tools")
    if isinstance(tools, list):
        try:
            chars += len(json.dumps(tools, default=str))
        except (TypeError, ValueError):
            pass
    return int(chars / 3.0)


def _clamp_max_tokens_to_context(body: dict, profile: ProfileConfig) -> None:
    """Cap the client's declared `max_tokens` / `max_output_tokens` so
    `prompt + max_tokens ≤ context_window − safety_margin`. Runs
    unconditionally on every request.

    Why: clients that compute their own
    `max_tokens = context_window − input_tokens` (e.g. opencode) hit a
    classic off-by-one when the input grows between turns — request
    arrives at exactly `context_window + N` and the upstream rejects
    with ContextWindowExceeded. Some clients don't recognise the
    litellm-shaped error as an overflow signal and silently hang
    instead of triggering compaction. We clamp here so the request
    always fits and the client's own retry/compaction logic doesn't
    matter.

    Idempotent: if the cap is already safe, does nothing.
    """
    if "messages" in body:
        field = "max_tokens"
    elif "input" in body:
        field = "max_output_tokens"
    else:
        return
    current = body.get(field)
    if not isinstance(current, int) or current <= 0:
        return
    upstream_cfg = CONFIG.upstreams.get(profile.upstream)
    if upstream_cfg is None:
        return
    prompt_est = _estimate_prompt_tokens(body)
    max_allowed = upstream_cfg.context_window - prompt_est - upstream_cfg.context_safety_margin
    # If the prompt is already over the limit, leave a tiny slot — the
    # upstream will reject either way, but at least don't ship a
    # negative cap.
    max_allowed = max(max_allowed, 256)
    if current > max_allowed:
        body[field] = max_allowed


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
    # Always run model aliasing first so downstream transforms (and the
    # upstream itself) see the resolved id.
    if profile.model_aliases:
        original = body.get("model")
        resolved = profile.resolve_model(original)
        if isinstance(resolved, str) and resolved != original:
            body["model"] = resolved
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
        _apply_effort_budget(body, profile)
    if profile.has("force_serial_tool_calls") and kind == "responses":
        _force_serial_tool_calls(body)
    if profile.has("drop_oai_only_fields"):
        _drop_oai_only_fields(body)
    # Always last: cap max_tokens so prompt+cap fits the context window.
    # Runs after every other transform (which may have bumped or
    # rewritten max_tokens) so the clamp sees the final value.
    _clamp_max_tokens_to_context(body, profile)
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
                async with _gated(profile):
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
    # Fired exclusively from the chat/completions handler, so the body has
    # `messages` natively. Fall back to converting `input` only as a
    # safety net for callers that might pass a responses-API body in the
    # future.
    raw_messages = original_body.get("messages")
    if isinstance(raw_messages, list) and raw_messages:
        messages = [m for m in raw_messages if isinstance(m, dict)]
    else:
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


def _required_fields_for_tool(tools_list: Any, tool_name: str | None) -> set[str]:
    """Extract the `required` set from the schema for ``tool_name``.

    Returns an empty set when the tool isn't found, the schema is
    malformed, or the schema doesn't declare any required fields. The
    caller treats "no required fields" as "any args dict is valid".
    """
    if not isinstance(tools_list, list) or not isinstance(tool_name, str):
        return set()
    for t in tools_list:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        if fn.get("name") != tool_name:
            continue
        params = fn.get("parameters") or {}
        if not isinstance(params, dict):
            return set()
        req = params.get("required")
        if isinstance(req, list):
            return {str(f) for f in req}
        return set()
    return set()


def _validate_tool_calls(tool_calls: Any, tools_list: Any) -> bool:
    """Return True when every tool_call's args satisfy its schema's
    required fields. Malformed JSON or unknown tool names also count as
    invalid (we want to retry those too)."""
    if not isinstance(tool_calls, list) or not tool_calls:
        return True  # nothing to validate
    for tc in tool_calls:
        if not isinstance(tc, dict):
            return False
        fn = tc.get("function") or {}
        name = fn.get("name")
        args_raw = fn.get("arguments")
        if not isinstance(args_raw, str):
            return False
        try:
            parsed = json.loads(args_raw)
        except (json.JSONDecodeError, ValueError):
            return False
        if not isinstance(parsed, dict):
            return False
        required = _required_fields_for_tool(tools_list, name)
        if required and not required.issubset(parsed.keys()):
            return False
    return True


def _client_already_disabled_thinking(body: dict) -> bool:
    """Return True if the body explicitly disables thinking, so retrying
    with thinking-off would be a no-op."""
    eb = body.get("extra_body") or {}
    if not isinstance(eb, dict):
        return False
    ctk = eb.get("chat_template_kwargs") or {}
    if isinstance(ctk, dict) and ctk.get("enable_thinking") is False:
        return True
    re_top = body.get("reasoning_effort")
    if isinstance(re_top, str) and re_top.strip().lower() in _CLIENT_DISABLE_EFFORTS:
        return True
    re_obj = body.get("reasoning")
    if isinstance(re_obj, dict):
        eff = re_obj.get("effort")
        if isinstance(eff, str) and eff.strip().lower() in _CLIENT_DISABLE_EFFORTS:
            return True
    return False


def _build_thinking_off_retry_body(body: dict) -> dict:
    """Clone the request body and force thinking off for the retry.

    Drops `thinking_token_budget` (irrelevant when thinking is off)
    and sets `chat_template_kwargs.enable_thinking=false`. The retry
    is non-streaming (caller wraps it via `_post_chat_for_text`-style
    helpers) so we also force `stream=False` defensively.
    """
    retry = copy.deepcopy(body)
    retry["stream"] = False
    retry.pop("stream_options", None)
    eb = retry.get("extra_body")
    if not isinstance(eb, dict):
        eb = {}
    eb.pop("thinking_token_budget", None)
    ctk = eb.get("chat_template_kwargs")
    if not isinstance(ctk, dict):
        ctk = {}
    ctk["enable_thinking"] = False
    eb["chat_template_kwargs"] = ctk
    retry["extra_body"] = eb
    return retry


async def _retry_chat_thinking_off(
    body: dict, headers: dict[str, str], profile: ProfileConfig
) -> tuple[int, dict]:
    """Run a non-streaming chat/completions retry with thinking disabled.

    Returns (status, payload) — caller validates the payload before
    deciding whether to swap.
    """
    cfg = CONFIG.upstreams[profile.upstream]
    url = f"{_upstream_url(profile)}/chat/completions"
    retry_body = _build_thinking_off_retry_body(body)
    last_status = 502
    payload: dict = {"error": {"message": "unreachable"}}
    try:
        async for attempt in _retry_policy(cfg):
            with attempt:
                async with _gated(profile):
                    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                        r = await client.post(url, json=retry_body, headers=headers)
                last_status = r.status_code
                if r.status_code in _RETRYABLE_STATUS:
                    raise _UpstreamHTTPError(r.status_code, r.text)
                payload = (
                    r.json()
                    if r.headers.get("content-type", "").startswith("application/json")
                    else {}
                )
                break
    except _QueueTimeout:
        return 503, {"error": {"message": "queue timeout on retry"}}
    except (RetryError, _UpstreamHTTPError):
        return last_status, payload if isinstance(payload, dict) else {"error": {}}
    except httpx.HTTPError as e:
        return 502, {"error": {"message": f"retry upstream error: {e}"}}
    return last_status, payload


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


def _coerce_error_payload(status: int, body_text: str) -> dict:
    """Parse the upstream's error text into an OpenAI-shaped envelope
    (`{error: {message, type, code}}`) so downstream clients can
    render it. Forwards the upstream's own envelope verbatim when
    it's already in that shape.

    The bridge previously wrapped errors in `{type:"error", status,
    body}`, which neither opencode (`@ai-sdk/openai-compatible`) nor
    codex CLI knows how to parse — both expect either `choices` or a
    nested `error` object. The Zod failure manifests in happy as a
    cryptic "expected array, received undefined" instead of the real
    upstream message.
    """
    try:
        parsed = json.loads(body_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
        return parsed
    if isinstance(parsed, dict) and "message" in parsed:
        return {
            "error": {
                "message": str(parsed.get("message")),
                "type": parsed.get("type") or "upstream_error",
                "code": str(parsed.get("code") or status),
            }
        }
    return {
        "error": {
            "message": body_text or f"upstream HTTP {status}",
            "type": "upstream_error",
            "code": str(status),
        }
    }


def _sse_chat_error(status: int, body_text: str) -> bytes:
    """Emit a chat/completions-style SSE error event. The shape
    `{error: {message, type, code}}` is what AI-SDK / OpenAI clients
    parse on streaming errors."""
    return _sse(_coerce_error_payload(status, body_text))


def _sse_responses_error(status: int, body_text: str) -> bytes:
    """Emit a responses-API style error event. OpenAI's spec uses an
    event with `type: "error"` plus `code`/`message` fields. We
    flatten the payload so both fields are at the top level."""
    payload = _coerce_error_payload(status, body_text)
    err = payload.get("error", {}) if isinstance(payload, dict) else {}
    return _sse(
        {
            "type": "error",
            "code": err.get("code") or str(status),
            "message": err.get("message") or body_text or "upstream error",
            "param": err.get("param"),
        }
    )


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
    state: dict | None = None,
) -> AsyncIterator[bytes]:
    """Stream a /v1/responses request, applying SSE rewrites and post-stream
    recovery. `state` is an optional out-dict the caller can pass in to read
    `state["recovery"]` (the recovery kind that fired, if any) after the
    generator drains."""
    if state is None:
        state = {}
    model_hint = body.get("model")
    cfg = CONFIG.upstreams[profile.upstream]
    upstream_url = f"{_upstream_url(profile)}/responses"

    # State for vLLM #39426 SSE rewrite (fc_state) and overflow recovery.
    fc_state: dict[int, dict] = {}
    reasoning_accum = ""
    message_text_accum = ""
    message_emitted = False
    completed_payload: dict | None = None
    # Track in-flight message items so we can synthesize the closing
    # events that NaN's /v1/responses stream sometimes omits. Keyed by
    # output_index. Each entry tracks {id, role, text, content_index,
    # last_seq, model, content_part_added, output_text_done,
    # content_part_done}. Codex CLI requires output_item.done for an
    # agentMessage to fire its `item/completed` event — without it,
    # happy never receives an `agent_message`, which manifests as
    # "session spawned but no response ever shows".
    open_messages: dict[int, dict] = {}
    last_seq = 0
    do_parallel_fix = profile.has("parallel_tool_sse_fix")

    async def _open_stream():
        # tenacity on the entire stream open is fine — we don't replay
        # mid-stream, we only retry the connect/headers phase.
        async for attempt in _retry_policy(cfg):
            with attempt:
                async with _gated(profile):
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
    except _QueueTimeout as exc:
        yield _sse_responses_error(503, str(exc))
        return
    except (RetryError, _UpstreamHTTPError) as exc:
        status = getattr(exc, "status", 502)
        body_text = getattr(exc, "body", str(exc))
        yield _sse_responses_error(status, body_text)
        return

    try:
        if response.status_code >= 400:
            error_text = (await response.aread()).decode("utf-8", errors="ignore")
            yield _sse_responses_error(response.status_code, error_text)
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
                if isinstance(payload.get("sequence_number"), int):
                    last_seq = max(last_seq, payload["sequence_number"])

                if etype == "response.output_item.added":
                    item = payload.get("item") or {}
                    if item.get("type") == "message":
                        idx = payload.get("output_index")
                        if isinstance(idx, int):
                            open_messages[idx] = {
                                "id": item.get("id") or f"msg_{idx}_{int(time.time()*1000)}",
                                "role": item.get("role") or "assistant",
                                "text": "",
                                "content_index": 0,
                                "model": payload.get("model"),
                                "content_part_added": False,
                                "output_text_done": False,
                                "content_part_done": False,
                            }
                if etype == "response.content_part.added":
                    idx = payload.get("output_index")
                    if isinstance(idx, int) and idx in open_messages:
                        open_messages[idx]["content_part_added"] = True
                        ci = payload.get("content_index")
                        if isinstance(ci, int):
                            open_messages[idx]["content_index"] = ci
                if etype == "response.output_text.delta":
                    delta = payload.get("delta")
                    if isinstance(delta, str) and delta:
                        message_emitted = True
                        message_text_accum += delta
                        idx = payload.get("output_index")
                        if isinstance(idx, int) and idx in open_messages:
                            open_messages[idx]["text"] += delta
                if etype == "response.output_text.done":
                    idx = payload.get("output_index")
                    if isinstance(idx, int) and idx in open_messages:
                        open_messages[idx]["output_text_done"] = True
                if etype == "response.content_part.done":
                    idx = payload.get("output_index")
                    if isinstance(idx, int) and idx in open_messages:
                        open_messages[idx]["content_part_done"] = True
                if etype == "response.output_item.done":
                    idx = payload.get("output_index")
                    if isinstance(idx, int) and idx in open_messages:
                        # Upstream closed it properly — nothing to
                        # synthesize for this index.
                        open_messages.pop(idx, None)

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

    # Diagnostic trace — useful for debugging silent/empty responses
    # like "happy spawns codex but no answer ever shows". Logged for
    # every responses-API completion. Cheap and easy to grep.
    if profile.name in {"codex", "default"}:
        items = response_obj.get("output") or []
        item_summary = []
        for it in items:
            if not isinstance(it, dict):
                item_summary.append(type(it).__name__)
                continue
            t = it.get("type", "?")
            if t == "message":
                texts = []
                for p in it.get("content") or []:
                    if isinstance(p, dict) and p.get("type") == "output_text":
                        texts.append((p.get("text") or "")[:80])
                item_summary.append(f"message[role={it.get('role')},text={texts!r}]")
            elif t == "function_call":
                item_summary.append(f"function_call[name={it.get('name')}]")
            elif t == "reasoning":
                summary_parts = it.get("summary") or []
                item_summary.append(
                    f"reasoning[summary_parts={len(summary_parts)},"
                    f"content_chars={sum(len((c.get('text') or '')) for c in it.get('content') or [] if isinstance(c, dict))}]"
                )
            else:
                item_summary.append(t)
        print(
            f"[codex-trace] profile={profile.name} status={response_obj.get('status')!r}"
            f" finish_reason={response_obj.get('incomplete_details') or 'OK'}"
            f" items={item_summary}"
            f" message_emitted={message_emitted}"
            f" message_text_chars={len(message_text_accum)}"
            f" message_text_preview={message_text_accum[:120]!r}"
            f" reasoning_chars={len(reasoning_accum)}",
            flush=True,
        )

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
                kind = "thinking_overflow"
            elif fake_invocation_kicker:
                kind = "fake_invocation"
            else:
                kind = "silent_completion"
            _record_recovery(kind)
            state["recovery"] = kind

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

    # NaN's /v1/responses stream sometimes omits the closing events for
    # message items (output_text.done, content_part.done,
    # output_item.done) — Codex CLI then never fires its own
    # `item/completed`, so happy never receives an `agent_message`
    # event. Synthesize the missing events here so the answer is
    # actually delivered to the client.
    seq = last_seq
    for idx, st in list(open_messages.items()):
        if not st["text"]:
            # The item was opened but never produced text — likely a
            # tool_call wrapper, leave it alone.
            continue
        if not st["output_text_done"]:
            seq += 1
            yield _sse(
                {
                    "type": "response.output_text.done",
                    "item_id": st["id"],
                    "output_index": idx,
                    "content_index": st["content_index"],
                    "text": st["text"],
                    "sequence_number": seq,
                    "model": st["model"],
                }
            )
        if not st["content_part_done"]:
            seq += 1
            yield _sse(
                {
                    "type": "response.content_part.done",
                    "item_id": st["id"],
                    "output_index": idx,
                    "content_index": st["content_index"],
                    "part": {
                        "type": "output_text",
                        "text": st["text"],
                        "annotations": [],
                    },
                    "sequence_number": seq,
                    "model": st["model"],
                }
            )
        seq += 1
        yield _sse(
            {
                "type": "response.output_item.done",
                "output_index": idx,
                "item": {
                    "id": st["id"],
                    "type": "message",
                    "role": st["role"],
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": st["text"],
                            "annotations": [],
                        }
                    ],
                },
                "sequence_number": seq,
                "model": st["model"],
            }
        )
        open_messages.pop(idx, None)
    if seq != last_seq:
        # We synthesized events — bump the sequence_number on the
        # buffered completed payload so it doesn't go backwards.
        completed_payload = dict(completed_payload)
        completed_payload["sequence_number"] = seq + 1

    # No recovery applied — emit the original completed event.
    record = _extract_usage(profile.name, model_hint, response_obj)
    if record:
        _broadcast_usage(record)
    yield _sse(completed_payload)


# =============================================================================
# /v1/chat/completions streaming
# =============================================================================


# Heartbeat sent to the SSE client when the upstream is silent. Most clients
# (opencode's @ai-sdk/openai-compatible) implement chunkTimeout as raw-bytes
# inactivity, so any bytes — including SSE comments — reset their watchdog.
# Comments are part of the SSE spec (line starting with `:`) and parsers are
# required to ignore their content.
_SSE_KEEPALIVE = b": keepalive\n\n"
# How long we let the upstream go silent before we emit a keepalive. Has to
# stay safely under client-side chunk timeouts (opencode default 30s, others
# usually similar) so the client never sees inactivity.
_STREAM_KEEPALIVE_INTERVAL_S = 15.0
# How long the upstream may go silent in total before we give up. Distinct
# from connect/idle limits in `httpx.Timeout`: we want to keep streams alive
# through long thinking pauses (heavy reasoning models can sit silent for a
# minute mid-tool-call) but a 5-minute black hole is a dead connection.
_STREAM_SILENCE_LIMIT_S = 300.0


class _StreamStalledError(RuntimeError):
    """Upstream stopped emitting bytes for `_STREAM_SILENCE_LIMIT_S`. We
    treat this like a hard upstream failure so the bridge can surface a
    proper SSE error to the client instead of letting it hang forever."""


async def _iter_bytes_with_keepalive(
    response: httpx.Response,
) -> AsyncIterator[bytes]:
    """Wrap `response.aiter_bytes()` with a heartbeat so a slow/stalled
    upstream doesn't trip the client's chunk-inactivity timeout.

    Heartbeats are SSE comments — they pass through any spec-conforming
    parser as a no-op, but keep raw bytes flowing so the client's
    `chunkTimeout` watchdog stays happy.

    Critical correctness rule: a keepalive byte **must only be emitted
    on an SSE event boundary** (right after a `\\n\\n`). Injecting one
    inside an in-flight `data: {...}` line splits the JSON across two
    parsed events on the client and the tool-call args end up corrupted
    (we observed opencode receiving `arguments="{"` and bailing with
    "JSON Parse error: Expected '}'" because the keepalive landed
    mid-event during a slow tool-call generation). When the upstream
    pauses mid-event we just hold the keepalive until the next boundary
    arrives — by then either the model resumed (no need for keepalive)
    or `_STREAM_SILENCE_LIMIT_S` is hit and we abort.
    """
    iterator = response.aiter_bytes().__aiter__()
    last_real_chunk = time.monotonic()
    # Track whether the most recent yielded chunk ended on a `\n\n` event
    # boundary. We start `True` so a slow first event still gets its
    # heartbeat (the client hasn't started a partial parse yet either).
    on_event_boundary = True
    while True:
        try:
            chunk = await asyncio.wait_for(
                iterator.__anext__(), timeout=_STREAM_KEEPALIVE_INTERVAL_S
            )
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            silence = time.monotonic() - last_real_chunk
            if silence >= _STREAM_SILENCE_LIMIT_S:
                raise _StreamStalledError(
                    f"upstream silent for {silence:.0f}s, aborting stream"
                )
            if on_event_boundary:
                yield _SSE_KEEPALIVE
            # Mid-event: skip this tick, wait for the upstream to either
            # resume (pushing us past the boundary) or hit the silence
            # limit. We do NOT inject — corrupting the event is worse
            # than letting the client's chunk timeout fire.
            continue
        last_real_chunk = time.monotonic()
        yield chunk
        on_event_boundary = chunk.endswith(b"\n\n")


async def _stream_chat_completions(
    body: dict,
    headers: dict[str, str],
    profile: ProfileConfig,
    state: dict | None = None,
) -> AsyncIterator[bytes]:
    """Forward chat/completions SSE byte-for-byte by default, sniffing
    usage along the way.

    When the profile has `tool_call_args_retry` enabled, switches to a
    BUFFERED mode: collects all upstream chunks first, parses to
    extract the final assistant message, validates tool_call args
    against the schemas in `body.tools`, and either:
      - emits the buffered chunks unchanged (valid), or
      - retries the same request with thinking disabled and synthesizes
        a fresh SSE stream from the retry response (invalid).

    Buffered mode pays for-correctness with latency: the client sees no
    intermediate deltas, only the final result. Keepalive comments are
    still emitted to the client so chunkTimeout doesn't fire.
    """
    if state is None:
        state = {}
    model_hint = body.get("model")
    cfg = CONFIG.upstreams[profile.upstream]
    upstream_url = f"{_upstream_url(profile)}/chat/completions"

    async def _open_stream():
        async for attempt in _retry_policy(cfg):
            with attempt:
                async with _gated(profile):
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
    except _QueueTimeout as exc:
        yield _sse_chat_error(503, str(exc))
        return
    except (RetryError, _UpstreamHTTPError) as exc:
        status = getattr(exc, "status", 502)
        body_text = getattr(exc, "body", str(exc))
        yield _sse_chat_error(status, body_text)
        return

    do_retry_recovery = profile.has("tool_call_args_retry") and not _client_already_disabled_thinking(body)

    try:
        if response.status_code >= 400:
            error_text = (await response.aread()).decode("utf-8", errors="ignore")
            yield _sse_chat_error(response.status_code, error_text)
            return

        if not do_retry_recovery:
            # Existing path: byte-for-byte forward, sniff usage as it
            # passes by.
            buffer = b""
            try:
                async for chunk in _iter_bytes_with_keepalive(response):
                    if not chunk:
                        continue
                    yield chunk
                    if chunk == _SSE_KEEPALIVE:
                        continue
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
            except _StreamStalledError as exc:
                yield _sse_chat_error(504, str(exc))
            return

        # Speculative passthrough path:
        #   - Stream every chunk to the client as it arrives, EXCEPT we
        #     hold the very first chunks in a tiny buffer until we know
        #     what kind of turn this is.
        #   - As soon as we see a `delta.tool_calls` event, switch to
        #     full-buffer mode (validate at the end, retry if needed).
        #   - As soon as we see a non-empty `delta.content` text event,
        #     switch to passthrough mode (model committed to a
        #     conversational reply; tool_call retry doesn't apply).
        #   - If neither is seen by the time `finish_reason` arrives,
        #     flush the held bytes and emit the rest passthrough.
        #
        # This preserves true-streaming for content-only turns and only
        # incurs the buffer/retry cost on tool_call turns, where the
        # client typically waits for `finish_reason: tool_calls` anyway.

        DECISION_PENDING = 0
        DECISION_PASSTHROUGH = 1
        DECISION_BUFFER = 2
        decision = DECISION_PENDING
        held_bytes = bytearray()       # bytes received before decision was made
        upstream_buffer = bytearray()  # full buffer when in DECISION_BUFFER mode
        sse_buf = b""                  # incremental SSE event-line splitter (pre-decision)
        usage_obj: dict | None = None  # sniffed usage for passthrough path

        async def _flush_held():
            # Helper: if there are held bytes, emit them now. The caller
            # uses this once at the moment of switching to PASSTHROUGH.
            nonlocal held_bytes
            if held_bytes:
                _b = bytes(held_bytes)
                held_bytes = bytearray()
                return _b
            return b""

        try:
            async for chunk in _iter_bytes_with_keepalive(response):
                if not chunk:
                    continue
                if chunk == _SSE_KEEPALIVE:
                    yield chunk
                    continue
                if decision == DECISION_PASSTHROUGH:
                    yield chunk
                    # still sniff usage in passthrough
                    sse_buf += chunk
                    while b"\n\n" in sse_buf:
                        ev, sse_buf = sse_buf.split(b"\n\n", 1)
                        for line in ev.splitlines():
                            if not line.startswith(b"data: "): continue
                            raw = line[6:].strip()
                            if raw in (b"[DONE]", b""): continue
                            try: pl = json.loads(raw)
                            except (ValueError, UnicodeDecodeError): continue
                            if isinstance(pl, dict) and isinstance(pl.get("usage"), dict):
                                usage_obj = pl["usage"]
                    continue
                if decision == DECISION_BUFFER:
                    upstream_buffer.extend(chunk)
                    continue
                # DECISION_PENDING — hold bytes and inspect each event
                held_bytes.extend(chunk)
                sse_buf += chunk
                while b"\n\n" in sse_buf:
                    ev, sse_buf = sse_buf.split(b"\n\n", 1)
                    for line in ev.splitlines():
                        if not line.startswith(b"data: "): continue
                        raw = line[6:].strip()
                        if raw == b"[DONE]" or not raw: continue
                        try: pl = json.loads(raw)
                        except (ValueError, UnicodeDecodeError): continue
                        if not isinstance(pl, dict): continue
                        for choice in (pl.get("choices") or []):
                            if not isinstance(choice, dict): continue
                            delta = choice.get("delta") or {}
                            if not isinstance(delta, dict): continue
                            if delta.get("tool_calls"):
                                decision = DECISION_BUFFER
                                break
                            content = delta.get("content")
                            if isinstance(content, str) and content.strip():
                                decision = DECISION_PASSTHROUGH
                                break
                        else:
                            continue
                        break
                    if decision != DECISION_PENDING:
                        break
                if decision == DECISION_PASSTHROUGH:
                    flushed = await _flush_held()
                    if flushed:
                        yield flushed
                elif decision == DECISION_BUFFER:
                    # Move held bytes into the full buffer; nothing emitted.
                    upstream_buffer.extend(held_bytes)
                    held_bytes = bytearray()
        except _StreamStalledError as exc:
            yield _sse_chat_error(504, str(exc))
            return

        # End of upstream stream.
        if decision == DECISION_PASSTHROUGH:
            # Already streamed everything (plus held flush).
            if usage_obj:
                rec = _extract_usage(profile.name, model_hint,
                                     {"usage": usage_obj, "model": model_hint})
                if rec: _broadcast_usage(rec)
            return
        if decision == DECISION_PENDING:
            # No content, no tool_calls — empty turn or weird upstream.
            # Flush whatever we held and exit; nothing to validate.
            flushed = await _flush_held()
            if flushed:
                yield flushed
            return

        # DECISION_BUFFER: full buffer. Parse, validate, maybe retry.
        assembled = _assemble_chat_sse(bytes(upstream_buffer), model_hint)
        upstream_payload = assembled["payload"]
        msg = ((upstream_payload.get("choices") or [{}])[0]).get("message") or {}
        tool_calls = msg.get("tool_calls")
        if not tool_calls or _validate_tool_calls(tool_calls, body.get("tools")):
            yield bytes(upstream_buffer)
            if isinstance(assembled.get("usage"), dict):
                rec = _extract_usage(profile.name, model_hint,
                                     {"usage": assembled["usage"], "model": model_hint})
                if rec: _broadcast_usage(rec)
            return

        # Invalid → retry with thinking off.
        r_status, retry_payload = await _retry_chat_thinking_off(body, headers, profile)
        if r_status >= 400 or not isinstance(retry_payload, dict):
            yield bytes(upstream_buffer)
            return
        retry_choice = (retry_payload.get("choices") or [{}])[0]
        retry_msg = retry_choice.get("message") or {}
        retry_tcs = retry_msg.get("tool_calls")
        if not retry_tcs or not _validate_tool_calls(retry_tcs, body.get("tools")):
            yield bytes(upstream_buffer)
            return

        for synth_chunk in _synthesize_chat_sse(retry_payload, model_hint):
            yield synth_chunk
        state["recovery"] = "tool_call_args_retry"
        _record_recovery("tool_call_args_retry")
        usage = retry_payload.get("usage")
        if isinstance(usage, dict):
            rec = _extract_usage(profile.name, model_hint, retry_payload)
            if rec: _broadcast_usage(rec)
    finally:
        try:
            await response.aclose()
        finally:
            await client.aclose()


def _assemble_chat_sse(buf: bytes, model_hint: str | None) -> dict:
    """Walk buffered SSE bytes from a chat/completions stream and
    reconstruct the equivalent non-streaming payload.

    Returns ``{"payload": <chat completion-shaped dict>, "usage": <usage dict|None>}``.
    Any malformed lines are skipped silently; we just rebuild what we
    can. The reconstructed payload has the same shape as a non-stream
    response so downstream validation can run against it.
    """
    msg: dict[str, Any] = {"role": "assistant", "content": ""}
    tcs_by_index: dict[int, dict] = {}
    finish: str | None = None
    rid: str | None = None
    created: int | None = None
    model_id = model_hint
    usage_obj: dict | None = None

    for event in buf.split(b"\n\n"):
        for line in event.splitlines():
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw in (b"[DONE]", b""):
                continue
            try:
                payload = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            if rid is None and isinstance(payload.get("id"), str):
                rid = payload["id"]
            if created is None and isinstance(payload.get("created"), int):
                created = payload["created"]
            if isinstance(payload.get("model"), str):
                model_id = payload["model"]
            if isinstance(payload.get("usage"), dict):
                usage_obj = payload["usage"]
            for choice in payload.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                if isinstance(choice.get("finish_reason"), str):
                    finish = choice["finish_reason"]
                delta = choice.get("delta") or choice.get("message") or {}
                if not isinstance(delta, dict):
                    continue
                if isinstance(delta.get("role"), str):
                    msg["role"] = delta["role"]
                if isinstance(delta.get("content"), str):
                    msg["content"] = (msg.get("content") or "") + delta["content"]
                for tc in delta.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    idx = tc.get("index", 0)
                    slot = tcs_by_index.setdefault(int(idx), {
                        "id": None, "type": "function",
                        "function": {"name": "", "arguments": ""},
                    })
                    if isinstance(tc.get("id"), str):
                        slot["id"] = tc["id"]
                    if isinstance(tc.get("type"), str):
                        slot["type"] = tc["type"]
                    fn = tc.get("function") or {}
                    if isinstance(fn.get("name"), str) and fn["name"]:
                        slot["function"]["name"] = fn["name"]
                    if isinstance(fn.get("arguments"), str):
                        slot["function"]["arguments"] = (
                            slot["function"]["arguments"] + fn["arguments"]
                        )
    if tcs_by_index:
        msg["tool_calls"] = [tcs_by_index[i] for i in sorted(tcs_by_index)]
    return {
        "payload": {
            "id": rid or f"chatcmpl_assembled_{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": created or int(time.time()),
            "model": model_id,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish or "stop"}],
            **({"usage": usage_obj} if usage_obj else {}),
        },
        "usage": usage_obj,
    }


def _synthesize_chat_sse(payload: dict, model_hint: str | None) -> list[bytes]:
    """Convert a non-streaming chat/completions payload into a sequence
    of SSE chunks compatible with streaming clients.

    Single-delta synthesis: one chunk announces the role + tool_calls +
    full content, a second chunk carries `finish_reason`, an optional
    third carries usage, and a final `data: [DONE]\\n\\n` closes.
    AI-SDK / OpenAI-compatible clients accept this shape because the
    spec allows merging deltas across events.
    """
    rid = payload.get("id") or f"chatcmpl_synth_{int(time.time()*1000)}"
    created = payload.get("created") or int(time.time())
    model_id = payload.get("model") or model_hint or "unknown"
    choice = (payload.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    finish = choice.get("finish_reason") or "stop"
    chunks: list[bytes] = []

    # Chunk 1: role announcement (some clients require this separately).
    chunks.append(_sse({
        "id": rid, "object": "chat.completion.chunk",
        "created": created, "model": model_id,
        "choices": [{"index": 0, "delta": {"role": msg.get("role") or "assistant"}, "finish_reason": None}],
    }))
    # Chunk 2: content (if any) and full tool_calls.
    delta: dict[str, Any] = {}
    if isinstance(msg.get("content"), str) and msg["content"]:
        delta["content"] = msg["content"]
    if isinstance(msg.get("tool_calls"), list):
        delta["tool_calls"] = []
        for i, tc in enumerate(msg["tool_calls"]):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            delta["tool_calls"].append({
                "index": i,
                "id": tc.get("id"),
                "type": tc.get("type") or "function",
                "function": {
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments") or "",
                },
            })
    if delta:
        chunks.append(_sse({
            "id": rid, "object": "chat.completion.chunk",
            "created": created, "model": model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }))
    # Chunk 3: finish_reason.
    chunks.append(_sse({
        "id": rid, "object": "chat.completion.chunk",
        "created": created, "model": model_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
    }))
    # Chunk 4 (optional): usage.
    usage = payload.get("usage")
    if isinstance(usage, dict):
        chunks.append(_sse({
            "id": rid, "object": "chat.completion.chunk",
            "created": created, "model": model_id,
            "choices": [],
            "usage": usage,
        }))
    chunks.append(b"data: [DONE]\n\n")
    return chunks


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
                async with _gated(profile):
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
    except _QueueTimeout as exc:
        return 503, {"error": {"message": str(exc), "retry_after_s": 10}}
    except (RetryError, _UpstreamHTTPError):
        return last_status, {"error": {"message": last_body}}
    except httpx.HTTPError as e:
        return 502, {"error": {"message": f"upstream error: {e}"}}
    return 502, {"error": {"message": "unreachable"}}


# =============================================================================
# FastAPI endpoints
# =============================================================================


app = FastAPI(title="resilient-llm-bridge")

_watchdog_task: asyncio.Task | None = None


@app.on_event("startup")
async def _start_watchdog() -> None:
    global _watchdog_task
    if _watchdog_task is None or _watchdog_task.done():
        _watchdog_task = asyncio.create_task(_gate_watchdog_loop())


@app.on_event("shutdown")
async def _stop_watchdog() -> None:
    global _watchdog_task
    if _watchdog_task and not _watchdog_task.done():
        _watchdog_task.cancel()
        try:
            await _watchdog_task
        except (asyncio.CancelledError, Exception):
            pass


def _redact_body(body: Any, max_bytes: int = 65536) -> dict:
    # Debug escape hatch: when BRIDGE_NO_REDACT is set, store the full
    # unredacted body in the activity row. Useful for replaying client
    # requests against the upstream directly without going through the
    # bridge. WARNING: Hermes-class bodies can be 250-500KB each; with
    # the 1000-row activity buffer, that's 250-500MB resident. Toggle
    # off when not actively debugging.
    if os.environ.get("BRIDGE_NO_REDACT"):
        return copy.deepcopy(body) if isinstance(body, dict) else {"_": "<non-dict>"}
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


def _diff_thinking_params(before: dict, after: dict) -> dict:
    """Merge a pre-transform and post-transform `_inspect_thinking_params`
    snapshot into a single map, marking fields the bridge injected /
    changed so the dashboard can render them clearly.

    Output shape: each value is `{"value": <effective>, "from": "client"|"bridge"|"changed"}`.
    The dashboard renders bridge-injected fields with a "+bridge" tag and
    changed values with a "→" indicator.
    """
    out: dict = {}
    for k, v in after.items():
        if k not in before:
            out[k] = {"value": v, "from": "bridge"}
        elif before[k] != v:
            out[k] = {"value": v, "from": "changed", "was": before[k]}
        else:
            out[k] = {"value": v, "from": "client"}
    # Keep client-only entries (they were stripped by transforms — usually
    # the (DROPPED) markers, which we still want to surface).
    for k, v in before.items():
        if k not in after:
            out[k] = {"value": v, "from": "client", "stripped": True}
    return out


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
        # Hermes / Nous Portal style reasoning hint (only sent when the
        # client thinks the upstream is reasoning-capable; for our
        # bridge URL Hermes won't send this, but it shows up if anyone
        # ever points at us via a whitelisted hostname).
        eb_reasoning = extra.get("reasoning")
        if isinstance(eb_reasoning, dict):
            if (eff := eb_reasoning.get("effort")) is not None:
                out["extra_body.reasoning.effort"] = eff
            if (en := eb_reasoning.get("enabled")) is not None:
                out["extra_body.reasoning.enabled"] = en
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
    forwarded: dict | None = None,
    model: str | None = None,
    recovery: str | None = None,
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
    if model:
        record["model"] = model
    if recovery:
        record["recovery"] = recovery
    if params:
        record["params"] = params
    if body is not None:
        record["body"] = body
    if forwarded is not None:
        record["forwarded"] = forwarded
    _broadcast_activity(record)


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "uptime_s": round(time.time() - _started_at, 1),
        "profiles": list(CONFIG.profiles.keys()),
        "upstreams": [g.snapshot() for g in _UPSTREAM_GATES.values()],
    }


@app.post("/compactions/_inject")
async def compactions_inject(payload: dict) -> dict:
    """Debug-only: push a synthetic compaction record into the buffer.
    Used to test the panel-side poll without waiting for an actual
    overflow-driven summarization (which only fires past ~80k tokens).
    Body: `{pid, profile?, model?, input_chars?}`. The pid value
    becomes `source_pid` so the panel filter matches when this pid is
    the user's opencode binary."""
    pid = payload.get("pid")
    if not isinstance(pid, int):
        return {"error": "pid (int) required"}
    record = {
        "ts": time.time(),
        "profile": payload.get("profile") or "opencode",
        "model": payload.get("model") or "qwen3.6",
        "source_pid": pid,
        "source_host": "127.0.0.1",
        "source_port": None,
        "input_chars": int(payload.get("input_chars") or 0),
        "injected": True,
    }
    _compaction_history.append(record)
    return {"ok": True, "record": record}


@app.get("/compactions/recent")
async def compactions_recent(
    pid: int | None = None,
    since_ts: float | None = None,
    limit: int = 50,
) -> dict:
    """Return recently-detected opencode summarization (compaction)
    requests. The panel polls this with `?pid=<opencode_pid>&since_ts=
    <last_seen_ts>` so it sees only its own session's events.

    Without `pid`: returns ALL recent events (callers can filter
    themselves or use this for diagnostics).
    Events whose source pid lookup failed are returned with
    `source_pid: null` and only included when no `pid` filter is set —
    the panel ignores those.
    """
    rows = list(_compaction_history)
    if since_ts is not None:
        rows = [r for r in rows if r["ts"] > since_ts]
    if pid is not None:
        rows = [r for r in rows if r.get("source_pid") == pid]
    rows.sort(key=lambda r: r["ts"])
    if limit > 0:
        rows = rows[-limit:]
    return {"events": rows, "now": time.time()}


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
                "total_duration_ms": 0,
            }
        return d[key]

    # Activity rows now carry both `profile` and `model` (post-transform
    # upstream id), so each request increments BOTH a profile bucket and
    # a model bucket — fixes the previous "Per-model" panel showing 0
    # requests / 0ms p50 because only by_profile was incremented from
    # activity and by_model only got tokens from usage rows.
    for row in activity_rows:
        b_p = _bucket(by_profile, row.get("profile") or "?")
        b_m = _bucket(by_model, row.get("model") or "?")
        for b in (b_p, b_m):
            b["requests"] += 1
            if (row.get("status") or 0) >= 400:
                b["errors"] += 1
            if "duration_ms" in row:
                b["duration_ms"].append(row["duration_ms"])
                b["total_duration_ms"] += row["duration_ms"]
    for row in usage_rows:
        b_p = _bucket(by_profile, row.get("profile") or "?")
        b_m = _bucket(by_model, row.get("model") or "?")
        for b in (b_p, b_m):
            b["tokens_in"] += int(row.get("input_tokens") or 0)
            b["tokens_out"] += int(row.get("total_output_tokens") or row.get("output_tokens") or 0)

    # Per-window recovery counts derived from activity rows. The lifetime
    # counter (_recovery_counts) survives the bounded activity buffer;
    # this view tracks recoveries inside the requested window so the
    # dashboard can answer "did one fire in the last 5m?".
    recoveries: dict[str, int] = {k: 0 for k in _recovery_counts}
    for row in activity_rows:
        kind = row.get("recovery")
        if isinstance(kind, str) and kind in recoveries:
            recoveries[kind] += 1

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
        "tokens_out": sum(int(r.get("total_output_tokens") or r.get("output_tokens") or 0) for r in usage_rows),
        "total_duration_ms": sum(r.get("duration_ms", 0) for r in activity_rows),
        "p50_ms": round(_percentile(durations, 0.50), 1),
        "p95_ms": round(_percentile(durations, 0.95), 1),
        "p99_ms": round(_percentile(durations, 0.99), 1),
        "by_profile": _finalize(by_profile),
        "by_model": _finalize(by_model),
        "recoveries": recoveries,
    }


@app.get("/stats")
async def stats() -> dict:
    now = time.time()
    windows = {"1m": 60.0, "5m": 300.0, "15m": 900.0, "1h": 3600.0}
    activity = list(_activity_history)
    usage = list(_usage_history)
    # Lifetime is just an unbounded aggregation — same shape as windows
    # so the dashboard can treat it uniformly (proper p50/p95, error
    # split, by_profile, by_model).
    lifetime = _aggregate_window(activity, usage)
    lifetime["recoveries"] = dict(_recovery_counts)
    out: dict = {
        "now": now,
        "uptime_s": round(now - _started_at, 1),
        "lifetime": lifetime,
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


@app.get("/history")
async def history() -> dict:
    """Return last entries from disk logs for dashboard initial load."""
    activity: list[dict] = []
    usage: list[dict] = []
    for fname in ("activity.jsonl", "usage.jsonl"):
        path = LOG_DIR / fname
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except OSError:
            continue
        # Strip body/forwarded from activity entries — too large for
        # dashboard initial load. Decode failures are tolerated (log
        # files can have a partially-flushed last line).
        for line in lines[-200:]:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if fname == "activity.jsonl":
                obj.pop("body", None)
                obj.pop("forwarded", None)
                activity.append(obj)
            else:
                usage.append(obj)
    return {"activity": activity, "usage": usage}


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


def _dump_config_yaml(cfg: BridgeConfig) -> str:
    """Serialize a `BridgeConfig` back to YAML.

    Round-trips through PyYAML's default dumper so the file stays
    human-editable. The shape mirrors `_load_config()`'s reader.
    """
    out: dict[str, Any] = {"upstreams": {}, "profiles": {}}
    for name, u in cfg.upstreams.items():
        out["upstreams"][name] = {
            "url": u.url,
            "rate_limit_rpm": u.rate_limit_rpm,
            "rate_limit_concurrent": u.rate_limit_concurrent,
            "queue_timeout_s": u.queue_timeout_s,
            "stuck_warn_s": u.stuck_warn_s,
            "retry_max_attempts": u.retry_max_attempts,
            "retry_initial_wait": u.retry_initial_wait,
            "retry_max_wait": u.retry_max_wait,
        }
    for name, p in cfg.profiles.items():
        entry: dict[str, Any] = {
            "upstream": p.upstream,
            "queue_priority": p.queue_priority,
            "features": sorted(p.features),
        }
        # Only emit thinking_enabled when the profile has set it
        # explicitly (True or False). None means "profile silent" —
        # omit the key so the YAML stays minimal.
        if p.thinking_enabled is not None:
            entry["thinking_enabled"] = p.thinking_enabled
        # Only persist `default_thinking_budget` when the profile has
        # one set — None is the canonical "no opinion, let upstream
        # decide" default and we don't want to round-trip nulls.
        if p.default_thinking_budget is not None:
            entry["default_thinking_budget"] = p.default_thinking_budget
        if p.model_aliases:
            entry["model_aliases"] = dict(p.model_aliases)
        out["profiles"][name] = entry
    out["default_profile"] = cfg.default_profile
    return yaml.safe_dump(out, sort_keys=False, default_flow_style=False)


def _persist_config() -> None:
    """Write the current in-memory `CONFIG` to disk."""
    DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_CONFIG_PATH.write_text(_dump_config_yaml(CONFIG), encoding="utf-8")


def _validate_profile_payload(name: str, body: dict) -> tuple[ProfileConfig | None, str | None]:
    """Validate JSON profile input from the UI. Returns (profile, error).
    On error, profile is None and error contains a user-facing message."""
    if not isinstance(name, str) or not name.strip():
        return None, "profile name is required"
    if "/" in name or " " in name:
        return None, "profile name can't contain spaces or slashes"
    upstream = str(body.get("upstream") or "")
    if upstream not in CONFIG.upstreams:
        return None, f"unknown upstream {upstream!r}; available: {sorted(CONFIG.upstreams)}"
    feats_raw = body.get("features") or []
    if not isinstance(feats_raw, list):
        return None, "features must be a list of strings"
    feats = set(str(f) for f in feats_raw)
    invalid = feats - ALL_FEATURES
    if invalid:
        return None, f"unknown features: {sorted(invalid)}"
    aliases_raw = body.get("model_aliases") or {}
    if not isinstance(aliases_raw, dict):
        return None, "model_aliases must be a string→string mapping"
    aliases = {str(k): str(v) for k, v in aliases_raw.items()}
    try:
        priority = int(body.get("queue_priority", 0))
    except (TypeError, ValueError):
        return None, "queue_priority must be an integer"
    # default_thinking_budget is now Optional[int]: missing/null means
    # "no budget injected — let upstream decide". Numeric value clamps
    # the reasoning depth.
    raw_budget = body.get("default_thinking_budget", None)
    if raw_budget is None or raw_budget == "":
        budget: int | None = None
    else:
        try:
            budget = int(raw_budget)
        except (TypeError, ValueError):
            return None, "default_thinking_budget must be an integer or null"
        if budget < 0 or budget > 64000:
            return None, "default_thinking_budget must be between 0 and 64000"
        if budget == 0:
            budget = None  # treat 0 as "no budget"
    if "thinking_enabled" not in body:
        thinking_enabled: bool | None = None
    else:
        raw_enabled = body.get("thinking_enabled")
        thinking_enabled = None if raw_enabled is None else bool(raw_enabled)
    return (
        ProfileConfig(
            name=name,
            upstream=upstream,
            features=feats,
            model_aliases=aliases,
            default_thinking_budget=budget,
            thinking_enabled=thinking_enabled,
            queue_priority=priority,
        ),
        None,
    )


@app.get("/config")
async def config_get() -> dict:
    """Return the current config in a UI-friendly shape, plus the list
    of available features so the editor can render checkboxes."""
    return {
        "upstreams": [
            {
                "name": u.name,
                "url": u.url,
                "rate_limit_rpm": u.rate_limit_rpm,
                "rate_limit_concurrent": u.rate_limit_concurrent,
                "queue_timeout_s": u.queue_timeout_s,
                "stuck_warn_s": u.stuck_warn_s,
            }
            for u in CONFIG.upstreams.values()
        ],
        "profiles": [
            {
                "name": p.name,
                "upstream": p.upstream,
                "features": sorted(p.features),
                "queue_priority": p.queue_priority,
                "default_thinking_budget": p.default_thinking_budget,
                "thinking_enabled": p.thinking_enabled,
                "model_aliases": dict(p.model_aliases),
            }
            for p in CONFIG.profiles.values()
        ],
        "default_profile": CONFIG.default_profile,
        "available_features": sorted(ALL_FEATURES),
        "config_path": str(DEFAULT_CONFIG_PATH),
    }


@app.put("/config/profiles/{name}")
async def config_profile_put(name: str, request: Request) -> JSONResponse:
    """Upsert a profile. Body is the JSON shape returned by /config."""
    body = await request.json()
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
    profile, err = _validate_profile_payload(name, body)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    CONFIG.profiles[name] = profile  # type: ignore[index]
    try:
        _persist_config()
    except OSError as e:
        return JSONResponse(
            {"error": f"profile updated in memory but YAML write failed: {e}"},
            status_code=500,
        )
    return JSONResponse({"ok": True, "profile": name})


@app.delete("/config/profiles/{name}")
async def config_profile_delete(name: str) -> JSONResponse:
    if name not in CONFIG.profiles:
        return JSONResponse({"error": "profile not found"}, status_code=404)
    if name == CONFIG.default_profile:
        return JSONResponse(
            {"error": "can't delete the default profile; set a different default first"},
            status_code=400,
        )
    del CONFIG.profiles[name]
    try:
        _persist_config()
    except OSError as e:
        return JSONResponse(
            {"error": f"profile removed from memory but YAML write failed: {e}"},
            status_code=500,
        )
    return JSONResponse({"ok": True, "deleted": name})


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
    client_params = _inspect_thinking_params(body, "responses")
    redacted = _redact_body(body)
    body = _apply_request_transforms(body, profile, kind="responses")
    forwarded = _redact_body(body)  # post-transform — what we sent upstream
    inspected = _diff_thinking_params(client_params, _inspect_thinking_params(body, "responses"))
    want_stream = bool(body.get("stream", False))
    headers = _build_outgoing_headers(request)
    model_id = body.get("model") if isinstance(body.get("model"), str) else None
    if want_stream:
        stream_state: dict = {}
        async def _gen():
            async for chunk in _stream_responses(body, headers, profile, stream_state):
                yield chunk
            _record_activity(
                profile,
                f"/{profile.name}/v1/responses",
                "POST",
                200,
                (time.monotonic() - started) * 1000,
                params=inspected,
                body=redacted,
                forwarded=forwarded,
                model=model_id,
                recovery=stream_state.get("recovery"),
            )
        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    status, payload = await _post_responses_nonstream(body, headers, profile)
    recovery_kind: str | None = None
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
                        recovery_kind = "thinking_overflow"
                    elif message_emitted:
                        recovery_kind = "fake_invocation"
                    else:
                        recovery_kind = "silent_completion"
                    _record_recovery(recovery_kind)
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
        forwarded=forwarded,
        model=model_id,
        recovery=recovery_kind,
    )
    return JSONResponse(content=payload, status_code=status)


async def _handle_chat_completions(
    request: Request, profile: ProfileConfig
) -> StreamingResponse | JSONResponse:
    started = time.monotonic()
    body = await request.json()
    client_params = _inspect_thinking_params(body, "chat_completions")
    redacted = _redact_body(body)
    # Compaction detection runs BEFORE the body is mutated by transforms
    # so the SUMMARY_TEMPLATE substring we match is the literal opencode
    # sent us — transforms could in theory add system messages later.
    if _looks_like_summary_request(body):
        client = request.client
        _record_compaction(
            profile_name=profile.name,
            model=body.get("model") if isinstance(body.get("model"), str) else None,
            source_host=client.host if client else None,
            source_port=client.port if client else None,
            body=body,
        )
    body = _apply_request_transforms(body, profile, kind="chat_completions")
    forwarded = _redact_body(body)  # post-transform — what we sent upstream
    inspected = _diff_thinking_params(client_params, _inspect_thinking_params(body, "chat_completions"))
    want_stream = bool(body.get("stream", False))
    headers = _build_outgoing_headers(request)
    model_id = body.get("model") if isinstance(body.get("model"), str) else None
    if want_stream:
        stream_state: dict = {}
        async def _gen():
            async for chunk in _stream_chat_completions(body, headers, profile, stream_state):
                yield chunk
            _record_activity(
                profile,
                f"/{profile.name}/v1/chat/completions",
                "POST",
                200,
                (time.monotonic() - started) * 1000,
                params=inspected,
                body=redacted,
                forwarded=forwarded,
                model=model_id,
                recovery=stream_state.get("recovery"),
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
                async with _gated(profile):
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
    except _QueueTimeout as exc:
        last_status = 503
        payload = {"error": {"message": str(exc), "retry_after_s": 10}}
    except (RetryError, _UpstreamHTTPError):
        payload = {"error": {"message": last_body}}
    except httpx.HTTPError as e:
        last_status = 502
        payload = {"error": {"message": f"upstream error: {e}"}}
    recovery_kind: str | None = None
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
                recovery_kind = "truncated_content"
                _record_recovery(recovery_kind)
        # tool_call_args_retry: model emitted tool_calls with args
        # missing required fields (cargo-cult on poisoned history /
        # Qwen3 #1817). Retry with thinking disabled — the no-thinking
        # path doesn't have this bug per upstream reports + our own
        # probing. Skip if the client already disabled thinking.
        elif (
            profile.has("tool_call_args_retry")
            and message.get("tool_calls")
            and not _validate_tool_calls(message.get("tool_calls"), body.get("tools"))
            and not _client_already_disabled_thinking(body)
        ):
            r_status, retry_payload = await _retry_chat_thinking_off(body, headers, profile)
            if r_status < 400 and isinstance(retry_payload, dict):
                retry_choice = (retry_payload.get("choices") or [{}])[0]
                retry_msg = retry_choice.get("message") or {}
                retry_tcs = retry_msg.get("tool_calls")
                if retry_tcs and _validate_tool_calls(retry_tcs, body.get("tools")):
                    payload = retry_payload
                    recovery_kind = "tool_call_args_retry"
                    _record_recovery(recovery_kind)
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
                async with _gated(profile):
                    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                        retry_r = await client.post(upstream_url, json=body, headers=headers)
                if retry_r.status_code < 400:
                    retry_payload = retry_r.json()
                    retry_choice = (retry_payload.get("choices") or [{}])[0]
                    retry_message = retry_choice.get("message") or {}
                    retry_content = retry_message.get("content") or ""
                    if retry_content.strip():
                        payload = retry_payload
                        recovery_kind = "empty_with_stop_retry"
                        _record_recovery(recovery_kind)
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
        forwarded=forwarded,
        model=model_id,
        recovery=recovery_kind,
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
                async with _gated(profile):
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
    except _QueueTimeout as exc:
        last_status = 503
        response_content = json.dumps(
            {"error": {"message": str(exc), "retry_after_s": 10}}
        ).encode("utf-8")
        response_media_type = "application/json"
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


def _synthesize_model_metadata(profile: ProfileConfig, model_id: str) -> dict:
    """Produce an OpenAI-style /v1/models/{id} response enriched with
    the per-upstream context_window so clients (notably hermes) can
    skip the 200K default fallback and use the real number for their
    compaction thresholds.

    Resolves aliases through the profile so a client asking for
    `nan-thinking` gets back `id: "nan-thinking"` (matches what they
    configured) but the metadata reflects the real upstream model.
    """
    cfg = CONFIG.upstreams[profile.upstream]
    return {
        "id": model_id,
        "object": "model",
        "created": int(_started_at),
        "owned_by": cfg.name,
        "context_length": cfg.context_window,
        "max_context_length": cfg.context_window,
        "max_completion_tokens": 32000,
        "capabilities": {
            "reasoning": True,
            "tool_call": True,
            "completion": True,
        },
    }


def _enrich_models_list(payload: dict, profile: ProfileConfig) -> dict:
    """Add context_length / max_completion_tokens to upstream's
    /v1/models response so hermes' recursive metadata walker picks
    them up. Idempotent — preserves whatever the upstream already
    declared and only fills missing fields.
    """
    if not isinstance(payload, dict):
        return payload
    models = payload.get("data")
    if not isinstance(models, list):
        return payload
    cfg = CONFIG.upstreams[profile.upstream]
    enriched: list = []
    for m in models:
        if not isinstance(m, dict):
            enriched.append(m)
            continue
        m = dict(m)
        m.setdefault("context_length", cfg.context_window)
        m.setdefault("max_context_length", cfg.context_window)
        m.setdefault("max_completion_tokens", 32000)
        enriched.append(m)
    payload = dict(payload)
    payload["data"] = enriched
    return payload


@app.get("/{profile_name}/v1/models/{model_id:path}")
async def profile_model_details(profile_name: str, model_id: str, request: Request):
    """Synthesize per-model metadata. NaN-style backends serve the
    list at /v1/models but return 404 here; hermes falls back to a
    200K context default which makes its compaction thresholds
    misalign with the real 131K limit. We give it the truth instead."""
    profile = CONFIG.profile(profile_name)
    return JSONResponse(_synthesize_model_metadata(profile, model_id))


@app.get("/{profile_name}/v1/models")
async def profile_models_list(profile_name: str, request: Request):
    """Proxy the upstream's /v1/models and enrich each entry with
    context_length etc. so clients don't have to guess."""
    profile = CONFIG.profile(profile_name)
    started = time.monotonic()
    upstream_url = f"{_upstream_url(profile)}/models"
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length"}
    }
    cfg = CONFIG.upstreams[profile.upstream]
    try:
        async for attempt in _retry_policy(cfg):
            with attempt:
                async with _gated(profile):
                    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                        r = await client.get(upstream_url, headers=headers)
                if r.status_code in _RETRYABLE_STATUS:
                    raise _UpstreamHTTPError(r.status_code, r.text)
                payload: Any
                try:
                    payload = r.json()
                except (ValueError, json.JSONDecodeError):
                    payload = None
                _record_activity(
                    profile,
                    f"/{profile.name}/v1/models",
                    "GET",
                    r.status_code,
                    (time.monotonic() - started) * 1000,
                )
                if r.status_code >= 400 or not isinstance(payload, dict):
                    return Response(
                        content=r.content,
                        status_code=r.status_code,
                        media_type=r.headers.get("content-type"),
                    )
                return JSONResponse(_enrich_models_list(payload, profile))
    except (RetryError, _UpstreamHTTPError) as exc:
        status = getattr(exc, "status", 502)
        body_text = getattr(exc, "body", str(exc))
        return JSONResponse(_coerce_error_payload(status, body_text), status_code=status)
    except _QueueTimeout as exc:
        return JSONResponse(_coerce_error_payload(503, str(exc)), status_code=503)
    return JSONResponse({"error": {"message": "unreachable"}}, status_code=502)


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
      border-radius: 6px; padding: 0.75rem 0.9rem; display: flex;
      flex-direction: column; gap: 0.15rem; }}
    .kpi-label {{ color: var(--dim); font-size: 0.7rem; text-transform: uppercase;
      letter-spacing: 0.08em; }}
    .kpi-value {{ font-size: 1.55rem; color: var(--accent); font-weight: 500;
      line-height: 1.1; }}
    .kpi-sub {{ color: var(--dim); font-size: 0.78rem; }}
    .kpi.good .kpi-value {{ color: var(--good); }}
    .kpi.warn .kpi-value {{ color: var(--warn); }}
    .kpi.bad .kpi-value {{ color: var(--bad); }}
    .kpi.accent2 .kpi-value {{ color: var(--accent2); }}
    .kpi-spark {{ display: block; width: 100%; height: 26px;
      margin-top: 0.4rem; opacity: 0.85; }}

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
    .params .pk-bridge {{ color: var(--accent2); }}
    .params .pk-changed {{ color: var(--warn); }}
    .params .pair {{ display: inline-block; margin-right: 0.6rem;
      background: var(--panel2); padding: 0.05rem 0.35rem; border-radius: 3px; }}
    .params .pp-tag {{ font-size: 0.78em; margin-left: 0.3rem;
      padding: 0.02rem 0.3rem; border-radius: 2px; opacity: 0.85; }}
    .params .pp-bridge {{ background: rgba(210, 168, 255, 0.15); color: var(--accent2); }}
    .params .pp-changed {{ background: rgba(240, 136, 62, 0.15); color: var(--warn); }}
    .params .pp-stripped {{ background: rgba(255, 123, 114, 0.12); color: var(--bad); }}
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
    .body-pre .jk-added {{ color: var(--good); font-weight: 500; }}
    .body-pre .jk-changed {{ color: var(--warn); font-weight: 500; }}
    .body-pre .jline-added {{ background: rgba(86, 211, 100, 0.06); }}
    .body-pre .jline-changed {{ background: rgba(240, 136, 62, 0.06); }}
    .body-pre .jbadge {{ font-size: 0.85em; margin-left: 0.4rem;
      padding: 0.02rem 0.3rem; border-radius: 2px; opacity: 0.85; }}
    .body-pre .jb-add {{ background: rgba(86, 211, 100, 0.15); color: var(--good); }}
    .body-pre .jb-change {{ background: rgba(240, 136, 62, 0.15); color: var(--warn); }}
    .body-label {{ color: var(--dim); font-size: 0.78rem;
      text-transform: uppercase; letter-spacing: 0.08em;
      margin: 0.4rem 0 0.2rem 0; }}
    .body-label .dim {{ text-transform: none; letter-spacing: 0; opacity: 0.6; }}

    /* Profile editor — compact */
    .editor-hint {{ color: var(--dim); font-size: 0.72em;
      margin: 0 0 0.35rem 0; line-height: 1.35; }}
    .editor-actions {{ margin-top: 0.35rem; display: flex; align-items: center; gap: 0.5rem; }}
    .editor-btn {{ background: var(--panel2); border: 1px solid var(--line);
      color: var(--fg); padding: 0.12rem 0.4rem; border-radius: 2px;
      cursor: pointer; font: inherit; font-size: 0.7rem; }}
    .editor-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
    .editor-btn.danger {{ color: var(--bad); }}
    .editor-btn.danger:hover {{ border-color: var(--bad); }}
    .editor-btn.primary {{ background: rgba(121, 192, 255, 0.1); color: var(--accent); border-color: var(--accent); }}
    .editor-btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
    .editor-status {{ font-size: 0.7rem; color: var(--dim); }}
    .editor-status.ok {{ color: var(--good); }}
    .editor-status.err {{ color: var(--bad); }}

    .profile-row {{ background: var(--panel2); border-radius: 3px;
      padding: 0.25rem 0.4rem; margin-bottom: 0.2rem;
      border: 1px solid transparent; }}
    .profile-row.dirty {{ border-color: var(--warn); }}
    .pf-head {{ display: flex; align-items: center;
      gap: 0.35rem; flex-wrap: wrap; }}
    .pf-head .pf-name {{ font-weight: 500; color: var(--accent);
      font-size: 0.78rem; min-width: 4rem; white-space: nowrap; }}
    .pf-head .pf-name input {{ background: transparent; border: none;
      color: var(--accent); font: inherit; font-size: 0.78rem;
      font-weight: 500; padding: 0.02rem 0.15rem; border-radius: 1px;
      border-bottom: 1px dashed var(--accent); width: 8rem; }}
    .pf-head .pf-default {{ color: var(--dim); font-size: 0.6rem;
      text-transform: uppercase; letter-spacing: 0.05em;
      padding: 0.02rem 0.2rem; border: 1px solid var(--line); border-radius: 1px; }}
    .pf-inline {{ display: inline-flex; align-items: center; gap: 0.15rem; }}
    .pf-inline label {{ font-size: 0.62rem; color: var(--dim); text-transform: uppercase; letter-spacing: 0.04em; }}
    .pf-inline select, .pf-inline input {{ background: var(--panel);
      border: 1px solid var(--line); color: var(--fg);
      padding: 0.06rem 0.2rem; font: inherit; font-size: 0.7rem;
      border-radius: 1px; }}
    .pf-inline input[type="number"] {{ width: 4rem; }}
    .pf-inline input:focus, .pf-inline select:focus {{ border-color: var(--accent); outline: none; }}
    .pf-actions-inline {{ margin-left: auto; display: inline-flex; gap: 0.25rem; }}

    .pf-secondary {{ display: flex; align-items: flex-start;
      gap: 0.4rem; margin-top: 0.2rem; }}
    .pf-secondary .pf-features {{ flex: 1; display: flex; flex-wrap: wrap;
      gap: 0.12rem 0.2rem; }}
    .pf-secondary .pf-aliases {{ flex: 0 0 auto; min-width: 10rem; }}
    .pf-feature {{ display: inline-flex; align-items: center; gap: 0.12rem;
      cursor: pointer; user-select: none; font-size: 0.62rem;
      padding: 0.03rem 0.2rem; border: 1px solid var(--line); border-radius: 1px;
      background: var(--panel); color: var(--dim); }}
    .pf-feature.on {{ color: var(--accent); border-color: var(--accent); }}
    .pf-feature input {{ display: none; }}

    .pf-aliases-label {{ font-size: 0.6rem; text-transform: uppercase;
      letter-spacing: 0.04em; color: var(--dim); margin-bottom: 0.08rem; }}
    .pf-alias-row {{ display: flex; gap: 0.12rem; margin-bottom: 0.08rem; align-items: center; }}
    .pf-alias-row input {{ flex: 1; background: var(--panel);
      border: 1px solid var(--line); color: var(--fg); padding: 0.05rem 0.2rem;
      font: inherit; font-size: 0.66rem; border-radius: 1px; min-width: 0; }}
    .pf-alias-arrow {{ color: var(--dim); font-size: 0.7em; }}
    .pf-alias-del {{ background: transparent; border: 1px solid var(--line);
      color: var(--bad); padding: 0 0.2rem; border-radius: 1px;
      font: inherit; font-size: 0.66rem; cursor: pointer; }}

    /* Sparkline */
    .spark {{ display: block; width: 100%; height: 40px; }}

    /* Recovery card */
    .recoveries {{ display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.4rem; }}
    .rec {{ background: var(--panel2); border-radius: 4px; padding: 0.5rem 0.6rem; }}
    .rec.zero {{ opacity: 0.55; }}
    .rec-name {{ color: var(--dim); font-size: 0.7rem; text-transform: uppercase;
      letter-spacing: 0.05em; }}
    .rec-count {{ color: var(--accent2); font-size: 1.1rem; margin-top: 0.1rem;
      display: flex; align-items: baseline; gap: 0.4rem; }}
    .rec-window {{ color: var(--accent2); font-weight: 500; }}
    .rec-lifetime {{ color: var(--dim); font-size: 0.78rem; font-weight: 400; }}

    /* Activity filter bar */
    .filter-bar {{ display: flex; flex-wrap: wrap; align-items: center;
      gap: 0.5rem; margin: 0 0 0.6rem 0; padding: 0.4rem 0.6rem;
      background: var(--panel2); border-radius: 4px;
      border: 1px solid var(--line); font-size: 0.78rem; }}
    .filter-bar label {{ color: var(--dim); font-size: 0.7rem;
      text-transform: uppercase; letter-spacing: 0.05em; }}
    .filter-bar select, .filter-bar input[type="text"] {{
      background: var(--panel); border: 1px solid var(--line);
      color: var(--fg); padding: 0.15rem 0.4rem; font: inherit;
      font-size: 0.78rem; border-radius: 3px; }}
    .filter-bar input[type="text"] {{ min-width: 8rem; }}
    .filter-bar select:focus, .filter-bar input:focus {{
      border-color: var(--accent); outline: none; }}
    .filter-toggle {{ background: transparent; border: 1px solid var(--line);
      color: var(--dim); padding: 0.15rem 0.55rem; border-radius: 3px;
      cursor: pointer; font: inherit; font-size: 0.75rem; }}
    .filter-toggle:hover {{ color: var(--fg); }}
    .filter-toggle.active {{ color: var(--accent); border-color: var(--accent); }}
    .filter-toggle.active.errors {{ color: var(--bad); border-color: var(--bad); }}
    .filter-toggle.active.recoveries {{ color: var(--accent2); border-color: var(--accent2); }}
    .filter-count {{ color: var(--dim); margin-left: auto; font-size: 0.74rem; }}
    .filter-clear {{ color: var(--dim); cursor: pointer; padding: 0.05rem 0.35rem;
      border-radius: 2px; font-size: 0.74rem; }}
    .filter-clear:hover {{ color: var(--bad); }}

    /* Recovery badge in the activity row */
    .rec-badge {{ display: inline-block; margin-left: 0.4rem;
      padding: 0.02rem 0.35rem; border-radius: 2px;
      background: rgba(210, 168, 255, 0.15); color: var(--accent2);
      font-size: 0.72rem; letter-spacing: 0.02em; }}

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

  <h2>Profiles</h2>
  <div class="card">
    <p class="editor-hint">
      Edit, add, or remove profiles. Saves go to
      <code id="editor-config-path">~/.config/resilient-llm-bridge/config.yaml</code>
      and apply immediately (no restart).
    </p>
    <div id="profile-editor"></div>
    <div class="editor-actions">
      <button class="editor-btn" id="profile-add">+ new profile</button>
      <span class="editor-status" id="editor-status"></span>
    </div>
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
        <th>rpm cap</th><th>concurrent</th><th>in flight</th>
        <th>waiting</th><th>oldest</th><th>utilization</th>
      </tr></thead>
      <tbody id="upstreams-body"><tr><td colspan="8" class="empty">loading…</td></tr></tbody>
    </table>
  </div>

  <h2>Recent activity</h2>
  <div class="card">
    <div class="filter-bar">
      <button type="button" class="filter-toggle errors" id="flt-errors">errors only</button>
      <button type="button" class="filter-toggle recoveries" id="flt-recoveries">recoveries only</button>
      <span class="pf-inline">
        <label>profile</label>
        <select id="flt-profile"><option value="">(all)</option></select>
      </span>
      <span class="pf-inline">
        <label>model</label>
        <select id="flt-model"><option value="">(all)</option></select>
      </span>
      <span class="pf-inline">
        <label>path</label>
        <input type="text" id="flt-path" placeholder="substring match">
      </span>
      <span class="filter-clear" id="flt-clear" title="clear all filters">clear</span>
      <span class="filter-count" id="flt-count">—</span>
    </div>
    <table>
      <thead><tr>
        <th>time</th><th>profile</th><th>method</th>
        <th>path</th><th>status</th><th class="num">ms</th>
        <th class="num">↑</th><th class="num">↓</th>
        <th>thinking params</th>
      </tr></thead>
      <tbody id="activity"><tr><td colspan="9" class="empty">no activity yet</td></tr></tbody>
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
    const fmtRate = (n) => {{
      if (!n || n === 0) return '0';
      const s = n.toFixed(4);
      return s.replace(/0+$/, '').replace(/\.$/, '');
    }};

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
        const entry = p[k];
        // Two value shapes are accepted: legacy primitive (still in
        // older activity records) and the new tagged object with
        // value/from/was/stripped fields.
        let val, source = null, was = null, stripped = false;
        if (entry && typeof entry === 'object' && 'value' in entry) {{
          val = entry.value;
          source = entry.from || null;
          was = entry.was;
          stripped = !!entry.stripped;
        }} else {{
          val = entry;
        }}
        const dropped = k.includes('DROPPED');
        let kCls = 'pk';
        let badge = '';
        if (dropped) {{
          kCls = 'pk dropped';
        }} else if (source === 'bridge') {{
          kCls = 'pk pk-bridge';
          badge = '<span class="pp-tag pp-bridge">+bridge</span>';
        }} else if (source === 'changed') {{
          kCls = 'pk pk-changed';
          badge = `<span class="pp-tag pp-changed">was ${{escapeHtml(was)}}</span>`;
        }} else if (stripped) {{
          kCls = 'pk dropped';
          badge = '<span class="pp-tag pp-stripped">stripped</span>';
        }}
        return `<span class="pair"><span class="${{kCls}}">${{escapeHtml(k)}}</span>=<span class="pv">${{escapeHtml(val)}}</span>${{badge}}</span>`;
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
    // Survives re-renders: keyed by activity row ts (stable per request).
    const expandedKeys = new Set();
    // Activity filter state. Mutated by the filter bar handlers; read
    // by `applyFilters()` on every render. `pathQuery` is a substring
    // match (case-insensitive); empty string disables.
    const filters = {{
      errorsOnly: false,
      recoveriesOnly: false,
      profile: '',
      model: '',
      pathQuery: '',
    }};

    document.querySelectorAll('.windows button').forEach(btn => {{
      btn.addEventListener('click', () => {{
        document.querySelectorAll('.windows button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeWindow = btn.dataset.win;
        if (lastStats) renderStats(lastStats);
      }});
    }});

    // Activity filter bar wiring. Each control updates `filters` and
    // re-renders. The dropdowns are kept in sync with the values seen
    // in the activity buffer (so a profile/model that hasn't appeared
    // yet doesn't show up). Path is a case-insensitive substring match.
    function matchesFilters(r) {{
      if (filters.errorsOnly && (Number(r.status) || 0) < 400) return false;
      if (filters.recoveriesOnly && !r.recovery) return false;
      if (filters.profile && r.profile !== filters.profile) return false;
      if (filters.model && r.model !== filters.model) return false;
      if (filters.pathQuery) {{
        const p = (r.path || '').toLowerCase();
        if (!p.includes(filters.pathQuery)) return false;
      }}
      return true;
    }}
    function syncDropdown(id, values, currentSelection) {{
      const sel = $(id);
      if (!sel) return;
      // Keep "(all)" as the first option, then sorted unique values.
      // Don't blow away the user's selection on every render.
      const desired = ['', ...[...values].sort()];
      const have = Array.from(sel.options).map(o => o.value);
      const same = desired.length === have.length && desired.every((v, i) => v === have[i]);
      if (same) return;
      const prev = currentSelection || sel.value || '';
      sel.innerHTML = desired.map(v =>
        `<option value="${{escapeHtml(v)}}"${{v === prev ? ' selected' : ''}}>` +
          (v === '' ? '(all)' : escapeHtml(v)) +
        `</option>`
      ).join('');
    }}
    function rerenderActivity() {{
      if (lastStats) renderStats(lastStats);
    }}
    $('flt-errors').addEventListener('click', () => {{
      filters.errorsOnly = !filters.errorsOnly;
      $('flt-errors').classList.toggle('active', filters.errorsOnly);
      rerenderActivity();
    }});
    $('flt-recoveries').addEventListener('click', () => {{
      filters.recoveriesOnly = !filters.recoveriesOnly;
      $('flt-recoveries').classList.toggle('active', filters.recoveriesOnly);
      rerenderActivity();
    }});
    $('flt-profile').addEventListener('change', (e) => {{
      filters.profile = e.target.value;
      rerenderActivity();
    }});
    $('flt-model').addEventListener('change', (e) => {{
      filters.model = e.target.value;
      rerenderActivity();
    }});
    $('flt-path').addEventListener('input', (e) => {{
      filters.pathQuery = (e.target.value || '').toLowerCase();
      rerenderActivity();
    }});
    $('flt-clear').addEventListener('click', () => {{
      filters.errorsOnly = false;
      filters.recoveriesOnly = false;
      filters.profile = '';
      filters.model = '';
      filters.pathQuery = '';
      $('flt-errors').classList.remove('active');
      $('flt-recoveries').classList.remove('active');
      $('flt-profile').value = '';
      $('flt-model').value = '';
      $('flt-path').value = '';
      rerenderActivity();
    }});

    function pickWindow(stats) {{
      if (activeWindow === 'lifetime') return stats.lifetime || {{}};
      return stats.windows[activeWindow] || {{}};
    }}

    function windowSeconds(stats) {{
      if (activeWindow === 'lifetime') {{
        return Math.max(stats?.uptime_s || 1, 1);
      }}
      return {{ '1m': 60, '5m': 300, '15m': 900, '1h': 3600 }}[activeWindow] || 300;
    }}

    function renderStats(stats) {{
      const w = pickWindow(stats);
      const span = windowSeconds(stats);
      const empty = !w.requests;
      $('kpi-req').textContent = empty ? '—' : fmt(w.requests);
      $('kpi-rps').textContent = empty
        ? 'no requests in window'
        : `${{fmtRate(w.requests / span)}} rps · ${{w.errors_4xx || 0}}×4xx · ${{w.errors_5xx || 0}}×5xx`;
      $('kpi-tok-out').textContent = w.tokens_out ? fmt(w.tokens_out) : '—';
      const procSec = (w.total_duration_ms || 0) / 1000;
      $('kpi-tok-rate').textContent = (w.tokens_out && procSec > 0)
        ? `${{fmtRate(w.tokens_out / procSec)}} tok/s avg`
        : '—';
      $('kpi-tok-in').textContent = w.tokens_in ? fmt(w.tokens_in) : '—';
      const ratio = w.tokens_in > 0 ? fmtRate(w.tokens_out / w.tokens_in) : '—';
      $('kpi-tok-ratio').textContent = `out/in ${{ratio}}`;
      $('kpi-p50').textContent = w.p50_ms ? fmtMs(w.p50_ms) : '—';
      $('kpi-p95').textContent = w.p95_ms
        ? `p95 ${{fmtMs(w.p95_ms)}} · p99 ${{fmtMs(w.p99_ms || 0)}}`
        : '—';
      const successRate = w.requests > 0
        ? (((w.requests - w.errors) / w.requests) * 100).toFixed(1) + '%'
        : '—';
      $('kpi-success').textContent = successRate;
      $('kpi-errors').textContent = w.requests > 0
        ? `${{w.errors || 0}} errors / ${{w.requests}} req`
        : '—';
      const recoveries = (stats.lifetime && stats.lifetime.recoveries) || {{}};
      const totalRec = Object.values(recoveries).reduce((a, b) => a + b, 0);
      $('kpi-rec').textContent = totalRec ? fmt(totalRec) : '—';
      const recParts = Object.entries(recoveries)
        .filter(([_, v]) => v > 0)
        .map(([k, v]) => `${{k.replace(/_/g, ' ')}}: ${{v}}`);
      $('kpi-rec-sub').textContent = recParts.length ? recParts.join(' · ') : 'none fired yet';

      // Sparklines from history (same span as the KPI window).
      const sec = span;
      const reqBins = bin(activityRows, sec, 24, _ => 1);
       const tokBins = bin(usageRows, sec, 24, e => (e.total_output_tokens || e.output_tokens || 0));
      const latBins = bin(activityRows, sec, 24, e => (e.duration_ms || 0));
      // Convert sums to averages for latency.
      const latCounts = bin(activityRows, sec, 24, _ => 1);
      const latAvg = latBins.map((s, i) => latCounts[i] ? s / latCounts[i] : 0);
      spark('spark-req', reqBins, '#79c0ff');
      spark('spark-tok', tokBins, '#d2a8ff');
      spark('spark-lat', latAvg, '#f0883e');

      // Recoveries grid: per-window (active) AND lifetime, side by side.
      // Lifetime comes from the global counter (survives buffer truncation);
      // window comes from aggregating activity rows in the slice.
      const lifetimeRec = (stats.lifetime && stats.lifetime.recoveries) || {{}};
      const windowRec = (w && w.recoveries) || {{}};
      const recKinds = Object.keys(lifetimeRec).length
        ? Object.keys(lifetimeRec)
        : Object.keys(windowRec);
      const winLabel = activeWindow === 'lifetime' ? 'all' : activeWindow;
      $('rec-grid').innerHTML = recKinds.map((k) => {{
        const wn = Number(windowRec[k] || 0);
        const lf = Number(lifetimeRec[k] || 0);
        const cls = (wn === 0 && lf === 0) ? 'rec zero' : 'rec';
        return `<div class="${{cls}}">` +
          `<div class="rec-name">${{escapeHtml(k.replace(/_/g, ' '))}}</div>` +
          `<div class="rec-count">` +
            `<span class="rec-window">${{fmt(wn)}}</span>` +
            `<span class="rec-lifetime">${{winLabel}} · ${{fmt(lf)}} all</span>` +
          `</div></div>`;
      }}).join('');

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
        const waiting = u.queue_waiting || 0;
        const cap = u.concurrent_limit || 1;
        const pct = Math.min(100, (inFlight / cap) * 100);
        const cls = pct >= 90 ? 'full' : pct >= 50 ? 'busy' : '';
        const oldest = u.oldest_in_flight_s || 0;
        const stuckWarn = u.stuck_warn_s || 300;
        const oldestCls = oldest > stuckWarn ? 'lat-slow' : oldest > stuckWarn * 0.5 ? 'lat-mid' : '';
        const oldestStr = oldest > 0 ? `${{oldest.toFixed(1)}}s` : '—';
        const waitCls = waiting > 0 ? 'lat-mid' : '';
        return `<tr><td>${{escapeHtml(u.name)}}</td>` +
          `<td><code>${{escapeHtml(u.url)}}</code></td>` +
          `<td class="num">${{u.rpm_remaining}}/${{u.rpm_capacity}}</td>` +
          `<td class="num">${{cap}}</td>` +
          `<td class="num">${{inFlight}}</td>` +
          `<td class="num ${{waitCls}}">${{waiting}}</td>` +
          `<td class="num ${{oldestCls}}">${{oldestStr}}</td>` +
          `<td><div class="rate-bar"><div class="rate-bar-fill ${{cls}}" style="width:${{pct.toFixed(1)}}%"></div></div></td></tr>`;
      }}).join('');
      $('upstreams-body').innerHTML = upRows ||
        `<tr><td colspan="8" class="empty">no upstreams</td></tr>`;

      // Activity table from rolling buffer. Each main row has a hidden
      // sibling row with the full redacted JSON body — expanded on
      // click. Expanded state is keyed by the row's ts (stable across
      // renders) so periodic re-renders don't collapse open rows.
      // Build a lookup: profile -> usage records sorted by ts.
      const usageByProfile = {{}};
      for (const u of usageRows) {{
        const p = u.profile || '?';
        if (!usageByProfile[p]) usageByProfile[p] = [];
        usageByProfile[p].push(u);
      }}
      // Sort each list by ts ascending.
      for (const p in usageByProfile) usageByProfile[p].sort((a, b) => a.ts - b.ts);

      function findUsageFor(profile, ts) {{
        const list = usageByProfile[profile];
        if (!list || list.length === 0) return null;
        // Walk backwards from the end — pick the last usage record
        // whose ts <= activity ts + 1s (usage arrives slightly after).
        let best = null;
        for (let i = list.length - 1; i >= 0; i--) {{
          if (list[i].ts <= ts + 1) {{
            best = list[i];
            break;
          }}
        }}
        return best;
      }}

      // Refresh the profile/model dropdowns from whatever's been seen.
      // Keep the user's current selection; just append any new options.
      const seenProfiles = new Set(activityRows.map((r) => r.profile).filter(Boolean));
      const seenModels = new Set(activityRows.map((r) => r.model).filter(Boolean));
      syncDropdown('flt-profile', seenProfiles, filters.profile);
      syncDropdown('flt-model', seenModels, filters.model);

      // Apply filters to the full activity buffer, then take the last 30
      // matching rows. Order matters: filtering first means we see 30
      // matches even if the latest 30 unfiltered rows have none of them.
      const filtered = activityRows.filter(matchesFilters);
      const recent = filtered.slice(-30).reverse();
      $('flt-count').textContent = filtered.length === activityRows.length
        ? `${{activityRows.length}} requests`
        : `${{filtered.length}} of ${{activityRows.length}} match`;
      $('activity').innerHTML = recent.length === 0
        ? `<tr><td colspan="9" class="empty">${{
            activityRows.length === 0 ? 'no activity yet' : 'no rows match the current filters'
          }}</td></tr>`
        : recent.map((r) => {{
            const key = String(r.ts);
            const targetId = `body-${{key.replace('.', '_')}}`;
            const isExpanded = expandedKeys.has(key);
            const mainCls = isExpanded ? 'activity-row expanded' : 'activity-row';
            const bodyCls = isExpanded ? 'body-row expanded' : 'body-row';
            const usage = findUsageFor(r.profile, r.ts);
            const tokIn = usage ? fmt(usage.input_tokens) : '—';
            const tokOut = usage ? fmt(usage.total_output_tokens || usage.output_tokens) : '—';
            const recBadge = r.recovery
              ? `<span class="rec-badge" title="recovery fired">${{escapeHtml(r.recovery.replace(/_/g, ' '))}}</span>`
              : '';
            const main = `<tr class="${{mainCls}}" data-key="${{key}}" data-target="${{targetId}}">` +
              `<td>${{fmtTime(r.ts)}}</td><td>${{escapeHtml(r.profile)}}</td>` +
              `<td>${{escapeHtml(r.method)}}</td><td><code>${{escapeHtml(r.path)}}</code>${{recBadge}}</td>` +
              `<td class="${{statusClass(r.status)}}">${{r.status}}</td>` +
              `<td class="num ${{latClass(r.duration_ms)}}">${{fmtMs(r.duration_ms)}}</td>` +
              `<td class="num">${{tokIn}}</td><td class="num">${{tokOut}}</td>` +
              `<td>${{renderParams(r.params)}}</td></tr>`;
            // Forwarded body is enough — it shows everything the
            // client sent plus inline diff markers (green +bridge for
            // injected fields, orange "was X" for changed values).
            const bodyJson = r.forwarded
              ? annotatedJson(r.forwarded, r.body)
              : (r.body
                  ? jsonHighlight(r.body)
                  : '<span class="jnull">no body captured</span>');
            const expand = `<tr class="${{bodyCls}}" id="${{targetId}}">` +
              `<td colspan="9"><div class="body-pre">${{bodyJson}}</div></td></tr>`;
            return main + expand;
          }}).join('');
      // Wire click handlers for expandable rows. Toggle both the
      // expandedKeys set (the source of truth across re-renders) and
      // the live DOM classes (for instant feedback before next render).
      document.querySelectorAll('#activity .activity-row').forEach(row => {{
        row.addEventListener('click', () => {{
          const key = row.dataset.key;
          const target = $(row.dataset.target);
          if (!target) return;
          if (expandedKeys.has(key)) {{
            expandedKeys.delete(key);
            row.classList.remove('expanded');
            target.classList.remove('expanded');
          }} else {{
            expandedKeys.add(key);
            row.classList.add('expanded');
            target.classList.add('expanded');
          }}
        }});
      }});
    }}

    // Diff-aware JSON renderer for the forwarded body. Walks the
    // forwarded object alongside the client body and tags each leaf
    // with `+added` (bridge introduced the path) or `≠ was X`
    // (bridge changed an existing value). Falls back to the plain
    // highlighter when no comparison body is supplied.
    function flattenJson(obj, prefix, out) {{
      if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) {{
        out.set(prefix, obj);
        return;
      }}
      if (Object.keys(obj).length === 0) {{
        out.set(prefix, obj);
        return;
      }}
      for (const k of Object.keys(obj)) {{
        const path = prefix ? `${{prefix}}.${{k}}` : k;
        flattenJson(obj[k], path, out);
      }}
    }}
    function diffJson(client, forwarded) {{
      const fw = new Map(); flattenJson(forwarded, '', fw);
      const cl = new Map(); flattenJson(client, '', cl);
      const added = new Set();
      const changed = new Map();
      for (const [k, v] of fw) {{
        if (!cl.has(k)) {{
          added.add(k);
        }} else if (JSON.stringify(cl.get(k)) !== JSON.stringify(v)) {{
          changed.set(k, cl.get(k));
        }}
      }}
      return {{ added, changed }};
    }}
    function annotatedJson(forwarded, client) {{
      if (!forwarded) return '<span class="jnull">no body</span>';
      if (!client) return jsonHighlight(forwarded);
      const {{ added, changed }} = diffJson(client, forwarded);

      function fmtVal(v) {{
        if (v === null) return '<span class="jnull">null</span>';
        if (typeof v === 'string') return `<span class="js">"${{escapeHtml(v)}}"</span>`;
        if (typeof v === 'number') return `<span class="jn">${{v}}</span>`;
        if (typeof v === 'boolean') return `<span class="jbool">${{v}}</span>`;
        return escapeHtml(String(v));
      }}
      function pathOfChild(prefix, key) {{
        return prefix ? `${{prefix}}.${{key}}` : key;
      }}
      function isUnderAdded(path) {{
        // A child path is implicitly "added" when its parent was added.
        let p = path;
        while (p) {{
          if (added.has(p)) return true;
          const i = p.lastIndexOf('.');
          if (i < 0) return false;
          p = p.slice(0, i);
        }}
        return false;
      }}
      function render(obj, indent, prefix, suppressBadge) {{
        if (obj === null || typeof obj !== 'object') return fmtVal(obj);
        if (Array.isArray(obj)) {{
          if (obj.length === 0) return '[]';
          const inner = obj.map((v) =>
            `${{indent}}  ${{render(v, indent + '  ', prefix, true)}}`
          ).join(',\\n');
          return `[\\n${{inner}}\\n${{indent}}]`;
        }}
        const keys = Object.keys(obj);
        if (keys.length === 0) return '{{}}';
        const lines = keys.map((k) => {{
          const path = pathOfChild(prefix, k);
          const isAdded = added.has(path) || isUnderAdded(path);
          const isChanged = changed.has(path);
          let cls = 'jk';
          let lineCls = '';
          let badge = '';
          if (isAdded) {{
            cls = 'jk jk-added';
            lineCls = 'jline-added';
            if (!suppressBadge) badge = '<span class="jbadge jb-add">+bridge</span>';
          }} else if (isChanged) {{
            cls = 'jk jk-changed';
            lineCls = 'jline-changed';
            badge = `<span class="jbadge jb-change">was ${{escapeHtml(JSON.stringify(changed.get(path)))}}</span>`;
          }}
          const valHtml = render(obj[k], indent + '  ', path, suppressBadge || isAdded);
          const linePrefix = lineCls ? `<span class="${{lineCls}}">` : '';
          const lineSuffix = lineCls ? `</span>` : '';
          return `${{indent}}  ${{linePrefix}}<span class="${{cls}}">"${{escapeHtml(k)}}"</span>: ${{valHtml}}${{badge}}${{lineSuffix}}`;
        }});
        return `{{\\n${{lines.join(',\\n')}}\\n${{indent}}}}`;
      }}
      return render(forwarded, '', '', false);
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
    setInterval(refreshStats, 1000);

    // Load disk history on initial page load.
    fetch('/history')
      .then(r => r.json())
      .then(d => {{
        if (Array.isArray(d.activity)) {{
          activityRows = d.activity;
          if (activityRows.length > 1000) activityRows = activityRows.slice(-1000);
        }}
        if (Array.isArray(d.usage)) {{
          usageRows = d.usage;
          if (usageRows.length > 1000) usageRows = usageRows.slice(-1000);
        }}
      }})
      .catch(() => {{}});

    /* ========================================================================
       Profile editor
       ======================================================================== */
    let editorState = null;  // {{profiles: [...], originals: Map<name,obj>, upstreams, available_features, default_profile}}
    const editorStatus = $('editor-status');

    function setStatus(text, kind) {{
      editorStatus.textContent = text || '';
      editorStatus.className = 'editor-status' + (kind ? ' ' + kind : '');
    }}
    function fadeStatus(text, kind, ms) {{
      setStatus(text, kind);
      window.setTimeout(() => {{
        if (editorStatus.textContent === text) setStatus('');
      }}, ms || 3000);
    }}

    function emptyAliases() {{ return []; }}
    function aliasesToList(map) {{
      return Object.entries(map || {{}}).map(([from, to]) => ({{ from, to }}));
    }}
    function aliasesToMap(list) {{
      const out = {{}};
      for (const item of (list || [])) {{
        const f = (item.from || '').trim();
        const t = (item.to || '').trim();
        if (f && t) out[f] = t;
      }}
      return out;
    }}

    function profileToEditable(p) {{
      return {{
        name: p.name,
        upstream: p.upstream,
        queue_priority: p.queue_priority,
        default_thinking_budget: p.default_thinking_budget,
        features: new Set(p.features || []),
        aliases: aliasesToList(p.model_aliases),
        isNew: false,
      }};
    }}

    async function loadEditor() {{
      try {{
        const r = await fetch('/config');
        if (!r.ok) throw new Error('config returned ' + r.status);
        const d = await r.json();
        $('editor-config-path').textContent = d.config_path || '~/.config/resilient-llm-bridge/config.yaml';
        editorState = {{
          profiles: (d.profiles || []).map(profileToEditable),
          originals: new Map((d.profiles || []).map((p) => [p.name, JSON.stringify(p)])),
          upstreams: (d.upstreams || []).map((u) => u.name),
          available_features: d.available_features || [],
          default_profile: d.default_profile,
        }};
        renderEditor();
      }} catch (e) {{
        setStatus('failed to load config: ' + (e?.message || e), 'err');
      }}
    }}

    function isDirty(prof) {{
      if (prof.isNew) return true;
      const original = editorState.originals.get(prof.name);
      if (!original) return true;
      const current = JSON.stringify({{
        name: prof.name,
        upstream: prof.upstream,
        features: [...prof.features].sort(),
        queue_priority: prof.queue_priority,
        default_thinking_budget: prof.default_thinking_budget,
        model_aliases: aliasesToMap(prof.aliases),
      }});
      return current !== original;
    }}

    function renderEditor() {{
      if (!editorState) return;
      const root = $('profile-editor');
      const upstreamOpts = editorState.upstreams
        .map((u) => `<option value="${{escapeHtml(u)}}">${{escapeHtml(u)}}</option>`).join('');
      const html = editorState.profiles.map((prof, idx) => {{
        const dirty = isDirty(prof);
        const featuresHtml = editorState.available_features.map((f) => {{
          const on = prof.features.has(f);
          return `<label class="pf-feature ${{on ? 'on' : ''}}" data-idx="${{idx}}" data-feature="${{escapeHtml(f)}}">` +
            `<input type="checkbox" ${{on ? 'checked' : ''}}>` +
            `${{escapeHtml(f)}}</label>`;
        }}).join('');
        const aliasesHtml = (prof.aliases || []).map((al, ai) =>
          `<div class="pf-alias-row">` +
            `<input data-idx="${{idx}}" data-alias-idx="${{ai}}" data-alias-key="from" placeholder="from" value="${{escapeHtml(al.from)}}">` +
            `<span class="pf-alias-arrow">→</span>` +
            `<input data-idx="${{idx}}" data-alias-idx="${{ai}}" data-alias-key="to" placeholder="to" value="${{escapeHtml(al.to)}}">` +
            `<button class="pf-alias-del" data-idx="${{idx}}" data-alias-del="${{ai}}" type="button">×</button>` +
          `</div>`
        ).join('');
        const selected = (val) => prof.upstream === val ? 'selected' : '';
        const upstreamSel = editorState.upstreams.map((u) =>
          `<option value="${{escapeHtml(u)}}" ${{selected(u)}}>${{escapeHtml(u)}}</option>`
        ).join('');
        const featuresTag = `<div class="pf-features">${{featuresHtml}}</div>`;
        const aliasesTag = `<div class="pf-aliases">` +
          `<div class="pf-aliases-label">aliases</div>` +
          `${{aliasesHtml}}` +
          `<button class="editor-btn" data-idx="${{idx}}" data-action="alias-add" type="button">+ alias</button>` +
        `</div>`;
        return `<div class="profile-row ${{dirty ? 'dirty' : ''}}" data-idx="${{idx}}">` +
          `<div class="pf-head">` +
            (prof.isNew
              ? `<span class="pf-name"><input data-idx="${{idx}}" data-key="name" value="${{escapeHtml(prof.name)}}" placeholder="name"></span>`
              : `<span class="pf-name">${{escapeHtml(prof.name)}}</span>`) +
            (prof.name === editorState.default_profile
              ? `<span class="pf-default">default</span>`
              : '') +
            `<span class="pf-inline"><label>upstream</label>` +
              `<select data-idx="${{idx}}" data-key="upstream">${{upstreamSel}}</select></span>` +
            `<span class="pf-inline" title="higher = jumps ahead"><label>pri</label>` +
              `<input type="number" data-idx="${{idx}}" data-key="queue_priority" value="${{prof.queue_priority}}" style="width:2.5rem"></span>` +
            `<span class="pf-inline"><label>budget</label>` +
              `<input type="number" min="0" max="64000" data-idx="${{idx}}" data-key="default_thinking_budget" value="${{prof.default_thinking_budget}}" style="width:4rem"></span>` +
            `<span class="pf-actions-inline">` +
              `<button class="editor-btn danger" data-idx="${{idx}}" data-action="delete" type="button">delete</button>` +
              `<button class="editor-btn primary" data-idx="${{idx}}" data-action="save" type="button" ${{dirty ? '' : 'disabled'}}>${{prof.isNew ? 'create' : 'save'}}</button>` +
            `</span>` +
          `</div>` +
          `<div class="pf-secondary">${{featuresTag}}${{aliasesTag}}</div>` +
        `</div>`;
      }}).join('');
      root.innerHTML = html;
      wireEditorHandlers();
    }}

    function wireEditorHandlers() {{
      const root = $('profile-editor');
      root.querySelectorAll('input[data-key], select[data-key]').forEach((el) => {{
        el.addEventListener('input', (e) => {{
          const t = e.target;
          const idx = Number(t.dataset.idx);
          const key = t.dataset.key;
          const prof = editorState.profiles[idx];
          if (!prof) return;
          if (key === 'queue_priority' || key === 'default_thinking_budget') {{
            prof[key] = Number(t.value);
          }} else {{
            prof[key] = t.value;
          }}
          renderEditor();
        }});
      }});
      root.querySelectorAll('.pf-feature').forEach((el) => {{
        el.addEventListener('click', (e) => {{
          if (e.target.tagName === 'INPUT') return; // let the checkbox handle itself
          e.preventDefault();
          const idx = Number(el.dataset.idx);
          const feature = el.dataset.feature;
          const prof = editorState.profiles[idx];
          if (!prof) return;
          if (prof.features.has(feature)) prof.features.delete(feature);
          else prof.features.add(feature);
          renderEditor();
        }});
      }});
      root.querySelectorAll('input[data-alias-key]').forEach((el) => {{
        el.addEventListener('input', (e) => {{
          const t = e.target;
          const idx = Number(t.dataset.idx);
          const ai = Number(t.dataset.aliasIdx);
          const which = t.dataset.aliasKey;
          const prof = editorState.profiles[idx];
          if (!prof || !prof.aliases[ai]) return;
          prof.aliases[ai][which] = t.value;
          // No re-render on every keystroke (cursor jumps); only refresh dirty flag.
          const row = root.querySelector(`.profile-row[data-idx="${{idx}}"]`);
          if (row) row.classList.toggle('dirty', isDirty(prof));
          const saveBtn = root.querySelector(`button[data-idx="${{idx}}"][data-action="save"]`);
          if (saveBtn) saveBtn.disabled = !isDirty(prof);
        }});
      }});
      root.querySelectorAll('button[data-action]').forEach((el) => {{
        el.addEventListener('click', () => {{
          const idx = Number(el.dataset.idx);
          const action = el.dataset.action;
          const prof = editorState.profiles[idx];
          if (!prof) return;
          if (action === 'alias-add') {{
            prof.aliases.push({{ from: '', to: '' }});
            renderEditor();
          }} else if (action === 'delete') {{
            void deleteProfile(idx);
          }} else if (action === 'save') {{
            void saveProfile(idx);
          }}
        }});
      }});
      root.querySelectorAll('button[data-alias-del]').forEach((el) => {{
        el.addEventListener('click', () => {{
          const idx = Number(el.dataset.idx);
          const ai = Number(el.dataset.aliasDel);
          const prof = editorState.profiles[idx];
          if (!prof) return;
          prof.aliases.splice(ai, 1);
          renderEditor();
        }});
      }});
    }}

    async function saveProfile(idx) {{
      const prof = editorState.profiles[idx];
      if (!prof) return;
      const trimmedName = (prof.name || '').trim();
      if (!trimmedName) {{
        setStatus('profile name required', 'err');
        return;
      }}
      setStatus(`saving ${{trimmedName}}…`);
      const payload = {{
        upstream: prof.upstream,
        features: [...prof.features],
        queue_priority: prof.queue_priority,
        default_thinking_budget: prof.default_thinking_budget,
        model_aliases: aliasesToMap(prof.aliases),
      }};
      try {{
        const r = await fetch('/config/profiles/' + encodeURIComponent(trimmedName), {{
          method: 'PUT',
          headers: {{ 'content-type': 'application/json' }},
          body: JSON.stringify(payload),
        }});
        const d = await r.json().catch(() => ({{}}));
        if (!r.ok) throw new Error(d.error || ('save returned ' + r.status));
        fadeStatus(`saved ${{trimmedName}}`, 'ok');
        await loadEditor();
      }} catch (e) {{
        setStatus(`save failed: ${{e?.message || e}}`, 'err');
      }}
    }}

    async function deleteProfile(idx) {{
      const prof = editorState.profiles[idx];
      if (!prof) return;
      if (prof.isNew) {{
        editorState.profiles.splice(idx, 1);
        renderEditor();
        return;
      }}
      if (!window.confirm(`Delete profile "${{prof.name}}"?`)) return;
      setStatus(`deleting ${{prof.name}}…`);
      try {{
        const r = await fetch('/config/profiles/' + encodeURIComponent(prof.name), {{
          method: 'DELETE',
        }});
        const d = await r.json().catch(() => ({{}}));
        if (!r.ok) throw new Error(d.error || ('delete returned ' + r.status));
        fadeStatus(`deleted ${{prof.name}}`, 'ok');
        await loadEditor();
      }} catch (e) {{
        setStatus(`delete failed: ${{e?.message || e}}`, 'err');
      }}
    }}

    $('profile-add').addEventListener('click', () => {{
      if (!editorState) return;
      const upstream = editorState.upstreams[0] || '';
      editorState.profiles.push({{
        name: '',
        upstream,
        queue_priority: 0,
        default_thinking_budget: 4096,
        features: new Set(['qwen_sampling_defaults', 'effort_to_thinking_budget',
          'thinking_overflow_recovery', 'silent_completion_recovery',
          'truncated_content_recovery', 'empty_with_stop_retry',
          'drop_oai_only_fields']),
        aliases: emptyAliases(),
        isNew: true,
      }});
      renderEditor();
    }});

    loadEditor();
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
