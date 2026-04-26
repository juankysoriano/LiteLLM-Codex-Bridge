# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "uvicorn[standard]",
#   "httpx",
# ]
# ///
"""
LiteLLM-Codex-Bridge — a thin local proxy that lets OpenAI Codex CLI talk to
a LiteLLM-fronted vLLM Qwen3 deployment without exploding.

Codex was designed against OpenAI's APIs. Pointing it at a non-OpenAI
LiteLLM proxy reveals a long list of incompatibilities: greedy decoding
defaults that make Qwen3 loop forever, an `/v1/responses` validator that
rejects half of Codex's request shapes, OpenAI-only fields the upstream
chokes on, broken streaming events for parallel tool calls, MCP namespace
tools that round-trip nowhere, and so on.

This bridge sits at `127.0.0.1:4100` (configurable). Codex points at it
instead of the upstream LiteLLM directly. Each request gets normalised
into a shape the upstream actually accepts, and each response stream gets
patched on the way back. Most clients (opencode, openclaw, etc.) don't
need this — Codex is the only popular harness that hits `/v1/responses`
hard enough to surface every edge case.

Architecture:

    Codex ──HTTP──> bridge.py (this file) ──HTTPS──> LiteLLM ──> vLLM
                       └─ /usage/stream SSE for live token counters

Endpoints exposed:

    GET  /health                  health check + upstream URL
    GET  /usage/stream            SSE feed of {input_tokens, output_tokens, total_tokens}
                                  whenever a completion finishes
    POST /v1/responses            transformed forward to upstream `/v1/responses`,
                                  with stream-rewrite for the parallel-tool-call SSE bug
    POST /v1/chat/completions     transformed forward to upstream `/v1/chat/completions`,
                                  passthrough stream
    *    /v1/{anything}           generic passthrough (embeddings, audio, models, …)

Env vars:

    BRIDGE_UPSTREAM_URL           default https://api.nan.builders/v1
    BRIDGE_USAGE_RING_SIZE        default 32 (last N usage records kept in memory)
    HOST                          default 127.0.0.1
    PORT                          default 4100
    LOG_LEVEL                     default info

Documentation of every transformation lives next to the function that
applies it. Read top-to-bottom for a guided tour of the bug list this
bridge exists to paper over.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


# --------------------------------------------------------------------- config --

# Backward compat: also accept the legacy NAN_UPSTREAM_URL env var for
# anyone migrating from the original `nan-stream-fixer.py` script.
UPSTREAM = (
    os.environ.get("BRIDGE_UPSTREAM_URL")
    or os.environ.get("NAN_UPSTREAM_URL")
    or "https://api.nan.builders/v1"
).rstrip("/")
USAGE_RING_SIZE = int(
    os.environ.get("BRIDGE_USAGE_RING_SIZE")
    or os.environ.get("NAN_USAGE_RING_SIZE")
    or "32"
)

app = FastAPI(title="LiteLLM-Codex-Bridge")


# --------------------------------------------------------------------- usage --
#
# We sniff the `usage` block from every completion the upstream emits and
# rebroadcast it on `/usage/stream` so a dashboard / TUI can show live token
# counters. Last N records are kept in memory so a freshly-attached client
# sees the most recent value immediately.

_usage_history: deque[dict] = deque(maxlen=USAGE_RING_SIZE)
_usage_subscribers: set[asyncio.Queue[dict]] = set()


def _broadcast_usage(record: dict) -> None:
    """Append a usage record to history and notify every connected subscriber.

    Uses `put_nowait` and silently drops on full queue — a slow consumer
    must never back-pressure live request handling.
    """
    _usage_history.append(record)
    print(
        f"usage model={record.get('model')} in={record.get('input_tokens')}"
        f" out={record.get('output_tokens')} total={record.get('total_tokens')}",
        flush=True,
    )
    for queue in list(_usage_subscribers):
        try:
            queue.put_nowait(record)
        except asyncio.QueueFull:
            pass


def _extract_usage(model: str | None, payload: dict) -> dict | None:
    """Parse OpenAI-style usage block out of a response or `response.completed`.

    Tolerates both the chat-completions shape (`prompt_tokens` /
    `completion_tokens`) and the responses-API shape (`input_tokens` /
    `output_tokens`). Falls back to summing input+output if `total_tokens`
    isn't reported.
    """
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    record = {
        "ts": time.time(),
        "model": model or payload.get("model"),
        "input_tokens": int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }
    if record["total_tokens"] == 0:
        record["total_tokens"] = record["input_tokens"] + record["output_tokens"]
    return record


# ------------------------------------------------------- request body shaping --
#
# Everything below this line is request-side surgery: we mutate Codex'
# JSON body before forwarding it upstream. Each transform exists to fix
# exactly one observed incompatibility — read the docstrings.

_SYSTEM_LIKE_ROLES = {"system", "developer"}
_SYSTEM_NOTE_PREFIX = "[system note]\n"


# --- thinking budget ---------------------------------------------------------
#
# Codex' `reasoning.effort` enum (low / medium / high / xhigh) is mapped to a
# Qwen3 `thinking_token_budget`. Numbers track the model card for
# Qwen3-30B-A3B-Thinking-2507 (same A3B family as the qwen3.6-35b-a3b that
# motivated this bridge), where the team uses 8192 to "avoid overly verbose
# reasoning" on benchmarks; 32K is competition-math territory and provokes
# the "thinks for kilometers" failure mode at xhigh — vLLM's
# thinking_token_budget is a hard cap, so the model fills the entire budget
# when it doesn't auto-emit `</think>`.
#
# IMPORTANT — empirically this is currently a no-op against api.nan.builders
# (probed 2026-04-26). NaN's vLLM is not started with --reasoning-parser
# qwen3, so the param is silently dropped: setting thinking_token_budget=128
# still produces ~4500 reasoning tokens. The hooks stay in place so they
# fire the moment vLLM gets the right startup flags. Until then, bound the
# combined output via `max_output_tokens` instead.
#
# Sources:
#   https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507
#   https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html#thinking-budget
#   https://docs.vllm.ai/en/latest/features/reasoning_outputs.html
_EFFORT_TO_THINKING_BUDGET: dict[str, int] = {
    "low": 1024,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
}
# Cap applied when the caller didn't pin a budget — protects clients
# (opencode, openclaw) that don't surface a reasoning-effort knob.
_DEFAULT_THINKING_BUDGET = 8192


# --- Qwen3 sampling defaults -------------------------------------------------
#
# Codex defaults to `temperature: 0` (greedy decoding). Qwen3's model card
# explicitly warns: "DO NOT use greedy decoding, as it can lead to
# performance degradation and endless repetitions." The shipped
# generation_config.json sets temperature=0.6, top_k=20, top_p=0.95, min_p=0
# and does NOT set presence_penalty (so the effective default is 0.0).
#
# We inject those defaults whenever the caller didn't set them. Qwen also
# accepts `presence_penalty` in [0, 2] to fight repetition, but the model
# card warns that high values cause language mixing on coding tasks — keep
# at 0.0 unless you observe actual repetition loops, then escalate to 0.5,
# then 1.0. Never above 1.0 for coding/tool-use.
#
# Source: https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507/raw/main/generation_config.json
_QWEN_TOP_LEVEL_DEFAULTS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "presence_penalty": 0.0,
}
_QWEN_EXTRA_BODY_DEFAULTS = {
    "top_k": 20,
    "min_p": 0,
}


# --- OpenAI-only fields to drop ----------------------------------------------
#
# Codex sends a handful of fields that only exist in OpenAI's flagship API.
# Forwarding them to a LiteLLM/vLLM upstream typically produces either a
# 400 validation error or an immediate `stream closed before
# response.completed`. Drop them rather than play whack-a-mole.
_DROP_FIELDS = {
    "client_metadata",
    "include",            # e.g. ["reasoning.encrypted_content"]
    "text",               # {"verbosity": "low"} — OpenAI verbosity hint
    "prompt_cache_key",   # OpenAI prompt-caching hint
    "store",              # whether OpenAI persists the response server-side
    "service_tier",
    "user",
    "metadata",
}


def _content_to_string(content) -> str:
    """Flatten an OpenAI message `content` field down to a plain string.

    Accepts the three canonical shapes:
      - bare string: returned as-is
      - list of {"type":"input_text"|"output_text"|..., "text":"..."}: joined
      - anything else: stringified

    Used by `_normalize_responses_input` to collapse Codex' verbose part
    arrays into the easy-form strings that the upstream Pydantic validator
    actually accepts (see that function's docstring).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n\n".join(parts)
    return str(content or "")


def _normalize_responses_input(input_value, *, fold_leading_into_instructions: bool):
    """Reshape a `/v1/responses` `input` array so the upstream validator accepts it.

    Returns a tuple `(new_input_list, folded_instructions_text)`. The
    folded text should be appended to the top-level `instructions` field
    if it's non-empty.

    LiteLLM's `/v1/responses` route bridges to chat/completions internally
    and is significantly stricter than OpenAI's reference implementation:

    * Rejects `{"type":"message","role":"assistant","content":[
      {"type":"output_text",...}]}` — the verbose form Codex prefers.
      Only the easy form `{"role":"assistant","content":"<string>"}`
      validates against `EasyInputMessageParam`. We collapse to easy form.
    * Rejects `reasoning` items entirely — they're meaningful only with
      `include: ["reasoning.encrypted_content"]`, which we drop upstream
      anyway. Codex re-injects them on every turn; we strip them.
    * Allows at most one leading `system` message (chat-completions
      semantics). Codex sends top-level `instructions` AND a leading
      `developer` / `system` message — that's two systems by the time
      LiteLLM bridges. We fold the leading messages into `instructions`
      to stay under the cap.
    * Mid-conversation `developer` / `system` messages get rewritten to
      `user` with a `[system note]` prefix so the validator stops
      complaining about role ordering while the model still sees the
      directive.
    * `function_call` / `function_call_output` items pass through —
      LiteLLM accepts those directly.
    * Empty assistant messages (Codex emits these when a turn was
      tool-calls only) are dropped — the validator rejects empty content
      arrays for assistant role.
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


def _apply_tool_transforms(data: dict) -> None:
    """Filter the request's `tools` array to what Codex+upstream can round-trip.

    Strategy: keep `type: "function"` only. Drop everything else.

    * **`namespace` (Codex's MCP server bundling)**: tempting to flatten
      each inner tool to a top-level function so the upstream accepts it,
      but Codex's MCP dispatcher only routes the resulting `function_call`
      back to the right server when the model emits the namespaced shape.
      Since we'd be sending the model a flat name, the response comes
      back as an unknown function and Codex emits `"unsupported call:
      mcp__SERVER__TOOL"`. The model often gives up and emits an empty
      assistant message after that, so the user sees no response on the
      first turn. Cleanest is to never advertise MCP tools to the model.

    * **`web_search`, `image_generation`, `mcp`, `mcp_tool`**: these are
      hosted-side capabilities (OpenAI runs them server-side). Most
      LiteLLM/vLLM upstreams either reject the type with a 400 or accept
      it but emit an unfulfilled `function_call` named after the type,
      which Codex similarly can't dispatch. Drop them and let the model
      improvise (e.g. `curl` via exec_command for a question that wanted
      `web_search`).

    Trade-off: MCP-routed features (Happy's title-rename, context7's doc
    lookup, OpenAI's developer-docs MCP) become invisible to the model.
    Everything text-based and exec-based still works.
    """
    tools = data.get("tools")
    if not isinstance(tools, list):
        return
    kept: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function":
            kept.append(tool)
    if kept:
        data["tools"] = kept
    else:
        data.pop("tools", None)
        data.pop("tool_choice", None)


def _apply_qwen_sampling_defaults(data: dict) -> None:
    """Inject Qwen3's recommended sampling whenever Codex left a slot blank.

    Codex defaults to `temperature: 0` (greedy decoding) for OpenAI's
    GPT-class models. Qwen3 *cannot* tolerate greedy decoding — the model
    card warns it leads to "performance degradation and endless
    repetitions". You'll observe Qwen falling into 30-second loops of
    `"step 1: analyze step 1: analyze step 1: analyze..."` until it
    exhausts `max_output_tokens`.

    We override only when the caller didn't set a value (or sent the
    greedy default of 0), so explicit overrides survive. Top-level
    `temperature` / `top_p` / `presence_penalty` go on the request body
    directly; vLLM-specific knobs (`top_k`, `min_p`) ride along in
    `extra_body` because they're not part of OpenAI's standard schema.
    """
    if data.get("temperature") in (None, 0, 0.0):
        data["temperature"] = _QWEN_TOP_LEVEL_DEFAULTS["temperature"]
    if data.get("top_p") is None:
        data["top_p"] = _QWEN_TOP_LEVEL_DEFAULTS["top_p"]
    if data.get("presence_penalty") in (None, 0, 0.0):
        data["presence_penalty"] = _QWEN_TOP_LEVEL_DEFAULTS["presence_penalty"]

    extra = data.get("extra_body")
    if not isinstance(extra, dict):
        extra = {}
    for key, value in _QWEN_EXTRA_BODY_DEFAULTS.items():
        extra.setdefault(key, value)
    data["extra_body"] = extra


def _apply_effort_budget(data: dict) -> None:
    """Translate `reasoning.effort` into a Qwen3 `thinking_token_budget`.

    Falls back to `_DEFAULT_THINKING_BUDGET` when no effort is sent.

    NOTE — this transform is currently a NO-OP against api.nan.builders.
    Empirical probe on 2026-04-26: setting `thinking_token_budget=128` in
    `extra_body.chat_template_kwargs` (or anywhere else) produces the
    same ~4500 reasoning tokens as no cap. Per vLLM docs, the budget is
    enforced only when the server is started with
    `--reasoning-parser qwen3`; without it, the param is silently
    dropped. The hooks below stay in place so the transform fires the
    moment the upstream is reconfigured. To bound thinking today, lower
    `max_output_tokens`.
    """
    effort: str | None = None
    reasoning = data.get("reasoning")
    if isinstance(reasoning, dict) and isinstance(reasoning.get("effort"), str):
        effort = reasoning["effort"]
    if effort is None and isinstance(data.get("reasoning_effort"), str):
        effort = data["reasoning_effort"]

    # Unknown effort strings (typos like "ultra", or future values we don't
    # know about yet) fall back to the default cap rather than skipping
    # the transform entirely — the caller asked for *some* thinking limit.
    if effort:
        budget = _EFFORT_TO_THINKING_BUDGET.get(effort, _DEFAULT_THINKING_BUDGET)
    else:
        budget = _DEFAULT_THINKING_BUDGET

    extra = data.get("extra_body")
    if not isinstance(extra, dict):
        extra = {}
    chat_kwargs = extra.get("chat_template_kwargs")
    if not isinstance(chat_kwargs, dict):
        chat_kwargs = {}
    chat_kwargs.setdefault("enable_thinking", True)
    chat_kwargs.setdefault("thinking_token_budget", budget)
    extra["chat_template_kwargs"] = chat_kwargs
    data["extra_body"] = extra


def _transform_body(body: dict) -> dict:
    """Apply every per-request transformation needed for `/v1/responses`.

    Order matters: input normalisation must run before the sampling and
    budget transforms because the latter mutate `extra_body`, and we
    want the dropped fields to disappear *after* everything else has
    consumed them. The forced `parallel_tool_calls=False` is a workaround
    for vLLM bug https://github.com/vllm-project/vllm/issues/39426 — see
    `_stream_passthrough` for the SSE-rewrite half of the same story.
    """
    if "input" in body:
        instructions = body.get("instructions")
        has_instructions = isinstance(instructions, str) and instructions.strip()
        new_input, folded = _normalize_responses_input(
            body["input"], fold_leading_into_instructions=bool(has_instructions)
        )
        body["input"] = new_input
        if folded:
            body["instructions"] = (instructions or "").rstrip() + "\n\n" + folded

    _apply_tool_transforms(body)
    _apply_qwen_sampling_defaults(body)
    _apply_effort_budget(body)

    # Force serial tool execution. Even though Codex sends
    # `parallel_tool_calls: True`, vLLM's `/v1/responses` SSE concatenates
    # the arguments JSON of every parallel call into a single
    # `response.function_call_arguments.done` event, which Codex then
    # fails to parse. Setting this to False would *normally* be honoured
    # by the upstream and the bug avoided — empirically it isn't, so we
    # also rewrite the broken SSE in `_stream_passthrough`. The override
    # is kept in case a future upstream version starts honouring it.
    body["parallel_tool_calls"] = False

    for field in _DROP_FIELDS:
        body.pop(field, None)

    return body


def _transform_chat_body(body: dict) -> dict:
    """Per-request transforms for `/v1/chat/completions`.

    Strict subset of `_transform_body`: chat-completions uses `messages`
    directly (no input/instructions reshaping needed), only knows
    `type: "function"` tools (no flattening needed), and uses a different
    streaming format that doesn't trigger vLLM's parallel-tool concat
    bug (so no `parallel_tool_calls=False` override). We still inject
    Qwen sampling defaults, the thinking-budget cap, and drop the
    OpenAI-only fields.
    """
    _apply_qwen_sampling_defaults(body)
    _apply_effort_budget(body)
    for field in _DROP_FIELDS:
        body.pop(field, None)
    return body


# ----------------------------------------------------------------- endpoints --


def _sse(obj: dict) -> bytes:
    """Serialise a dict as a single Server-Sent Event line."""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")


@app.get("/health")
async def health():
    """Liveness probe. Returns the upstream URL so callers can sanity-check config."""
    return {"ok": True, "upstream": UPSTREAM}


@app.get("/usage/stream")
async def usage_stream():
    """SSE feed of `{input_tokens, output_tokens, total_tokens}` per completion.

    Emits the latest historical record on connect (so a freshly-attached
    client sees the most recent value immediately), then any new record
    as it arrives. Sends a `: ping` comment every 15s of silence to keep
    proxies and idle-timeout middleware from closing the connection.
    """
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


def _build_outgoing_headers(request: Request) -> dict[str, str]:
    """Build headers for the upstream request.

    The non-obvious bit is `accept-encoding: identity`. Many LiteLLM
    deployments (and any Cloudflare in front of them) gzip SSE responses
    by default. When this bridge is in the proxy chain, gzip means
    re-compress / re-decompress at every hop, and the slightest byte
    mismatch between content-encoding header and actual bytes yields
    "stream closed before response.completed" failures — Codex shows
    them as `ERROR: Reconnecting... 1/5`. Forcing identity gets us raw
    SSE bytes from upstream that we forward verbatim, no decompression
    mismatch possible.
    """
    headers: dict[str, str] = {
        "content-type": "application/json",
        "accept": "text/event-stream",
        "accept-encoding": "identity",
    }
    auth = request.headers.get("authorization")
    if auth:
        headers["authorization"] = auth
    return headers


async def _stream_passthrough(
    body: dict,
    headers: dict[str, str],
) -> AsyncIterator[bytes]:
    """Forward upstream `/v1/responses` SSE, fixing vLLM's parallel-tool bugs.

    Background: vLLM's native `/v1/responses` streaming has two bugs around
    parallel tool calls (tracked as
    https://github.com/vllm-project/vllm/issues/39426 and
    https://github.com/vllm-project/vllm/issues/39584; fix in
    https://github.com/vllm-project/vllm/pull/39600 unmerged as of
    2026-04-26):

    1. When the model emits N parallel `function_call` items in one turn,
       vLLM concatenates the arguments JSON of every call into a single
       `response.function_call_arguments.done` event:

           {"cmd":"a"}{"cmd":"b"}{"cmd":"c"}

       Codex tries to JSON-parse that and dies with "trailing characters
       at column N".

    2. The `response.output_item.done` event for each function_call
       carries `item.type: "message"` with empty fields, instead of the
       real `function_call` shape Codex expects.

    What this function does: track per-`output_index` argument deltas as
    they stream in, drop the bogus concatenated `done`, and synthesize a
    correct per-call `function_call_arguments.done` plus a corrected
    `output_item.done` with the right `type`, `name`, `call_id`, and
    accumulated `arguments` at the right time. Non-function-call events
    are re-serialised from the parsed JSON payload (so identical content
    but not necessarily byte-identical to upstream — JSON whitespace and
    key ordering may differ); SSE comments / keepalive lines are
    forwarded byte-for-byte.

    Server-side alternative for upstream operators: configure LiteLLM to
    use `use_chat_completions_api: true` on the model entry — that
    routes `/v1/responses` through LiteLLM's bridge from chat/completions
    (which works correctly for parallel tool calls since LiteLLM 1.82.6)
    instead of vLLM's broken native path. Once that's in place, this
    function becomes a no-op.
    """
    model_hint = body.get("model")
    buffer = b""
    # output_index → {"id","call_id","name","args"} for in-flight function_call items.
    fc_state: dict[int, dict] = {}

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, read=None)) as client:
        async with client.stream(
            "POST", f"{UPSTREAM}/responses", json=body, headers=headers
        ) as response:
            if response.status_code >= 400:
                error_text = (await response.aread()).decode("utf-8", errors="ignore")
                yield _sse({"type": "error", "status": response.status_code, "body": error_text})
                return
            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                buffer += chunk
                while b"\n\n" in buffer:
                    event, buffer = buffer.split(b"\n\n", 1)
                    raw_lines = event.splitlines()
                    payload: dict | None = None
                    for line in raw_lines:
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
                        # SSE comment / keepalive line — pass through unchanged.
                        yield event + b"\n\n"
                        continue

                    etype = payload.get("type")

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
                        # Drop vLLM's bogus concatenated `done` per turn —
                        # we emit correct per-call dones at output_item.done time.
                        continue

                    if etype == "response.output_item.done":
                        idx = payload.get("output_index")
                        st = fc_state.get(idx)
                        if st is not None:
                            seq = payload.get("sequence_number")
                            yield _sse({
                                "type": "response.function_call_arguments.done",
                                "item_id": st["id"],
                                "output_index": idx,
                                "name": st["name"],
                                "arguments": st["args"],
                                "sequence_number": seq,
                                "model": payload.get("model"),
                            })
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
                        yield _sse(payload)
                        continue

                    if etype == "response.completed":
                        response_obj = payload.get("response")
                        if isinstance(response_obj, dict):
                            record = _extract_usage(model_hint, response_obj)
                            if record:
                                _broadcast_usage(record)
                        yield _sse(payload)
                        continue

                    yield _sse(payload)


@app.post("/v1/responses")
async def responses(request: Request):
    """Codex's primary endpoint. Transform body, forward, fix the SSE on the way back."""
    body = await request.json()
    body = _transform_body(body)
    want_stream = bool(body.get("stream", False))
    headers = _build_outgoing_headers(request)

    if want_stream:
        return StreamingResponse(
            _stream_passthrough(body, headers),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        try:
            r = await client.post(f"{UPSTREAM}/responses", json=body, headers=headers)
        except httpx.HTTPError as e:
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"upstream error: {e}"}},
            )

    payload: dict = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if r.status_code < 400 and isinstance(payload, dict):
        record = _extract_usage(body.get("model"), payload)
        if record:
            _broadcast_usage(record)

    return JSONResponse(content=payload, status_code=r.status_code)


async def _chat_stream_passthrough(
    body: dict,
    headers: dict[str, str],
) -> AsyncIterator[bytes]:
    """Forward upstream `/v1/chat/completions` SSE, sniffing usage on the way.

    Unlike `/v1/responses`, the chat-completions stream is well-formed —
    we only forward bytes and watch for the final `usage` block to update
    the live token counter. No SSE rewriting needed.
    """
    model_hint = body.get("model")
    buffer = b""
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, read=None)) as client:
        async with client.stream(
            "POST", f"{UPSTREAM}/chat/completions", json=body, headers=headers
        ) as response:
            if response.status_code >= 400:
                error_text = (await response.aread()).decode("utf-8", errors="ignore")
                yield _sse({"type": "error", "status": response.status_code, "body": error_text})
                return
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
                        usage = payload.get("usage")
                        if isinstance(usage, dict):
                            record = _extract_usage(model_hint, payload)
                            if record:
                                _broadcast_usage(record)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat endpoint with sampling + budget transforms applied."""
    body = await request.json()
    body = _transform_chat_body(body)
    want_stream = bool(body.get("stream", False))
    headers = _build_outgoing_headers(request)

    if want_stream:
        return StreamingResponse(
            _chat_stream_passthrough(body, headers),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        try:
            r = await client.post(f"{UPSTREAM}/chat/completions", json=body, headers=headers)
        except httpx.HTTPError as e:
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"upstream error: {e}"}},
            )

    payload: dict = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if r.status_code < 400 and isinstance(payload, dict):
        record = _extract_usage(body.get("model"), payload)
        if record:
            _broadcast_usage(record)

    return JSONResponse(content=payload, status_code=r.status_code)


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    include_in_schema=False,
)
async def proxy_passthrough(request: Request, path: str):
    """Generic passthrough for the rest of the upstream's surface.

    Forwards request body, query params, and most headers to
    `{UPSTREAM}/{path}` without transformation. Strips
    `host`/`content-length` on the way out (httpx sets them) and
    `content-encoding`/`content-length`/`transfer-encoding` on the way
    back (the forwarded body is already decoded, so the original
    encoding headers no longer match).

    Lets a single base URL cover `audio/transcriptions`, `embeddings`,
    `models`, `audio/speech`, etc. without us having to know each one.
    """
    upstream_url = f"{UPSTREAM}/{path}"
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }
    body_bytes = await request.body()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, read=None)) as client:
            r = await client.request(
                request.method,
                upstream_url,
                content=body_bytes,
                headers=headers,
                params=dict(request.query_params),
            )
    except httpx.HTTPError as e:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"upstream error: {e}"}},
        )
    return Response(
        content=r.content,
        status_code=r.status_code,
        headers={
            k: v for k, v in r.headers.items()
            if k.lower() not in {"content-encoding", "content-length", "transfer-encoding"}
        },
        media_type=r.headers.get("content-type"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "4100")),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
