# LiteLLM-Codex-Bridge

A small local proxy that lets [OpenAI Codex CLI](https://github.com/openai/codex) talk to a [LiteLLM](https://github.com/BerriAI/litellm) proxy serving a [vLLM](https://github.com/vllm-project/vllm)-hosted Qwen3 model — without exploding.

You point Codex at this bridge (`http://127.0.0.1:4100/v1`); the bridge transforms each request into a shape the upstream actually accepts, patches the response stream on the way back, and forwards everything else verbatim. Most of what's in here is wire-level compatibility surgery you'd otherwise have to debug yourself across half a dozen GitHub issues.

Originally built for Qwen3.6-35B-A3B served via [NaN Builders](https://api.nan.builders), but it's generic enough to work against any LiteLLM-fronted vLLM endpoint serving a Qwen3 thinking-mode model.

---

## Why this exists

Codex was designed against OpenAI's flagship API. Pointing it at a non-OpenAI LiteLLM proxy reveals a long list of incompatibilities. Here's the full menu of bugs you hit on day one, and what the bridge does about each:

| Symptom | Cause | What the bridge does |
| --- | --- | --- |
| Model loops forever, repeats phrases until `max_tokens` runs out | Codex sends `temperature: 0` (greedy decoding); Qwen3 explicitly warns against this | Override to `temperature=0.6, top_p=0.95, top_k=20, min_p=0` (Qwen's own [`generation_config.json`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507/raw/main/generation_config.json) defaults) |
| `400 - Input should be a valid string` on `body.instructions` | Codex sends both top-level `instructions` and a leading `developer`/`system` message — that's two systems by the time LiteLLM bridges to chat/completions | Fold every leading developer/system message into `instructions` |
| `400 - Unexpected message role: developer` | Codex injects `developer` messages mid-conversation | Rewrite mid-chat developer/system to `user` with `[system note]` prefix |
| `400 - Reasoning items not supported` | Codex re-sends its own reasoning chain on every turn | Strip `reasoning` items from input |
| `400 - Input should be a valid string` on assistant content | Codex sends the verbose `{type:"message", content:[{type:"output_text",...}]}` form; LiteLLM's responses-validator only accepts the easy form | Convert to easy form `{role:"assistant", content:"<text>"}` |
| `400` on assistant message with empty content | Codex emits empty assistants when the previous turn was tool-calls only | Drop empty assistant messages |
| `400` on `client_metadata`, `include`, `text`, `prompt_cache_key`, `store`, `service_tier`, `user`, `metadata` | Codex sends OpenAI-only fields; the upstream rejects unknown keys | Drop them server-side before forwarding |
| `failed to parse function arguments: trailing characters at column N` | vLLM bug ([#39426](https://github.com/vllm-project/vllm/issues/39426), [#39584](https://github.com/vllm-project/vllm/issues/39584), fix [#39600](https://github.com/vllm-project/vllm/pull/39600) unmerged) — concatenates parallel-tool-call argument JSON into one event | Track per-`output_index` argument deltas, drop the bogus concatenated `done`, synthesize correct per-call `function_call_arguments.done` events |
| `output_item.done` for function_call carries `type:"message"` instead of `type:"function_call"` | Same vLLM bug | Rewrite the item shape using deltas captured at `output_item.added` |
| `ERROR: Reconnecting... 1/5 ... stream closed before response.completed` | gzip-encoding mismatch when there's a proxy chain; common with Cloudflare in front | Force `Accept-Encoding: identity` on the upstream request |
| `unsupported call: mcp__SERVER__TOOL` (Codex MCP tools never round-trip) | Codex bundles MCP servers as `type:"namespace"` tools; flattening to flat function names confuses Codex's MCP dispatcher on the way back | Drop all non-`function` tools (namespace, mcp, web_search, image_generation) |
| Thinking budget knob silently ignored | vLLM not started with `--reasoning-parser qwen3` | Inject `extra_body.chat_template_kwargs.thinking_token_budget` (no-op until upstream is reconfigured, but ready) |

The transformations are documented inline in [`bridge.py`](bridge.py) — every function has a docstring explaining the bug it papers over and the source link.

---

## Architecture

```
Codex CLI ──HTTP──> 127.0.0.1:4100 ──HTTPS──> LiteLLM ──> vLLM ──> Qwen3
            (this bridge)
              │
              └─> /usage/stream SSE feed (live token counter)
```

All traffic goes through one local entry point. The bridge is stateless — kill the systemd unit, restart, you lose nothing. Each request gets `_transform_body` (or `_transform_chat_body`) applied before forwarding; responses get `_stream_passthrough` (for `/v1/responses` parallel-tool fix) or pass through verbatim (`/v1/chat/completions` and the rest).

**Endpoints exposed:**
- `GET  /health` — liveness + upstream URL
- `GET  /usage/stream` — SSE feed of `{input_tokens, output_tokens, total_tokens}` per completion
- `POST /v1/responses` — Codex's primary endpoint, transformed + SSE-rewritten
- `POST /v1/chat/completions` — opencode/openclaw/etc. compatibility
- `*    /v1/{anything}` — generic passthrough (embeddings, audio, models, …)

---

## Quick start

### 1. Install dependencies

The bridge is a single-file Python script with [`uv`](https://github.com/astral-sh/uv) inline metadata. If you have `uv`, you can run it directly with no setup:

```bash
uv run bridge.py
```

`uv` reads the `# /// script` block at the top, creates an isolated venv with FastAPI/uvicorn/httpx, and runs the script.

If you don't have `uv`:

```bash
pip install fastapi 'uvicorn[standard]' httpx
python3 bridge.py
```

The bridge listens on `127.0.0.1:4100` by default.

### 2. Sanity-check it's up

```bash
curl http://127.0.0.1:4100/health
# {"ok":true,"upstream":"https://api.nan.builders/v1"}
```

### 3. Point Codex at it

Copy [`examples/codex-config.toml`](examples/codex-config.toml) to `~/.codex/config.toml` (or merge if you already have one), then export your upstream API key:

```bash
export X_NAN_KEY="sk-..."   # or whatever your upstream provider gave you
```

Run Codex:

```bash
codex exec --skip-git-repo-check 'Reply with exactly "ok"'
# ok
```

### 4. (Optional) Run as a systemd user service

```bash
mkdir -p ~/.config/litellm-codex-bridge
cp bridge.py ~/.config/litellm-codex-bridge/

mkdir -p ~/.config/systemd/user
cp systemd/litellm-codex-bridge.service ~/.config/systemd/user/

systemctl --user daemon-reload
systemctl --user enable --now litellm-codex-bridge.service
journalctl --user -u litellm-codex-bridge.service -f
```

Make sure `loginctl enable-linger $USER` is set if you want it to survive logout.

---

## Configuration

Everything is environment-variable driven. None of these are required if you're hitting `https://api.nan.builders`:

| Env var | Default | Notes |
| --- | --- | --- |
| `BRIDGE_UPSTREAM_URL` | `https://api.nan.builders/v1` | Where to forward requests |
| `BRIDGE_USAGE_RING_SIZE` | `32` | How many recent usage records to keep in memory for newly-attached `/usage/stream` clients |
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `4100` | Bind port |
| `LOG_LEVEL` | `info` | uvicorn log level |

Backward compat: the bridge also accepts the legacy `NAN_UPSTREAM_URL` and `NAN_USAGE_RING_SIZE` variable names if you're migrating from the original `nan-stream-fixer.py` script.

---

## What this bridge is *not* for

- **opencode, openclaw, OpenWebUI, Cline, etc.** — these clients send well-formed OpenAI requests and don't trip over the validator quirks. They work fine talking directly to LiteLLM. The bridge has a `/v1/chat/completions` route mostly for completeness; you can route them through it if you want centralized Qwen-sampling injection, but it's not required.
- **Hosted features the upstream doesn't run** — `web_search`, `image_generation`, file_search, the OpenAI hosted MCP gateway. The bridge drops the tool definitions so Codex doesn't try to use them, but it doesn't *implement* them. If your upstream LiteLLM exposes equivalents, declare those as plain `function` tools.
- **Authentication.** The bridge is a dumb forwarder for the `Authorization` header — it doesn't read or validate API keys. Don't expose it on a public port; it's a localhost-only tool.

---

## Reporting upstream

Several of the bugs the bridge papers over should ideally be fixed at the source. Two that are worth filing:

1. **vLLM's parallel-tool-call SSE bug** — tracked as [#39426](https://github.com/vllm-project/vllm/issues/39426) / [#39584](https://github.com/vllm-project/vllm/issues/39584); fix in [#39600](https://github.com/vllm-project/vllm/pull/39600) (unmerged at time of writing). Once shipped, the bridge's `_stream_passthrough` SSE rewriting becomes a no-op.

2. **vLLM started without `--reasoning-parser qwen3`** is a per-deployment config problem; the bridge's `_apply_effort_budget` has been a no-op against deployments missing this flag. Ask your upstream operator to add:

    ```
    --reasoning-parser qwen3 --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
    ```

Operators can also work around the parallel-tool bug without touching vLLM by switching their LiteLLM `model_list` entry to `openai/chat_completions/<model>` with `use_chat_completions_api: true` — that routes `/v1/responses` through LiteLLM's chat-completions bridge instead of vLLM's broken native path. Per [LiteLLM docs](https://docs.litellm.ai/docs/response_api).

---

## License

MIT — see [LICENSE](LICENSE).
