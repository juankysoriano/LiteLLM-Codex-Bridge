# resilient-llm-bridge

A local HTTP proxy that adds **resilience** between OpenAI-compatible clients (Codex CLI, opencode, hermes, OpenAI Agents SDK, Cline-style harnesses, custom integrations) and OpenAI-compatible upstreams (LiteLLM proxy, [vLLM](https://github.com/vllm-project/vllm), llama.cpp, Together, Fireworks, [Groq](https://groq.com), [NaN Builders](https://nan.builders), …).

You point each client at a profile-specific path on the bridge (`http://127.0.0.1:4242/codex/v1`, `http://127.0.0.1:4242/opencode/v1`, …); the bridge runs a configurable transform/recovery/retry pipeline per profile against the upstream and stitches everything together transparently.

Originally extracted from a Codex+vLLM+Qwen3 setup that needed ~12 wire-level workarounds to function. The same fixes turn out to be useful far beyond Codex.

---

## What "resilience" means concretely

| Concern | What the bridge does |
| --- | --- |
| **Different clients need different fixes** | Per-client profiles select which transformations apply. Codex needs heavy massaging for its OpenAI-native quirks; opencode/hermes are well-behaved chat/completions clients that need almost nothing. Same proxy, different paths. |
| **Transient upstream failures (5xx, 429, network errors)** | [Tenacity](https://tenacity.readthedocs.io)-based retry with exponential backoff and jitter. Configurable per-upstream `retry_max_attempts`, `retry_initial_wait`, `retry_max_wait`. |
| **Empty responses from thinking-mode models that overflow `max_tokens` during `<think>`** | Detects the pattern (`status: incomplete` + `incomplete_details.reason: max_output_tokens` + no message item, or `finish_reason: length` + content cut mid-thought, or empty + stop), runs a two-tier recovery via vLLM's `continue_final_message` (Qwen-official force-close pattern). Stitches the recovered answer back into the original response shape. |
| **Provider rate limits** | Per-upstream token bucket for RPM and `asyncio.Semaphore` for max concurrent. Queues requests when the bucket is empty, doesn't reject. |
| **vLLM bugs the upstream hasn't shipped a fix for** | Rewrites the malformed parallel-tool-call SSE that [vllm-project/vllm#39426](https://github.com/vllm-project/vllm/issues/39426) produces, until [PR #39600](https://github.com/vllm-project/vllm/pull/39600) ships. |
| **Strict-validator quirks in providers' Responses-API implementations** | Reshapes Codex' verbose request shapes (drops `reasoning` items, easy-form for assistants, folds `developer`/`system` into `instructions`, mid-chat developer→user, drops empty assistants), strips OpenAI-only fields the upstream rejects (`include`, `prompt_cache_key`, `store`, `text.verbosity`, `client_metadata`, `metadata`, `service_tier`, `user`). |
| **Greedy-decoding loops on Qwen3** | Codex defaults to `temperature: 0` (which Qwen3's docs explicitly warn against). The bridge injects Qwen-recommended sampling (`temperature=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0`) when missing. |
| **gzip mismatches in proxy chains (Cloudflare-fronted upstreams)** | Forces `Accept-Encoding: identity` upstream so SSE bytes pass through verbatim — no `stream closed before response.completed` failures. |
| **Live observability** | `/usage/stream` SSE feed for live token counter; `/activity/stream` for recent-request metadata; HTML dashboard at `/` showing it all. |

---

## Architecture

```
            ┌─────────────────────┐
client(s) ──>│ /codex/v1/...       │── path-based ─┐
            │ /opencode/v1/...    │   profile     │
            │ /hermes/v1/...      │   selection   │
            │ /v1/... (default)   │               │
            └─────────────────────┘               │
                                                   v
                            ┌────────────────────────────────┐
                            │ resilient-llm-bridge :4242     │
                            │ ─────────────────────────────  │
                            │ - request transforms (per      │
                            │   profile features list)       │
                            │ - rate limit (per-upstream     │
                            │   token bucket + semaphore)    │
                            │ - retry policy (tenacity)      │
                            │ - SSE rewrite + recovery       │
                            │ - dashboard at /               │
                            │ - /usage/stream + /activity    │
                            └────────────────┬───────────────┘
                                             │
                          ┌──────────────────┼──────────────────┐
                          v                  v                  v
                       upstream A         upstream B         upstream C
                  (e.g. NaN Builders) (e.g. Together)   (e.g. local llama.cpp)
```

Each upstream has its own rate-limit/retry config. Each profile picks one upstream and a set of features. Backward compat: `/v1/...` (no profile prefix) routes through the `default` profile.

---

## Quick start

### 1. Run

Single-file script with [`uv`](https://github.com/astral-sh/uv) inline metadata. If you have `uv`:

```bash
uv run bridge.py
```

`uv` reads the `# /// script` block at the top, creates an isolated venv with FastAPI/uvicorn/httpx/tenacity/pyyaml, and runs.

If you don't have `uv`:

```bash
pip install fastapi 'uvicorn[standard]' httpx 'tenacity>=8.0' pyyaml
python3 bridge.py
```

The bridge listens on `127.0.0.1:4242` by default.

### 2. Open the dashboard

Visit [http://127.0.0.1:4242/](http://127.0.0.1:4242/). You'll see configured upstreams, profiles, the live token counter, and recent activity.

### 3. Point your client at a profile

#### Codex CLI (`~/.codex/config.toml`)

```toml
[model_providers.nan]
base_url = "http://127.0.0.1:4242/codex/v1"
env_key = "X_NAN_KEY"
wire_api = "responses"
```

#### opencode (`~/.config/opencode/opencode.json`)

```json
{
  "provider": {
    "nan": {
      "options": {
        "baseURL": "http://127.0.0.1:4242/opencode/v1",
        "apiKey": "{env:X_NAN_KEY}"
      }
    }
  }
}
```

#### hermes (`~/.hermes/config.yaml`)

```yaml
model:
  base_url: http://127.0.0.1:4242/hermes/v1
```

Then run your client normally — it has no idea the bridge exists. Watch the dashboard while you work.

### 4. (Optional) Run as a systemd user service

```bash
mkdir -p ~/.config/resilient-llm-bridge
cp bridge.py ~/.config/resilient-llm-bridge/

mkdir -p ~/.config/systemd/user
cp systemd/resilient-llm-bridge.service ~/.config/systemd/user/

systemctl --user daemon-reload
systemctl --user enable --now resilient-llm-bridge.service
journalctl --user -u resilient-llm-bridge.service -f
```

Make sure `loginctl enable-linger $USER` is set if you want it to survive logout.

---

## Configuration

The bridge ships with **sensible defaults** (single upstream pointing at NaN Builders, profiles for codex/opencode/hermes/default). You only need a config file if you want to customize.

Default config path: `~/.config/resilient-llm-bridge/config.yaml`. Override with `BRIDGE_CONFIG_PATH`.

### Example: multiple upstreams, custom rate limits

```yaml
upstreams:
  nan:
    url: https://api.nan.builders/v1
    rate_limit_rpm: 100
    rate_limit_concurrent: 5
    retry_max_attempts: 3
    retry_initial_wait: 1.0
    retry_max_wait: 20.0
  groq:
    url: https://api.groq.com/openai/v1
    rate_limit_rpm: 30
    rate_limit_concurrent: 3
  local_vllm:
    url: http://192.168.1.50:8000/v1
    # no rate limit on a private deployment
    rate_limit_rpm: 0

profiles:
  default:
    upstream: nan
    features:
      - qwen_sampling_defaults
      - drop_oai_only_fields
      - effort_to_thinking_budget
      - thinking_overflow_recovery
      - truncated_content_recovery
      - empty_with_stop_retry

  codex:
    upstream: nan
    features:
      # default features
      - qwen_sampling_defaults
      - drop_oai_only_fields
      - effort_to_thinking_budget
      - thinking_overflow_recovery
      - truncated_content_recovery
      - empty_with_stop_retry
      # Codex-specific extras
      - normalize_responses_input
      - drop_namespace_tools
      - force_serial_tool_calls
      - parallel_tool_sse_fix

  hermes:
    upstream: nan
    # leaner — well-behaved chat/completions client
    features:
      - qwen_sampling_defaults
      - thinking_overflow_recovery
      - truncated_content_recovery

  fast:
    upstream: groq
    features: []   # Groq's strict — let it through unchanged

default_profile: default
```

### Available features

| Feature | What it does | Affects |
| --- | --- | --- |
| `qwen_sampling_defaults` | Inject Qwen3 sampling when caller didn't set them (`temperature=0.6`, `top_p`, `top_k`, `min_p`, `presence_penalty=0`) | Both endpoints |
| `drop_oai_only_fields` | Strip OpenAI-only fields the upstream rejects | Both endpoints |
| `effort_to_thinking_budget` | Translate `reasoning.effort` → `chat_template_kwargs.thinking_token_budget` | Both endpoints |
| `normalize_responses_input` | Reshape `input` items for strict validators | `/v1/responses` only |
| `drop_namespace_tools` | Drop `namespace`, `web_search`, `image_generation` etc. tools | `/v1/responses` only |
| `force_serial_tool_calls` | Set `parallel_tool_calls: false` | `/v1/responses` only |
| `parallel_tool_sse_fix` | Rewrite vLLM #39426 malformed parallel-tool SSE | `/v1/responses` streaming only |
| `thinking_overflow_recovery` | Two-tier recovery on `incomplete + max_output_tokens` | Both endpoints |
| `truncated_content_recovery` | `continue_final_message` when content cut mid-thought | Chat/completions non-stream |
| `empty_with_stop_retry` | One cheap retry on empty content + `finish_reason=stop` | Chat/completions non-stream |

### Environment variables

| Var | Default | Notes |
| --- | --- | --- |
| `BRIDGE_CONFIG_PATH` | `~/.config/resilient-llm-bridge/config.yaml` | Override config location |
| `PORT` | `4242` | Bind port |
| `HOST` | `127.0.0.1` | Bind address |
| `LOG_LEVEL` | `info` | uvicorn log level |
| `BRIDGE_USAGE_RING_SIZE` | `32` | Last-N usage records kept in memory |
| `BRIDGE_ACTIVITY_RING_SIZE` | `20` | Last-N activity records for the dashboard |

---

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | HTML dashboard (status + activity) |
| `GET` | `/health` | Liveness check + upstream/profile listing |
| `GET` | `/usage/stream` | SSE feed of `{input_tokens, output_tokens, total_tokens, profile, model}` per completion |
| `GET` | `/activity/stream` | SSE feed of recent request metadata `{ts, profile, upstream, path, method, status, duration_ms}` |
| `POST` | `/{profile}/v1/responses` | Profile-routed Responses-API |
| `POST` | `/{profile}/v1/chat/completions` | Profile-routed chat/completions |
| `*` | `/{profile}/v1/{path}` | Profile-routed catch-all (embeddings, audio/transcriptions, models, etc.) |
| `*` | `/v1/{path}` | Default-profile catch-all (backward compat) |

---

## Reporting upstream

Several of the bugs the bridge papers over should ideally be fixed at the source. Two worth filing:

1. **vLLM's parallel-tool-call SSE bug** — [vllm-project/vllm#39426](https://github.com/vllm-project/vllm/issues/39426) and [#39584](https://github.com/vllm-project/vllm/issues/39584); fix in [PR #39600](https://github.com/vllm-project/vllm/pull/39600) (unmerged at time of writing). Once shipped, `parallel_tool_sse_fix` becomes a no-op.

2. **vLLM started without `--reasoning-parser qwen3`** is a per-deployment config problem. Ask your provider operator to add:

    ```
    --reasoning-parser qwen3 --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
    ```

Operators using LiteLLM in front of vLLM can also avoid the parallel-tool bug without touching vLLM by switching their LiteLLM `model_list` entry to `openai/<model>` with `use_chat_completions_api: true` — see [LiteLLM docs](https://docs.litellm.ai/docs/response_api).

---

## License

MIT — see [LICENSE](LICENSE).
