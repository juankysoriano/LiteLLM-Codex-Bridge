# NaN LLM Bridge

NaN LLM Bridge is a local OpenAI-compatible proxy for routing multiple clients through the NaN Builders upstream with per-profile policy, retries, streaming safeguards, queueing, and live observability.

It is meant to be a practical compatibility layer: point clients at `localhost:4242`, give each client or workload a profile, and let the bridge handle the small but important differences between what clients send and what the upstream expects.

![NaN LLM Bridge dashboard](docs/dashboard.png)

```text
http://localhost:4242/v1                  # default profile
http://localhost:4242/myproject/v1        # example profile
http://localhost:4242/hermes/v1           # local profile you can add
http://localhost:4242/opencode/v1         # local profile you can add
```

Only `default` and `myproject` are shipped as public example profiles. Real local profiles are meant to live in your own `~/.config/resilient-llm-bridge/config.yaml`, outside the repo.

## Features

- OpenAI-compatible endpoints for `/v1/chat/completions`, `/v1/responses`, and pass-through routes such as `/v1/models`.
- Profile routing via `/{profile}/v1/...`, so different clients can have different behavior.
- NaN Builders upstream by default, with configurable upstreams if you want to point elsewhere.
- Per-profile policies for thinking, reasoning effort translation, output-token defaults, model overrides, sampling overrides, forced streaming, and retries.
- Internal queue with profile priorities for upstream limits such as `100 rpm` and `5` concurrent requests.
- Stream-first behavior to reduce Cloudflare or proxy idle timeouts when a client forgets `stream=true`.
- Recovery and retry mechanisms for common failure cases.
- Dashboard at `/` with recent activity, active requests, profiles, models, token usage, latency, recoveries, and upstream load.

## Why Profiles Help

Profiles let one bridge behave differently for different clients or workloads.

Examples:

- `default`: conservative catch-all for unknown clients.
- `myproject`: a project-specific profile with higher queue priority than background work.
- `hermes`: a local profile you might configure for fast interactive chat.
- `opencode`: a local profile that respects the selected model but translates `reasoning_effort` into the upstream thinking fields.
- `batch-heavy`: a local profile for offline jobs that need more thinking budget and lower priority.
- `fast-no-thinking`: a local profile for cheap, low-latency calls where thinking should stay off.

Clients call the profile directly:

```text
http://localhost:4242/myproject/v1/chat/completions
```

The bridge strips the profile prefix, applies that profile's policy, and forwards the request to the configured upstream.

## Retry And Recovery

The bridge can retry or recover several common cases:

- transient upstream errors before bytes reach the client
- 524-style timeout failures
- empty responses
- completions cut by `max_tokens`
- streamed requests that need buffered recovery
- malformed or incomplete tool-call arguments
- Qwen-style XML tool-call residue leaking into text or reasoning
- payload fields that need compatibility cleanup before reaching the upstream

Retries are profile-controlled via `auto_retries`. Streaming can also be forced per profile via `force_stream`, including for bridge-internal recovery calls.

## Queueing And Priorities

Each upstream has a rate/concurrency gate:

- `rate_limit_rpm`
- `rate_limit_concurrent`
- `reserved_priority_slots`
- `reserved_priority_threshold`
- `first_byte_timeout_s`
- `queue_timeout_s`
- `stuck_warn_s`

When the upstream is saturated, requests wait in an internal priority queue instead of failing immediately. Higher `queue_priority` profiles go first.

This is useful when the upstream has limits such as `100 rpm` and `5` parallel requests. You can let interactive clients jump ahead of batch jobs while still keeping background work queued.

`reserved_priority_slots` keeps part of the concurrency pool unavailable to low-priority profiles. For example, with `rate_limit_concurrent: 5`, `reserved_priority_slots: 2`, and `reserved_priority_threshold: 1`, priority `0` work can use at most 3 slots while priority `1+` can use all 5.

`first_byte_timeout_s` aborts and retries streaming requests that never produce upstream bytes. Set it around `60` seconds to avoid “waiting / 0 chunks / 0 bytes” calls holding a slot indefinitely.

## Quick Start

Run directly with `uv`:

```bash
uv run bridge.py
```

Or install dependencies yourself:

```bash
pip install fastapi 'uvicorn[standard]' httpx 'tenacity>=8.0' pyyaml
python3 bridge.py
```

The bridge listens on `127.0.0.1:4242` by default.

Open the dashboard:

```text
http://127.0.0.1:4242/
```

Health check:

```bash
curl http://127.0.0.1:4242/health
```

## Configuration

Default config path:

```text
~/.config/resilient-llm-bridge/config.yaml
```

Override it with:

```bash
BRIDGE_CONFIG_PATH=/path/to/config.yaml uv run bridge.py
```

Public example:

```yaml
upstreams:
  nan:
    url: "https://api.nan.builders/v1"
    rate_limit_rpm: 100
    rate_limit_concurrent: 5
    reserved_priority_slots: 2
    reserved_priority_threshold: 1
    first_byte_timeout_s: 60.0
    queue_timeout_s: 1200.0
    stuck_warn_s: 1200.0

profiles:
  default:
    upstream: nan
    queue_priority: 0
    auto_retries: true
    force_stream: true
    model_fallback_enabled: false
    features:
      - model_sampling_defaults
      - drop_oai_only_fields
      - effort_to_thinking_budget
      - thinking_overflow_recovery
      - silent_completion_recovery
      - truncated_content_recovery
      - empty_with_stop_retry
      - gemma_thought_leak_retry

  myproject:
    upstream: nan
    queue_priority: 5
    auto_retries: true
    force_stream: true
    model_fallback_enabled: false
    features:
      - model_sampling_defaults
      - drop_oai_only_fields
      - effort_to_thinking_budget
      - thinking_overflow_recovery
      - silent_completion_recovery
      - truncated_content_recovery
      - empty_with_stop_retry
      - gemma_thought_leak_retry
    disabled_features: []

default_profile: default
```

The same file is available at [examples/config.yaml](examples/config.yaml).

## Profile Options

Common profile fields:

| Field | Meaning |
| --- | --- |
| `upstream` | Upstream name from `upstreams` |
| `features` | Transform/recovery features enabled for the profile |
| `disabled_features` | Explicitly disable default-on features, currently `gemma_thought_leak_retry` |
| `queue_priority` | Higher values jump ahead in the upstream queue |
| `auto_retries` | Retry transient upstream failures before bytes reach the client |
| `force_stream` | Send `stream=true` upstream even when the client did not |
| `model_fallback_enabled` | Fallback to another active model when the selected model is unhealthy or fails before bytes reach the client |
| `force_model` | Optional hard model override, e.g. `qwen3.6` or `gemma4` |
| `thinking_enabled` | `true`, `false`, or omitted to respect client/upstream defaults |
| `default_thinking_effort` | Optional `low`, `medium`, `high`, or `xhigh` when profile enables thinking by default |
| `default_max_output_tokens` | Fill output cap only when the client is silent |
| `force_max_output_tokens` | Hard override for output cap |
| `force_temperature` | Hard temperature override |
| `force_top_p` | Hard top-p override |
| `force_presence_penalty` | Hard presence-penalty override |
| `model_aliases` | Optional client-model-id to upstream-model-id mapping |

## Model Health And Fallback

The bridge probes configured NaN models once per minute with a small `ping` chat request and thinking disabled. A model is marked active only when it returns bytes within `30` seconds; otherwise it is treated as inactive. The status is exposed in `/health` and `/stats`.

Set `model_fallback_enabled: true` on a profile to use the health table. If the selected model is inactive and another configured model is active, the bridge rewrites the request to the active model. If every model is inactive, it keeps the requested model.

Fallback can also trigger before client-visible bytes are sent when the upstream returns a retryable failure such as `524` or times out waiting for the first byte. Health checks need an auth token from `X_NAN_KEY`, `NAN_API_KEY`, `OPENAI_API_KEY`, or an env file pointed to by `BRIDGE_AUTH_ENV_PATH`.

## Features Reference

| Feature | What it does |
| --- | --- |
| `model_sampling_defaults` | Inject model-aware sampling defaults when the client is silent |
| `drop_oai_only_fields` | Remove OpenAI-only fields the upstream rejects |
| `effort_to_thinking_budget` | Translate `reasoning_effort` / `reasoning.effort` into model-specific thinking fields; Gemma4 gets `enable_thinking` without an invented budget |
| `thinking_overflow_recovery` | Recover when reasoning consumes the output budget before a final answer |
| `silent_completion_recovery` | Recover completed responses that contain no useful message |
| `truncated_content_recovery` | Continue answers that were cut by length mid-content |
| `empty_with_stop_retry` | Retry once when the upstream returns empty content with `finish_reason=stop` |
| `tool_call_args_retry` | Retry with thinking disabled when tool-call arguments miss required schema fields |
| `xml_tool_residue_retry` | Retry when XML tool-call templates leak into text/reasoning |
| `gemma_thought_leak_retry` | Retry Gemma post-tool turns with thinking disabled when hidden thought leaks into visible content; enabled by default unless disabled per profile |

## Client Examples

Any OpenAI-compatible client can point at a profile URL.

Hermes-style:

```yaml
model:
  base_url: http://127.0.0.1:4242/hermes/v1
```

opencode-style:

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

Project-specific client:

```text
http://127.0.0.1:4242/myproject/v1
```

Do not commit real API keys. Use environment variables such as `X_NAN_KEY` in your client config.

## Dashboard

The dashboard at `/` shows:

- request count and success rate by time window
- tokens in/out
- latency percentiles
- recovery counts
- active in-flight requests
- recent requests with model/profile/path/status
- per-model stats
- upstream queue and concurrency state
- editable local profiles

Recent activity respects the selected time window: `1m`, `5m`, `15m`, `1h`, or `all`.

## Running As A systemd User Service

```bash
mkdir -p ~/.config/resilient-llm-bridge
cp bridge.py ~/.config/resilient-llm-bridge/

mkdir -p ~/.config/systemd/user
cp systemd/resilient-llm-bridge.service ~/.config/systemd/user/

systemctl --user daemon-reload
systemctl --user enable --now resilient-llm-bridge.service
journalctl --user -u resilient-llm-bridge.service -f
```

If you want the service to survive logout:

```bash
loginctl enable-linger "$USER"
```

## Environment Variables

| Var | Default | Notes |
| --- | --- | --- |
| `BRIDGE_CONFIG_PATH` | `~/.config/resilient-llm-bridge/config.yaml` | Config file path |
| `PORT` | `4242` | Bind port |
| `HOST` | `127.0.0.1` | Bind host |
| `LOG_LEVEL` | `info` | uvicorn log level |
| `BRIDGE_USAGE_RING_SIZE` | `1000` | In-memory usage rows |
| `BRIDGE_ACTIVITY_RING_SIZE` | `1000` | In-memory activity rows |

## Security Notes

- The repo should not contain API keys, bearer tokens, local private profiles, or personal configs.
- Keep real configs in `~/.config/resilient-llm-bridge/config.yaml`.
- Keep keys in environment variables or your secret manager.
- The bridge forwards `Authorization` headers to the upstream; do not expose it publicly without your own network controls.
- `BRIDGE_NO_REDACT=1` is useful for local debugging but can store full request bodies in activity history. Do not enable it for shared or public deployments.

## License

MIT. See [LICENSE](LICENSE).
