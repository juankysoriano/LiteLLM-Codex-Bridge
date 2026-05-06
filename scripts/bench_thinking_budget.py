# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""
Benchmark `thinking_token_budget` enforcement on a Qwen3-Thinking upstream.

Runs the same prompt at a sweep of budget values and reports per-run:
  - completion_tokens            (the upstream's count of all output tokens)
  - reasoning_tokens / reasoning_chars
  - answer_tokens / answer_chars
  - duration                     wall-clock seconds
  - tokens/s                     simple completion_tokens / duration

If the upstream is honoring `thinking_token_budget` correctly, you should
see `reasoning_tokens` track the requested budget within ~5%. If reasoning
keeps roughly the same value regardless of budget, the cap isn't being
enforced (likely vLLM running without `--reasoning-parser qwen3` or with
MTP speculative decoding hitting issue #39573).

Usage:
  X_NAN_KEY=<your-key> uv run bench_thinking_budget.py
  X_NAN_KEY=<your-key> uv run bench_thinking_budget.py --target https://api.nan.builders/v1
  X_NAN_KEY=<your-key> uv run bench_thinking_budget.py --budgets 128 1024 8192

By default it hits NaN Builders directly (bypassing any local bridge) so
you measure the upstream behavior cleanly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import httpx


DEFAULT_PROMPT = (
    "Plan a complete strategy to refactor a 50k LOC monolith into "
    "microservices. Think very carefully step by step before answering, "
    "considering all architectural concerns: data flow, transactions, "
    "observability, deployment topology, rollback strategy."
)
DEFAULT_BUDGETS = [128, 1024, 4096, 8192, 16384]
DEFAULT_MAX_TOKENS = 24000
THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
APPROX_CHARS_PER_TOKEN = 4.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--target",
        default="https://api.nan.builders/v1",
        help="Upstream base URL (default: https://api.nan.builders/v1)",
    )
    p.add_argument(
        "--model",
        default="qwen3.6",
        help="Model id (default: qwen3.6)",
    )
    p.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=DEFAULT_BUDGETS,
        help=f"Budgets to test in tokens (default: {DEFAULT_BUDGETS})",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"max_tokens cap for the request (default: {DEFAULT_MAX_TOKENS})",
    )
    p.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send (default: monolith refactor question)",
    )
    p.add_argument(
        "--baseline",
        action="store_true",
        help="Also run a baseline request without any thinking budget for comparison",
    )
    p.add_argument(
        "--include-fixed",
        type=str,
        choices=["false", "true"],
        default="false",
        help="Include enable_thinking=false reference run (default: false; instant baseline)",
    )
    return p.parse_args()


def split_reasoning_and_answer(message: dict) -> tuple[str, str]:
    """Extract (reasoning_text, answer_text) from an OpenAI chat message.

    Tries `reasoning_content` first (the field vLLM exposes when
    `--reasoning-parser` is configured). Falls back to parsing
    `<think>...</think>` blocks from the content (what you get when no
    reasoning parser is configured — Qwen emits the tags inline).
    """
    reasoning = (message.get("reasoning_content") or "").strip()
    raw_content = (message.get("content") or "")
    if reasoning:
        return reasoning, raw_content.strip()
    matches = THINK_TAG_RE.findall(raw_content)
    if matches:
        reasoning = "\n".join(m.strip() for m in matches)
        answer = THINK_TAG_RE.sub("", raw_content).strip()
        return reasoning, answer
    # Open <think> with no closing — model ran out of tokens during thinking.
    if "<think>" in raw_content:
        idx = raw_content.find("<think>")
        return raw_content[idx + len("<think>"):].strip(), ""
    return "", raw_content.strip()


def run_one(
    client: httpx.Client,
    *,
    target: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    budget: int | None,
    enable_thinking: bool = True,
) -> dict:
    body: dict = {
        "model": model,
        "stream": False,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
    }
    if budget is not None:
        body["extra_body"]["thinking_token_budget"] = budget

    label = f"budget={budget}" if budget is not None else "no-budget"
    if not enable_thinking:
        label = "thinking=off"

    t0 = time.monotonic()
    try:
        r = client.post(
            f"{target}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=600,
        )
    except httpx.HTTPError as e:
        return {"label": label, "error": str(e)}
    duration = time.monotonic() - t0

    if r.status_code >= 400:
        return {
            "label": label,
            "status": r.status_code,
            "error": r.text[:200],
            "duration_s": duration,
        }
    data = r.json()
    usage = data.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    finish_reason = (data.get("choices") or [{}])[0].get("finish_reason")
    message = (data.get("choices") or [{}])[0].get("message") or {}
    reasoning_text, answer_text = split_reasoning_and_answer(message)

    reasoning_chars = len(reasoning_text)
    answer_chars = len(answer_text)
    # Approx token estimate when the upstream doesn't break it down.
    reasoning_tokens_est = round(reasoning_chars / APPROX_CHARS_PER_TOKEN)
    answer_tokens_est = round(answer_chars / APPROX_CHARS_PER_TOKEN)
    tps = (completion_tokens / duration) if duration > 0 else 0.0

    return {
        "label": label,
        "budget": budget,
        "duration_s": round(duration, 2),
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "reasoning_chars": reasoning_chars,
        "reasoning_tokens_est": reasoning_tokens_est,
        "answer_chars": answer_chars,
        "answer_tokens_est": answer_tokens_est,
        "finish_reason": finish_reason,
        "tokens_per_s": round(tps, 1),
    }


def print_table(rows: list[dict]) -> None:
    cols = [
        ("label", 16),
        ("duration_s", 10),
        ("completion_tokens", 10),
        ("reasoning_tokens_est", 10),
        ("answer_tokens_est", 10),
        ("tokens_per_s", 10),
        ("finish_reason", 16),
    ]
    headers = "  ".join(f"{name:>{width}}" for name, width in cols)
    print(headers)
    print("  ".join("-" * width for _, width in cols))
    for row in rows:
        if "error" in row:
            label = row.get("label", "?")
            err = row["error"]
            print(f"{label:>16}  ERROR: {err[:80]}")
            continue
        line = "  ".join(
            f"{str(row.get(name, '—')):>{width}}" for name, width in cols
        )
        print(line)


def diagnose(rows: list[dict]) -> str:
    """Decide whether the upstream is honoring thinking_token_budget."""
    budgeted = [r for r in rows if r.get("budget") is not None and "error" not in r]
    if len(budgeted) < 2:
        return "Not enough successful budget runs to diagnose."
    reasoning_values = [r["reasoning_tokens_est"] for r in budgeted]
    diff = max(reasoning_values) - min(reasoning_values)
    avg = sum(reasoning_values) / len(reasoning_values)
    spread_ratio = diff / avg if avg > 0 else 0
    if spread_ratio < 0.25:
        return (
            f"Budget IGNORED: reasoning tokens are basically constant "
            f"(spread {spread_ratio:.0%} of mean) regardless of budget value. "
            f"Likely vLLM lacks --reasoning-parser qwen3 or has MTP enabled "
            f"(see vllm-project/vllm#39573)."
        )
    # Check correlation with budget
    sorted_b = sorted(budgeted, key=lambda r: r["budget"])
    monotonic = all(
        sorted_b[i + 1]["reasoning_tokens_est"] >= sorted_b[i]["reasoning_tokens_est"] - 100
        for i in range(len(sorted_b) - 1)
    )
    if monotonic:
        return (
            f"Budget HONORED: reasoning tokens roughly track requested budget "
            f"(spread {spread_ratio:.0%}). The cap is being enforced upstream."
        )
    return (
        f"Budget partially honored / noisy (spread {spread_ratio:.0%}). "
        f"Re-run a few times to confirm."
    )


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("X_NAN_KEY") or os.environ.get("UPSTREAM_API_KEY")
    if not api_key:
        print("Set X_NAN_KEY (or UPSTREAM_API_KEY) in your env.", file=sys.stderr)
        return 1

    target = args.target.rstrip("/")
    print(f"target: {target}")
    print(f"model:  {args.model}")
    print(f"prompt: {args.prompt[:80]}{'…' if len(args.prompt) > 80 else ''}")
    print(f"max_tokens: {args.max_tokens}")
    print()

    rows: list[dict] = []
    with httpx.Client() as client:
        if args.baseline:
            print("running: baseline (no budget) ...", flush=True)
            rows.append(
                run_one(
                    client,
                    target=target,
                    api_key=api_key,
                    model=args.model,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    budget=None,
                )
            )
        for b in args.budgets:
            print(f"running: budget={b} ...", flush=True)
            rows.append(
                run_one(
                    client,
                    target=target,
                    api_key=api_key,
                    model=args.model,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    budget=b,
                )
            )
        if args.include_fixed == "true":
            print("running: enable_thinking=false (reference) ...", flush=True)
            rows.append(
                run_one(
                    client,
                    target=target,
                    api_key=api_key,
                    model=args.model,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    budget=None,
                    enable_thinking=False,
                )
            )

    print()
    print_table(rows)
    print()
    print("diagnosis:", diagnose(rows))
    print()
    print("(JSON dump for tooling:)")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
