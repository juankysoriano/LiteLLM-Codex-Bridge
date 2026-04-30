#!/usr/bin/env python3
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
"""Unit-style tests for the thinking-injection policy in bridge._apply_request_transforms.

No HTTP, no upstream — just exercise the in-process transform with
synthetic profiles and assert the resulting body matches policy.

Run:
    uv run scripts/test_thinking_policy.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bridge


def make_profile(
    name: str,
    *,
    thinking_enabled: bool | None = None,
    default_thinking_budget: int | None = None,
    features: list[str] | None = None,
) -> bridge.ProfileConfig:
    return bridge.ProfileConfig(
        name=name,
        upstream="nan",
        features=set(features or ["effort_to_thinking_budget"]),
        thinking_enabled=thinking_enabled,
        default_thinking_budget=default_thinking_budget,
    )


def transform(body: dict, profile: bridge.ProfileConfig) -> dict:
    return bridge._apply_request_transforms(copy.deepcopy(body), profile, kind="chat_completions")


def fail(label: str, msg: str) -> None:
    print(f"  ✗ {label}: {msg}")
    raise AssertionError(f"[{label}] {msg}")


def ok(label: str, detail: str = "") -> None:
    print(f"  ✓ {label}{(' — ' + detail) if detail else ''}")


# ----------------------------------------------------------------------------
# Cases
# ----------------------------------------------------------------------------
def case_named_profile_no_signals() -> None:
    """Named profile (no thinking config), client sends nothing → bridge
    injects nothing. Pure passthrough — upstream sees no thinking
    fields → upstream defaults to no thinking."""
    label = "named profile + no client signals"
    profile = make_profile("hermes-like")  # defaults: thinking_enabled=False, budget=None
    body = {"model": "qwen3.6", "messages": [{"role": "user", "content": "hi"}]}
    out = transform(body, profile)
    eb = out.get("extra_body") or {}
    ctk = eb.get("chat_template_kwargs") or {}
    if "enable_thinking" in ctk:
        fail(label, f"expected NO enable_thinking key, got {ctk!r}")
    if "thinking_token_budget" in eb:
        fail(label, f"expected NO budget, got {eb.get('thinking_token_budget')}")
    if "max_tokens" in out:
        fail(label, f"expected NO max_tokens, got {out['max_tokens']}")
    ok(label, "no overrides — pure passthrough")


def case_default_profile_no_signals() -> None:
    """`default`-style profile (thinking_enabled=true, budget=4096), client
    sends nothing → bridge injects both."""
    label = "default profile + no client signals"
    profile = make_profile("default-like", thinking_enabled=True, default_thinking_budget=4096)
    body = {"model": "qwen3.6", "messages": [{"role": "user", "content": "hi"}]}
    out = transform(body, profile)
    eb = out.get("extra_body") or {}
    if eb.get("thinking_token_budget") != 4096:
        fail(label, f"expected budget=4096, got {eb.get('thinking_token_budget')}")
    if eb.get("chat_template_kwargs", {}).get("enable_thinking") is not True:
        fail(label, f"expected enable_thinking=True, got {eb.get('chat_template_kwargs')}")
    ok(label, "forced thinking + budget=4096")


def case_named_profile_client_effort_high() -> None:
    """Named profile (no overrides), client sends reasoning_effort=high.
    Bridge translates → budget=8192. enable_thinking is NOT injected
    (profile says no override) but upstream activates thinking
    implicitly via budget alone (verified empirically)."""
    label = "named profile + client effort=high"
    profile = make_profile("opencode-like")
    body = {
        "model": "qwen3.6",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "high",
    }
    out = transform(body, profile)
    eb = out.get("extra_body") or {}
    if eb.get("thinking_token_budget") != 8192:
        fail(label, f"expected budget=8192 (translated from high), got {eb.get('thinking_token_budget')}")
    ctk = eb.get("chat_template_kwargs") or {}
    if "enable_thinking" in ctk:
        fail(label, f"expected NO enable_thinking key (no override), got {ctk!r}")
    ok(label, "effort=high → budget=8192, no enable_thinking key")


def case_named_profile_client_responses_effort() -> None:
    """Named profile + client reasoning.effort=low (responses-API)."""
    label = "named profile + client reasoning.effort=low"
    profile = make_profile("codex-like")
    body = {
        "model": "qwen3.6",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning": {"effort": "low"},
    }
    out = transform(body, profile)
    eb = out.get("extra_body") or {}
    if eb.get("thinking_token_budget") != 2048:
        fail(label, f"expected budget=2048 (low), got {eb.get('thinking_token_budget')}")
    ok(label, "reasoning.effort=low → budget=2048")


def case_default_profile_client_disable_via_enable_thinking() -> None:
    """Profile authoritative: thinking_enabled=True overrides client's
    enable_thinking=False. Profile wins."""
    label = "default profile (forced) + client enable_thinking=False"
    profile = make_profile("default-like", thinking_enabled=True, default_thinking_budget=4096)
    body = {
        "model": "qwen3.6",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    out = transform(body, profile)
    eb = out.get("extra_body") or {}
    if eb.get("chat_template_kwargs", {}).get("enable_thinking") is not True:
        fail(label, f"expected profile to force enable_thinking=True, got {eb.get('chat_template_kwargs')}")
    if eb.get("thinking_token_budget") != 4096:
        fail(label, f"expected profile budget=4096, got {eb.get('thinking_token_budget')}")
    ok(label, "profile overrides client disable signal")


def case_default_profile_client_disable_via_effort_none() -> None:
    """Profile authoritative: client effort=none ignored when profile
    forces thinking on. Translation only happens when profile is silent."""
    label = "default profile (forced) + client effort=none"
    profile = make_profile("default-like", thinking_enabled=True, default_thinking_budget=4096)
    body = {
        "model": "qwen3.6",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "none",
    }
    out = transform(body, profile)
    eb = out.get("extra_body") or {}
    if eb.get("chat_template_kwargs", {}).get("enable_thinking") is not True:
        fail(label, f"expected profile-forced enable_thinking=True, got {eb.get('chat_template_kwargs')}")
    if eb.get("thinking_token_budget") != 4096:
        fail(label, f"expected profile budget=4096, got {eb.get('thinking_token_budget')}")
    ok(label, "profile overrides effort=none")


def case_default_profile_client_set_budget_explicit() -> None:
    """Profile authoritative: client budget=512 ignored when profile
    sets default_thinking_budget=4096. Profile wins."""
    label = "default profile (forced) + client extra_body.thinking_token_budget=512"
    profile = make_profile("default-like", thinking_enabled=True, default_thinking_budget=4096)
    body = {
        "model": "qwen3.6",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_body": {"thinking_token_budget": 512},
    }
    out = transform(body, profile)
    eb = out.get("extra_body") or {}
    if eb.get("thinking_token_budget") != 4096:
        fail(label, f"expected profile budget=4096 to override client 512, got {eb.get('thinking_token_budget')}")
    if eb.get("chat_template_kwargs", {}).get("enable_thinking") is not True:
        fail(label, f"expected enable_thinking=True (profile authoritative), got {eb.get('chat_template_kwargs')}")
    ok(label, "profile budget overrides client budget")


def case_named_profile_client_disable_respected() -> None:
    """Named profile (no override): client says enable_thinking=false →
    bridge respects, forwards as false."""
    label = "named profile + client enable_thinking=False"
    profile = make_profile("hermes-like")
    body = {
        "model": "qwen3.6",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    out = transform(body, profile)
    eb = out.get("extra_body") or {}
    if eb.get("chat_template_kwargs", {}).get("enable_thinking") is not False:
        fail(label, f"expected client False to pass through, got {eb.get('chat_template_kwargs')}")
    ok(label, "named profile respects client disable")


def case_named_profile_client_effort_none_respected() -> None:
    """Named profile (no override): client effort=none → bridge sets
    enable_thinking=False (translation of disable signal)."""
    label = "named profile + client effort=none"
    profile = make_profile("hermes-like")
    body = {
        "model": "qwen3.6",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "none",
    }
    out = transform(body, profile)
    eb = out.get("extra_body") or {}
    if eb.get("chat_template_kwargs", {}).get("enable_thinking") is not False:
        fail(label, f"expected enable_thinking=False (translated), got {eb.get('chat_template_kwargs')}")
    if "thinking_token_budget" in eb:
        fail(label, f"expected no budget when disabled, got {eb.get('thinking_token_budget')}")
    ok(label, "effort=none translated to disable")


def case_named_profile_client_set_budget_explicit() -> None:
    """On a named profile (no overrides), client-supplied budget is
    respected as-is."""
    label = "named profile + client extra_body.thinking_token_budget=512"
    profile = make_profile("hermes-like")
    body = {
        "model": "qwen3.6",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_body": {"thinking_token_budget": 512},
    }
    out = transform(body, profile)
    eb = out.get("extra_body") or {}
    if eb.get("thinking_token_budget") != 512:
        fail(label, f"expected client budget=512, got {eb.get('thinking_token_budget')}")
    ok(label, "named profile preserves client budget")


def case_no_max_tokens_injection() -> None:
    """Bridge no longer inflates max_tokens. With no client max_tokens,
    none should appear in the forwarded body."""
    label = "no max_tokens injection"
    for profile_args in [
        {"thinking_enabled": True, "default_thinking_budget": None},
        {"thinking_enabled": True, "default_thinking_budget": 4096},
        {"thinking_enabled": False, "default_thinking_budget": None},
    ]:
        profile = make_profile("p", **profile_args)
        body = {"model": "qwen3.6", "messages": [{"role": "user", "content": "hi"}]}
        out = transform(body, profile)
        if "max_tokens" in out:
            fail(label, f"profile={profile_args} unexpected max_tokens={out['max_tokens']}")
    ok(label, "verified across 3 profile configs")


def case_client_max_tokens_clamp_only() -> None:
    """When client sends max_tokens, _clamp_max_tokens_to_context still
    applies but only against context window, no budget addition."""
    label = "client max_tokens preserved (within context)"
    profile = make_profile("p", default_thinking_budget=4096)
    body = {
        "model": "qwen3.6",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 12345,
    }
    out = transform(body, profile)
    if out.get("max_tokens") != 12345:
        fail(label, f"expected 12345 preserved, got {out.get('max_tokens')}")
    ok(label, "max_tokens=12345 unchanged (no budget inflation)")


def main() -> int:
    cases = [
        case_named_profile_no_signals,
        case_default_profile_no_signals,
        case_named_profile_client_effort_high,
        case_named_profile_client_responses_effort,
        case_default_profile_client_disable_via_enable_thinking,
        case_default_profile_client_disable_via_effort_none,
        case_default_profile_client_set_budget_explicit,
        case_named_profile_client_set_budget_explicit,
        case_named_profile_client_disable_respected,
        case_named_profile_client_effort_none_respected,
        case_no_max_tokens_injection,
        case_client_max_tokens_clamp_only,
    ]
    print(f"Running {len(cases)} thinking-policy cases...")
    failed = 0
    for fn in cases:
        try:
            fn()
        except AssertionError:
            failed += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: unexpected {type(e).__name__}: {e}")
            failed += 1
    print()
    if failed:
        print(f"FAILED: {failed}/{len(cases)}")
        return 1
    print(f"All {len(cases)} cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
