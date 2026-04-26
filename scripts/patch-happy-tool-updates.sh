#!/usr/bin/env bash
#
# patch-happy-tool-updates.sh — re-apply our patch to happy's bundled
# AcpBackend so subsequent tool_call_update events get re-emitted instead
# of silently dropped (the upstream behaviour leaves opencode tool calls
# stuck on `args: {locations: []}` forever, since the actual file paths
# only arrive on later updates).
#
# Run after every `npm i -g happy@<version>`. Idempotent: rerunning the
# script when the patch is already applied is a no-op.
#
# What it changes:
#   ~/.npm-global/lib/node_modules/happy/dist/AcpBackend-*.{mjs,cjs}
#
# In each, the function `handleToolCallUpdate` has an `else` branch that
# logs "already tracked, skipping" and emits nothing. We replace that
# with a re-call to startToolCall so the new args/locations get emitted
# downstream as a fresh tool-call event. The HappyNativePanel ToolCard
# merge keeps the latest args (so the timeline shows the latest paths).
#
set -euo pipefail

DIST_DIR="${HAPPY_DIST_DIR:-$HOME/.npm-global/lib/node_modules/happy/dist}"
if [ ! -d "$DIST_DIR" ]; then
  echo "[patch] $DIST_DIR not found — is happy installed via npm-global?" >&2
  exit 1
fi

# Sentinel so we can detect already-patched files without diff-ing.
SENTINEL="LITELLM_BRIDGE_PATCH_TOOL_UPDATES"

patch_one() {
  local file="$1"
  if grep -q "$SENTINEL" "$file"; then
    echo "[patch] $(basename "$file"): already patched, skipping"
    return 0
  fi

  # Original snippet (with normal whitespace) — both .mjs and .cjs share
  # the same source modulo formatting:
  #
  #   } else {
  #     logger.debug(`[AcpBackend] Tool call ${toolCallId} already tracked, status: ${status}`);
  #   }
  #
  # We replace the body of the `else` with a startToolCall re-call.
  python3 - "$file" "$SENTINEL" <<'PY'
import re, sys
path, sentinel = sys.argv[1:]
src = open(path, "r", encoding="utf-8").read()
# .mjs uses `logger.debug(...)`; .cjs uses `api.logger.debug(...)` — match both.
needle = re.compile(
    r"\} else \{\s*"
    r"(?:api\.)?logger\.debug\(`\[AcpBackend\] Tool call \$\{toolCallId\} already tracked,"
    r' status: \$\{status\}`\);\s*\}',
    re.DOTALL,
)
m = needle.search(src)
if not m:
    print(f"[patch] {path}: did not find handleToolCallUpdate else-branch — skipping", file=sys.stderr)
    sys.exit(2)
replacement = (
    "} else {\n"
    "      // " + sentinel + ": re-emit on subsequent updates so the UI sees\n"
    "      // updated args/locations from opencode (and other ACP agents).\n"
    "      startToolCall(toolCallId, toolKind, update, ctx, \"tool_call_update_continued\");\n"
    "    }"
)
patched = src[:m.start()] + replacement + src[m.end():]
open(path, "w", encoding="utf-8").write(patched)
print(f"[patch] {path}: applied")
PY
}

shopt -s nullglob
patched_any=0
for f in "$DIST_DIR"/AcpBackend-*.mjs "$DIST_DIR"/AcpBackend-*.cjs; do
  patch_one "$f"
  patched_any=1
done

if [ "$patched_any" -eq 0 ]; then
  echo "[patch] no AcpBackend-*.{mjs,cjs} files in $DIST_DIR" >&2
  exit 1
fi

echo "[patch] done. Restart happy daemon to pick up: systemctl --user restart happy-daemon (or kill its PID)."
