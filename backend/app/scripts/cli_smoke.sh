#!/usr/bin/env bash
# Drive the real CLI through plain stdin, twice, as two separate processes:
# the second run is a restart that resumes the first run's conversation.
#
#   bash scripts/cli_smoke.sh [user]
#
# Uses the real model, database and corpus; conversations are created for
# the given user (default atlas-e2e) so they are easy to find and remove.
set -euo pipefail
cd "$(dirname "$0")/.."
USER_ID="${1:-atlas-e2e}"

echo "=== run 1: new conversation, status commands ==="
printf '%s\n' \
  "status --time" \
  "status --retries" \
  "How does write-ahead logging help crash recovery?" \
  "status --time" \
  "status --retries" \
  "status" \
  "exit" \
  | uv run python main.py --no-acquire --user "$USER_ID" start

CONVERSATION=$(uv run python - "$USER_ID" <<'EOF'
import sys
from memory.conversation_registry import open_conversation_registry
print(open_conversation_registry().list(sys.argv[1])[0].conversation_id)
EOF
)

echo
echo "=== run 2 (restart): switch to $CONVERSATION and continue it ==="
printf '%s\n' \
  "switch" \
  "$CONVERSATION" \
  "And what is a checkpoint in that process?" \
  "status --time" \
  "exit" \
  | uv run python main.py --no-acquire --user "$USER_ID" start
