#!/bin/sh
# Speak text on the connected StackChan:  ./announce.sh "早晨，今日..."
# Reads the token from bridge/.env so callers (Hermes cron) never hold it.
set -eu
[ $# -eq 1 ] && [ -n "$1" ] || { echo "usage: $0 TEXT" >&2; exit 2; }
BRIDGE_DIR="$(cd "$(dirname "$0")" && pwd)"
TOKEN="$(sed -n 's/^BRIDGE_DEVICE_TOKEN=//p' "$BRIDGE_DIR/.env" | tr -d "\"'")"
PORT="$(sed -n 's/^BRIDGE_PORT=//p' "$BRIDGE_DIR/.env")"
curl -sS --fail-with-body --get "http://127.0.0.1:${PORT:-8765}/announce" \
  -H "Authorization: Bearer $TOKEN" --data-urlencode "text=$1"
