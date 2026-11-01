#!/bin/sh
# Install (or reinstall) the bridge launchd agent for the current user. Reads bridge/.env.
set -eu
BRIDGE_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$HOME/Library/Logs"
label=space.hermes.stackchan-bridge
plist="$HOME/Library/LaunchAgents/$label.plist"

[ -f "$BRIDGE_DIR/.env" ] || { echo "missing $BRIDGE_DIR/.env (copy .env.example)" >&2; exit 1; }
(cd "$BRIDGE_DIR" && uv sync --frozen)
sed -e "s|__BRIDGE_DIR__|$BRIDGE_DIR|g" -e "s|__LOG_DIR__|$LOG_DIR|g" \
    "$BRIDGE_DIR/launchd/$label.plist" > "$plist"
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
while launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1; do sleep 0.5; done  # bootout is async
launchctl bootstrap "gui/$(id -u)" "$plist"
echo "loaded $label"
