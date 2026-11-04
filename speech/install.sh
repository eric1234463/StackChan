#!/bin/sh
# Install (or reinstall) the STT/TTS launchd agents for the current user.
#   TTS_MODEL_DIR=/path/to/qwen3-tts-...-8bit [TTS_VOICE=eric] [TTS_SPEED=1.15] \
#     [STT_ENGINE=qwen3-asr|whisper] [STT_MODEL=hf-repo-or-dir] ./install.sh
set -eu
: "${TTS_MODEL_DIR:?set TTS_MODEL_DIR to the merged Qwen3-TTS HK Cantonese MLX model directory}"
STT_ENGINE="${STT_ENGINE:-qwen3-asr}"
if [ "$STT_ENGINE" = whisper ]; then
  STT_MODEL="${STT_MODEL:-WinsonLi/whisper-large-v3-cantonese-mlx-4bit}"
else
  STT_MODEL="${STT_MODEL:-$HOME/models/liujgoj-qwen3-asr-1.7b-mlx-8bit}"
fi
TTS_VOICE="${TTS_VOICE:-hk_voice_343}"
TTS_SPEED="${TTS_SPEED:-1.0}"
SPEECH_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$HOME/Library/Logs"
AGENTS="$HOME/Library/LaunchAgents"

(cd "$SPEECH_DIR" && uv sync --frozen)
for template in "$SPEECH_DIR"/launchd/*.plist; do
  label="$(basename "$template" .plist)"
  sed -e "s|__SPEECH_DIR__|$SPEECH_DIR|g" -e "s|__LOG_DIR__|$LOG_DIR|g" \
      -e "s|__TTS_MODEL_DIR__|$TTS_MODEL_DIR|g" -e "s|__STT_MODEL__|$STT_MODEL|g" -e "s|__STT_ENGINE__|$STT_ENGINE|g" -e "s|__TTS_VOICE__|$TTS_VOICE|g" -e "s|__TTS_SPEED__|$TTS_SPEED|g" \
      "$template" > "$AGENTS/$label.plist"
  launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
  while launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1; do sleep 0.5; done  # bootout is async
  launchctl bootstrap "gui/$(id -u)" "$AGENTS/$label.plist"
  echo "loaded $label"
done
