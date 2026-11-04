# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of `m5stack/StackChan` turned into a Cantonese voice front-end for the local Hermes agent:

```
"喂Eric" (on-device microWakeWord) → Opus over xiaozhi WebSocket → bridge/ on the Mac
  → Wyoming STT :10300 → Hermes api_server :8642 /v1/responses → Wyoming TTS :10201 → Opus back → StackChan speaks
```

All AI runs on the Mac; the robot only does audio I/O, wake word, screen, servos. `spec.md` is the design doc
(decisions D1–D6, acceptance criteria, open questions). The code has moved on from it in places (e.g. Hermes is
called non-streaming, conversation name gets a daily suffix, ack phrase + follow-up window) — trust the code.

Our work lives in `bridge/`, `speech/`, `voice/`, and the Hermes parts of `firmware/`. `app/`, `remote/`, `server/`
are upstream and out of scope (spec non-goal) — don't modify them.

## Components and commands

Python projects use `uv` (each has its own `pyproject.toml` / `uv.lock`; `pythonpath = ["."]`).

| Dir | Purpose | Test |
|---|---|---|
| `bridge/` | xiaozhi WS server (`stackchan_bridge`), the core of the system | `cd bridge && uv run pytest` |
| `speech/` | Wyoming STT (`mlx-whisper` Cantonese 4-bit; optional Qwen3-ASR Liujgoj romanization) and TTS (Qwen3-TTS HK Cantonese, MLX 8-bit) servers; Apple Silicon only | `cd speech && uv run pytest` |
| `voice/` | Speaker dataset recorder (`record.py`) + speaker-LoRA training (`train/train.sh`, MPS) for the `eric` TTS voice | `cd voice && uv run pytest` |
| `firmware/` | ESP-IDF v5.5.4 firmware for M5Stack CoreS3 | host tests below |

Single test: `uv run pytest tests/test_session.py::test_name -x`.

Services run as launchd agents; install scripts render the plist templates in `*/launchd/` and (re)load them:

- `bridge/install.sh` — needs `bridge/.env` (copy `.env.example`; `BRIDGE_DEVICE_TOKEN`, `HERMES_API_KEY` required).
  Label `space.hermes.stackchan-bridge`, log `~/Library/Logs/stackchan-bridge.log`. Run locally: `uv run python -m stackchan_bridge`.
- `speech/install.sh` — `TTS_MODEL_DIR=... [TTS_VOICE=eric] [TTS_SPEED=...] [STT_MODEL=...] ./install.sh`.
  `STT_ENGINE=whisper|qwen3-asr`. Labels `space.hermes.whisper-yue` (either engine) / `space.hermes.canto-tts`, logs `~/Library/Logs/stackchan-{stt,tts}.log`.
  These Wyoming servers are also used by Home Assistant: keep their ports and protocol compatible.
- `bridge/announce.sh "TEXT"` — make the connected robot speak unprompted (`GET /announce` with the device token).

Firmware:

```sh
cd firmware
python3 ./fetch_repos.py          # clones deps from repos.json, applies patches/xiaozhi-esp32.patch to xiaozhi-esp32/
idf.py reconfigure build          # reconfigure is required after editing sdkconfig.defaults.local
idf.py erase-flash flash          # first install: NVS from older firmware must not win over baked config
# host-side unit tests (no hardware):
cmake -S tests -B build-host-tests && cmake --build build-host-tests && ctest --test-dir build-host-tests --output-on-failure
```

## Architecture notes

### Bridge (`bridge/stackchan_bridge/`)

- `server.py` — one `Session` per connected device. Speaks the upstream xiaozhi WS protocol unchanged (JSON
  `hello`/`listen`/`stt`/`llm`/`tts`/`abort`/`mcp` + binary Opus; uplink 16 kHz, downlink 24 kHz, 60 ms frames).
  Bridge-side VAD ends utterances (end-of-speech silence, speculative STT started earlier and reused if speech
  doesn't resume), filters noise/empty transcripts, plays an ack phrase while Hermes works, then streams reply
  sentences paced to real time, synthesising sentence N+1 while N plays. After a reply the device re-listens for a
  short follow-up window; an acknowledging follow-up (`text.is_closing`) sends the device to idle.
- `Backends` dataclass (transcribe/ask/synthesize) is the seam for tests: `tests/test_session.py` runs a real
  WebSocket against fake backends; `tests/test_units.py` covers text/audio/config helpers.
- `services.py` — Wyoming STT/TTS clients and the Hermes `/v1/responses` call; all failures become `ServiceError`.
- `text.py` — cleaning (markdown/URL/emoji) and sentence splitting before TTS; `audio.py` — Opus, resampling, VAD.
- `config.py` — `Config` dataclass with defaults; loaded from env + `bridge/.env`. Tuned values carry comments with
  the measurement that justified them — keep that convention when changing them.
- The device must never hang: every failure path ends with an `alert`/`tts stop`/`system idle` message.
- The bridge also sends custom `llm` emotions the patched firmware interprets as gestures/screens (`nod`, `working`)
  and `system` commands (`idle`, `keepalive`). Changing these requires matching firmware changes.

### Firmware

- AI runtime is `xiaozhi-esp32` v2.2.4, fetched (not committed, gitignored) and modified only through
  `firmware/patches/xiaozhi-esp32.patch`. To change xiaozhi code: edit in `firmware/xiaozhi-esp32/`, then regenerate
  the patch with `git -C xiaozhi-esp32 diff > patches/xiaozhi-esp32.patch`. `fetch_repos.py` silently skips a patch
  that doesn't apply cleanly.
- Hermes-specific code: `firmware/main/hal/hermes/` (`micro_wake_word` — TFLite Micro wake word, not ESPHome's GPL
  code; `state_screen` — one full-screen PNG per state; `head_cues`) plus hooks in `hal/board/*.cc`. Assets in
  `hal/hermes/assets/` (`hermes_<state>.png`, 320×240; `hermes_wei_eric.tflite`) are embedded via `EMBED_FILES` in
  `main/CMakeLists.txt`.
- Config is baked at build time (`CONFIG_HERMES_*` in `main/Kconfig.projbuild`): Wi-Fi, WS URL, token. Values go in
  `firmware/sdkconfig.defaults.local` (gitignored, auto-applied by `firmware/CMakeLists.txt`; template
  `sdkconfig.defaults.local.example`). No provisioning or OTA activation on device.

## Secrets

- Never commit or log `BRIDGE_DEVICE_TOKEN`, `HERMES_API_KEY`, Wi-Fi credentials, or full URLs with credentials.
  `firmware/scan_secrets.py` must pass. Built `.bin` files contain credentials in plain text — never publish them.
- Bridge logs timestamps in UTC (`...Z`).

## Conventions

- Commits: Conventional Commits with scope by component, e.g. `fix(speech): ...`, `feat(bridge): ...`, `feat(firmware): ...`.
