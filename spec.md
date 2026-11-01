# StackChan × Hermes — Custom Voice Build Spec

Status: draft · Owner: Eric · Date: 2026-11-01 (UTC+8)

## 1. Goal

Fork [`m5stack/StackChan`](https://github.com/m5stack/StackChan) into our own custom build so the robot works as a
Cantonese voice front-end for the local Hermes agent:

```
"喂Eric" (on-device wake word) → record → local STT → Hermes API → local TTS → StackChan speaks + animates
```

All AI runs on the Mac (`eric-mac-studio.local`). The robot handles only audio I/O, wake word, face/servo/LED.

This build **replaces** `/Users/eric/personal-project/Hermes-StackChan` (fork of `circlemouth/Hermes-StackChan`).
See §9 for decommissioning.

### Non-goals (v1)

- No cloud STT/TTS/LLM, no xiaozhi.me cloud.
- No barge-in (interrupt while speaking). Wake word is disabled during playback.
- No voiceprint / multi-user.
- No new MCP robot tools beyond what upstream firmware already exposes (§7 is optional, phase 2).
- No Flutter `app/` or `remote/` changes. The Flutter setup app is not needed for this build.
- No on-device provisioning (BLE, SoftAP, SD config import, OTA activation): config is baked at build time (D5).

## 2. Existing assets (verified on this Mac)

| Asset | Location | Interface |
|---|---|---|
| Custom wake word `喂Eric` (source of truth) | `/Users/eric/personal-project/wakeword-喂Eric/wei_eric.tflite` + `training_parameters.yaml` | microWakeWord, trained `window_step_ms 10`, `clip_duration_ms 1500`. Runtime params (from the ESPHome manifest `stackchan-fw/wei_eric.json`, same model SHA-1 `5f0c5397…`): `probability_cutoff 0.2`, `sliding_window_size 5`, `tensor_arena_size 80000` |
| STT (Cantonese) | launchd `space.hermes.whisper-yue`, `speech/stackchan_speech/stt.py` (installed by `speech/install.sh`) | Wyoming `tcp://0.0.0.0:10300`, `mlx-whisper`, model `WinsonLi/whisper-large-v3-cantonese-mlx-4bit`, ~0.2 s per utterance |
| TTS (Cantonese) | launchd `space.hermes.canto-tts`, `speech/stackchan_speech/tts.py` (installed by `speech/install.sh`) | Wyoming `tcp://0.0.0.0:10201`, sherpa-onnx VITS voice `yue_HK-xiaomaiiwn-medium`, 16-bit mono PCM |
| TTS (Mandarin, unused) | launchd `space.hermes.piper-tts` | Wyoming `:10200`, `zh_CN-chaowen-medium` |
| Hermes agent | profile `homeassistant`, gateway running | `api_server` platform (OpenAI-compatible, default port `8642`) — **not enabled yet** (only `webhook :8644` is on) |

All three Wyoming servers are also used by Home Assistant; this build must not change them.

## 3. Upstream baseline

`m5stack/StackChan` `main` (ESP-IDF v5.5.4, M5Stack CoreS3):

- `firmware/` — Mooncake UI apps + HAL (servo, display, touch, IMU, LED, ESP-NOW).
- AI runtime = `xiaozhi-esp32` **v2.2.4** pulled by `firmware/fetch_repos.py` and patched by `firmware/patches/xiaozhi-esp32.patch`.
- Voice transport = xiaozhi WebSocket protocol (`docs/websocket.md` in xiaozhi): JSON text frames (`hello`, `listen`, `stt`, `llm`, `tts`, `mcp`, `abort`) + binary Opus frames.
- Wake word = xiaozhi `WakeWord` interface (`main/audio/wake_word.h`) with ESP-SR implementations only (`AfeWakeWord`, `EspWakeWord`, `CustomWakeWord` via MultiNet). **No microWakeWord support.**

## 4. Architecture

```
┌──────────── StackChan (CoreS3) ────────────┐        ┌──────────────── Mac ─────────────────────────┐
│ mic → AFE(AEC/NS) → MicroWakeWord("喂Eric")│        │ bridge/ (Python, asyncio)                    │
│        │ detected                          │  WS    │  ├─ xiaozhi WS server  :8765                 │
│        ▼                                   │◀──────▶│ xiaozhi Application: listen → Opus uplink  │ Opus + │  ├─ Opus decode → Wyoming STT :10300          │
│ ← Opus downlink → speaker, face, servos    │  JSON  │  ├─ Hermes api_server :8642 (stream)          │
└────────────────────────────────────────────┘        │  └─ sentence split → Wyoming TTS :10201 → Opus│
                                                      └──────────────────────────────────────────────┘
```

### Decisions

- **D1 — Wake word runs on-device as a new `MicroWakeWord : WakeWord` class.**
  Alternative: stream audio to the Mac and detect there. Rejected: always-on Wi-Fi uplink, higher latency, and the model is already trained for microWakeWord on ESP32-S3.
- **D2 — Keep the upstream xiaozhi WebSocket protocol unchanged; write a new server (`bridge/`) that speaks it.**
  Alternative: reuse `Hermes-StackChan/ai-server` (TypeScript). Rejected: it is being removed, carries Japanese sanoTTS/dashboard_ws/children_http paths we do not use, and its STT/TTS go through an HTTP shim that loads a second copy of whisper large-v3 (~3.7 GB RSS).
- **D3 — Bridge in Python.** Deciding fact: the `wyoming` client library and `opuslib`/`numpy` are already available in `/Users/eric/.hermes/profiles/homeassistant/whisper-env`; Wyoming STT/TTS are used directly with no shim.
- **D4 — Hermes via `api_server` `/v1/responses` with a fixed `conversation` name**, so Hermes keeps conversation history server-side and the bridge stays stateless. Fallback if streaming on `/v1/responses` is unsuitable: `/v1/chat/completions` with `stream: true` and bridge-held history.
  Alternative considered: async Hermes webhook (existing route `stackchan-voice` on `:8644`, HMAC `X-Webhook-Signature-V2`, returns `202` and replies later via `deliver`). Rejected (stability): whether the robot speaks at all would depend on the LLM choosing to call a speak tool — the existing route prompt already has to insist "ALWAYS also speak", i.e. it can be skipped; with the API the reply text *is* the response, so speech is guaranteed. Also `202` only means "queued", so there is no signal for "Hermes finished"; `thinking` would end on a timeout guess, Hermes would need a callback tool to make the robot speak, and the device would need a permanently open WebSocket. The synchronous stream gives exact start/first-token/end events for the display phases (§4.1).

### 4.1 Turn phases → display

| # | Phase | Event that starts it | Image |
|---|---|---|---|
| 1 | Request received | wake word `喂Eric` detected (or head tap) | `listening.png` |
| 2 | Command received | utterance ended, STT transcript sent to Hermes (bridge sends `stt` to device) | `thinking.png` |
| 3 | Answering | first TTS audio of the reply starts (`tts start`) | `replying.png` |
| 4 | Done | `tts stop` | `idle.png` |
| — | Failure in 2 or 3 | §7.6 | `error.png` (3 s) → `idle.png` |

`thinking` stays on while Hermes runs tools (e.g. Home Assistant); it ends only on a real event (first audio or error), never on a fixed timer shorter than `HERMES_TIMEOUT_S`.

## 5. Repository layout

```
stackchan-hermes/
├── spec.md
├── firmware/                 # fork of upstream firmware/ (+ our patches)
│   ├── main/audio/micro_wake_word.{h,cc}    # or as a patch on xiaozhi-esp32, see §6.1
│   ├── models/wei_eric.tflite, wei_eric.json
│   ├── assets/states/*.png                  # 9 state illustrations (§6.3)
│   ├── sdkconfig.defaults.local.example          # committed; real sdkconfig.defaults.local gitignored (§6.2)
│   └── patches/xiaozhi-esp32.patch          # upstream patch + our additions
├── bridge/
│   ├── pyproject.toml
│   ├── stackchan_bridge/{server.py, protocol.py, stt.py, tts.py, hermes.py, audio.py, config.py}
│   ├── tests/
│   └── launchd/space.hermes.stackchan-bridge.plist
├── app/, remote/, server/    # upstream, untouched
└── .github/                  # upstream CI kept if it builds
```

Git: fork `m5stack/StackChan` to our GitHub account, clone here, add `upstream` remote. Our changes live on `main` of the fork; upstream syncs by merge.

## 6. Firmware changes

### 6.1 MicroWakeWord

- Add `MicroWakeWord` implementing xiaozhi `WakeWord` (`Initialize`, `Feed`, `OnWakeWordDetected`, `Start`, `Stop`, `GetFeedSize`, `EncodeWakeWordData`, `GetWakeWordOpus`, `GetLastDetectedWakeWord`).
- Inference: `espressif/esp-tflite-micro` managed component. Feature extraction: TensorFlow Lite Micro `micro_frontend` (40 mel channels, 30 ms window, 10 ms step — must match `feature_step_size 10`).
- Detection: sliding mean of last `sliding_window_size` (5) probabilities ≥ `probability_cutoff` (0.20) → callback with `"喂Eric"`. Then 1 s refractory period.
- Model is embedded via `EMBED_FILES` (no SD card — see upstream SD/LCD shared-SPI hazard).
- Tensor arena 80 000 bytes in PSRAM.
- Selection: Kconfig `CONFIG_USE_MICRO_WAKE_WORD=y` makes the CoreS3 board create `MicroWakeWord` instead of the ESP-SR wake word.
- Licensing: do **not** copy ESPHome's `micro_wake_word` C++ (GPLv3). Write against esp-tflite-micro / TFLM frontend (Apache-2.0) directly.
- Wake word is stopped during `speaking` state (no barge-in, D-non-goal) and restarted on return to `idle`.

### 6.2 Zero-setup: flash and use (D5)

**D5 — All runtime config is baked in at build time; the device needs no provisioning.**
Alternative: upstream setup flow (setup app, BLE/AP Wi-Fi provisioning, SD config import, OTA activation). Rejected: we want "flash → it works", and the SD import path is a known LCD/SPI hazard on CoreS3.

Build-time values (Kconfig `CONFIG_HERMES_*`, set from a gitignored file):

| Kconfig | Example | Notes |
|---|---|---|
| `CONFIG_HERMES_WIFI_SSID` | `MyHomeWiFi` | 2.4 GHz only (ESP32-S3) |
| `CONFIG_HERMES_WIFI_PASSWORD` | — | |
| `CONFIG_HERMES_WS_URL` | `ws://192.168.1.10:8765/xiaozhi/v1/` | Mac LAN IP; reserve it in the router (DHCP reservation) |
| `CONFIG_HERMES_WS_TOKEN` | — | = bridge `BRIDGE_DEVICE_TOKEN` |

Secret handling:

- Values live in `firmware/sdkconfig.defaults.local` (gitignored, auto-applied by upstream `CMakeLists.txt`; `firmware/sdkconfig.defaults.local.example` committed with placeholders).
- Build: `idf.py reconfigure build` (reconfigure is needed after editing the overlay; plain `build` does not re-read it)`. Build fails fast if any `CONFIG_HERMES_*` value is empty.
- Firmware never logs SSID password, token, or the full WS URL (log host only). Upstream `firmware/scan_secrets.py` must pass on every commit.
- The resulting `.bin` contains the credentials in plain text — never upload it to GitHub releases or share it.

Boot behaviour:

1. Connect Wi-Fi with the baked credentials (retry with back-off forever; show `no_wifi` face while disconnected). No AP/BLE provisioning mode.
2. **Skip xiaozhi OTA check/activation**; use `CONFIG_HERMES_WS_URL` + token directly as the WebSocket protocol settings.
3. Boot straight into the voice runtime (avatar + wake word). Launcher / setup / account / app-center UI are not shown. A long press on head touch (or reboot holding a button — pick at M0) may still open the upstream launcher for debugging.
4. NVS values from any earlier firmware (old Wi-Fi, old URL) are ignored: baked config always wins. Erase flash on first install: `idf.py erase-flash flash`.

Changing Wi-Fi or the Mac IP = edit `sdkconfig.defaults.local` and reflash.

Implementation points (verified in upstream source at `f901cc2`):

| Concern | Upstream mechanism | Our change |
|---|---|---|
| Wi-Fi credentials | `SsidManager` (NVS, `78__esp-wifi-connect`), written by BLE setup `hal_ble.cpp`; read by `xiaozhi-esp32/main/boards/common/wifi_board.cc` | At boot, before Wi-Fi start: clear list, `SsidManager::AddSsid(CONFIG_HERMES_WIFI_SSID, CONFIG_HERMES_WIFI_PASSWORD)` |
| WS URL + token | NVS `Settings("websocket")` keys `url`/`token`, filled from the OTA response; read in `websocket_protocol.cc` | Write Kconfig values into `Settings("websocket", true)` at boot |
| OTA / activation | `Application::CheckNewVersion()` → `InitializeProtocol()` picks MQTT unless OTA returned websocket config (`application.cc`) | Under `CONFIG_HERMES_BAKED_CONFIG`: skip `CheckNewVersion()`, always create `WebsocketProtocol` (extend `patches/xiaozhi-esp32.patch`) |
| Boot into voice | `main.cpp`: `startAiAgentOnBoot` (NVS via `hal_bridge.cc`) skips Mooncake and calls `startXiaozhi()` | Force `startAiAgentOnBoot = true` when baked; debug launcher entry deferred | That is the accepted cost of D5.

### 6.3 State illustrations (D6)

**D6 — The screen shows one full-screen custom illustration per state instead of the upstream drawn avatar face.**
Source: `/Users/eric/Downloads/ap/` (identical to `stackchan-fw/ap/`), copied into `firmware/assets/states/`. All 9 files are PNG, 320×240 (= CoreS3 LCD), no alpha.

| Image | Shown when | Trigger |
|---|---|---|
| `loading.png` | boot, Wi-Fi connecting, WS connecting | power-on until first WS `hello` reply |
| `idle.png` | waiting for wake word | xiaozhi state `idle` |
| `listening.png` | recording the user | wake word detected / `listening` |
| `thinking.png` | waiting for Hermes | bridge `stt` message received, until `tts start` |
| `replying.png` | speaking | `tts start` … `tts stop` (`speaking`) |
| `muted.png` | mic muted, wake word off | head-touch tap toggles mute (Assume: no upstream mute trigger to reuse; settle at M0) |
| `error.png` | turn failed | bridge sends error (§7.6), WS protocol error; shown 3 s then back to `idle` |
| `no_wifi.png` | Wi-Fi down | Wi-Fi disconnected / not yet associated after 10 s; retries continue |
| `no_ha.png` | bridge unreachable | Wi-Fi up but WS connect/handshake to `CONFIG_HERMES_WS_URL` fails; retries every 5 s |

Rules:

- Exactly one image on screen; a state change swaps it under the LVGL lock (`LvglLockGuard` / display lock), never by direct `esp_lcd_panel_draw_bitmap()`.
- Priority when several apply: `no_wifi` > `no_ha` > `error` > `muted` > activity states (`listening`/`thinking`/`replying`) > `idle`.
- No chat-bubble / subtitle text over the image in v1 (xiaozhi `SetChatMessage` and `SetEmotion` become no-ops on this display).
- Servo idle motion, LEDs and lip-sync-free head movement stay as upstream.
- Storage: embed the PNGs as-is (~530 KB total) and decode with LVGL `lodepng` (already `CONFIG_LV_USE_LODEPNG=y` upstream). Decoded images (9 × 150 KB RGB565 ≈ 1.35 MB) are cached in PSRAM so swaps after the first are instant. Alternative: pre-convert to RGB565 C arrays (no decode, +1.35 MB app size); use only if decode latency > 200 ms, after checking the build still fits the 0x4F0000 `ota_0` partition.
- Replacing an image = drop a new 320×240 PNG with the same name into `firmware/assets/states/` and rebuild.

### 6.4 Unchanged

Servos, HAL, MCP tools — kept as upstream. Mooncake launcher kept in the image but not shown on boot (§6.2). The upstream avatar face code stays in the tree but is not used (§6.3).

## 7. Bridge (`bridge/`) behaviour

### 7.1 Protocol (per xiaozhi `docs/websocket.md`, binary protocol v1)

1. Device `hello` (`audio_params: opus, 16000 Hz, mono, 60 ms`) → bridge replies `hello` with `transport: websocket`, `session_id`, `audio_params: opus, 24000 Hz, mono, 60 ms`.
2. Device `listen state=detect text=喂Eric` → ignore text (optionally log); `listen state=start mode=auto` → start buffering uplink Opus.
3. End of utterance: device `listen state=stop`, **or** bridge-side VAD (silence ≥ 700 ms after speech, max 15 s).
4. Pipeline (§7.2). Send `stt {text}`, `llm {emotion}`, `tts start`, per sentence `tts sentence_start {text}` + Opus frames, `tts stop`.
5. Device `abort` → cancel in-flight Hermes request and TTS, send `tts stop`.
6. Unknown message types are logged and ignored; protocol errors close the socket with a logged reason.

### 7.2 Pipeline

| Step | Implementation |
|---|---|
| Decode | Opus 16 kHz → PCM s16le mono |
| STT | Wyoming `Transcribe(language="yue")` + `AudioStart/Chunk/Stop` to `tcp://127.0.0.1:10300` → `Transcript.text` |
| Filter | empty / < 2 chars / filler-only transcript → no Hermes call; send `tts stop` and return to listening once |
| Hermes | `POST http://127.0.0.1:8642/v1/responses`, `Authorization: Bearer $API_SERVER_KEY`, `conversation: "stackchan"`, `stream: true`, `instructions` = voice prompt (§7.3) |
| Split | stream text → sentences on `。！？!?\n` (or 40 chars) |
| Clean | strip markdown, URLs, emoji, code blocks before TTS |
| TTS | Wyoming `Synthesize(text)` to `tcp://127.0.0.1:10201` → PCM at the server's `rate` |
| Encode | resample to 24 kHz → Opus 60 ms frames, paced at real time (+ ≤ 300 ms lead) |

Sentence 1 TTS starts while Hermes is still streaming sentence 2.

### 7.3 Voice prompt (default, configurable)

> 你而家係透過 StackChan 機械人同 Eric 用語音傾偈。用廣東話口語答，一至兩句，唔好用 markdown、表情符號或者列表。你有 Home Assistant 工具，可以直接控制家居。語音轉文字有好多同音錯字（例如「幫我識客聽燈槽」即係「幫我開客廳燈」），要按上文估最可能嘅指令去做，唔好話聽唔明。

Carried over from the existing `stackchan-voice` webhook route prompt; that route can stay for other uses but this build does not call it.

### 7.4 Emotion

The display ignores emotions (§6.3). Bridge always sends `llm {emotion: "neutral"}` once before `tts start` to stay protocol-compatible; strip emoji from the reply before TTS.

### 7.4a Errors to device

On any §7.6 failure the bridge sends `{"type":"system","command":"error"}`-style message agreed at M3 (or reuses an existing xiaozhi alert message) so the device shows `error.png`.

### 7.5 Config (`bridge/.env`, never committed)

| Var | Default |
|---|---|
| `BRIDGE_HOST` / `BRIDGE_PORT` | `0.0.0.0` / `8765` |
| `BRIDGE_DEVICE_TOKEN` | required |
| `STT_URI` | `tcp://127.0.0.1:10300` |
| `STT_LANGUAGE` | `yue` |
| `TTS_URI` | `tcp://127.0.0.1:10201` |
| `TTS_VOICE` | `yue_HK-xiaomaiiwn-medium` |
| `HERMES_URL` | `http://127.0.0.1:8642` |
| `HERMES_API_KEY` | required (= profile `API_SERVER_KEY`) |
| `HERMES_CONVERSATION` | `stackchan` |
| `HERMES_TIMEOUT_S` | `120` |
| `VAD_END_SILENCE_MS` / `MAX_UTTERANCE_MS` | `700` / `15000` |

### 7.6 Errors (device must never hang in `thinking`)

| Failure | Bridge behaviour |
|---|---|
| STT/TTS server down or timeout (30 s) | speak/send fixed fallback text via `stt`-style bubble + `tts stop`, log error |
| Hermes 401/5xx/timeout | spoken fallback `我而家連唔到 Hermes。` (if TTS up) then `tts stop` |
| WebSocket drop mid-turn | cancel turn, discard audio |
| Bad device token | reject handshake `401`, no log of the token |

Logs never contain tokens, `API_SERVER_KEY` or full URLs with credentials. Timestamps logged in UTC.

### 7.7 Service

launchd `space.hermes.stackchan-bridge` (template `bridge/launchd/`, installed by `bridge/install.sh`), runs with `bridge/.venv`, reads `bridge/.env`, `KeepAlive=true`, logs to `~/Library/Logs/stackchan-bridge.log`.

## 8. Hermes setup (done 2026-11-01)

- `gateway.multiplex_profiles: true` → `api_server` can only be enabled on the **default** profile (`~/.hermes/config.yaml`), which serves all profiles. Per-profile routes (`/p/<profile>/v1`) are not used; the bridge calls the default profile at `http://127.0.0.1:8642/v1`.
- Enabled: `platforms.api_server` `host 127.0.0.1`, `port 8642`; `API_SERVER_KEY` in `~/.hermes/.env` (bridge reads the same value into `HERMES_API_KEY`). Backup of the previous config: `~/.hermes/config.yaml.bak-before-api-server`.
- Default profile already has the `homeassistant` plugin enabled.
- Verified: `GET /v1/health` → `{"status":"ok","version":"0.22.5"}`; `POST /v1/responses` (read-only "客廳燈而家開咗未？") → `開咗，客廳燈槽而家係開住。` in **26 s**, via `skill_view` + several `terminal` / `read_file` calls rather than a direct HA tool call.
- Future Hermes config changes should go through the `hermes` CLI where it covers them.

**Latency risk (O5):** 26 s for a simple state query is far above acceptance criterion 3 (≤ 4 s to first audio). To address in M2: (a) instructions that name the HA tools directly, (b) a faster model for voice turns if the API allows per-request model selection, (c) a short spoken filler (e.g. 「等我睇下。」) when no text arrives within 3 s, so `thinking` is not silent.

## 9. Decommission `Hermes-StackChan`

Order matters — stop services first, then delete.

1. `launchctl bootout gui/$(id -u)/space.hermes.aiserver` and `space.hermes.canto-shims`; remove both plists from `~/Library/LaunchAgents/`. This frees ports `8765` (reused by our bridge), `8766`, `18077` and ~3.7 GB RAM.
2. **Keep** `space.hermes.whisper-yue`, `space.hermes.canto-tts`, `space.hermes.piper-tts` (shared with Home Assistant and this build).
3. The old repo has uncommitted changes (`ai-server/src/session.ts`, `firmware/dependencies.lock`, `firmware/main/idf_component.yml`, `firmware/patches/esp-ml307-tcp-shutdown.patch`) and untracked `canto-shims.py`. Archive (`tar`) or discard deliberately, then delete `/Users/eric/personal-project/Hermes-StackChan`.
4. Copy `wei_eric.tflite` from `/Users/eric/personal-project/wakeword-喂Eric/` into `firmware/models/`, and carry its runtime params into our own `firmware/models/wei_eric.json` (the only existing copy is `stackchan-fw/wei_eric.json`; `stackchan-fw/` is the ESPHome attempt, out of scope here).
5. Device currently flashed with the old firmware will stop working after step 1 until the new firmware is flashed.

## 10. Milestones

| # | Deliverable | Done when |
|---|---|---|
| M0 | Fork + clone + upstream builds unmodified | `idf.py build` passes for CoreS3; flashed device boots to launcher |
| M0.5 | Baked config + direct boot (§6.2) | `erase-flash flash` → device joins Wi-Fi and opens the WS to the bridge with no user interaction |
| M1 | Hermes `api_server` enabled | `curl /v1/health` 200; a `/v1/responses` call returns Cantonese text |
| M2 | Bridge with text-only test client | pytest: fake device sends Opus of a recorded WAV → receives `stt`, `tts start/stop` and decodable Opus |
| M3 | Device ↔ bridge with upstream wake word / touch-to-talk | full voice turn on real device |
| M3.5 | State illustrations (§6.3) | each of the 9 images observed on the physical LCD in its trigger state |
| M4 | `MicroWakeWord` on device | `喂Eric` triggers listening; acceptance below |
| M5 | Decommission old project (§9), launchd service | reboot Mac → bridge auto-starts, device works |

## 11. Acceptance criteria

1. Saying `喂Eric` at 1–2 m in a quiet room triggers listening in ≥ 9/10 tries.
2. ≤ 1 false wake per hour with TV/conversation in background (measured over 2 h).
3. Wake → first TTS audio ≤ 4 s for a one-sentence answer (STT + Hermes first sentence + TTS), measured from bridge logs.
4. A follow-up question referencing the previous answer is answered in context (conversation continuity via `HERMES_CONVERSATION`).
5. Stopping any one of STT / TTS / Hermes → device returns to idle within 35 s with a visible/audible error, no reboot needed.
6. Device token and `API_SERVER_KEY` do not appear in any log (`grep` over bridge logs and serial monitor).
7. Fresh device: `idf.py erase-flash flash` → within 30 s of power-on it shows the idle face and answers `喂Eric`, with zero taps/app/setup steps.
8. `git grep` for the real SSID, Wi-Fi password and device token returns nothing; `sdkconfig.defaults.local` is gitignored.
9. Bridge unit tests (protocol handshake, sentence splitter, text cleaner, error paths with fake Wyoming/Hermes servers) pass in CI-less local run: `cd bridge && pytest`.

## 12. Open questions

- ~~**O1**~~ Resolved at M0 (read from source) — injection points listed in §6.2 "Implementation points".
- **O2** `/v1/responses` stream event format in this Hermes version, and whether it runs tools (e.g. Home Assistant) inside the turn — verify at M1; latency of tool calls may break criterion 3.
- **O3** canto-vits drops English tokens as OOV (seen in `canto-tts.log`: `OOV done. Ignore it!`). Mixed Cantonese/English replies lose English words. Options: prompt Hermes to avoid English, or route English spans to another voice (phase 2).
- **O5** Hermes turn latency (§8): measured 26 s for a one-step HA query. Must reach criterion 3 or relax it.
  - 2026-11-01 update: Hermes via `/p/homeassistant/v1` (profile with `HASS_TOKEN`, `homeassistant` toolset on `api_server`) → 5.7 s. STT switched to `alvanlii/whisper-small-cantonese` (CTranslate2 int8 at `whisper-env/share/whisper-small-cantonese-int8`, Wyoming `127.0.0.1:10301`, `language=zh` because whisper-small has no `yue` token) → 0.6–0.8 s vs 4.4 s for large-v3, similar accuracy on 5 test phrases. End of speech → first audio: 25 s → 11.0 s → **5.9 s**.
  - 2026-11-02 update: STT and TTS moved into `speech/`. STT is now `mlx-whisper` with `WinsonLi/whisper-large-v3-cantonese-mlx-4bit` on `:10300` (`language=yue`): 0.15–0.4 s vs 4.4–7 s for faster-whisper large-v3 on 20 recorded utterances, similar accuracy. A self-converted MLX FP16 was no more accurate, so 4-bit stays.
- **O4** AFE output (post-AEC/NS) vs raw mic for microWakeWord — model was trained on ESPHome's pipeline; pick whichever meets criteria 1–2.
