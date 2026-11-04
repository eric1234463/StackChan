"""Bridge configuration from environment (+ optional bridge/.env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INSTRUCTIONS = (
    "你而家係透過 StackChan 機械人同 Eric 用語音傾偈。用廣東話口語答，一至兩句，"
    "唔好用 markdown、表情符號、列表或者英文。"
    "你有 Home Assistant 工具：查詢或者控制家居時，直接用 ha_get_state / ha_call_service"
    "（唔識 entity 先用 ha 相關工具搵），唔好用 terminal、唔好讀 skill 檔案。"
    "開關設備只需要 ha_call_service 一次，做完直接答，唔使再用 ha_get_state 確認。"
    "語音轉文字有好多同音錯字（例如「幫我識客聽燈槽」即係「幫我開客廳燈」），"
    "要按上文估最可能嘅指令去做，唔好話聽唔明。"
)
# Appended unless STT_ROMANIZED=0: the Liujgoj Qwen3-ASR writes romanization instead of Chinese characters.
ROMANIZED_NOTE = (
    "語音轉文字係粵語拼音（Liujgoj，聲調用字尾字母標，例如「Laanqhei yiqgingj gaauzor heoi yih saph ngq douh」"
    "即係「冷氣已經較咗去二十五度」），當廣東話咁理解，回覆照用中文字。"
)


@dataclass(frozen=True)
class Config:
    device_token: str
    hermes_api_key: str
    host: str = "0.0.0.0"
    port: int = 8765
    stt_uri: str = "tcp://127.0.0.1:10300"
    stt_language: str = "yue"
    stt_romanized: bool = True  # the STT writes Liujgoj romanization (speech engine qwen3-asr); 0 for whisper
    tts_uri: str = "tcp://127.0.0.1:10201"
    hermes_url: str = "http://127.0.0.1:8642"
    hermes_conversation: str = "stackchan"
    # One Hermes session per burst of use: a day-long session reached ~170 messages / 67k tokens per call.
    hermes_session_idle_s: float = 900.0
    hermes_model: str = ""  # with hermes_provider, overrides the profile's model for voice turns; "" = Hermes default
    hermes_provider: str = ""
    # The profile default is "high"; on 5 HA turns low/minimal took 19-36 s vs 38 s (provider noise dominates).
    hermes_reasoning_effort: str = ""  # none|minimal|low|medium|high; "" = Hermes default
    hermes_timeout_s: float = 120.0
    instructions: str = DEFAULT_INSTRUCTIONS
    stt_timeout_s: float = 30.0
    tts_timeout_s: float = 30.0
    vad_rms_threshold: float = 0.015  # device noise floor ~0.007 (measured 2026-11-01)
    vad_start_speech_ms: int = 180
    min_voiced_ms: int = 300  # utterances with less voiced audio are noise (whisper hallucinates on them)
    vad_end_silence_ms: int = 800  # longest mid-sentence pause in 55 real takes: 780 ms (700 cut 4 of them)
    max_utterance_ms: int = 15000
    listen_timeout_ms: int = 8000  # after the wake word: time to start talking
    followup_timeout_ms: int = 2000  # after a reply or announcement: time to START a follow-up question
    # A follow-up quieter than this (speech RMS p90) is someone else talking, not a reply to the device: on
    # 2026-11-02 background-talk follow-ups were p90 0.016-0.034 and real ones 0.062-0.100.
    followup_min_rms: float = 0.045
    fallback_text: str = "我而家連唔到 Hermes。"
    ack_text: str = "收到，幫緊你。"  # spoken while Hermes works; empty disables
    device_volume: int | None = None  # 0-100, set on the device at every connect; None leaves its own setting
    debug_audio_dir: str = ""  # if set, each utterance is saved as WAV there (local debugging only)
    wake_audio_dir: str = ""  # if set, the device's audio leading up to each wake word is saved as WAV there
    device_log_file: str = ""  # if set, the device's own log (uploaded at every connect) is appended there


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load(env: dict[str, str] | None = None, env_file: Path | None = None) -> Config:
    if env is None:
        file_values = _read_env_file(env_file or Path(__file__).resolve().parent.parent / ".env")
        env = {**file_values, **os.environ}

    def need(name: str) -> str:
        value = env.get(name, "")
        if not value:
            raise SystemExit(f"{name} is required (set it in bridge/.env)")
        return value

    defaults = Config(device_token="", hermes_api_key="")
    romanized = env.get("STT_ROMANIZED", "1").lower() not in ("0", "false", "no")
    return Config(
        device_token=need("BRIDGE_DEVICE_TOKEN"),
        hermes_api_key=need("HERMES_API_KEY"),
        host=env.get("BRIDGE_HOST", defaults.host),
        port=int(env.get("BRIDGE_PORT", defaults.port)),
        stt_uri=env.get("STT_URI", defaults.stt_uri),
        stt_language=env.get("STT_LANGUAGE", defaults.stt_language),
        stt_romanized=romanized,
        tts_uri=env.get("TTS_URI", defaults.tts_uri),
        hermes_url=env.get("HERMES_URL", defaults.hermes_url).rstrip("/"),
        hermes_conversation=env.get("HERMES_CONVERSATION", defaults.hermes_conversation),
        hermes_session_idle_s=float(env.get("HERMES_SESSION_IDLE_S", defaults.hermes_session_idle_s)),
        hermes_model=env.get("HERMES_MODEL", ""),
        hermes_provider=env.get("HERMES_PROVIDER", ""),
        hermes_reasoning_effort=env.get("HERMES_REASONING_EFFORT", ""),
        hermes_timeout_s=float(env.get("HERMES_TIMEOUT_S", defaults.hermes_timeout_s)),
        instructions=env.get("HERMES_INSTRUCTIONS", defaults.instructions) + (ROMANIZED_NOTE if romanized else ""),
        vad_rms_threshold=float(env.get("VAD_RMS_THRESHOLD", defaults.vad_rms_threshold)),
        vad_end_silence_ms=int(env.get("VAD_END_SILENCE_MS", defaults.vad_end_silence_ms)),
        max_utterance_ms=int(env.get("MAX_UTTERANCE_MS", defaults.max_utterance_ms)),
        listen_timeout_ms=int(env.get("LISTEN_TIMEOUT_MS", defaults.listen_timeout_ms)),
        followup_timeout_ms=int(env.get("FOLLOWUP_TIMEOUT_MS", defaults.followup_timeout_ms)),
        followup_min_rms=float(env.get("FOLLOWUP_MIN_RMS", defaults.followup_min_rms)),
        debug_audio_dir=env.get("BRIDGE_DEBUG_AUDIO_DIR", ""),
        wake_audio_dir=env.get("BRIDGE_WAKE_AUDIO_DIR", ""),
        device_log_file=env.get("BRIDGE_DEVICE_LOG_FILE", ""),
        ack_text=env.get("ACK_TEXT", defaults.ack_text),
        device_volume=int(env["DEVICE_VOLUME"]) if env.get("DEVICE_VOLUME") else None,
    )
