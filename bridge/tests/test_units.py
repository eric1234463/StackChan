import json
import time

import httpx
import numpy as np

from stackchan_bridge import config, services
from stackchan_bridge.audio import FRAME_MS, IN_RATE, Vad
from stackchan_bridge.text import clean, is_meaningful, split_sentences

FRAME = IN_RATE * FRAME_MS // 1000


def tone(amplitude: float, hz: float = 440) -> bytes:
    t = np.arange(FRAME) / IN_RATE
    return (np.sin(2 * np.pi * hz * t) * amplitude * 32767).astype(np.int16).tobytes()


def test_vad_detects_start_and_end():
    vad = Vad(threshold=0.025, start_ms=120, end_silence_ms=700)
    events = [vad.feed(tone(0.0)) for _ in range(5)]
    events += [vad.feed(tone(0.3)) for _ in range(10)]
    events += [vad.feed(tone(0.0)) for _ in range(12)]
    assert events.count("start") == 1 and events.count("end") == 1
    assert events.index("start") < events.index("end")


def test_vad_ignores_high_frequency_whine():
    # Device whine measured 2026-11-02: ~4.5 kHz tone, full-band RMS 0.02-0.03, kept the VAD open forever.
    vad = Vad(threshold=0.015, start_ms=180, end_silence_ms=900)
    assert all(vad.feed(tone(0.04, hz=4500)) is None for _ in range(50)) and not vad.in_speech


def test_vad_ignores_short_click():
    vad = Vad(threshold=0.025, start_ms=120, end_silence_ms=700)
    assert [vad.feed(tone(0.3)), vad.feed(tone(0.0))] == [None, None]


def test_clean_strips_markdown_urls_emoji():
    assert clean("**開咗** 😀 客廳燈 https://x.io/a `code`") == "開咗 客廳燈 code"


def test_split_keeps_punctuation_and_limits_length():
    parts = split_sentences("好呀。我幫你開咗客廳燈！仲有冇其他嘢？", max_chars=40)
    assert parts == ["好呀。", "我幫你開咗客廳燈！", "仲有冇其他嘢？"]
    long = "一" * 100
    assert all(len(p) <= 40 for p in split_sentences(long)) and "".join(split_sentences(long)) == long


def test_is_meaningful():
    assert not is_meaningful("嗯。")
    assert not is_meaningful("  ")
    assert is_meaningful("開燈")
    assert is_meaningful("冇嘢 laa")
    assert not is_meaningful("All right. Then we try it.")
    assert not is_meaningful("안녕히 가세요")
    assert not is_meaningful("字幕由 Amara.org 社群提供")
    assert not is_meaningful("MING PAO CANADA // MING PAO TORONTO")
    assert not is_meaningful("多謝您收睇!")
    assert not is_meaningful("謝謝觀賞,下次見。")


def test_config_requires_secrets():
    import pytest
    with pytest.raises(SystemExit, match="BRIDGE_DEVICE_TOKEN"):
        config.load(env={"HERMES_API_KEY": "x"})
    cfg = config.load(env={"BRIDGE_DEVICE_TOKEN": "t", "HERMES_API_KEY": "k", "BRIDGE_PORT": "9000"})
    assert cfg.port == 9000 and cfg.stt_language == "yue"


def test_normalize_boosts_quiet_audio_with_gain_cap():
    from stackchan_bridge.audio import normalize, rms
    quiet = tone(0.2)  # needs 4.5x, under the 8x cap
    assert abs(max(abs(x) for x in np.frombuffer(normalize(quiet), np.int16)) / 32767 - 0.9) < 0.01
    very_quiet = tone(0.01)
    assert rms(normalize(very_quiet)) < rms(very_quiet) * 8.01  # gain capped at 8x


def test_conversation_rotates_after_idle_gap_only():
    from stackchan_bridge.server import Conversation
    t0 = time.mktime((2026, 11, 2, 17, 44, 11, 0, 0, -1))
    conv = Conversation("stackchan", idle_s=900)
    first = conv.name(t0)
    assert first == "stackchan-20261102-174411"
    assert conv.name(t0 + 600) == first  # follow-up within the window keeps the context
    assert conv.name(t0 + 600 + 899) == first  # the window counts from the last turn
    assert conv.name(t0 + 600 + 899 + 901) == "stackchan-20261102-182411"


async def test_ask_hermes_sends_model_provider_effort_only_when_set():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "completed", "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "好"}]}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await services.ask_hermes(client, "http://h", "k", "c", "i", "開燈", 5)
        await services.ask_hermes(client, "http://h", "k", "c", "i", "開燈", 5,
                                  model="muse-spark-1.3-contributor", provider="opencode-go",
                                  reasoning_effort="low")
    assert sent[0]["model"] == "hermes-agent" and "provider" not in sent[0] and "model_options" not in sent[0]
    assert sent[1]["model"] == "muse-spark-1.3-contributor" and sent[1]["provider"] == "opencode-go"
    assert sent[1]["model_options"] == {"reasoning_effort": "low"}


def test_loud_hits_target_level_and_never_clips():
    from stackchan_bridge.server import LOUD_RMS_DB, loud
    t = np.arange(24000 * 2) / 24000
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) ** 2  # syllable-like amplitude swings
    for peak_db in (-4, -20):  # typical TTS output, and a much quieter one
        x = np.sin(2 * np.pi * 200 * t) * envelope
        x = (x / np.abs(x).max() * 10 ** (peak_db / 20) * 32767).astype(np.int16)
        y = loud(x).astype(np.float32) / 32768
        assert np.abs(y).max() < 1.0 and abs(20 * np.log10(np.sqrt(np.mean(y ** 2))) - LOUD_RMS_DB) < 2


def test_closing_phrases():
    from stackchan_bridge.text import is_closing
    for said in ("OK", "ok。", "收到", "收到，多謝", "好", "好啦", "好嘅，唔該", "明白", "冇嘢喇", "得喇", "OK 收到", "冇問題", "好，冇問題。", "OK收到"):
        assert is_closing(said), said
    for said in ("好，咁聽日呢？", "收到未？", "幫我開燈", "OK Google 開燈", "好凍呀"):
        assert not is_closing(said), said
