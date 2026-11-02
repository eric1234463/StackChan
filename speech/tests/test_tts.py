import numpy as np

from stackchan_speech import tts


def test_chunks_split_at_punctuation_within_limit():
    text = "早晨。" + "今日天文台話多雲，" * 6 + "記得帶遮！"
    pieces = tts.chunks(text)
    assert pieces[0] == "早晨。" and pieces[-1].endswith("記得帶遮！")
    assert all(len(p) <= tts.MAX_CHUNK_CHARS for p in pieces) and "".join(pieces) == text


class FakeModel:
    """Yields lead-in silence + voice of the given lengths, one per generate() call."""

    def __init__(self, durations_s):
        self.durations_s, self.calls = list(durations_s), 0

    def generate(self, text, voice, lang_code):
        seconds = self.durations_s[self.calls]
        self.calls += 1
        voice_part = np.full(int(tts.RATE * seconds), 0.5, np.float32)
        yield type("R", (), {"audio": np.concatenate([np.zeros(tts.RATE, np.float32), voice_part])})()


def test_overlong_generation_is_retried_and_lead_silence_trimmed(monkeypatch):
    model = FakeModel([4.8, 1.6])  # babble, then a normal take
    monkeypatch.setattr(tts, "load", lambda _: model)
    pcm = tts.synth_chunk("unused", "收到，幫緊你。")
    seconds = len(pcm) / 2 / tts.RATE
    assert model.calls == 2 and 1.6 <= seconds < 1.6 + tts.LEAD_KEEP_S + tts.TAIL_KEEP_S + 0.05


def test_keeps_shortest_when_all_attempts_are_long(monkeypatch):
    model = FakeModel([5.0, 4.0, 4.5])
    monkeypatch.setattr(tts, "load", lambda _: model)
    seconds = len(tts.synth_chunk("unused", "收到。")) / 2 / tts.RATE
    assert model.calls == tts.MAX_ATTEMPTS and 4.0 <= seconds < 4.3


def test_time_stretch_shortens_without_changing_pitch():
    t = np.arange(tts.RATE * 2) / tts.RATE
    tone = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    fast = tts.time_stretch(tone, 1.25)
    assert abs(len(fast) / len(tone) - 1 / 1.25) < 0.02
    spectrum = np.abs(np.fft.rfft(fast * np.hanning(len(fast))))
    assert abs(np.fft.rfftfreq(len(fast), 1 / tts.RATE)[spectrum.argmax()] - 200) < 3  # same pitch
    assert tts.time_stretch(tone, 1.0) is tone
