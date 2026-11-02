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


def test_trim_cuts_learned_room_noise_not_just_digital_silence():
    # The eric voice reproduces its recording room: ~0.5 s of -50 dBFS hiss around speech peaking at -13 dBFS
    # (only ~37 dB below peak, so a fixed peak-relative threshold kept all of it).
    rng = np.random.default_rng(0)
    rate = tts.RATE
    noise = lambda s: (rng.standard_normal(int(rate * s)) * 10 ** (-50 / 20)).astype(np.float32)
    t = np.arange(int(rate * 1.0)) / rate
    speech = (np.sin(2 * np.pi * 220 * t) * 10 ** (-13 / 20) * np.sqrt(2)).astype(np.float32)
    out = tts.trim(np.concatenate([noise(0.5), speech, noise(0.6)]))
    assert len(out) / rate < 1.0 + tts.LEAD_KEEP_S + tts.TAIL_KEEP_S + 0.05


def test_trim_noise_floor_ignores_digital_silence():
    # Some generations end hiss -> exact zeros -> hiss; the zeros must not pull the noise-floor estimate down.
    rng = np.random.default_rng(1)
    rate = tts.RATE
    noise = lambda s: (rng.standard_normal(int(rate * s)) * 10 ** (-50 / 20)).astype(np.float32)
    t = np.arange(int(rate * 1.0)) / rate
    speech = (np.sin(2 * np.pi * 220 * t) * 10 ** (-13 / 20) * np.sqrt(2)).astype(np.float32)
    out = tts.trim(np.concatenate([noise(0.5), speech, noise(0.4), np.zeros(int(rate * 0.4), np.float32), noise(0.2)]))
    assert len(out) / rate < 1.0 + tts.LEAD_KEEP_S + tts.TAIL_KEEP_S + 0.05


def test_long_internal_pauses_are_shortened_short_ones_kept():
    # The eric voice copies read-aloud comma pauses (400-680 ms); conversation wants ~200 ms.
    rng = np.random.default_rng(2)
    rate = tts.RATE
    noise = lambda s: (rng.standard_normal(int(rate * s)) * 10 ** (-50 / 20)).astype(np.float32)
    tone = lambda s: (np.sin(2 * np.pi * 220 * np.arange(int(rate * s)) / rate) * 0.3).astype(np.float32)
    audio = np.concatenate([tone(0.5), noise(0.6), tone(0.5), noise(0.1), tone(0.5)])
    out = tts.cap_pauses(audio)
    assert abs(len(out) / rate - (1.5 + tts.MAX_PAUSE_S + 0.1)) < 0.05
