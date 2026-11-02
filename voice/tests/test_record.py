import numpy as np

from record import check

RATE = 48000


def take(seconds: float, voice: float, noise: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(int(RATE * seconds)) / RATE
    audio = rng.normal(0, noise, t.size).astype(np.float32)
    audio[RATE // 2:-RATE // 2] += (voice * np.sin(2 * np.pi * 220 * t[RATE // 2:-RATE // 2])).astype(np.float32)
    return audio


def test_clean_take_has_no_problems():
    stats, problems = check(take(6, voice=0.3, noise=0.0005), RATE)
    assert problems == [] and stats["duration"] == 6.0


def test_flags_short_clipped_and_noisy_takes():
    assert any("length" in p for p in check(take(1.5, 0.3, 0.0005), RATE)[1])
    assert any("clipping" in p for p in check(np.clip(take(6, 1.2, 0.0005), -1, 1), RATE)[1])
    assert any("too quiet" in p for p in check(take(6, 0.01, 0.0001), RATE)[1])
    assert any("background noise" in p for p in check(take(6, 0.3, 0.01), RATE)[1])


def test_flags_bluetooth_band_limit_and_noise_gate():
    audio = take(6, voice=0.3, noise=0.0005)
    spectrum = np.fft.rfft(audio)
    spectrum[np.fft.rfftfreq(audio.size, 1 / RATE) > 7400] = 0  # HFP-style low-pass
    gated = np.fft.irfft(spectrum, audio.size).astype(np.float32)
    gated[: RATE // 2] = 0  # noise gate between words
    problems = check(gated, RATE)[1]
    assert any("Bluetooth" in p for p in problems) and any("noise gate" in p for p in problems)
