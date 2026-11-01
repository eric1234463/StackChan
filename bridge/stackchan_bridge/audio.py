"""Opus codec, resampling and a simple speech-band RMS voice-activity detector."""
from __future__ import annotations

import os

import numpy as np
import soxr

# opuslib locates libopus via ctypes find_library, which ignores Homebrew's prefix on macOS.
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib:/usr/local/lib:/usr/lib")
import opuslib  # noqa: E402

IN_RATE = 16000  # device uplink
OUT_RATE = 24000  # downlink, announced in our hello
FRAME_MS = 60


class OpusDecoder:
    def __init__(self, rate: int = IN_RATE) -> None:
        self._rate = rate
        self._dec = opuslib.Decoder(rate, 1)

    def decode(self, packet: bytes) -> bytes:
        return self._dec.decode(packet, self._rate * FRAME_MS // 1000)


def encode_opus_frames(samples: np.ndarray, rate: int) -> list[bytes]:
    """int16 mono PCM at `rate` -> list of 60 ms Opus packets at OUT_RATE."""
    if rate != OUT_RATE:
        samples = soxr.resample(samples.astype(np.float32) / 32768.0, rate, OUT_RATE)
        samples = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)
    frame = OUT_RATE * FRAME_MS // 1000
    pad = (-len(samples)) % frame
    if pad:
        samples = np.concatenate([samples, np.zeros(pad, np.int16)])
    enc = opuslib.Encoder(OUT_RATE, 1, opuslib.APPLICATION_VOIP)
    return [enc.encode(samples[i:i + frame].tobytes(), frame) for i in range(0, len(samples), frame)]


def normalize(pcm: bytes, target_peak: float = 0.9, max_gain: float = 8.0) -> bytes:
    """Scale int16 PCM so its peak reaches target_peak (device mic peaks ~0.1); gain capped to avoid boosting noise."""
    data = np.frombuffer(pcm, np.int16).astype(np.float32)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 0:
        return pcm
    gain = min(max_gain, target_peak * 32767.0 / peak)
    return np.clip(data * gain, -32768, 32767).astype(np.int16).tobytes()


def rms(pcm: bytes) -> float:
    data = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(data * data))) if data.size else 0.0


SPEECH_BAND_HZ = (300, 3400)


def speech_rms(pcm: bytes, rate: int = IN_RATE) -> float:
    """RMS of the 300-3400 Hz band only. The device emits a drifting ~4.5 kHz whine (90% of its energy
    above 3 kHz) that is louder than the threshold in full-band RMS; in this band it is ~5% of that."""
    data = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
    if not data.size:
        return 0.0
    spectrum = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(data.size, 1 / rate)
    spectrum[(freqs < SPEECH_BAND_HZ[0]) | (freqs > SPEECH_BAND_HZ[1])] = 0
    band = np.fft.irfft(spectrum, data.size)
    return float(np.sqrt(np.mean(band * band)))


class Vad:
    """Speech-band RMS VAD over fixed-size frames. feed() returns 'start', 'end' or None."""

    # SHORTCUT: fixed RMS threshold; switch to silero/webrtc VAD if noisy rooms cause false ends.

    def __init__(self, threshold: float, start_ms: int, end_silence_ms: int) -> None:
        self.threshold = threshold
        self.start_ms = start_ms
        self.end_silence_ms = end_silence_ms
        self.in_speech = False
        self._voiced_ms = 0
        self._silence_ms = 0

    def feed(self, pcm: bytes, frame_ms: int = FRAME_MS) -> str | None:
        voiced = speech_rms(pcm) >= self.threshold
        if not self.in_speech:
            self._voiced_ms = self._voiced_ms + frame_ms if voiced else 0
            if self._voiced_ms >= self.start_ms:
                self.in_speech = True
                self._silence_ms = 0
                return "start"
            return None
        self._silence_ms = 0 if voiced else self._silence_ms + frame_ms
        if self._silence_ms >= self.end_silence_ms:
            self.in_speech = False
            self._voiced_ms = 0
            return "end"
        return None
