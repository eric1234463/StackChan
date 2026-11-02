# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "pymicro-features", "ai-edge-litert"]
# ///
"""Replay audio through the 喂Eric microWakeWord model the way firmware MicroWakeWord does, to choose a cutoff.

    uv run voice/wakeword/wakeeval.py --pos ~/Downloads/eric ~/Downloads/sze --neg ~/voice-data/eric/wavs bridge/debug
    uv run voice/wakeword/wakeeval.py --trace ~/Library/Logs/stackchan-wake/wake-*.wav

--pos dirs hold takes of the wake word (each should trigger once), --neg dirs hold speech without it. Any format
afconvert reads is accepted (m4a, 48 kHz WAV, ...). The model defaults to the one embedded in the firmware.
Replaying a device wake clip reproduced its on-device window within ~0.04 (2026-11-02).
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parents[2] / "firmware/main/hal/hermes/assets/hermes_wei_eric.tflite"
WINDOW = 5  # sliding_window_size, as kSlidingWindow in micro_wake_word.h
REFRACTORY = 33  # kRefractoryInferences
CUTOFFS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load(path: Path, pad_s: float = 0.0) -> bytes:
    """16 kHz mono int16 PCM, converting with afconvert when needed, plus pad_s of trailing silence."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            w = wave.open(str(path))
            ok = w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        except (wave.Error, EOFError):
            ok = False
        if not ok:
            out = Path(tmp) / "a.wav"
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(path), str(out)], check=True)
            w = wave.open(str(out))
        pcm = w.readframes(w.getnframes())
    return pcm + b"\0" * (int(pad_s * 16000) * 2)


def probabilities(pcm: bytes, model: Path = MODEL) -> np.ndarray:
    """Per-inference wake probability (0..1), with the firmware's frontend quantisation and stride."""
    import pymicro_features
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=str(model))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    stride = inp["shape"][1]
    frontend = pymicro_features.MicroFrontend()
    frames, probs, i = [], [], 0
    while i + 320 <= len(pcm):
        r = frontend.process_samples(pcm[i:i + 320])  # 10 ms
        i += r.samples_read * 2
        if not r.features:
            continue
        v = np.round(np.array(r.features) * 25.6).astype(np.int32)  # back to the frontend's uint16 scale
        frames.append(np.clip((v * 256 + 333) // 666 - 128, -128, 127).astype(np.int8))  # QuantizeFeature
        if len(frames) == stride:
            interp.set_tensor(inp["index"], np.stack(frames)[None])
            interp.invoke()
            probs.append(interp.get_tensor(out["index"])[0][0] / 255.0)
            frames = []
    return np.array(probs)


def detections(p: np.ndarray, cutoff: float) -> list[int]:
    """Inference indices where the firmware would fire: the first WINDOW are skipped after a reset, then the mean
    of the last WINDOW must reach cutoff, then REFRACTORY inferences are skipped."""
    hits, ignore = [], WINDOW
    for k in range(len(p)):
        if ignore > 0:
            ignore -= 1
            continue
        if k + 1 >= WINDOW and p[k + 1 - WINDOW:k + 1].mean() >= cutoff:
            hits.append(k)
            ignore = REFRACTORY
    return hits


def files(dirs: list[str]) -> list[Path]:
    return sorted(f for d in dirs for f in Path(d).expanduser().iterdir() if f.suffix.lower() in (".wav", ".m4a"))


def evaluate(pos: list[str], neg: list[str], model: Path) -> None:
    print("set", "files", "hours", *(f"c{c:.1f}" for c in CUTOFFS), sep="\t")
    for label, dirs in (("pos", pos), ("neg", neg)):
        for d in dirs:
            paths = files([d])
            probs = [probabilities(load(f, pad_s=0.5), model) for f in paths]
            hours = sum(len(p) for p in probs) * 0.03 / 3600
            # files that fired at least once (what matters for pos) / total firings (what matters for neg)
            row = [f"{sum(1 for p in probs if detections(p, c))}/{sum(len(detections(p, c)) for p in probs)}"
                   for c in CUTOFFS]
            print(f"{label}:{Path(d).name}", len(paths), f"{hours:.3f}", *row, sep="\t")
            for f, p in zip(paths, probs):
                peak = np.convolve(p, np.ones(WINDOW) / WINDOW, "valid").max(initial=0.0)
                if (label == "pos" and peak < CUTOFFS[-1]) or (label == "neg" and detections(p, CUTOFFS[0])):
                    print(f"  {f.name}: peak window mean {peak:.2f}")


def trace(paths: list[str], model: Path, cutoff: float) -> None:
    for f in paths:
        p = probabilities(load(Path(f)), model)
        hits = detections(p, cutoff)
        k = hits[0] if hits else int(np.argmax(p))
        print(Path(f).name, "first hit" if hits else "no hit, peak", k,
              "window", np.round(p[max(0, k - WINDOW + 1):k + 1], 2).tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pos", nargs="*", default=[], help="dirs of wake word takes")
    ap.add_argument("--neg", nargs="*", default=[], help="dirs of speech without the wake word")
    ap.add_argument("--trace", nargs="*", default=[], help="files to show the window at the first detection")
    ap.add_argument("--cutoff", type=float, default=0.9, help="cutoff for --trace (firmware kProbabilityCutoff)")
    ap.add_argument("--model", type=Path, default=MODEL)
    args = ap.parse_args()
    if args.trace:
        trace(args.trace, args.model, args.cutoff)
    if args.pos or args.neg:
        evaluate(args.pos, args.neg, args.model)
