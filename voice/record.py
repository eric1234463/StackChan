"""Record the speaker dataset for a custom TTS voice, one prompted sentence per take.

    uv run record.py --list-devices
    uv run record.py --device "USB"          # name substring or index; resumes where you stopped

Takes go to <out>/wavs/NNNN.wav (mono PCM16 at the device rate) and <out>/metadata.jsonl.
Keep recordings out of the repo: they are personal biometric data.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SENTENCES = Path(__file__).with_name("sentences.txt")
MIN_S, MAX_S = 3.5, 20.0  # recipe keeps 3-30 s clips with >= 4 s decoded; leave room for edge trimming
CLIP_DB, QUIET_DB, NOISY_DB = -1.0, -30.0, -50.0


def db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-9))


def check(audio: np.ndarray, rate: int) -> tuple[dict, list[str]]:
    """Level stats and the problems that make a take worth redoing."""
    frame = int(rate * 0.05)
    frames = np.sqrt([np.mean(audio[i:i + frame] ** 2) for i in range(0, max(len(audio) - frame, 1), frame)])
    stats = {"duration": round(len(audio) / rate, 2), "peak_db": round(db(float(np.abs(audio).max(initial=0))), 1),
             "floor_db": round(db(float(np.percentile(frames, 10))), 1)}
    problems = []
    if not MIN_S <= stats["duration"] <= MAX_S:
        problems.append(f"length {stats['duration']} s (want {MIN_S}-{MAX_S} s)")
    if stats["peak_db"] > CLIP_DB:
        problems.append(f"clipping (peak {stats['peak_db']} dBFS): move back or lower the input gain")
    elif stats["peak_db"] < QUIET_DB:
        problems.append(f"too quiet (peak {stats['peak_db']} dBFS): move closer or raise the input gain")
    if stats["floor_db"] > NOISY_DB:
        problems.append(f"background noise {stats['floor_db']} dBFS (want < {NOISY_DB}): TV, fan, aircon?")
    return stats, problems


def record(device, rate: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with sd.InputStream(device=device, samplerate=rate, channels=1, dtype="float32",
                        callback=lambda data, *_: chunks.append(data.copy())):
        input("  ● recording… press Enter when done ")
        time.sleep(0.3)  # keep the tail of the last syllable
    return np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, np.float32)


def find_device(spec: str | None):
    if spec is None:
        return None
    if spec.isdigit():
        return int(spec)
    matches = [i for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0 and spec.lower() in d["name"].lower()]
    if len(matches) != 1:
        sys.exit(f"--device {spec!r} matches {len(matches)} input devices; see --list-devices")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path.home() / "voice-data" / "eric")
    parser.add_argument("--device", help="input device name substring or index (default: system input)")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
    if args.list_devices:
        print("\n".join(f"{i}: {d['name']} ({d['max_input_channels']} in)" for i, d in enumerate(sd.query_devices())
                        if d["max_input_channels"] > 0))
        return

    device = find_device(args.device)
    info = sd.query_devices(device, "input")
    rate = int(info["default_samplerate"])
    sentences = [s.strip() for s in SENTENCES.read_text(encoding="utf-8").splitlines() if s.strip()]
    (args.out / "wavs").mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "metadata.jsonl"
    done = {json.loads(line)["id"] for line in meta_path.open(encoding="utf-8")} if meta_path.exists() else set()
    print(f"Mic: {info['name']} @ {rate} Hz. {len(done)}/{len(sentences)} done. Out: {args.out}")
    print("Same room, same distance (~20 cm), normal speaking voice. TV off.\n")

    for idx, text in enumerate(sentences):
        if idx in done:
            continue
        while True:
            print(f"[{len(done) + 1}/{len(sentences)}]  {text}")
            key = input("  Enter = record, s = skip, q = quit: ").strip().lower()
            if key == "q":
                return
            if key == "s":
                break
            audio = record(device, rate)
            stats, problems = check(audio, rate)
            print(f"  {stats['duration']} s, peak {stats['peak_db']} dBFS, noise {stats['floor_db']} dBFS"
                  + "".join(f"\n  ! {p}" for p in problems))
            key = input("  Enter = keep, r = redo, p = play back: ").strip().lower()
            while key == "p":
                sd.play(audio, rate)
                sd.wait()
                key = input("  Enter = keep, r = redo, p = play back: ").strip().lower()
            if key == "r":
                continue
            name = f"wavs/{idx:04d}.wav"
            sf.write(args.out / name, audio, rate, subtype="PCM_16")
            with meta_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": idx, "file": name, "text": text, "rate": rate, **stats,
                                    "problems": problems, "device": info["name"]}, ensure_ascii=False) + "\n")
            done.add(idx)
            break
    print(f"All done: {len(done)} takes in {args.out}")


if __name__ == "__main__":
    main()
