"""Data and bake steps of train.sh (run by it, inside the work dir's torch venv).

    pipeline.py prep --recipe R --data D --tokenizer T --out O [--exclude 28] [--holdout 10 --seed 1]
    pipeline.py bake --start S --export E --name eric --slot 3001

prep: the recipe's speaker-clip path without Sidon (CUDA-only) or its dynamics step: 80 Hz high-pass at the
source rate (mains hum, 36-134 Hz in the takes) -> librosa resample to 24 kHz -> ingest trim_edges(0.15, 0.20) ->
energy -> edges v2 -> trim (0.10 s before, 0.30 s after the speech core) -> pad back to the ingested length with
digital silence -> per-clip loudness: speech RMS to TARGET_RMS_DB, capped so the peak stays <= PEAK_CAP_DB (the
model reproduces training loudness: a fixed -7 dB on quietly recorded takes gave voices ~11 dB quieter) ->
12 Hz codec codes. Writes train.jsonl / holdout.jsonl (trainer rows) and clips.json.
bake: the trainer wrote the new voice's mean speaker vector at --slot; hk_voice_343 stays at row 3000. Refuses unless
every codec_embedding row except --slot equals the start point bit for bit, then names both voices in spk_id.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

TARGET_RMS_DB = -20.0  # speech-frame RMS; hk_voice_343 outputs sit around -24 dBFS RMS / -5 dBFS peak
PEAK_CAP_DB = -1.0
RATE = 24000
KEY = "talker.model.codec_embedding.weight"


def prep(a) -> None:
    import numpy as np
    import librosa
    import scipy.signal as ss
    import soundfile as sf
    sys.path.insert(0, str(Path(a.recipe) / "src"))
    import steps as st
    from ingest_recordings import trim_edges

    out = Path(a.out).resolve()
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    exclude = {int(x) for x in a.exclude.split(",") if x}
    meta = [json.loads(l) for l in open(Path(a.data) / "metadata.jsonl", encoding="utf-8") if l.strip()]
    meta = [r for r in meta if r["id"] not in exclude]
    held = set(random.Random(a.seed).sample(sorted(r["id"] for r in meta), a.holdout)) if a.holdout else set()

    def loudness_gain(x):
        frame = int(RATE * 0.05)
        e = np.sqrt([np.mean(x[i:i + frame] ** 2) for i in range(0, max(len(x) - frame, 1), frame)])
        speech_rms = float(np.mean(e[e > 0.1 * e.max()])) if e.max() > 0 else 1.0
        peak = float(np.abs(x).max()) or 1.0
        return min(10 ** (TARGET_RMS_DB / 20) / speech_rms, 10 ** (PEAK_CAP_DB / 20) / peak)

    clips = []
    for r in meta:
        y, sr = sf.read(Path(a.data) / r["file"], dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = ss.sosfiltfilt(ss.butter(4, 80, "highpass", fs=sr, output="sos"), y).astype(np.float32)
        y = librosa.resample(y=y, orig_sr=sr, target_sr=RATE) if sr != RATE else y
        y = trim_edges(y, RATE, 0.15, 0.20)
        n = len(y)
        segs = st.segments(st.frames_db(y, RATE, 0.01), 0.01, -40.0, 0.15, 0.04)   # recipe energy step params
        ed = st.edges_of(y, RATE, segs, st.EDGES_PARAMS)
        z = np.zeros(n, np.float32)
        if ed.get("speech"):
            s0 = max(0, int(np.floor((ed["core_start_s"] - ed.get("lead_attached_s", 0.0) - 0.10) * RATE)))
            s1 = min(n, int(np.ceil((ed["core_end_s"] + ed.get("trail_attached_s", 0.0) + 0.30) * RATE)))
            z[s0:s1] = y[s0:s1]
        else:
            z[:] = y
        z *= loudness_gain(z)
        path = out / "wavs" / f"{r['id']:04d}.wav"
        sf.write(path, z, RATE, subtype="PCM_16")
        clips.append({"id": r["id"], "audio": str(path), "text": r["text"], "duration_sec": round(n / RATE, 2),
                      "peak_dbfs": round(float(20 * np.log10(np.abs(z).max() + 1e-12)), 1),
                      "split": "holdout" if r["id"] in held else "train"})
    if max(c["peak_dbfs"] for c in clips) > PEAK_CAP_DB + 0.1:
        raise SystemExit(f"a clip exceeds {PEAK_CAP_DB} dBFS after loudness normalisation")

    from qwen_tts import Qwen3TTSTokenizer
    tok = Qwen3TTSTokenizer.from_pretrained(a.tokenizer, device_map=a.device)
    files = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "holdout")}
    for c in clips:   # one clip per encode call, as the recipe's encode step
        codes = tok.encode([c["audio"]]).audio_codes[0].cpu().tolist()
        files[c["split"]].write(json.dumps({"audio": c["audio"], "text": c["text"], "ref_audio": c["audio"],
                                            "speaker_id": a.name, "duration_sec": c["duration_sec"],
                                            "audio_codes": codes}, ensure_ascii=False) + "\n")
    for f in files.values():
        f.close()
    json.dump(clips, open(out / "clips.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n_train = sum(c["split"] == "train" for c in clips)
    print(f"prep: {n_train} train, {len(clips) - n_train} holdout {sorted(held)}, excluded {sorted(exclude)}, "
          f"{sum(c['duration_sec'] for c in clips if c['split'] == 'train') / 60:.1f} min train audio")


def bake(a) -> None:
    import torch
    from safetensors import safe_open
    start = safe_open(str(Path(a.start) / "model.safetensors"), "pt").get_tensor(KEY)
    trained = safe_open(str(Path(a.export) / "model.safetensors"), "pt").get_tensor(KEY)
    keep = torch.ones(start.shape[0], dtype=torch.bool)
    keep[a.slot] = False
    if not torch.equal(trained[keep], start[keep]):
        raise SystemExit("codec_embedding rows other than the new voice's differ from the start point")
    if torch.equal(trained[a.slot], start[a.slot]):
        raise SystemExit(f"row {a.slot} was not written by the trainer")
    cfg_p = Path(a.export) / "config.json"
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    if cfg.get("tts_model_size") != "1b7" or cfg.get("tts_model_type") != "custom_voice":
        raise SystemExit(f"unexpected export config: {cfg.get('tts_model_size')} {cfg.get('tts_model_type')}")
    voices = {"hk_voice_343": 3000, a.name: a.slot}
    cfg["talker_config"]["spk_id"] = voices
    cfg["talker_config"]["spk_is_dialect"] = {v: False for v in voices}
    cfg_p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cos = torch.nn.functional.cosine_similarity(trained[a.slot].float(), start[3000].float(), dim=0)
    print(f"bake: spk_id {voices}; hk_voice_343 row bit-identical to the start point; "
          f"cos({a.name}, hk_voice_343) = {float(cos):.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--recipe", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--exclude", default="")
    p.add_argument("--holdout", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--name", default="eric")
    p.add_argument("--device", default="mps")
    b = sub.add_parser("bake")
    b.add_argument("--start", required=True)
    b.add_argument("--export", required=True)
    b.add_argument("--name", default="eric")
    b.add_argument("--slot", type=int, default=3001)
    a = ap.parse_args()
    {"prep": prep, "bake": bake}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
