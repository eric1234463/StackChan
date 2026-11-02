# voice

Records the speaker dataset for a custom StackChan TTS voice (Qwen3-TTS HK Cantonese, speaker LoRA trained on
this Mac with torch MPS — feasibility checked 2026-11-02: ~0.9 s per optimizer step, ~15 GB peak).

```sh
uv run record.py --list-devices
uv run record.py --device "<external mic name>"   # resumable; q to stop, rerun to continue
# Start with the first 120 (60 assistant lines + 60 everyday lines, alternating), train, listen;
# record further only if the voice is not close enough.
```

- Output: `~/voice-data/eric/wavs/NNNN.wav` + `metadata.jsonl` (text, levels, problems). Not in the repo.
- Each take is checked for length (2–20 s), clipping, low level and background noise; redo flagged takes.
  The recipe's Sidon clean-up stage is CUDA-only, so recordings must be clean as recorded.
- `sentences.txt`: 60 assistant-style lines written for this project, plus 280 colloquial sentences from
  Common Voice `zh-HK` sentence-collector (CC0), filtered to 12–32 characters without 左/地 spellings.

## Training the voice

```sh
voice/train/train.sh --data ~/voice-data/eric --out ~/models/qwen3-tts-eric-8bit --exclude 28
# then: TTS_MODEL_DIR=~/models/qwen3-tts-eric-8bit, voice name `eric` (hk_voice_343 is still in the model)
```

- Fetches [qwen3-tts-cantonese-recipe](https://github.com/hobpond/qwen3-tts-cantonese-recipe), Qwen3-TTS-12Hz-1.7B-Base
  and `jmtl/hk-cantonese-cc0-qwen3-tts` at the revisions pinned in `train.sh`, into `--work`
  (default `~/models/qwen3-train-check/full`, with its own uv venvs). Nothing is written to the repo or `--data`.
- `recipe-mps.patch`: the recipe trainer's CUDA guard also accepts MPS, per-step MPS timing/memory lines,
  `--speaker_slot` (the voice is baked at row 3001, so row 3000 / hk_voice_343 stays bit-identical) and
  `--save_adapter_every_epoch`.
- `pipeline.py prep`: 80 Hz high-pass, 24 kHz, the recipe's ingest trim + edges/trim/pad (no Sidon: CUDA-only),
  fixed −7 dB, 12 Hz codec codes. `pipeline.py bake`: checks the codec rows and names both voices.
- Speaker phase with the recipe's `[train]` values (r32/α64, lr 1e-4, batch 1 × accum 4, seed 1) for `--epochs`
  (default 2: on 109 takes the held-out loss is lowest at 2 epochs and rises from the third).
  About 2 min end to end on an M5 Max, ~9 GB MPS memory.
- Choosing epochs again (more takes): `--holdout 10 --seed 1 --epochs 6 --save-epochs`, then
  `val_loss.py --trainer WORK/recipe/trainer --start <apply_adapter --trainable dir> --jsonl WORK/data/holdout.jsonl
  WORK/train/adapter-epoch-*` and pick the epoch before the held-out loss turns up.
