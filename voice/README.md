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
- Each take is checked for length (3.5–20 s), clipping, low level and background noise; redo flagged takes.
  The recipe's Sidon clean-up stage is CUDA-only, so recordings must be clean as recorded.
- `sentences.txt`: 60 assistant-style lines written for this project, plus 280 colloquial sentences from
  Common Voice `zh-HK` sentence-collector (CC0), filtered to 12–32 characters without 左/地 spellings.
