# speech

Local Wyoming servers used by the bridge (and Home Assistant):

| Service | Port | Engine | Model |
|---|---|---|---|
| STT `stackchan_speech.stt` | 10300 | mlx-whisper (Apple Silicon) | `WinsonLi/whisper-large-v3-cantonese-mlx-4bit` (HF cache, ~880 MB) |
| STT `stackchan_speech.stt --engine qwen3-asr` | 10301 | mlx-audio (Apple Silicon) | `Yvthyvq/Liujgoj-Cantonese-Qwen3-ASR-1.7B-V1` converted to MLX 8-bit (local dir, ~2 GB); bridge only |
| TTS `stackchan_speech.tts` | 10201 | mlx-audio (Apple Silicon) | Qwen3-TTS-12Hz-1.7B-Base + `jmtl/hk-cantonese-cc0-qwen3-tts` LoRA, merged and converted to MLX 8-bit (local dir, ~2.9 GB) |

```sh
TTS_MODEL_DIR=/path/to/qwen3-tts-...-8bit [TTS_VOICE=eric] ./install.sh   # uv sync + (re)load launchd agents
tail -f ~/Library/Logs/stackchan-stt.log ~/Library/Logs/stackchan-tts.log
```

`STT_MODEL` overrides the STT model (HF repo id or local MLX dir). The STT `--initial-prompt`
default biases toward HK Cantonese with English tech terms; edit `DEFAULT_PROMPT` in `stt.py`.

The Qwen3-ASR STT writes Liujgoj romanization (`Bongj ngoq hoij dangj`), not Chinese, so Home Assistant stays on
Whisper :10300 and the bridge uses :10301 with `STT_ROMANIZED=1`. A/B on 2026-11-04 (120 recorded lines, toneless
Jyutping letter error): Whisper 11.4%, Qwen3-ASR 10.5%; device clips p90 latency 1.24 s vs 0.18 s.
Build it once (4-bit breaks the fine-tune: it falls back to Simplified Chinese; 8-bit matches bf16):
`uv run python -m mlx_audio.convert --hf-path Yvthyvq/Liujgoj-Cantonese-Qwen3-ASR-1.7B-V1 --mlx-path ~/models/liujgoj-qwen3-asr-1.7b-mlx-8bit -q --q-bits 8 --model-domain stt`,
then `QWEN_ASR_MODEL_DIR=~/models/liujgoj-qwen3-asr-1.7b-mlx-8bit` for `install.sh`.

Building the TTS model (once): apply the LoRA with the adapter repo's `apply_adapter.py`
(`--base Qwen/Qwen3-TTS-12Hz-1.7B-Base --adapter jmtl/hk-cantonese-cc0-qwen3-tts --out hk-cantonese`), then
`python -m mlx_audio.convert --hf-path hk-cantonese --mlx-path qwen3-tts-hk-cantonese-8bit -q --q-bits 8 --model-domain tts`.
The voice is a real Common Voice CC-0 donor; the model card asks that output be marked as synthetic.

A custom voice (e.g. `eric`, trained with `voice/train/train.sh`) lives in the same model next to `hk_voice_343`;
select it with `TTS_VOICE` (default `hk_voice_343`).
