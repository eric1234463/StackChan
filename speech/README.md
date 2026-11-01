# speech

Local Wyoming servers used by the bridge (and Home Assistant):

| Service | Port | Engine | Model |
|---|---|---|---|
| STT `stackchan_speech.stt` | 10300 | mlx-whisper (Apple Silicon) | `WinsonLi/whisper-large-v3-cantonese-mlx-4bit` (HF cache, ~880 MB) |
| TTS `stackchan_speech.tts` | 10201 | sherpa-onnx VITS | `csukuangfj/vits-cantonese-hf-xiaomaiiwn` (local dir) |

```sh
TTS_MODEL_DIR=/path/to/canto-vits ./install.sh   # uv sync + (re)load launchd agents
tail -f ~/Library/Logs/stackchan-stt.log ~/Library/Logs/stackchan-tts.log
```

`STT_MODEL` overrides the STT model (HF repo id or local MLX dir). The STT `--initial-prompt`
default biases toward HK Cantonese with English tech terms; edit `DEFAULT_PROMPT` in `stt.py`.
