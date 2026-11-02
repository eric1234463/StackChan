#!/usr/bin/env bash
# Train the speaker voice `eric` (codec row 3001) on top of the HK Cantonese Qwen3-TTS weights, on this Mac (torch
# MPS), and convert the result to MLX 8-bit for stackchan_speech.tts. hk_voice_343 (row 3000) is kept bit-identical.
#
#   voice/train/train.sh --data ~/voice-data/eric --out ~/models/qwen3-tts-eric-8bit [--exclude 28] [--epochs 2]
#                        [--holdout 10 --seed 1] [--save-epochs] [--work ~/models/qwen3-train-check/full]
#
# Everything downloaded or written lives under --work (venvs, recipe clone, weights, preprocessed audio) or --out;
# nothing is written to the repo or to --data.
set -euo pipefail

RECIPE_URL=https://github.com/hobpond/qwen3-tts-cantonese-recipe
RECIPE_REV=4ec93058accd0e27a42a9d745821a94cde2c5837
BASE_REPO=Qwen/Qwen3-TTS-12Hz-1.7B-Base
BASE_REV=fd4b254389122332181a7c3db7f27e918eec64e3
ADAPTER_REPO=jmtl/hk-cantonese-cc0-qwen3-tts
ADAPTER_REV=86226d022bc793473ee57aa91642fa590fb1d63f
START_SHA256=23345ce612733b1d0861a9313021735e15ff5bb6eaa029757bf360905d087083   # apply_adapter --trainable output
TORCH_PKGS=(torch==2.15.0 torchaudio==2.11.1 transformers==4.57.3 accelerate==1.12.0 peft==0.20.0
            safetensors==0.8.0 qwen-tts==0.1.1 mlflow==3.15.1 librosa==1.0.0 soundfile==0.14.0 scipy==1.18.1
            numpy==2.5.4 huggingface-hub==0.36.2)
MLX_PKGS=(mlx-audio==0.6.3 mlx==0.32.4)

DATA= OUT= EXCLUDE= EPOCHS=2 HOLDOUT=0 SEED=1 SAVE_EPOCHS= NAME=eric SLOT=3001
WORK="$HOME/models/qwen3-train-check/full"
while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATA=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --exclude) EXCLUDE=$2; shift 2 ;;
    --epochs) EPOCHS=$2; shift 2 ;;
    --holdout) HOLDOUT=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --save-epochs) SAVE_EPOCHS=--save_adapter_every_epoch; shift ;;
    --work) WORK=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$DATA" ] && [ -n "$OUT" ] || { sed -n '5,6p' "$0" >&2; exit 2; }
[ -e "$OUT" ] && { echo "$OUT exists; move it away first" >&2; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=$(cd "$DATA" && pwd)
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
PY=$WORK/venv-torch/bin/python
MLXPY=$WORK/venv-mlx/bin/python

# 1. venvs (pinned)
if [ ! -x "$PY" ]; then
  uv venv -q -p 3.12 "$WORK/venv-torch" && VIRTUAL_ENV=$WORK/venv-torch uv pip install -q "${TORCH_PKGS[@]}"
fi
if [ ! -x "$MLXPY" ]; then
  uv venv -q -p 3.12 "$WORK/venv-mlx" && VIRTUAL_ENV=$WORK/venv-mlx uv pip install -q "${MLX_PKGS[@]}"
fi

# 2. recipe at the pinned commit, plus our MPS patch
R=$WORK/recipe
if [ ! -d "$R/.git" ]; then
  git clone -q "$RECIPE_URL" "$R"
  git -C "$R" checkout -q "$RECIPE_REV"
  git -C "$R" apply "$HERE/recipe-mps.patch"
fi
[ "$(git -C "$R" rev-parse HEAD)" = "$RECIPE_REV" ] || { echo "$R is not at $RECIPE_REV" >&2; exit 1; }
git -C "$R" diff --quiet -- trainer && { echo "$R: recipe-mps.patch is not applied" >&2; exit 1; }

# 3. weights at pinned revisions, and the trainable start point (HK Cantonese weights + speaker encoder)
"$WORK/venv-torch/bin/hf" download --quiet "$BASE_REPO" --revision "$BASE_REV" --local-dir "$WORK/hf/base" >/dev/null
"$WORK/venv-torch/bin/hf" download --quiet "$ADAPTER_REPO" --revision "$ADAPTER_REV" --local-dir "$WORK/hf/adapter" >/dev/null
START=$WORK/start
rm -rf "$START"
"$PY" -I "$WORK/hf/adapter/scripts/apply_adapter.py" --base "$WORK/hf/base" --adapter "$WORK/hf/adapter" \
  --out "$START" --trainable --expect-sha256 "$START_SHA256"

# 4. preprocess + encode (the user's clips are read, never written)
rm -rf "$WORK/data"
"$PY" -I "$HERE/pipeline.py" prep --recipe "$R" --data "$DATA" --tokenizer "$WORK/hf/base/speech_tokenizer" \
  --out "$WORK/data" --exclude "$EXCLUDE" --holdout "$HOLDOUT" --seed "$SEED" --name "$NAME"

# 5. speaker phase: the recipe's [train] values, one voice at row $SLOT
T=$WORK/train
rm -rf "$T"
(cd "$R/trainer" && "$PY" sft_12hz_lora_mlflow.py --init_model_path "$START" --output_model_path "$T" \
  --train_jsonl "$WORK/data/train.jsonl" --batch_size 1 --gradient_accumulation_steps 4 --lr 1e-4 \
  --num_epochs "$EPOCHS" --warmup_ratio 0.03 --lora_rank 32 --lora_alpha 64 --checkpoint_every_epochs 0 \
  --checkpoint_every_steps 500 --sub_talker_weight 0 --attn_implementation sdpa --seed 1 \
  --mlflow_tracking_uri "sqlite:///$T.mlflow.db" --speaker_name "$NAME" --speaker_slot "$SLOT" $SAVE_EPOCHS) \
  2>&1 | tee "$T.log" | grep -E --line-buffered "MPSSTEP|Epoch .* finished|Traceback|Error"
[ -f "$T/final_merged/model.safetensors" ] || { echo "training failed; see $T.log" >&2; exit 1; }

# 6. bake: name both voices, verify row 3000 is untouched; 7. MLX 8-bit
"$PY" -I "$HERE/pipeline.py" bake --start "$START" --export "$T/final_merged" --name "$NAME" --slot "$SLOT"
"$MLXPY" -I -m mlx_audio.convert --hf-path "$T/final_merged" --mlx-path "$OUT" -q --q-bits 8 --model-domain tts

# keep the LoRA adapter (small) and logs; drop the merged checkpoints
mv "$T/final_merged/lora_adapter" "$T/lora_adapter"
rm -rf "$T/final_merged" "$START"
echo "$OUT"
