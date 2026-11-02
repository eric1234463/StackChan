"""Held-out talker loss of the start point and of each per-epoch adapter, for choosing --epochs.

    val_loss.py --trainer RECIPE/trainer --start START --jsonl holdout.jsonl ADAPTER_DIR ...

The forward is the trainer's (sft_12hz_lora_mlflow.py, sub-talker weight 0): teacher-forced cross-entropy on the
first codebook, the speaker vector from each clip's own speaker-encoder embedding. No dropout, no gradients.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("adapters", nargs="*")
    a = ap.parse_args()
    sys.path.insert(0, a.trainer)
    from dataset import TTSDataset
    from peft import PeftModel
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from transformers import AutoConfig

    tts = Qwen3TTSModel.from_pretrained(a.start, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = tts.model.to("mps").eval()
    rows = [json.loads(l) for l in open(a.jsonl, encoding="utf-8") if l.strip()]
    ds = TTSDataset(rows, tts.processor, AutoConfig.from_pretrained(a.start))
    batches = [ds.collate_fn([ds[i]]) for i in range(len(ds))]

    @torch.no_grad()
    def loss_of(m) -> float:
        tot, n = 0.0, 0
        for b in batches:
            b = {k: v.to("mps") for k, v in b.items()}
            spk = m.speaker_encoder(b["ref_mels"].to(torch.bfloat16))
            ids = b["input_ids"]
            emb = m.talker.text_projection(m.talker.model.text_embedding(ids[:, :, 0])) * b["text_embedding_mask"]
            cemb = m.talker.model.codec_embedding(ids[:, :, 1]) * b["codec_embedding_mask"]
            cemb[:, 6, :] = spk
            emb = emb + cemb
            for i in range(1, 16):
                emb = emb + m.talker.code_predictor.get_input_embeddings()[i - 1](
                    b["codec_ids"][:, :, i]) * b["codec_mask"].unsqueeze(-1)
            logits = m.talker(inputs_embeds=emb[:, :-1, :], attention_mask=b["attention_mask"][:, :-1]).logits
            labels = b["codec_0_labels"][:, 1:]
            tot += float(torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                                           labels.reshape(-1), ignore_index=-100, reduction="sum"))
            n += int((labels != -100).sum())
        return tot / n

    print(f"start {loss_of(model):.4f}", flush=True)
    peft = None
    for i, path in enumerate(a.adapters):
        if peft is None:
            peft = PeftModel.from_pretrained(model, path, adapter_name=f"a{i}").eval()
        else:
            peft.load_adapter(path, adapter_name=f"a{i}")
            peft.set_adapter(f"a{i}")
        print(f"{path} {loss_of(peft):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
