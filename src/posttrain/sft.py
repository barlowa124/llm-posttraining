"""Supervised fine-tune: completion-only causal LM loss."""

import sys
from pathlib import Path

import pandas as pd
import torch

from posttrain.config import load_config
from posttrain.lm import collate, encode_pair


def sft_train(model, tok, df, cfg, out_path):
    m = cfg["model"]
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=m["lr_sft"])
    examples = [
        encode_pair(tok, r["prompt"], r["completion"], m["max_len"])
        for _, r in df.iterrows()
    ]
    for ep in range(m["sft_epochs"]):
        perm = torch.randperm(len(examples))
        tot, nb = 0.0, 0
        for i in range(0, len(examples), m["batch_size"]):
            batch = [examples[j] for j in perm[i : i + m["batch_size"]]]
            ids, labels, attn = collate(batch, tok.pad_token_id)
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            tot += out.loss.item()
            nb += 1
        print(f"sft epoch {ep + 1}/{m['sft_epochs']} loss {tot / nb:.4f}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    return model


def main(sft_parquet: str, out_model: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config()
    torch.manual_seed(cfg["model"]["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"])
    df = pd.read_parquet(sft_parquet)
    sft_train(model, tok, df, cfg, out_model)
    print(f"saved {out_model}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
