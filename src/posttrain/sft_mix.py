"""SFT on both context regimes: the original baked-context training set
concatenated with its retrieval-rewritten counterpart (sft_rag), so the
answer/abstain supervision holds under clean context AND retrieved
context rather than trading one for the other.

    PYTHONPATH=src python -m posttrain.sft_mix data/processed/sft.parquet \
        data/processed/sft_mix.parquet data/processed/sft_mix.pt
"""

import sys
from pathlib import Path

import pandas as pd
import torch

from posttrain.config import load_config
from posttrain.rag import build_corpus, build_sft_rag_dataframe
from posttrain.sft import sft_train


def main(sft_parquet: str, out_parquet: str, out_model: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config()
    k = int(cfg.get("retrieval", {}).get("k", 3))
    torch.manual_seed(cfg["model"]["seed"])

    df = pd.read_parquet(sft_parquet)
    corpus = build_corpus(sft_parquet, sft_parquet)
    rdf = build_sft_rag_dataframe(df, corpus, k)
    base_cols = list(df.columns)
    mixed = pd.concat([df, rdf[base_cols]], ignore_index=True)
    mixed = mixed.sample(frac=1.0, random_state=cfg["data"]["seed"]
                         ).reset_index(drop=True)
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    mixed.to_parquet(out_parquet, index=False)
    abstain = mixed["completion"].str.contains("enough information").mean()
    print(f"sft_mix rows {len(mixed)} ({len(df)} clean + {len(rdf)} rag) | "
          f"abstain-target share {abstain:.2%}")

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"])
    sft_train(model, tok, mixed, cfg, out_model)
    print(f"saved {out_model}")


if __name__ == "__main__":
    main(*sys.argv[1:])
