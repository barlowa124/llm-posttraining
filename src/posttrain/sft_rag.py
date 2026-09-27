"""SFT under retrieval noise: rebuild the training set with each prompt's
Context line replaced by BM25-retrieved docs, then fine-tune from base.

Repair attempt for the failure eval_rag measured. Abstention learned on
clean context did not transfer to retrieved context, so the training
distribution is shifted to match the deployment distribution.
The corpus is built from the train split only, so no held-out eval fact
can leak into a training context.

    PYTHONPATH=src python -m posttrain.sft_rag data/processed/sft.parquet \
        data/processed/sft_rag.parquet data/processed/sft_rag.pt
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
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    rdf.to_parquet(out_parquet, index=False)
    hit = (rdf["n_retrieved"] > 0).mean()
    flips = rdf.groupby("kind")["completion_flipped"].mean().to_dict()
    abstain = rdf["completion"].str.contains(
        "enough information").mean()
    print(f"sft_rag rows {len(rdf)} | corpus {len(corpus)} docs | k={k} | "
          f"rows with >=1 doc: {hit:.2%} | flips {flips} | "
          f"abstain-target share {abstain:.2%}")

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"])
    sft_train(model, tok, rdf, cfg, out_model)
    print(f"saved {out_model}")


if __name__ == "__main__":
    main(*sys.argv[1:])
