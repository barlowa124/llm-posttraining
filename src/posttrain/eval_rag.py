"""RAG eval: same stages and held-out entities as evaluate.py, but the
Context line comes from retrieval over a fact corpus rather than being
handed to the model.

    PYTHONPATH=src python -m posttrain.eval_rag data/processed/eval.parquet \
        data/processed/sft.parquet results/rag_eval.json

Outputs: results/rag_eval.json (per-stage per-kind rates + retrieval hit
rate), results/responses_<stage>_rag.csv (raw generations + retrieved docs,
inspectable), provenance.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch

from posttrain.config import load_config
from posttrain.provenance import write_manifest
from posttrain.rag import build_corpus, eval_rag


def main(eval_parquet: str, sft_parquet: str, out_json: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config()
    k = int(cfg.get("retrieval", {}).get("k", 3))
    torch.manual_seed(cfg["model"]["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ev = pd.read_parquet(eval_parquet)
    corpus = build_corpus(sft_parquet, eval_parquet)
    print(f"corpus: {len(corpus)} fact docs, k={k}")

    ckpts = [("base", None), ("sft", "data/processed/sft.pt"),
             ("dpo", "data/processed/dpo.pt"),
             ("grpo_s105", "data/processed/grpo_s105.pt")]
    result, responses = {}, {}
    for stage, ckpt in ckpts:
        if ckpt and not Path(ckpt).exists():
            print(f"skip {stage}: {ckpt} missing")
            continue
        model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"])
        if ckpt:
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
        result[stage], rdf = eval_rag(model, tok, ev, corpus, k)
        responses[stage] = rdf
        print(stage, result[stage])

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    for stage, rdf in responses.items():
        rdf.to_csv(f"results/responses_{stage}_rag.csv", index=False)
    write_manifest(
        "results/rag_eval_provenance.json",
        inputs=[eval_parquet, sft_parquet]
        + [c for _, c in ckpts if c and Path(c).exists()],
    )
    print(f"rag eval -> {out_json}")


if __name__ == "__main__":
    main(*sys.argv[1:])
