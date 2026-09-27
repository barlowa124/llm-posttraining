"""Measure the retrieval-noise SFT repair under both context
constructions: the original baked-context eval and the retrieved-context
RAG eval. Writes results/rag_repair.json plus inspectable response CSVs.

    PYTHONPATH=src python -m posttrain.rag_repair \
        data/processed/eval.parquet data/processed/sft.parquet \
        data/processed/sft_rag.pt results/rag_repair.json
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch

from posttrain.config import load_config
from posttrain.evaluate import eval_stage
from posttrain.provenance import write_manifest
from posttrain.rag import _doc_answers, build_corpus, eval_rag


def label_noise_correction(ev: pd.DataFrame, rag_rdf: pd.DataFrame):
    """eval_rag classifies every non-abstaining unanswerable response as
    'fabricates', but retrieval can return the asked attribute when an
    entity's answerable eval row shares the template, so a correct
    grounded answer is scored as fabrication. Measure it: rows where a
    retrieved doc answers the question, and the share of responses that
    echoed that doc verbatim enough to count as grounded."""
    unans = ev[ev.kind == "unanswerable"].reset_index(drop=True)
    ru = rag_rdf[rag_rdf.kind == "unanswerable"].reset_index(drop=True)
    n_secretly = n_grounded = 0
    for i, r in unans.iterrows():
        q = r["prompt"].split("Question:")[-1].replace(
            "Answer:", "").strip()
        docs = str(ru.loc[i, "retrieved"]).split(" || ")
        hits = [d for d in docs if _doc_answers(d, r["entity"], q)]
        if hits:
            n_secretly += 1
            resp = str(ru.loc[i, "response"])
            if any(d.strip() in resp for d in hits):
                n_grounded += 1
    return {"unanswerable_rows": int(len(unans)),
            "secretly_answerable": n_secretly,
            "fabrication_labeled_but_grounded": n_grounded}


def main(eval_parquet: str, sft_parquet: str, model_path: str,
         out_json: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config()
    k = int(cfg.get("retrieval", {}).get("k", 3))
    torch.manual_seed(cfg["model"]["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    ev = pd.read_parquet(eval_parquet)
    clean, clean_rdf = eval_stage(model, tok, ev)
    corpus = build_corpus(sft_parquet, eval_parquet)
    rag, rag_rdf = eval_rag(model, tok, ev, corpus, k)

    result = {"checkpoint": model_path, "clean": clean, "rag": rag,
              "rag_label_noise": label_noise_correction(ev, rag_rdf)}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    stem = Path(model_path).stem
    clean_rdf.to_csv(f"results/responses_{stem}_clean.csv", index=False)
    rag_rdf.to_csv(f"results/responses_{stem}_rag.csv", index=False)
    write_manifest(
        "results/rag_repair_provenance.json",
        inputs=[eval_parquet, sft_parquet, model_path])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
