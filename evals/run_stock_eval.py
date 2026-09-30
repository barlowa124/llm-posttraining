"""Run the abstention eval against stock open-weights models.

Same protocol as posttrain.evaluate: greedy decode (do_sample=False),
the repo's classify() labels each response abstains / correct /
degenerates / fabricates. Instruct models get the raw eval prompt —
no chat template — so the comparison is same-protocol, not per-model
prompt engineering. Results land in evals/results/; the report
generator binds every number in EVAL_REPORT.md to them.

Run from the repo root:

    PYTHONPATH=src:evals .venv/bin/python evals/run_stock_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.insert(0, "evals")
from claims import flatten_results, verify_markdown  # noqa: E402

from posttrain.evaluate import eval_stage  # noqa: E402
from posttrain.provenance import write_manifest  # noqa: E402


def _slug(name: str) -> str:
    return name.split("/")[-1].lower().replace("-", "_")


def main(report_only: bool = False) -> None:
    cfg = yaml.safe_load(open("evals/eval_config.yaml"))
    ev = pd.read_parquet(cfg["eval_parquet"])
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    if report_only:
        res = json.loads((results_dir / "eval_stock.json").read_text())
        write_report(res, ev, cfg)
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    per_model = {}
    for m in cfg["models"]:
        slug = _slug(m["name"])
        tok = AutoTokenizer.from_pretrained(m["name"])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(m["name"])
        model.eval()
        metrics, rdf = eval_stage(model, tok, ev)
        per_model[slug] = {
            "name": m["name"], "license": m["license"],
            "params_m": m["params_m"], **metrics,
        }
        rdf.to_csv(results_dir / f"responses_{slug}.csv", index=False)
        print(slug, metrics)
        del model

    out = {
        "eval": "abstention_eval",
        "max_new": cfg["max_new"],
        "n_prompts": int(len(ev)),
        "n_answerable": int((ev.kind == "answerable").sum()),
        "n_unanswerable": int((ev.kind == "unanswerable").sum()),
        "protocol": "greedy decode, raw completion prompt, "
                    "classify() abstains/correct/degenerate/fabricates",
        "models": per_model,
    }
    with open(results_dir / "eval_stock.json", "w") as f:
        json.dump(out, f, indent=2)
    write_manifest(
        str(results_dir / "provenance.json"),
        inputs=[cfg["eval_parquet"]]
        + [m["name"] for m in cfg["models"]],
    )
    write_report(out, ev, cfg)
    print("wrote", results_dir / "eval_stock.json")


def _fmt_rate(x: float) -> str:
    """Exact-precision rate: n=80 quanta are exact at 4 decimals; strip
    trailing zeros so every printed token equals the recorded leaf."""
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def write_report(res: dict, ev: pd.DataFrame, cfg: dict) -> None:
    results_dir = Path(cfg["results_dir"])
    models = res["models"]
    abst = {k: m["unanswerable"]["abstains"] for k, m in models.items()}
    fab = {k: m["unanswerable"]["fabricates"] for k, m in models.items()}
    lo, hi = min(abst, key=abst.get), max(abst, key=abst.get)
    hi_fab = max(fab, key=fab.get)

    rows = [
        f"| {m['name']} ({m['license'].split('-')[0]} license) "
        f"| {m['params_m']} "
        f"| {_fmt_rate(m['answerable']['correct'])} "
        f"| {_fmt_rate(m['answerable']['fabricates'])} "
        f"| {_fmt_rate(m['unanswerable']['abstains'])} "
        f"| {_fmt_rate(m['unanswerable']['fabricates'])} |"
        for m in models.values()
    ]
    md = "\n".join([
        "# Eval report: stock models on the abstention eval",
        "",
        "The repo's held-out abstention suite run end-to-end on stock "
        "open-weights checkpoints. Both models are small and the suite "
        "is small, so the report reads as a measured abstain/fabricate "
        "split for these checkpoints, not a benchmark ranking.",
        "",
        f"Each of the {res['n_prompts']} prompts "
        f"({res['n_answerable']} answerable, {res['n_unanswerable']} "
        "unanswerable) is judged under one protocol: "
        f"{res['protocol']}, max {res['max_new']} new tokens per "
        "response.",
        "",
        "## Rates",
        "",
        "| model | params (M) | ans. correct | ans. fabricates "
        "| unans. abstains | unans. fabricates |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "On unanswerable prompts the abstention rate spans from "
        f"{_fmt_rate(abst[lo])} ({models[lo]['name']}) to "
        f"{_fmt_rate(abst[hi])} ({models[hi]['name']}). "
        f"{models[hi_fab]['name']} fabricates on "
        f"{_fmt_rate(fab[hi_fab])} of them.",
        "",
        "Each generation's label is inspectable in "
        "`evals/results/responses_<model>.csv`. Degenerate "
        "(repetition-collapse) rates are in eval_stock.json.",
        "",
        "## Reading the table",
        "",
        "On answerable prompts the desired label is `correct`. On "
        "unanswerable prompts it is `abstains`. A stock instruct model "
        "typically answers regardless, so the unanswerable fabricate "
        "rate is the quantity this eval exists to expose.",
        "",
        "## Provenance",
        "",
        f"- eval set: `data/processed/eval.parquet` "
        f"({res['n_answerable']} held-out synthetic entities, never "
        "used in training)",
        "- results: `evals/results/eval_stock.json`",
        "- claims check: `evals/results/claims_check.json`. Every "
        "number above re-derives from the results JSON at display "
        "tolerance.",
        "- scope: synthetic pharmacology facts, small eval, greedy "
        "decoding. Findings describe this eval only.",
    ]) + "\n"
    Path(cfg["report_md"]).write_text(md)

    pool = {}
    for name in ("eval_stock",):
        src = json.loads((results_dir / f"{name}.json").read_text())
        for k, v in flatten_results(src).items():
            pool[f"{name}.{k}"] = v
    verdict = verify_markdown(md, pool)
    verdict["verified_by"] = "evals.claims (vendored, parity-pinned)"
    with open(results_dir / "claims_check.json", "w") as f:
        json.dump(verdict, f, indent=2)
    if not verdict["passed"]:
        print("UNBOUND:", [u["token"] for u in verdict["unbound_claims"]])


if __name__ == "__main__":
    sys.exit(main(report_only="--report-only" in sys.argv))
