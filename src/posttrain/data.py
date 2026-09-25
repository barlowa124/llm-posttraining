"""Synthetic abstention-task data.

Templated pharmacology-style facts over invented entity names (so the
held-out entity split is a true generalization test, not memorization).
The task is deliberately synthetic — what is being demonstrated is the
post-training *mechanics* and measured behavior change, not a new dataset.

Each example: context + question + completion. Answerable examples place
the queried fact in context; unanswerable examples contain only *other*
entities' facts, so copying is a fabrication.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from posttrain.config import load_config

ABSTAIN = "I don't have enough information to answer."

PROMPT = (
    "Answer the question using only the context. If the context does not "
    "contain the answer, say exactly: \"" + ABSTAIN + "\"\n\n"
    "Context: {ctx}\nQuestion: {q}\nAnswer:"
)

FACT_TEMPLATES = [
    ("The half-life of {e} is {v} hours.",
     "What is the half-life of {e}?"),
    ("The molecular weight of {e} is {v} daltons.",
     "What is the molecular weight of {e}?"),
    ("{e} is primarily cleared by the {v}.",
     "How is {e} primarily cleared?"),
]

FILLERS = [
    "The assay was run in triplicate at room temperature.",
    "Buffer conditions were held constant across replicates.",
    "The compound was stable for at least one week at 4 C.",
]


def entity_names(n: int, seed: int):
    rng = np.random.default_rng(seed)
    pre = ["vel", "dora", "tam", "osu", "len", "ribo", "nira", "selu",
           "capma", "eni", "lorla", "pemi", "soto", "zano"]
    suf = ["tinib", "mab", "statin", "ciclib", "parib", "lukast", "xaban",
           "gliptin", "sartan", "prazole"]
    names = set()
    while len(names) < n:
        names.add(rng.choice(pre) + rng.choice(suf) + "-" + str(rng.integers(100, 999)))
    return sorted(names)


def _value(template_i: int, rng):
    if template_i == 0:
        return f"{rng.integers(2, 96)}"
    if template_i == 1:
        return f"{rng.integers(200, 900)}"
    return str(rng.choice(["liver", "kidney"]))


def build_examples(cfg) -> pd.DataFrame:
    d = cfg["data"]
    rng = np.random.default_rng(d["seed"])
    names = entity_names(d["n_entities"], d["seed"])
    rows = []

    def fact(e, i, rng):
        t, q = FACT_TEMPLATES[i]
        v = _value(i, rng)
        return t.format(e=e, v=v), q.format(e=e), t.format(e=e, v=v)

    for e in names:
        for _ in range(d["n_answerable_per_entity"]):
            i = int(rng.integers(len(FACT_TEMPLATES)))
            ctx, q, ans = fact(e, i, rng)
            ctx += " " + str(rng.choice(FILLERS))
            rows.append({"entity": e, "kind": "answerable",
                         "prompt": PROMPT.format(ctx=ctx, q=q),
                         "completion": " " + ans, "gold": ans})
        # unanswerable: context carries a different entity's fact
        for _ in range(d["n_unanswerable_train"] // d["n_entities"]):
            i = int(rng.integers(len(FACT_TEMPLATES)))
            other = str(rng.choice([x for x in names if x != e]))
            ctx, q, _ = fact(other, i, rng)
            _, q_e, _ = fact(e, i, rng)
            rows.append({"entity": e, "kind": "unanswerable",
                         "prompt": PROMPT.format(ctx=ctx, q=q_e),
                         "completion": " " + ABSTAIN, "gold": ABSTAIN})
    return pd.DataFrame(rows)


def build_dpo_pairs(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Preference pairs: chosen vs rejected completion for the same prompt."""
    rng = np.random.default_rng(cfg["data"]["seed"] + 1)
    rows = []
    ans = df[df.kind == "answerable"]
    unans = df[df.kind == "unanswerable"]
    n = cfg["data"]["n_dpo_pairs"] // 2
    for _, r in ans.sample(n=min(n, len(ans)), random_state=cfg["data"]["seed"]).iterrows():
        # rejected: same fact shape, wrong value
        wrong = r["gold"]
        while wrong == r["gold"]:
            i = int(rng.integers(len(FACT_TEMPLATES)))
            wrong = FACT_TEMPLATES[i][0].format(e=r["entity"], v=_value(i, rng))
        rows.append({"prompt": r["prompt"], "chosen": r["completion"],
                     "rejected": " " + wrong})
    for _, r in unans.sample(n=min(n, len(unans)),
                             random_state=cfg["data"]["seed"] + 2).iterrows():
        # rejected: fabricated plausible answer naming the queried entity
        i = int(rng.integers(len(FACT_TEMPLATES)))
        fab = FACT_TEMPLATES[i][0].format(e=r["entity"], v=_value(i, rng))
        rows.append({"prompt": r["prompt"], "chosen": r["completion"],
                     "rejected": " " + fab})
    return pd.DataFrame(rows)


def build_eval(cfg) -> pd.DataFrame:
    """Held-out entities only: unseen names at eval time."""
    d = cfg["data"]
    rng = np.random.default_rng(d["seed"] + 3)
    names = entity_names(d["n_entities"], d["seed"])[-d["heldout_entities"]:]
    all_names = entity_names(d["n_entities"], d["seed"])
    rows = []
    for e in names:
        for kind in ("answerable", "unanswerable"):
            i = int(rng.integers(len(FACT_TEMPLATES)))
            t, q = FACT_TEMPLATES[i]
            v = _value(i, rng)
            if kind == "answerable":
                ctx = t.format(e=e, v=v)
                gold = t.format(e=e, v=v)
            else:
                other = str(rng.choice([x for x in all_names if x != e]))
                ctx = t.format(e=other, v=v)
                gold = ABSTAIN
            rows.append({"entity": e, "kind": kind,
                         "prompt": PROMPT.format(ctx=ctx, q=q.format(e=e)),
                         "gold": gold, "gold_value": v})
    return pd.DataFrame(rows)


def main(out_dir: str):
    cfg = load_config()
    df = build_examples(cfg)
    heldout = set(entity_names(cfg["data"]["n_entities"],
                               cfg["data"]["seed"])[-cfg["data"]["heldout_entities"]:])
    train = df[~df.entity.isin(heldout)].reset_index(drop=True)
    pairs = build_dpo_pairs(train, cfg)
    ev = build_eval(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "sft.parquet", index=False)
    pairs.to_parquet(out / "dpo.parquet", index=False)
    ev.to_parquet(out / "eval.parquet", index=False)
    print(f"sft {len(train)} | dpo {len(pairs)} | eval {len(ev)} "
          f"({len(heldout)} held-out entities)")


if __name__ == "__main__":
    main(sys.argv[1])
