"""Reward-variance probe: does a checkpoint's rollouts give GRPO any
within-group signal at a given sampling temperature?

This is the measurement behind the README's shaped-reward claim: at the
configured temperature a saturated policy produces all-+1 groups and a
collapsed one produces nearly-uniform -1 groups — both zero advantage.
The probe rolls each named checkpoint over the RL prompt pool at one or
more temperatures and reports, per cell, how many prompt groups carry
nonzero advantage.

    PYTHONPATH=src python -m posttrain.probe_variance \
        data/processed/rl.parquet sft=data/processed/sft.pt \
        dpo=data/processed/dpo.pt \
        --temperatures 1.0 2.5 --prompts 48 \
        --output results/reward_variance_probe.json

Not a Snakefile stage: it is a diagnostic over existing checkpoints, not
part of the pipeline. Checkpoint inputs are hashed into the output so
the artifact stays bound to the weights it measured.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from posttrain.config import load_config
from posttrain.grpo import group_advantages, reward, rollout


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe(policy, tok, pool, n_prompts: int, k: int, max_new: int,
          temperature: float, pad_id: int, shaped: bool) -> dict:
    batch = pool.iloc[:n_prompts]
    prompts = batch["prompt"].tolist()
    texts, _ = rollout(policy, tok, prompts, k, max_new, temperature, pad_id)
    rewards = np.array(
        [
            reward(t, batch.iloc[i // k]["entity"], batch.iloc[i // k]["gold_value"],
                   batch.iloc[i // k]["kind"], shaped)
            for i, t in enumerate(texts)
        ],
        dtype=np.float32,
    )
    adv = group_advantages(rewards, k).reshape(-1, k)
    std = rewards.reshape(-1, k).std(axis=1)
    return {
        "groups": int(len(batch)),
        "groups_with_variance": int((std > 0).sum()),
        "mean_reward": float(rewards.mean()),
        "max_abs_advantage": float(np.abs(adv).max()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("rl_parquet")
    p.add_argument("checkpoints", nargs="+",
                   help="label=path pairs, e.g. sft=data/processed/sft.pt")
    p.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 2.5])
    p.add_argument("--prompts", type=int, default=48)
    p.add_argument("--output")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config()
    g = cfg["grpo"]
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id
    pool = pd.read_parquet(args.rl_parquet)
    n_prompts = min(args.prompts, len(pool))
    shaped = g.get("shaped_reward", True)

    cells = {}
    for spec in args.checkpoints:
        label, _, path = spec.partition("=")
        if not path:
            p.error(f"checkpoint spec {spec!r} needs label=path")
        policy = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"])
        policy.load_state_dict(torch.load(path))
        policy.eval()
        for temp in args.temperatures:
            # same seed for every checkpoint at a given temperature: groups
            # and sampling noise are matched across cells
            torch.manual_seed(cfg["model"]["seed"] + int(temp * 1000))
            cells[f"{label}@T={temp}"] = probe(
                policy, tok, pool, n_prompts, g["n_samples"], g["max_new"],
                temp, pad_id, shaped,
            )
            print(f"{label} T={temp}: {cells[f'{label}@T={temp}']}")
        del policy

    report = {
        "kind": "reward_variance_probe_v1",
        "model": cfg["model"]["name"],
        "n_samples_per_prompt": g["n_samples"],
        "shaped_reward": shaped,
        "prompts": n_prompts,
        "seed": cfg["model"]["seed"],
        "checkpoints": {
            s.partition("=")[0]: {"path": s.partition("=")[2],
                                  "sha256": _sha256(s.partition("=")[2])}
            for s in args.checkpoints
        },
        "cells": cells,
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
