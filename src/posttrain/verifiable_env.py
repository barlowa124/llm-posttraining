"""Verifiable-reward RL: single-operation arithmetic word problems.

The abstention GRPO (`grpo.py`) rewards through the eval classifier —
a model-written judgment. This environment replaces the reward with a
programmatic check: the response's last integer must equal the computed
answer. Nothing about the check requires a model, so the reward cannot
inherit classifier quirks.

Reward tiers (kept because the same variance argument as grpo.py
applies): correct final number +1, a wrong number 0 (attempted an
answer), no number at all -1. The task is deliberately small: operands
<= 20, one operation, lab-flavored nouns. A 135M model can reach the
reward through format alone; arithmetic ability is not the claim.

Run:
    python -m posttrain.verifiable_env sft.pt sft.pt env.pt
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from posttrain.config import load_config
from posttrain.grpo import completion_logp, group_advantages, rollout

TEMPLATES = [
    ("A bioreactor holds {a} liters of medium. {b} more liters are "
     "pumped in. How many liters total?", lambda a, b: a + b),
    ("A flask starts with {a} mL of culture. {b} mL are removed for "
     "sampling. How many mL remain?", lambda a, b: a - b),
    ("A plate has {a} rows with {b} wells in each row. How many wells "
     "in total?", lambda a, b: a * b),
    ("{a} samples arrive and are split evenly into {b} batches. How "
     "many samples per batch?", lambda a, b: a // b),
]

_SUFFIX = " Answer with just the number."


def make_pool(n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        tpl, fn = TEMPLATES[rng.randrange(len(TEMPLATES))]
        if "split evenly" in tpl:
            b = rng.randint(2, 9)
            a = b * rng.randint(1, 9)   # keep division exact
        else:
            a, b = rng.randint(1, 20), rng.randint(1, 20)
        rows.append({"prompt": tpl.format(a=a, b=b) + _SUFFIX,
                     "answer": fn(a, b)})
    return pd.DataFrame(rows)


_NUM = re.compile(r"-?\d+")


def reward(response: str, answer: int) -> float:
    """+1 last number correct; 0 a wrong number; -1 no number."""
    nums = _NUM.findall(response)
    if not nums:
        return -1.0
    return 1.0 if int(nums[-1]) == int(answer) else 0.0


def train_grpo_env(init_ckpt: str, ref_ckpt: str, out_path: str,
                   seed_offset: int = 0) -> None:
    """GRPO on the verifiable env. Same optimizer shape as grpo.py:
    one step per on-policy batch, group-normalized advantage, KL to
    the frozen reference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config()
    g = cfg["grpo_env"]
    torch.manual_seed(cfg["model"]["seed"] + seed_offset)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id
    max_len = cfg["model"]["max_len"]

    policy = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"])
    policy.load_state_dict(torch.load(init_ckpt, weights_only=True))
    ref = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"])
    ref.load_state_dict(torch.load(ref_ckpt, weights_only=True))
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    pool = make_pool(g["n_prompts"], cfg["data"]["seed"])
    opt = torch.optim.Adam(policy.parameters(), lr=g["lr"])
    rng = np.random.default_rng(cfg["model"]["seed"] + seed_offset)

    hist = []
    for step in range(g["steps"]):
        batch = pool.iloc[rng.integers(0, len(pool), g["prompts_per_step"])]
        prompts = batch["prompt"].tolist()
        texts, gen_ids = rollout(
            policy, tok, prompts, g["n_samples"], g["max_new"],
            g["temperature"], pad_id,
        )
        rewards = np.array(
            [reward(t, int(batch.iloc[i // g["n_samples"]]["answer"]))
             for i, t in enumerate(texts)],
            dtype=np.float32,
        )
        adv = group_advantages(rewards, g["n_samples"])
        if adv.abs().max() == 0:
            hist.append({"step": step,
                         "mean_reward": float(rewards.mean()),
                         "loss": None, "skipped": True})
            continue

        prompts_flat = [p for p in prompts for _ in range(g["n_samples"])]
        lp_pi = completion_logp(
            policy, tok, prompts_flat, gen_ids, max_len, pad_id)
        with torch.no_grad():
            lp_ref = completion_logp(
                ref, tok, prompts_flat, gen_ids, max_len, pad_id)
        loss = -(adv * lp_pi).mean() + g["kl_coef"] * (lp_pi - lp_ref).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append({"step": step,
                     "mean_reward": float(rewards.mean()),
                     "loss": float(loss.detach()), "skipped": False})
        if step % 10 == 0 or step == g["steps"] - 1:
            print(f"env-grpo step {step}: mean_reward={rewards.mean():.3f} "
                  f"loss={loss.item():.4f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), out_path)
    log = pd.DataFrame(hist)
    Path("results").mkdir(exist_ok=True)
    log.to_csv(f"results/{Path(out_path).stem}_log.csv", index=False)
    n_skipped = int(log["skipped"].sum()) if len(log) else 0
    print(f"env-grpo: {len(hist)} steps ({n_skipped} no-signal) "
          f"-> {out_path}")


if __name__ == "__main__":
    train_grpo_env(sys.argv[1], sys.argv[2], sys.argv[3],
                   int(sys.argv[4]) if len(sys.argv) > 4 else 0)
