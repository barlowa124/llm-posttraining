"""GRPO-style RL post-training, hand-rolled like the DPO objective.

Per step: sample `prompts_per_step` prompts, generate `n_samples` rollouts
each (temperature sampling; the group is the point), score every rollout
with the *eval classifier* as the reward, and take a policy-gradient step
on the group-relative advantage:

    A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
    loss = -mean_i( A_i * mean_t logp_pi(y_i | x) )
           + kl_coef * mean_t( logp_pi - logp_ref )      # KL to frozen SFT

Simplifications, stated plainly: one gradient step per rollout batch
(on-policy, so the importance ratio is 1 and PPO clipping is unnecessary),
seq-level teacher-forced logprobs, k1-style KL estimate. Rewards:

    answerable:   correct -> +1, partial (names entity or value) -> 0, else -1
    unanswerable: abstains -> +1, abstain fragment -> 0, else -1

Binary +1/-1 rewards carry no signal in either failure regime observed
here: a saturated policy scores +1 on every rollout and a collapsed one
scores -1 on every rollout. Both give zero group advantage. The shaped
middle tier is what creates reward variance for RL to exploit. The reward
reuses `evaluate.classify` plus two partial-credit checks, so
the loop optimizes the exact metric that is reported, on train entities.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from posttrain.config import load_config
from posttrain.evaluate import classify
from posttrain.lm import collate


def reward(response: str, entity: str, gold_value: str, kind: str,
           shaped: bool = True) -> float:
    label = classify(response, entity, gold_value, kind)
    desired = "correct" if kind == "answerable" else "abstains"
    if label == desired:
        return 1.0
    if not shaped:
        return -1.0
    if kind == "answerable":
        # partial: names the entity or the value, but not both correctly
        if entity in response or (
            gold_value
            and re.search(rf"\b{re.escape(gold_value)}\b", response)
        ):
            return 0.0
    elif "don't have enough information" in response or "don't have" in response:
        # the negation, not the affirmative: "I have enough information"
        # followed by a fabricated answer must not score partial credit
        return 0.0
    return -1.0


def group_advantages(rewards: np.ndarray, k: int) -> torch.Tensor:
    """Group-normalize rewards within each prompt's k rollouts."""
    r = rewards.reshape(-1, k)
    a = (r - r.mean(axis=1, keepdims=True)) / (r.std(axis=1, keepdims=True) + 1e-6)
    # a prompt whose rollouts all score the same contributes no signal
    a[r.std(axis=1) == 0] = 0.0
    return torch.tensor(a.reshape(-1), dtype=torch.float32)


@torch.no_grad()
def rollout(model, tok, prompts, k, max_new, temperature, pad_id):
    """Sample k completions per prompt. Returns flat lists aligned to groups."""
    tok.padding_side = "left"  # pads must not sit between prompt and continuation
    enc = tok(prompts, return_tensors="pt", padding=True)
    out = model.generate(
        enc["input_ids"].repeat_interleave(k, dim=0),
        attention_mask=enc["attention_mask"].repeat_interleave(k, dim=0),
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temperature,
        pad_token_id=pad_id,
    )
    plen = enc["input_ids"].shape[1]
    texts, gen_ids = [], []
    for row in out:
        gen_ids.append(row[plen:])
        texts.append(tok.decode(row[plen:], skip_special_tokens=True))
    return texts, gen_ids


def completion_logp(model, tok, prompts_flat, gen_ids, max_len, pad_id):
    """Mean per-token logp of each sampled completion, teacher-forced."""
    pairs = []
    for p, g in zip(prompts_flat, gen_ids):
        p_ids = tok(p, add_special_tokens=True)["input_ids"]
        # pad ids in the sampled tail are not scored
        g_lab = [t if t != pad_id else -100 for t in g.tolist()]
        full = (p_ids + g.tolist())[:max_len]
        labels = ([-100] * len(p_ids) + g_lab)[:max_len]
        pairs.append((torch.tensor(full), torch.tensor(labels)))
    ids, labels, attn = collate(pairs, pad_id)
    logits = model(input_ids=ids, attention_mask=attn).logits
    logp = torch.log_softmax(logits[:, :-1], dim=-1)
    tgt = ids[:, 1:]
    mask = labels[:, 1:] != -100
    tok_lp = logp.gather(-1, tgt[:, :, None]).squeeze(-1)
    denom = mask.sum(dim=1).clamp(min=1)
    return (tok_lp * mask).sum(dim=1) / denom  # per-completion mean logp


def train_grpo(rl_path: str, init_ckpt: str, ref_ckpt: str, out_path: str,
               seed_offset: int = 0):
    """GRPO from `init_ckpt`, with KL measured against `ref_ckpt`.

    The wiring is the Snakefile's: init = DPO checkpoint (a repair
    attempt on the collapsed policy), ref = frozen SFT policy.
    `seed_offset` shifts all sampling for ablation replicates.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config()
    g = cfg["grpo"]
    torch.manual_seed(cfg["model"]["seed"] + seed_offset)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id
    max_len = cfg["model"]["max_len"]

    policy = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"])
    policy.load_state_dict(torch.load(init_ckpt))
    ref = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"])
    ref.load_state_dict(torch.load(ref_ckpt))
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    pool = pd.read_parquet(rl_path)
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
        shaped = g.get("shaped_reward", True)
        rewards = np.array(
            [
                reward(t, batch.iloc[i // g["n_samples"]]["entity"],
                       batch.iloc[i // g["n_samples"]]["gold_value"],
                       batch.iloc[i // g["n_samples"]]["kind"], shaped)
                for i, t in enumerate(texts)
            ],
            dtype=np.float32,
        )
        adv = group_advantages(rewards, g["n_samples"])
        if adv.abs().max() == 0:
            hist.append({"step": step, "mean_reward": float(rewards.mean()),
                         "loss": None, "skipped": True})
            continue  # all groups uniform this step; no signal

        prompts_flat = [p for p in prompts for _ in range(g["n_samples"])]
        lp_pi = completion_logp(
            policy, tok, prompts_flat, gen_ids, max_len, pad_id
        )
        with torch.no_grad():
            lp_ref = completion_logp(
                ref, tok, prompts_flat, gen_ids, max_len, pad_id
            )
        loss = -(adv * lp_pi).mean() + g["kl_coef"] * (lp_pi - lp_ref).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(
            {
                "step": step,
                "mean_reward": float(rewards.mean()),
                "loss": float(loss.detach()),
                "skipped": False,
            }
        )
        if step % 10 == 0 or step == g["steps"] - 1:
            print(f"grpo step {step}: mean_reward={rewards.mean():.3f} "
                  f"loss={loss.item():.4f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), out_path)
    log = pd.DataFrame(hist)
    log.to_csv(str(Path(out_path).with_suffix(".log.csv")), index=False)
    # no-signal steps are evidence for the reward-variance finding. Keep
    # the log in results/ where it gets committed
    Path("results").mkdir(exist_ok=True)
    stem = Path(out_path).stem
    log_name = "grpo_log.csv" if stem == "grpo" else f"{stem}_log.csv"
    log.to_csv(f"results/{log_name}", index=False)
    n_skipped = int(log["skipped"].sum()) if len(log) else 0
    print(f"grpo: {len(hist)} steps ({n_skipped} no-signal) -> {out_path}")


if __name__ == "__main__":
    train_grpo(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
               int(sys.argv[5]) if len(sys.argv) > 5 else 0)
