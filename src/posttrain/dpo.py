"""Direct preference optimization, implemented from the DPO objective.

loss = -log sigmoid( beta * [ (lp_pi(chosen) - lp_ref(chosen))
                            - (lp_pi(rejected) - lp_ref(rejected)) ] )

The reference model is the frozen SFT checkpoint — beta controls how far
the policy may drift from it while shifting probability mass toward the
preferred completion. Hand-rolled rather than `trl` so the objective is
inspectable in ~40 lines.
"""

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from posttrain.config import load_config
from posttrain.lm import encode_pair


def seq_logp_batch(model, ids, labels, attn):
    """Sum log p over label!=-100 positions, per row. Differentiable."""
    logits = model(input_ids=ids, attention_mask=attn).logits
    logp = torch.log_softmax(logits[:, :-1], dim=-1)
    tgt = ids[:, 1:]
    mask = labels[:, 1:] != -100
    tok_logp = logp.gather(-1, tgt[:, :, None]).squeeze(-1)
    return (tok_logp * mask).sum(dim=1)


def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta):
    logits = beta * ((pi_c - ref_c) - (pi_r - ref_r))
    return -F.logsigmoid(logits).mean(), (logits > 0).float().mean()


def _encode_batch(tok, prompts, completions, max_len, pad_id):
    pairs = [encode_pair(tok, p, c, max_len) for p, c in zip(prompts, completions)]
    maxlen = max(len(i) for i, _ in pairs)
    B = len(pairs)
    ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    labels = torch.full((B, maxlen), -100, dtype=torch.long)
    attn = torch.zeros((B, maxlen), dtype=torch.long)
    for i, (ii, ll) in enumerate(pairs):
        ids[i, : len(ii)] = ii
        labels[i, : len(ll)] = ll
        attn[i, : len(ii)] = 1
    return ids, labels, attn


def dpo_train(policy, ref, tok, pairs, cfg, out_path):
    m = cfg["model"]
    policy.train()
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=m["lr_dpo"])
    for ep in range(m["dpo_epochs"]):
        order = torch.randperm(len(pairs)).tolist()
        tot, acc, nb = 0.0, 0.0, 0
        for i in range(0, len(pairs), m["batch_size"]):
            b = pairs.iloc[order[i : i + m["batch_size"]]]
            ci, cl, ca = _encode_batch(
                tok, b["prompt"].tolist(), b["chosen"].tolist(),
                m["max_len"], tok.pad_token_id)
            ri, rl, ra = _encode_batch(
                tok, b["prompt"].tolist(), b["rejected"].tolist(),
                m["max_len"], tok.pad_token_id)
            pi_c = seq_logp_batch(policy, ci, cl, ca)
            pi_r = seq_logp_batch(policy, ri, rl, ra)
            with torch.no_grad():
                ref_c = seq_logp_batch(ref, ci, cl, ca)
                ref_r = seq_logp_batch(ref, ri, rl, ra)
            loss, acc_b = dpo_loss(pi_c, pi_r, ref_c, ref_r, m["beta"])
            loss.backward()
            opt.step()
            opt.zero_grad()
            tot += loss.item()
            acc += acc_b.item()
            nb += 1
        print(f"dpo epoch {ep + 1}/{m['dpo_epochs']} loss {tot / nb:.4f} "
              f"pref-acc {acc / nb:.3f}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), out_path)
    return policy


def main(dpo_parquet: str, sft_model: str, out_model: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config()
    torch.manual_seed(cfg["model"]["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    policy = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"])
    policy.load_state_dict(torch.load(sft_model))
    ref = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"])
    ref.load_state_dict(torch.load(sft_model))
    pairs = pd.read_parquet(dpo_parquet)
    dpo_train(policy, ref, tok, pairs, cfg, out_model)
    print(f"saved {out_model}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
