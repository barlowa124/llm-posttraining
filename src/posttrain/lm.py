"""Tokenization and log-prob helpers shared by SFT and DPO.

SFT trains on completion tokens only: the prompt is encoded but its label
positions are masked to -100, standard instruction-tuning practice.
"""

import torch


def encode_pair(tok, prompt: str, completion: str, max_len: int):
    """Return input_ids and labels (-100 on prompt tokens)."""
    p = tok(prompt, add_special_tokens=True)["input_ids"]
    c = tok(completion, add_special_tokens=False)["input_ids"]
    ids = (p + c)[:max_len]
    labels = ([-100] * len(p) + c)[:max_len]
    return torch.tensor(ids), torch.tensor(labels)


def seq_logprob(model, tok, prompt: str, completion: str, max_len: int):
    """Sum log p(completion | prompt) under the model."""
    ids, labels = encode_pair(tok, prompt, completion, max_len)
    with torch.no_grad():
        logits = model(ids[None, :]).logits[0]
    logp = torch.log_softmax(logits, dim=-1)
    total = 0.0
    for i in range(1, len(ids)):
        if labels[i] != -100:
            total += logp[i - 1, ids[i]].item()
    return float(total)


def collate(examples, pad_id):
    maxlen = max(len(e[0]) for e in examples)
    input_ids = torch.full((len(examples), maxlen), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), maxlen), -100, dtype=torch.long)
    attn = torch.zeros((len(examples), maxlen), dtype=torch.long)
    for i, (ids, lab) in enumerate(examples):
        input_ids[i, : len(ids)] = ids
        labels[i, : len(lab)] = lab
        attn[i, : len(ids)] = 1
    return input_ids, labels, attn
