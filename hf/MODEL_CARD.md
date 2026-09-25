---
license: apache-2.0
base_model: HuggingFaceTB/SmolLM2-135M
tags:
  - post-training
  - dpo
  - grpo
  - abstention
  - calibration
  - mechanics-demo
---

# SmolLM2-135M abstention post-training: SFT / DPO / GRPO checkpoints

A mechanics demonstration of the full post-training arc on a small LM:
supervised fine-tuning, hand-rolled DPO, and GRPO repair, on a synthetic
abstention-vs-fabrication task. The value is the *measured failure and
recovery*, not the model's capability — the task is templated synthetic
pharmacology facts.

Source: https://github.com/barlowa124/llm-posttraining

## Checkpoints

| Subfolder | What it is | Held-out answerable | Held-out unanswerable |
|---|---|---|---|
| `sft/` | Supervised, 3 epochs | 100% correct | 100% abstains |
| `dpo/` | DPO at lr=1e-4, beta=0.1 — **the collapse artifact** | 55% correct | 5% abstains, 95% degenerate |
| `grpo/` | GRPO initialized *from the collapsed DPO* checkpoint | 98.75% correct | 100% abstains |
| `dpo_lr1e5/` | Ablated DPO at lr=1e-5 (no collapse) | — | — |
| `grpo_s105/` | GRPO repair at rollout seed 105 — replicates | 98.75% correct | 100% abstains |
| `grpo_from_sft/` | GRPO from the healthy SFT policy — **provably inert** (60/60 no-signal steps, bit-identical output) | = SFT | = SFT |

## The finding these weights encode

- DPO at too-aggressive lr collapses *deployed* behavior while train
  metrics look fine — 95% degenerate outputs on held-out unanswerable
  prompts at 100% train preference accuracy.
- Binary ±1 rewards give GRPO **zero gradient in both regimes**: a
  saturated policy rolls all +1s, a collapsed policy all −1s; no
  within-group variance, no advantage, no update.
- A shaped reward tier (partial credit for the right shape of wrong
  answer) creates variance in the middle regime only — 8 gradient steps
  repaired held-out abstention 5% → 100%.
- The same GRPO applied to a healthy policy is provably inert: 60/60
  skipped steps, bit-identical checkpoint.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "barlowa124/smollm2-135m-abstention-posttrain", subfolder="grpo")
tok = AutoTokenizer.from_pretrained(
    "barlowa124/smollm2-135m-abstention-posttrain", subfolder="grpo")
```

## Scope

Synthetic single-task training; one seed, one temperature, one group
size for the main run. Not a capability model, not for domain use —
this is a documented post-training failure-and-repair artifact.
