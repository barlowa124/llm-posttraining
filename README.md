# llm-posttraining

SFT + **hand-rolled DPO** on a small open LM, on the behavior this
portfolio cares most about: **answering when the evidence supports it and
abstaining instead of fabricating when it does not** — measured before and
after each stage on held-out entities.

**Status: working demonstration.** Snakemake DAG runs synthetic data ->
SFT -> DPO -> three-stage evaluation on `SmolLM2-135M` (base, CPU-trainable).

## Design

- **Task (synthetic by construction)**: pharmacology-flavored QA over
  invented entity names — "What is the half-life of veltinib-123?" against
  a context that either contains the fact or contains only *other*
  entities' facts. Held-out entity names at eval make it a generalization
  test, not memorization.
- **SFT**: standard causal-LM fine-tune with prompt tokens masked —
  completion-only loss.
- **DPO**: implemented from the objective itself (~40 lines, `dpo.py`),
  not via `trl`: frozen SFT reference, `-log sigmoid(β·Δlogp)` on
  chosen/rejected completion pairs (correct-vs-wrong-value on answerable
  prompts; abstain-vs-fabricated-answer on unanswerable).
- **Eval**: greedy decode; each response classified `correct` (entity +
  gold value, word-boundary), `abstains`, or `fabricates` — per stage,
  on entities never seen in training.

## What it found (committed in `results/summary.json`)

| Stage | Answerable | Unanswerable | What happened |
|---|---|---|---|
| base | 100% "correct" (parroting context) | 100% fabricates | copies when it can, invents when it can't |
| SFT | 100% correct | 100% abstains | solves the task cleanly |
| DPO (lr 1e-4) | 55% correct, 31% abstains | 5% abstains, **95% degenerate** | train pref-acc 1.0, deployed behavior collapses |
| DPO (lr 1e-5) | 0% correct — abstains all | 86% abstains | collapses the *other* way |

The headline is a negative result, measured properly: **preference accuracy
on training pairs does not predict deployed behavior.** Both DPO variants
hit 100% preference accuracy on the training pairs while destabilizing the
held-out policy in *opposite* directions — repetition collapse
(`I I I I I…`) at 1e-4, blanket over-abstention at 1e-5. On a 135M model
with clean synthetic pairs, the abstain/answer balance is fragile and DPO
moves it globally, not per-prompt-type.

Two eval-iteration artifacts are kept visible:

- The first classifier required the full gold sentence as substring and
  counted correct-value-with-degenerate-tail as fabrication. It now scores
  entity + gold value with word boundaries, and repetition collapse is its
  own `degenerate` label separate from semantic fabrication.
- Raw generations are preserved (`results/responses_*.csv`) so every rate
  in the table traces to actual model output.

## Caveats

- The task is templated synthetic; it demonstrates post-training mechanics
  and measured behavior change, not a real-domain capability.
- One model size, one seed, greedy decode; no beta sweep, no KL tracking.
- The 135M base is very small — part of the instability is capacity;
  a slightly larger base or KL-annealed schedule is the next honest knob.
- DPO pair construction is idealized (clean chosen/rejected); real
  preference data is noisier and would likely destabilize further.
- SFT run-to-run variance exists on CPU (nondeterministic reductions):
  an earlier run collapsed to always-abstain at the same seed.

## Run

```bash
.venv/bin/snakemake -j1          # data -> sft -> dpo -> evaluate
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

`POSTTRAIN_CONFIG` env var selects an alternate config. Outputs:
`results/summary.json` (per-stage rates), `results/responses_*.csv`
(inspectable raw generations), `results/provenance.json`. Checkpoints are
regenerable and gitignored.
