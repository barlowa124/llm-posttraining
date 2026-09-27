# llm-posttraining

SFT + **hand-rolled DPO + hand-rolled GRPO** on a small open LM, on the
behavior this portfolio is built around: **answering when the evidence
supports it and abstaining instead of fabricating when it does not**.
Each stage is measured before and after on held-out entities.

**Status: working demonstration.** Snakemake DAG runs synthetic data ->
SFT -> DPO -> GRPO -> four-stage evaluation on `SmolLM2-135M` (base,
CPU-trainable).

## Design

- **Task (synthetic by construction)**: pharmacology-flavored QA over
  invented entity names ("What is the half-life of veltinib-123?") against
  a context that either contains the fact or contains only *other*
  entities' facts. Held-out entity names at eval make it a generalization
  test, not memorization.
- **SFT**: standard causal-LM fine-tune with prompt tokens masked,
  so only completion tokens get loss.
- **DPO**: implemented from the objective itself (~40 lines, `dpo.py`),
  not via `trl`: frozen SFT reference, `-log sigmoid(β·Δlogp)` on
  chosen/rejected completion pairs (correct-vs-wrong-value on answerable
  prompts; abstain-vs-fabricated-answer on unanswerable).
- **GRPO** (`grpo.py`): group-relative advantage over `k=6` temperature
  rollouts per prompt, reward = the eval classifier with a shaped middle
  tier, plus KL to the frozen SFT reference. Initialized from the
  *collapsed* DPO checkpoint as a repair attempt.
  One gradient step per rollout batch (on-policy, ratio=1, no clip
  needed), teacher-forced seq logprobs.
- **Eval**: greedy decode. Each response classified `correct` (entity +
  gold value, word-boundary), `abstains`, or `fabricates`, per stage,
  on entities never seen in training.

## What it found (committed in `results/summary.json`)

| Stage | Answerable | Unanswerable | What happened |
|---|---|---|---|
| base | 100% "correct" (parroting context) | 100% fabricates | copies when it can, invents when it can't |
| SFT | 100% correct | 100% abstains | solves the task cleanly |
| DPO (lr 1e-4) | 55% correct, 31% abstains | 5% abstains, **95% degenerate** | train pref-acc 1.0, deployed behavior collapses |
| DPO (lr 1e-5) † | 0% correct, abstains all | 86% abstains | collapses the *other* way |
| **GRPO (shaped, from collapsed DPO)** | **98.75% correct** | **100% abstains** | RL *repairs* the collapse |

† manual variant eval, in `results/responses_dpo_lr1e5.csv`, not the DAG's
`summary.json`.

The DPO headline is a negative result: **preference
accuracy on training pairs does not predict deployed behavior.** Both DPO
variants hit 100% preference accuracy on the training pairs while
destabilizing the held-out policy in *opposite* directions: repetition
collapse (`I I I I I…`) at 1e-4 and blanket over-abstention at 1e-5.

The RL result, and the subtler one underneath:

- **Shaped GRPO repairs the collapse on held-out entities**: 95%
  degenerate -> 100% abstain on unanswerable, 55% -> 98.75% correct on
  answerable, in only **8 gradient steps** (52 of 60 steps found zero
  within-group reward variance and were skipped). KL-to-SFT is part of
  the repair mechanism. The recovery is not attributable to reward alone.
- **The prerequisite finding:** binary +1/-1 rewards produce *no gradient
  at all* when a policy saturates. On the saturated SFT policy every
  rollout in a group scores +1 (verified: probe batches at the run's
  temperature give all-uniform groups, zero advantage). On the collapsed
  DPO policy most groups are uniform because degenerate repeats score
  identically, with only occasional splits on answerable prompts. Zero
  variance -> zero advantage -> zero gradient. The shaped middle tier
  (partial credit for naming the entity or emitting an abstain fragment)
  is what creates usable variance at the operating temperature. Even so,
  52/60 steps still skipped. (Measured caveat: raising temperature
  restores variance too, and a T=2.5 probe produces mixed groups on both
  checkpoints — `results/reward_variance_probe.json`, produced by
  `python -m posttrain.probe_variance`. Shaping is not the *only* fix.
  It is the one that works without degrading rollout quality.)

Two measured ablations (`results/grpo_ablations.json`):

- **Seed replication**: a second GRPO repair with a different rollout
  seed (s105) lands the identical held-out result, 98.75% correct /
  100% abstains, from 10 gradient steps (50 skipped). The repair
  replicates. It isn't a lucky roll.
- **SFT-init**: the same shaped GRPO from the *healthy* SFT checkpoint
  does nothing: 60/60 steps skipped, every rollout scores 1.0,
  and the output checkpoint is bit-identical to the input (verified).
  The shaped middle tier creates variance only in the *middle* competence
  regime: a policy that already maximizes the rubric has no gradient to
  find. The zero-signal finding cuts both ways. RL can't hurt this
  policy, but only because it can't touch it at all.

Two eval-iteration artifacts are kept visible:

- The first classifier required the full gold sentence as substring and
  counted correct-value-with-degenerate-tail as fabrication. It now scores
  entity + gold value with word boundaries, and repetition collapse is its
  own `degenerate` label separate from semantic fabrication.
- Raw generations are preserved (`results/responses_*.csv`) so every rate
  in the table traces to model output.

## Retrieval-augmented eval (`results/rag_eval.json`)

`eval_rag.py` reruns the same held-out eval but replaces the baked Context
line with top-3 BM25 retrieval over a 1016-doc fact corpus built from the
committed splits (`config.retrieval.k`). Retrieval itself is perfect at
this scale: the gold doc lands in the top-3 for 100% of answerable rows.

The result is a negative one for the abstention pipeline: **abstention
learned on a single clean context does not transfer to retrieved
context.** With three concatenated docs (the queried entity's other
attributes plus near-name distractors), every stage degrades:

| Stage | Answerable correct | Unanswerable abstains | Unanswerable fabricates |
|---|---|---|---|
| baked context (SFT) | 100% | 100% | 0% |
| base + RAG | 71% | 0% | 94% |
| SFT + RAG | 56% | 0% | **99%** |
| DPO + RAG | 42% | 12% | 78% |
| GRPO + RAG | 44% | 0% | 90% |

On unanswerable rows the retrieved docs are topically related but lack the
asked attribute, and the model echoes a related fact instead of abstaining.
That is the standard production RAG failure (related-but-insufficient
context induces hallucination), reproduced here on a 135M model with raw
outputs committed per stage (`results/responses_*_rag.csv`).

Label-noise caveat on the table above: for 31 of the 80 "unanswerable"
eval rows the corpus contains the answer, because the entity's
answerable eval row shares the fact template. The classifier still scores
unanswerable rows as abstain-or-fail, so a correct grounded echo counts
as "fabricates". For SFT + RAG, 14 of the 79 labeled fabrications were
verbatim retrieved facts — the true fabrication rate is ~81%, not 99%.
The failure remains; the number was overstated. `rag_repair.py` reports
this correction alongside its own eval.

## Retrieval-noise repair (`results/rag_repair.json`)

`sft_rag.py` rebuilds the SFT set with the baked Context line replaced by
top-3 retrieval over a **train-only** corpus (held-out facts cannot leak
into a context), and makes the supervision consistent with what retrieval
supplies: an answerable row whose gold fact is not retrieved flips to an
abstain target, and an unanswerable row where retrieval surfaced the asked
attribute is rescued to that fact. The model then trains from base with
the same schedule (`posttrain.sft`).

`rag_repair.py` scores the resulting checkpoint under both context
constructions — the original baked eval and the retrieved-context eval —
so the artifact answers whether the fix recovers abstention under
retrieval and at what cost to clean-context behavior.

Measured results (`results/rag_repair.json`, `rag_mix_repair.json`):

| Stage | Clean unans. abstain | RAG unans. abstain | RAG true fabricate* |
|---|---|---|---|
| SFT (clean-trained) | 100% | 0% | ~81% |
| SFT-rag (retrieval-trained) | **0%** | 61% | **0%** |
| SFT-mix (clean + retrieved) | **100%** | 61% | **0%** |

\* after the label-noise correction: every "fabricate" and "degenerate"
label on unanswerable rows was a verbatim echo of a retrieved doc that
actually answered the question (31/80 rows were secretly answerable).
De-noised, the retrieval-trained models are exactly context-faithful:
they abstain on precisely the 49 truly-unanswerable rows.

Two real findings fall out. First, the repair works but only on the
distribution it was trained on: retrieval-noise SFT alone traded
clean-context abstention (100% → 0%) for retrieval abstention, so the
mixed set (clean + retrieved, 1920 rows) is what holds both regimes.
Second, the eval's "unanswerable" label is noisy under retrieval: the
corpus can contain the asked attribute, and `rag_repair.py` reports that
correction as a measured field, not a footnote.

## Caveats

- The task is templated synthetic, and it demonstrates post-training mechanics
  and measured behavior change, not a real-domain capability.
- One model size, one seed, greedy decode. No beta sweep, no KL tracking.
- GRPO ran 60 steps at one shaped-reward scheme, one temperature, one
  KL coefficient. Two ablations landed (a second rollout seed, identical
  result; an SFT-init run, provably inert). Still open: temperature and
  KL sweeps, and reward-scheme variants.
- The 135M base is small. Part of the instability is capacity.
  A slightly larger base or KL-annealed schedule is the next knob.
- DPO pair construction is idealized (clean chosen/rejected). Real
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

## Publishing checkpoints

Published at
[huggingface.co/barlowa/smollm2-135m-abstention-posttrain](https://huggingface.co/barlowa/smollm2-135m-abstention-posttrain)
— six checkpoint dirs (`sft/`, `dpo/`, `grpo/` + ablations) loadable via
`AutoModelForCausalLM.from_pretrained(repo, subfolder="sft")`. The task
splits and **raw per-stage generations** are at
[datasets/barlowa/smollm2-135m-abstention-posttrain-data](https://huggingface.co/datasets/barlowa/smollm2-135m-abstention-posttrain-data)
— every classified rate in `results/summary.json` traces to an output
there.

`hf/` contains the HuggingFace packaging: `MODEL_CARD.md` plus
`upload_hf.py`, which converts each `data/processed/*.pt` into a
self-contained safetensors model dir (`<repo>/<stage>/`) and uploads the
repo. All six checkpoints stage by default with `--include-ablations`
(the collapsed `dpo/` checkpoint ships deliberately — it is the failure
artifact the card documents).

```bash
HF_TOKEN=hf_... PYTHONPATH=src .venv/bin/python hf/upload_hf.py \
    --repo <user>/smollm2-135m-abstention-posttrain --include-ablations
# --dry-run stages locally under data/processed/hf_repo/ without pushing
```
