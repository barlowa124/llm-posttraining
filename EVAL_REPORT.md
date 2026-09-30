# Eval report: stock models on the abstention eval

The repo's held-out abstention suite run end-to-end on stock open-weights checkpoints. Both models are small and the suite is small, so the report reads as a measured abstain/fabricate split for these checkpoints, not a benchmark ranking.

Each of the 160 prompts (80 answerable, 80 unanswerable) is judged under one protocol: greedy decode, raw completion prompt, classify() abstains/correct/degenerate/fabricates, max 40 new tokens per response.

## Rates

| model | params (M) | ans. correct | ans. fabricates | unans. abstains | unans. fabricates |
|---|---:|---:|---:|---:|---:|
| HuggingFaceTB/SmolLM2-135M-Instruct (Apache license) | 135 | 0.625 | 0.375 | 0 | 0.9875 |
| Qwen/Qwen2.5-0.5B-Instruct (Apache license) | 494 | 0.725 | 0.275 | 0.8875 | 0.1125 |

On unanswerable prompts the abstention rate spans from 0 (HuggingFaceTB/SmolLM2-135M-Instruct) to 0.8875 (Qwen/Qwen2.5-0.5B-Instruct). HuggingFaceTB/SmolLM2-135M-Instruct fabricates on 0.9875 of them.

Each generation's label is inspectable in `evals/results/responses_<model>.csv`. Degenerate (repetition-collapse) rates are in eval_stock.json.

## Reading the table

On answerable prompts the desired label is `correct`. On unanswerable prompts it is `abstains`. A stock instruct model typically answers regardless, so the unanswerable fabricate rate is the quantity this eval exists to expose.

## Provenance

- eval set: `data/processed/eval.parquet` (80 held-out synthetic entities, never used in training)
- results: `evals/results/eval_stock.json`
- claims check: `evals/results/claims_check.json`. Every number above re-derives from the results JSON at display tolerance.
- scope: synthetic pharmacology facts, small eval, greedy decoding. Findings describe this eval only.
