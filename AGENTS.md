# Project Guidance

- Synthetic task by design; describe it as a mechanics demonstration, not
  a real-domain capability.
- Report all three stages (base/SFT/DPO) per eval kind — never collapse
  to a single "accuracy" that hides the abstain/fabricate trade-off.
- Raw sample generations stay inspectable under `results/`; classified
  rates must be traceable to actual model output.
- Checkpoints and generated parquets stay out of Git.
- Keep `config/config.yaml` the single source of truth for model, data,
  and training settings.
- Run `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q` and
  `.venv/bin/snakemake -n` after changes. Snakemake does not track source
  edits — use `-F` to force reruns.
- `evals/run_stock_eval.py` evaluates stock open-weights models on the
  same abstention suite; `EVAL_REPORT.md` is generated from
  `evals/results/eval_stock.json` and every number in it must bind via
  `evals/claims.py` (vendored; digest pinned with bio-qc). Regenerate
  with `--report-only` after result JSON changes.
