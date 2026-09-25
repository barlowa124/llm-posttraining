"""Stage checkpoints as a multi-subfolder HF model repo and upload.

Each stage becomes a self-contained model dir (config + tokenizer +
model.safetensors) so `from_pretrained(repo, subfolder="<stage>")` works.

Usage:
    HF_TOKEN=hf_... PYTHONPATH=src .venv/bin/python hf/upload_hf.py \
        --repo barlowa124/smollm2-135m-abstention-posttrain --dry-run
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STAGES = ["sft", "dpo", "grpo"]
ABLATIONS = ["dpo_lr1e5", "grpo_s105", "grpo_from_sft"]
STAGING = Path("data/processed/hf_repo")


def stage_checkpoints(ckpt_dir: Path, stages: list, base_model: str,
                      work: Path) -> list:
    # save_pretrained handles SmolLM2's tied lm_head/embed_tokens and
    # writes config.json + generation_config.json per subfolder
    tok = AutoTokenizer.from_pretrained(base_model)
    staged = []
    for name in stages:
        src = ckpt_dir / f"{name}.pt"
        if not src.exists():
            sys.exit(f"missing checkpoint {src}")
        d = work / name
        d.mkdir(parents=True, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(base_model)
        model.load_state_dict(torch.load(src, map_location="cpu"))
        model.save_pretrained(d, safe_serialization=True)
        tok.save_pretrained(d)
        staged.append(name)
        n = sum(f.stat().st_size for f in d.glob("*.safetensors"))
        print(f"staged {name}/: {n / 1e6:.0f} MB safetensors")
        del model
    return staged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. user/repo-name")
    ap.add_argument("--ckpt-dir", default="data/processed")
    ap.add_argument("--include-ablations", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="stage locally, print plan, do not upload")
    ap.add_argument("--base", default="HuggingFaceTB/SmolLM2-135M")
    args = ap.parse_args()

    stages = STAGES + (ABLATIONS if args.include_ablations else [])
    staged = stage_checkpoints(Path(args.ckpt_dir), stages, args.base,
                               STAGING)
    card = Path(__file__).parent / "MODEL_CARD.md"
    target = STAGING / "README.md"
    target.write_text(card.read_text())
    print(f"model card -> {target}")

    if args.dry_run:
        total = sum(f.stat().st_size for f in STAGING.rglob("*"))
        print(f"dry-run: staged {staged} in {STAGING} "
              f"({total / 1e9:.2f} GB); would upload to {args.repo}")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, exist_ok=True)
    api.upload_folder(folder_path=str(STAGING), repo_id=args.repo,
                      repo_type="model")
    print(f"uploaded {len(staged)} checkpoint(s) to {args.repo}")


if __name__ == "__main__":
    main()
