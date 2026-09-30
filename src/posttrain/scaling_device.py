"""Device benchmark: forward+backward step time on cpu vs mps.

Single process, no distribution — complements scaling_ddp by measuring
what the local hardware can actually do. Records hardware, torch version,
and per-device step times. MPS rows exist only where the backend is
available.

    PYTHONPATH=src .venv/bin/python -m posttrain.scaling_device --steps 6
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import torch

from .scaling_ddp import WARMUP_STEPS, SyntheticTokens, build_model


def bench_device(device: str, args: dict) -> dict:
    torch.manual_seed(args["seed"])
    model = build_model(args["model"], args["tiny"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args["lr"])
    ds = SyntheticTokens(
        n_samples=args["batch"] * (args["steps"] + WARMUP_STEPS),
        seq_len=args["seq_len"], vocab=args["vocab"], seed=args["seed"])

    step_times = []
    for step in range(WARMUP_STEPS + args["steps"]):
        lo, hi = step * args["batch"], (step + 1) * args["batch"]
        batch = [ds[i] for i in range(lo, hi)]
        ids = torch.stack([b[0] for b in batch]).to(device)
        labels = torch.stack([b[1] for b in batch]).to(device)
        t0 = time.perf_counter()
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        if device == "mps":
            torch.mps.synchronize()
        if step >= WARMUP_STEPS:
            step_times.append(time.perf_counter() - t0)

    return {
        "device": device,
        "n_params": n_params,
        "batch": args["batch"],
        "seq_len": args["seq_len"],
        "timed_steps": args["steps"],
        "step_times_s": [round(t, 4) for t in step_times],
        "mean_step_s": round(sum(step_times) / len(step_times), 4),
        "samples_per_s": round(args["batch"] * len(step_times)
                               / sum(step_times), 2),
        "final_loss": float(out.loss.detach().cpu()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=160)
    p.add_argument("--vocab", type=int, default=49152)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--out", type=str, default="results/scaling/device.json")
    args = vars(p.parse_args())

    runs = []
    for dev in args["devices"]:
        if dev == "mps" and not torch.backends.mps.is_available():
            print("mps not available, skipping")
            continue
        if dev == "cuda" and not torch.cuda.is_available():
            print("cuda not available, skipping")
            continue
        rec = bench_device(dev, args)
        runs.append(rec)
        print(f"{dev}: {rec['mean_step_s']}s/step, "
              f"{rec['samples_per_s']} samples/s")

    out = {
        "benchmark": "single-process forward+backward step time",
        "model": args["model"] if not args["tiny"]
                 else f"{args['model']} (tiny config)",
        "hardware": {"machine": platform.machine(), "device_note":
                     "mps = Apple Silicon GPU via Metal Performance Shaders"},
        "torch": torch.__version__,
        "runs": runs,
    }
    Path(args["out"]).parent.mkdir(parents=True, exist_ok=True)
    with open(args["out"], "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
