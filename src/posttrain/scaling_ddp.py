"""DDP benchmark: SmolLM2 forward+backward under torch.distributed.

Measures real distributed-training mechanics (process group,
DistributedSampler partitioning, gradient all-reduce in backward) with
the repo's actual model. Scope is honest: gloo backend on one CPU
socket, not a multi-GPU cluster. World sizes >1 show synchronization
overhead as well as parallel throughput, which is the useful number for
someone evaluating whether the mechanism is understood.

    PYTHONPATH=src .venv/bin/python -m posttrain.scaling_ddp \
        --world-sizes 1 2 4 --steps 8

Tests use --tiny (random-init ~1M-param model, no HF download).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

WARMUP_STEPS = 2
BENCH_MODEL = "HuggingFaceTB/SmolLM2-135M"


class SyntheticTokens(Dataset):
    """Deterministic token dataset; each index is a fixed random draw."""

    def __init__(self, n_samples: int, seq_len: int, vocab: int, seed: int):
        self.n = n_samples
        self.seq_len = seq_len
        self.vocab = vocab
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + i)
        ids = torch.randint(0, self.vocab, (self.seq_len,), generator=g)
        labels = ids.clone()
        labels[:-1] = ids[1:]
        labels[-1] = -100
        return ids, labels


def build_model(name: str, tiny: bool):
    from transformers import AutoConfig, AutoModelForCausalLM
    if tiny:
        cfg = AutoConfig.from_pretrained(name)
        cfg.hidden_size = 64
        cfg.intermediate_size = 128
        cfg.num_hidden_layers = 2
        cfg.num_attention_heads = 2
        cfg.num_key_value_heads = 2
        cfg.vocab_size = 512
        cfg.max_position_embeddings = 256
        torch.manual_seed(0)
        return AutoModelForCausalLM.from_config(cfg)
    return AutoModelForCausalLM.from_pretrained(name)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ddp_worker(rank: int, world_size: int, port: int, args: dict,
               results_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Cap intra-op threads per rank: without this each worker claims all
    # cores and the ranks starve each other (measured: ~21x slowdown at
    # world_size=2 on a 10-core socket before this cap).
    n_threads = max(1, (os.cpu_count() or 1) // world_size)
    torch.set_num_threads(n_threads)

    model = build_model(args["model"], args["tiny"])
    n_params = sum(p.numel() for p in model.parameters())
    ddp = DDP(model)
    opt = torch.optim.AdamW(ddp.parameters(), lr=args["lr"])

    ds = SyntheticTokens(
        n_samples=args["batch"] * (args["steps"] + WARMUP_STEPS) * world_size,
        seq_len=args["seq_len"], vocab=args["vocab"], seed=args["seed"])
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                 shuffle=False)
    loader = DataLoader(ds, batch_size=args["batch"], sampler=sampler)
    it = iter(loader)

    step_times = []
    for step in range(WARMUP_STEPS + args["steps"]):
        ids, labels = next(it)
        t0 = time.perf_counter()
        out = ddp(input_ids=ids, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        if step >= WARMUP_STEPS:
            step_times.append(time.perf_counter() - t0)

    dist.barrier()
    if rank == 0:
        rec = {
            "world_size": world_size,
            "n_params": n_params,
            "batch_per_rank": args["batch"],
            "seq_len": args["seq_len"],
            "timed_steps": args["steps"],
            "step_times_s": [round(t, 4) for t in step_times],
            "mean_step_s": round(sum(step_times) / len(step_times), 4),
            "samples_per_s": round(
                args["batch"] * world_size * len(step_times)
                / sum(step_times), 2),
            "final_loss": float(out.loss.detach()),
            "threads_per_rank": n_threads,
        }
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(results_dir) / f"rank0_ws{world_size}.json", "w") as f:
            json.dump(rec, f, indent=2)
    dist.destroy_process_group()


def run_benchmark(args: dict) -> dict:
    results_dir = args["results_dir"]
    runs = []
    for ws in args["world_sizes"]:
        port = free_port()
        t0 = time.perf_counter()
        mp.spawn(ddp_worker, args=(ws, port, args, results_dir),
                 nprocs=ws, join=True)
        wall = time.perf_counter() - t0
        with open(Path(results_dir) / f"rank0_ws{ws}.json") as f:
            rec = json.load(f)
        rec["wall_s"] = round(wall, 2)
        runs.append(rec)

    out = {
        "benchmark": "ddp forward+backward, gloo backend",
        "model": args["model"] if not args["tiny"] else f"{args['model']} (tiny config)",
        "hardware": {
            "machine": platform.machine(),
            "cpu": platform.processor() or "unknown",
            "physical_cores": os.cpu_count(),
            "device": "cpu",
        },
        "torch": torch.__version__,
        "backend": "gloo",
        "runs": runs,
        "scope": ("single-socket CPU DDP; demonstrates the mechanism and "
                  "measures sync overhead. Not multi-GPU evidence."),
    }
    with open(Path(results_dir) / "scaling_ddp.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--world-sizes", type=int, nargs="+", default=[1, 2])
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=160)
    p.add_argument("--vocab", type=int, default=49152)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--model", type=str, default=BENCH_MODEL)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--results-dir", type=str, default="results/scaling")
    args = vars(p.parse_args())
    summary = run_benchmark(args)
    for r in summary["runs"]:
        print(f"world_size={r['world_size']}: {r['mean_step_s']}s/step, "
              f"{r['samples_per_s']} samples/s")


if __name__ == "__main__":
    main()
