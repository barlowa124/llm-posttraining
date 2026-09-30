"""scaling_ddp: deterministic dataset, worker contract, benchmark schema.

The heavy path (mp.spawn over the real 135M model) is exercised by the
committed benchmark run, not in CI — tests use the --tiny config and a
world_size=1 direct worker call.
"""

import json

import torch
from torch.utils.data import DistributedSampler

from posttrain.scaling_ddp import (SyntheticTokens, build_model,
                                   ddp_worker, free_port)


def test_dataset_deterministic():
    a = SyntheticTokens(10, 16, 512, seed=5)
    b = SyntheticTokens(10, 16, 512, seed=5)
    assert torch.equal(a[0][0], b[0][0])
    assert not torch.equal(a[0][0], a[1][0])


def test_dataset_labels_shifted():
    ds = SyntheticTokens(4, 8, 512, seed=1)
    ids, labels = ds[0]
    assert labels[-1].item() == -100
    assert torch.equal(labels[:-1], ids[1:])


def test_sampler_partitions_without_overlap():
    ds = SyntheticTokens(12, 8, 512, seed=1)
    s0 = list(DistributedSampler(ds, num_replicas=2, rank=0, shuffle=False))
    s1 = list(DistributedSampler(ds, num_replicas=2, rank=1, shuffle=False))
    assert not set(s0) & set(s1)
    assert len(s0) + len(s1) == 12


TINY_ARGS = {
    "model": "HuggingFaceTB/SmolLM2-135M",
    "tiny": True,
    "batch": 2, "seq_len": 16, "vocab": 512,
    "steps": 2, "lr": 1e-5, "seed": 5,
    "results_dir": None,
}


def test_tiny_model_builds_without_download():
    model = build_model(TINY_ARGS["model"], tiny=True)
    n = sum(p.numel() for p in model.parameters())
    assert n < 5_000_000
    out = model(input_ids=torch.randint(0, 512, (1, 8)))
    assert out.logits.shape[-1] == 512


def test_worker_world_size_1(tmp_path):
    args = {**TINY_ARGS, "results_dir": str(tmp_path)}
    ddp_worker(rank=0, world_size=1, port=free_port(), args=args,
               results_dir=str(tmp_path))
    rec = json.loads((tmp_path / "rank0_ws1.json").read_text())
    assert rec["world_size"] == 1
    assert rec["timed_steps"] == 2
    assert len(rec["step_times_s"]) == 2
    assert rec["samples_per_s"] > 0
    assert all(t > 0 for t in rec["step_times_s"])
