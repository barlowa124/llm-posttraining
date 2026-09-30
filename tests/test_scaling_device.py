"""scaling_device: per-device step timing (tiny model, cpu only in tests)."""

import json

from posttrain.scaling_device import bench_device

TINY = {
    "model": "HuggingFaceTB/SmolLM2-135M",
    "tiny": True,
    "batch": 2, "seq_len": 16, "vocab": 512,
    "steps": 2, "lr": 1e-5, "seed": 5,
}


def test_cpu_bench_record_schema():
    rec = bench_device("cpu", TINY)
    assert rec["device"] == "cpu"
    assert rec["timed_steps"] == 2
    assert len(rec["step_times_s"]) == 2
    assert rec["samples_per_s"] > 0
    assert rec["n_params"] < 5_000_000


def test_deterministic_same_seed():
    a = bench_device("cpu", TINY)
    b = bench_device("cpu", TINY)
    assert a["final_loss"] == b["final_loss"]
