"""Provenance manifest: git state, package versions, input/config hashes."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()[:10]
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True
            ).stdout.strip()
        )
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def write_manifest(out_path, inputs: list, config_path: str = "config/config.yaml"):
    import pandas
    import torch
    import transformers

    m = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_commit(),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "pandas": pandas.__version__,
        },
        "config_sha256": sha256_file(config_path)
        if Path(config_path).exists()
        else None,
        "inputs": [
            {"path": str(p), "sha256": sha256_file(p)}
            for p in inputs
            if Path(p).exists()
        ],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(m, f, indent=2)
    return m
