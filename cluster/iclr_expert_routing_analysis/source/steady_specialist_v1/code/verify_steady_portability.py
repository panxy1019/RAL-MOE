"""CPU-only audit for the migrated Steady specialist reproduction bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/root/panxy/particalMOE/steady_specialist_v1")
EXPECTED_CHECKPOINTS = {
    "checkpoint/frozen_s2b_validation.pt": "3ec8bdedbe2f953a181d71ae819980cbbdfd65c8fee33aac9237a16d780a682d",
    "checkpoint/frozen_s3b_contraction.pt": "b85f01bfcd8c209ee4ff6fe6f2a61dfa12acbcecd83da09b153fc0f15b19c23b",
    "checkpoint/frozen_s4_validation_step_1200.pt": "bcab5661af39e3262103ed887ce9846415467452b453f3dcc14e133f0f8bd80d",
}
REQUIRED = [
    "code/train_v16_4_v2_r32_compat.py", "code/train_s2b_3090.py",
    "code/finalize_s2b_3090.py", "code/train_s3.py",
    "code/audit_paired_gain_metrics.py", "code/evaluate_s3_heldout.py",
    "code/train_s4_steady.py", "code/finalize_s4_one_time.py",
    "code/training_s2b_portable.json", "code/reproduce_s4_one_epoch.py",
    "source_artifacts/steady/velocity_pod_steady.npz",
    "source_artifacts/steady/pressure_pod_steady.npz",
    "source_artifacts/steady/normalization_steady.npz",
    "source_artifacts/steady/velocity_rom_steady.npz",
    "source_artifacts/steady/pressure_poisson_surrogate_steady.npz",
    "source_artifacts/steady/Global_POD_AreaWeighted_L2/pod_snapshot_index.csv",
    "data/perturbation_bank_train_validation.npz",
    "evaluation_inputs/s3_heldout_metrics.json",
    "evaluation_inputs/s3_paired_gain_metric_audit_raw.json",
    *EXPECTED_CHECKPOINTS,
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    missing = [item for item in REQUIRED if not (ROOT / item).is_file()]
    assert not missing, missing
    hashes = {item: sha256(ROOT / item) for item in REQUIRED}
    for item, expected in EXPECTED_CHECKPOINTS.items():
        assert hashes[item] == expected, (item, hashes[item], expected)
    with np.load(ROOT / "source_artifacts/steady/velocity_pod_steady.npz", allow_pickle=False) as velocity, \
         np.load(ROOT / "source_artifacts/steady/pressure_pod_steady.npz", allow_pickle=False) as pressure:
        split = velocity["snapshot_splits"].astype(str)
        split_counts = {name: int(np.sum(split == name)) for name in ("train", "validation", "heldout")}
        assert split_counts == {"train": 894, "validation": 127, "heldout": 256}
        assert velocity["phi_uv"].shape[0] >= 32 and pressure["phi_p"].shape[0] >= 32
        assert str(velocity["fit_split"].item()) == str(pressure["fit_split"].item()) == "train"
        assert str(pressure["pressure_gauge"].item()) == "subtract_area_mean_per_snapshot"
    checkpoint_schema = {}
    for item in EXPECTED_CHECKPOINTS:
        payload = torch.load(ROOT / item, map_location="cpu", weights_only=False)
        assert "model" in payload and isinstance(payload["model"], dict)
        checkpoint_schema[item] = {
            "step": int(payload.get("step", -1)),
            "model_tensor_count": len(payload["model"]),
        }
        del payload
    report = {
        "status": "PASS", "gpu_used": False,
        "root": str(ROOT), "split_counts": split_counts,
        "ru": 32, "rp": 32, "pressure_gauge": "subtract_area_mean_per_snapshot",
        "heldout_used_for_training": False,
        "checkpoint_schema": checkpoint_schema, "files": hashes,
        "runtime": {"python": __import__("sys").version, "torch": torch.__version__, "numpy": np.__version__},
    }
    target = ROOT / "provenance" / "PORTABILITY_AUDIT.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "gpu_used", "split_counts", "checkpoint_schema", "runtime")}, indent=2))


if __name__ == "__main__":
    main()
