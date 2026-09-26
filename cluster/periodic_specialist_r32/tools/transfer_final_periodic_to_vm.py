#!/usr/bin/env python3
"""Direct cluster-to-VM minimal Periodic specialist transfer with SHA256 verification."""

import hashlib
import os
from pathlib import PurePosixPath

import paramiko


DEST_ROOT = "/home/ray/Desktop/transfer/periodic_specialist_r32"
FILES = [
    ("/root/V17indepentMOEV2/periodic_v16_public/runs/20260719_004132/ru32_rp32/FINAL_PERIODIC_SPECIALIST.pt", "checkpoint/FINAL_PERIODIC_SPECIALIST.pt", "b2052a746fdaab60f83e65d1b91d038a9dded8255ce74f41e3f6fca4c9e717c5"),
    ("/root/V17indepentMOEV2/periodic_v16_public/FINAL_PERIODIC_SPECIALIST.json", "metadata/FINAL_PERIODIC_SPECIALIST.json", "a34735218289b245edd0eeab71c40d89d4ada6c4b47f3476672e6e234f917e7a"),
    ("/root/V17indepentMOEV2/periodic_v16_public/code/train_periodic_moe.py", "code/train_periodic_moe.py", "500c2179734e6c3a6db6469629d4727637ae4552843b4d3154ca7b281a102314"),
    ("/root/V17indepentMOEV2/periodic_v16_public/code/evaluate_periodic_r32.py", "code/evaluate_periodic_r32.py", "3b694c5e23e4b81e9fc68987ee2b084cdfce4d63cfe44def66728850502f2380"),
    ("/root/V17indepentMOEV2/periodic_v16_public/runs/20260719_004132/ru32_rp32/multihorizon_evaluation/periodic_r32_multihorizon_evaluation.json", "evaluation/periodic_r32_multihorizon_evaluation.json", "3c1973e1ea2c54d3c8dc5e016cff4d91fd84252d2c7ab789771ac321e0d5c0db"),
    ("/root/V17indepentMOEV2/periodic_v16_public/runs/20260719_004132/ru32_rp32/multihorizon_evaluation/PERIODIC_R32_RP32_HELDOUT_EVALUATION_REPORT.md", "evaluation/PERIODIC_R32_RP32_HELDOUT_EVALUATION_REPORT.md", "9afc583e3df75b482698cf08b6fda7fb3cef3af2a6e26c7e93918582c7f3d33b"),
    ("/cephfs/shared/V17_IndependentMoEROM_v2/artifacts/periodic_r32_handoff/periodic/Global_POD_AreaWeighted_L2/global_velocity_pod_area_weighted_l2.npz", "assets/Global_POD_AreaWeighted_L2/global_velocity_pod_area_weighted_l2.npz", "a2c12dc3ea25772e7bd4216b37683df9dbcce804063f9900bdaebac3e25dc7a2"),
    ("/cephfs/shared/V17_IndependentMoEROM_v2/artifacts/periodic_r32_handoff/periodic/Global_POD_AreaWeighted_L2/global_pressure_pod_area_weighted_l2.npz", "assets/Global_POD_AreaWeighted_L2/global_pressure_pod_area_weighted_l2.npz", "f6e0a23432581c78e9f649d60bb0fceb9087a6a1d0631778bde444338c16239f"),
    ("/cephfs/shared/V17_IndependentMoEROM_v2/artifacts/periodic_r32_handoff/periodic/normalization_periodic.npz", "assets/normalization_periodic.npz", "a938909e01a6f16a5310b343a91e7077f86d0ca570388f9ea134f3bd115770c2"),
    ("/cephfs/shared/V17_IndependentMoEROM_v2/artifacts/periodic_r32_handoff/periodic/velocity_rom_periodic.npz", "assets/velocity_rom_periodic.npz", "8af11eaaab376d7c146fa45884f5d56d8bc4a6e48780cbc57cd8cb3f8f4ce87b"),
    ("/cephfs/shared/V17_IndependentMoEROM_v2/artifacts/periodic_r32_handoff/periodic/pressure_poisson_surrogate_periodic.npz", "assets/pressure_poisson_surrogate_periodic.npz", "bf64858428733e90bda185b504ab98ef7fc96f5e0f3bb3d6d788d758e2b15a09"),
    ("/cephfs/shared/V17_IndependentMoEROM_v2/artifacts/periodic_r32_handoff/periodic/periodic_r32_handoff_manifest.json", "assets/periodic_r32_handoff_manifest.json", "991562eaa6cc55aa9ac23c2d53d20619b6a696ed85b549425956e818ae1b9203"),
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("10.210.22.202", port=31283, username="root", password="123", timeout=30)
    source = client.open_sftp()
    os.makedirs(DEST_ROOT, exist_ok=True)
    manifest = []
    for remote, relative, expected in FILES:
        target = os.path.join(DEST_ROOT, relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.isfile(target) and sha256(target) == expected:
            print(f"SKIP {relative}", flush=True)
            manifest.append((expected, relative))
            continue
        partial = target + ".partial"
        if os.path.exists(partial):
            os.remove(partial)
        digest = hashlib.sha256()
        with source.open(remote, "rb") as src, open(partial, "wb") as dst:
            size = source.stat(remote).st_size
            src.prefetch(size)
            copied = 0
            while True:
                block = src.read(8 * 1024 * 1024)
                if not block:
                    break
                dst.write(block)
                digest.update(block)
                copied += len(block)
                if copied == size or copied % (128 * 1024 * 1024) < len(block):
                    print(f"COPY {relative} {copied}/{size}", flush=True)
        if copied != size or digest.hexdigest() != expected:
            raise RuntimeError(f"integrity failure: {relative}")
        os.replace(partial, target)
        manifest.append((expected, relative))
        print(f"OK {relative}", flush=True)
    manifest_path = os.path.join(DEST_ROOT, "TRANSFER_MANIFEST.sha256")
    with open(manifest_path + ".partial", "w", encoding="utf-8") as handle:
        handle.write("# Final Periodic r32/rp32 minimal transfer manifest\n")
        for digest, relative in manifest:
            handle.write(f"{digest}  {relative}\n")
    os.replace(manifest_path + ".partial", manifest_path)
    source.close(); client.close()
    print("TRANSFER_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
