#!/usr/bin/env python3
"""
Precompute per-utterance F0 statistics and frame-level sequences for prosody anchor.

Reads a manifest (JSON-lines, must have prosody_physics_filepath + manifest_id fields),
runs librosa.pyin per utterance (via utils.compute_phys_for_wav).

Outputs (two modes, can be combined):
  --out     f0_stats_train.pt   (N, 2) float32  utterance-level [mean_f0, mean_energy]
  --out_seq f0_seq_train.pt     dict with keys:
                                  "seq": (N, T_max, 2) float16  frame-level [f0_hz, energy]
                                  "len": (N,)          int32    actual mel frame counts
                                  "f0_mean": float     mean voiced F0 over training set (Hz)
                                  "f0_std":  float     std  voiced F0 over training set (Hz)
                                  "en_mean": float     mean energy over training set
                                  "en_std":  float     std  energy over training set

Unvoiced frames are stored as f0=0.0 so that voiced/unvoiced can be recovered at train time
via: vuv_mask = (seq[:,:,0] > 10.0).

Indexed as: table[manifest_id - 1]  (manifest_id is 1-based per NeMo convention)

Usage:
    # utterance-level only (disen_mode=5)
    python precompute_f0.py \\
        --manifest data/train_100/manifests/train.json \\
        --out      data/train_100/manifests/f0_stats_train.pt \\
        --workers  8

    # frame-level sequences (disen_mode=6)
    python precompute_f0.py \\
        --manifest data/train_100/manifests/train.json \\
        --out_seq  data/train_100/manifests/f0_seq_train.pt \\
        --workers  8

    # both at once
    python precompute_f0.py \\
        --manifest data/train_100/manifests/train.json \\
        --out      data/train_100/manifests/f0_stats_train.pt \\
        --out_seq  data/train_100/manifests/f0_seq_train.pt \\
        --workers  8
"""

import os
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from utils import compute_phys_for_wav


def _worker(args):
    """Returns utterance-level stats. Top-level for picklability."""
    manifest_id, wav_path, phys_path, sr = args
    phys_path = Path(phys_path)

    try:
        if not phys_path.exists():
            phys_path.parent.mkdir(parents=True, exist_ok=True)
            f0, energy, vuv = compute_phys_for_wav(wav_path, SR=sr)
            phys = np.stack([f0, energy, vuv], axis=0).astype(np.float16)
            np.save(phys_path, phys)
        else:
            phys = np.load(phys_path).astype(np.float32)  # (3, T)
            f0     = phys[0]
            energy = phys[1]
            vuv    = phys[2]

        voiced_mask = vuv > 0.5
        mean_f0     = float(f0[voiced_mask].mean()) if voiced_mask.any() else 0.0
        mean_energy = float(energy.mean())

        return manifest_id, mean_f0, mean_energy, None

    except Exception as e:
        return manifest_id, 0.0, 0.0, str(e)


def _worker_seq(args):
    """Returns frame-level [f0_masked, energy] sequence. Top-level for picklability."""
    manifest_id, wav_path, phys_path, sr = args
    phys_path = Path(phys_path)

    try:
        if not phys_path.exists():
            phys_path.parent.mkdir(parents=True, exist_ok=True)
            f0, energy, vuv = compute_phys_for_wav(wav_path, SR=sr)
            phys = np.stack([f0, energy, vuv], axis=0).astype(np.float16)
            np.save(phys_path, phys)
        else:
            phys = np.load(phys_path).astype(np.float32)  # (3, T)
            f0     = phys[0]
            energy = phys[1]
            vuv    = phys[2]

        # zero out unvoiced F0 so vuv can be recovered via f0 > threshold
        f0_masked = f0.copy()
        f0_masked[vuv <= 0.5] = 0.0

        seq = np.stack([f0_masked, energy], axis=1).astype(np.float16)  # (T, 2)
        return manifest_id, seq, None

    except Exception as e:
        return manifest_id, None, str(e)


def build_f0_stats(manifest_path: str, out_path: str, sr: int = 16000, workers: int = 4):
    """Utterance-level [mean_f0, mean_energy] → (N, 2) float32 tensor."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    for e in entries[:1]:
        for key in ("manifest_id", "audio_filepath", "prosody_physics_filepath"):
            if key not in e:
                raise KeyError(
                    f"manifest entry missing '{key}'. "
                    f"Re-generate manifest with build_manifest_from_hf_with_meta()."
                )

    max_id = max(int(e["manifest_id"]) for e in entries)
    stats  = np.zeros((max_id, 2), dtype=np.float32)

    jobs = [
        (int(e["manifest_id"]), e["audio_filepath"], e["prosody_physics_filepath"], sr)
        for e in entries
    ]

    n_fail = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, j): j[0] for j in jobs}
        pbar = tqdm(as_completed(futures), total=len(jobs), desc="F0 stats", unit="utt")
        for fut in pbar:
            mid, mean_f0, mean_energy, err = fut.result()
            if err:
                n_fail += 1
                pbar.set_postfix(fail=n_fail)
            else:
                stats[mid - 1, 0] = mean_f0
                stats[mid - 1, 1] = mean_energy

    tensor = torch.from_numpy(stats)
    torch.save(tensor, out_path)
    print(f"[INFO] Saved f0_stats: shape={tuple(tensor.shape)}, fail={n_fail}/{len(jobs)}")
    voiced = tensor[:, 0][tensor[:, 0] > 0]
    print(f"[INFO] F0  — mean={voiced.mean():.1f} Hz  std={voiced.std():.1f} Hz")
    print(f"[INFO] Energy — mean={tensor[:,1].mean():.4f}  std={tensor[:,1].std():.4f}")


def build_f0_seq(manifest_path: str, out_path: str, sr: int = 16000, workers: int = 4):
    """
    Frame-level [f0_masked, energy] → dict saved at out_path:
      "seq": (N, T_max, 2) float16
      "len": (N,)          int32
      "f0_mean", "f0_std", "en_mean", "en_std": float (global norm stats)
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    for e in entries[:1]:
        for key in ("manifest_id", "audio_filepath", "prosody_physics_filepath"):
            if key not in e:
                raise KeyError(f"manifest entry missing '{key}'.")

    max_id = max(int(e["manifest_id"]) for e in entries)

    jobs = [
        (int(e["manifest_id"]), e["audio_filepath"], e["prosody_physics_filepath"], sr)
        for e in entries
    ]

    # Collect sequences
    seqs = [None] * max_id   # seqs[manifest_id - 1] = np.array (T, 2)
    lens = np.zeros(max_id, dtype=np.int32)

    n_fail = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker_seq, j): j[0] for j in jobs}
        pbar = tqdm(as_completed(futures), total=len(jobs), desc="F0 seq", unit="utt")
        for fut in pbar:
            mid, seq, err = fut.result()
            if err or seq is None:
                n_fail += 1
                pbar.set_postfix(fail=n_fail)
            else:
                seqs[mid - 1] = seq
                lens[mid - 1] = len(seq)

    T_max = int(lens.max())
    print(f"[INFO] T_max={T_max} frames, N={max_id}, fail={n_fail}/{len(jobs)}")

    # Pad to (N, T_max, 2) float16
    padded = np.zeros((max_id, T_max, 2), dtype=np.float16)
    for i, seq in enumerate(seqs):
        if seq is not None:
            padded[i, :len(seq)] = seq

    tensor_seq = torch.from_numpy(padded)   # (N, T_max, 2) float16
    tensor_len = torch.from_numpy(lens)     # (N,) int32

    # Global norm stats from voiced frames and all energy frames
    voiced_f0 = tensor_seq[:, :, 0].float()
    voiced_f0 = voiced_f0[voiced_f0 > 10.0]   # voiced threshold: 10 Hz
    f0_mean = float(voiced_f0.mean()) if len(voiced_f0) > 0 else 100.0
    f0_std  = float(voiced_f0.std().clamp(min=1.0)) if len(voiced_f0) > 0 else 50.0

    # Energy stats: only from actual frames (not padding zeros)
    en_vals = []
    for i in range(max_id):
        l = int(lens[i])
        if l > 0:
            en_vals.append(tensor_seq[i, :l, 1].float())
    en_all  = torch.cat(en_vals) if en_vals else torch.zeros(1)
    en_mean = float(en_all.mean())
    en_std  = float(en_all.std().clamp(min=1e-6))

    out_dict = {
        "seq":     tensor_seq,
        "len":     tensor_len,
        "f0_mean": f0_mean,
        "f0_std":  f0_std,
        "en_mean": en_mean,
        "en_std":  en_std,
    }
    torch.save(out_dict, out_path)
    print(f"[INFO] Saved f0_seq: seq={tuple(tensor_seq.shape)}, "
          f"f0={f0_mean:.1f}±{f0_std:.1f} Hz, en={en_mean:.4f}±{en_std:.4f}")


def main():
    p = argparse.ArgumentParser("Precompute F0 statistics / frame sequences")
    p.add_argument("--manifest",  type=str, required=True)
    p.add_argument("--out",       type=str, default=None,
                   help="Output path for utterance-level stats (.pt, shape=(N,2))")
    p.add_argument("--out_seq",   type=str, default=None,
                   help="Output path for frame-level sequence dict (.pt). disen_mode=6용.")
    p.add_argument("--sr",        type=int, default=16000)
    p.add_argument("--workers",   type=int, default=4)
    args = p.parse_args()

    if args.out is None and args.out_seq is None:
        p.error("--out 또는 --out_seq 중 하나 이상을 지정해야 합니다.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        build_f0_stats(args.manifest, args.out, sr=args.sr, workers=args.workers)

    if args.out_seq:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_seq)), exist_ok=True)
        build_f0_seq(args.manifest, args.out_seq, sr=args.sr, workers=args.workers)


if __name__ == "__main__":
    main()
