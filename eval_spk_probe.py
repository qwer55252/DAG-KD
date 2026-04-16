#!/usr/bin/env python3
"""
eval_spk_probe.py
─────────────────
체크포인트에서 enc_text_t(teacher_feature) → z_t_text를 추출하고
fresh linear probe로 speaker classification accuracy를 측정한다.

z_t_spk도 같은 방식으로 측정하여 sanity check로 활용.

사용법:
  python eval_spk_probe.py \
    --ckpt outputs/grp_kd_based/E2_disen_orth/checkpoints/last.ckpt \
    --disen_mode 1 \
    --out probe_results/E2

  python eval_spk_probe.py \
    --ckpt outputs/grp_kd_based/E4_grl/checkpoints/last.ckpt \
    --disen_mode 3 \
    --out probe_results/E4
"""

import os
import gc
import json
import aiohttp
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import matplotlib
matplotlib.use("Agg")
from copy import deepcopy
from pathlib import Path
from omegaconf import OmegaConf
from nemo.collections import asr as nemo_asr
from datasets import load_dataset, DownloadConfig, config as hf_config
from torch.utils.data import DataLoader, TensorDataset

from train_grp_kd import DistilFlowMatchingCTCModelBPE
from models import GradientReversalLayer
from utils import (
    scan_speakers,
    build_manifest_from_hf_with_meta,
    str2bool,
    release_nemoAPI,
    load_speaker_table_from_manifest,
    snapshot_sources,
)


# ── Feature Extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, teacher, dataloader, spk_table, device):
    """
    전체 데이터로더를 순회하여 z_t_text / z_t_spk utterance-level 임베딩 수집.

    z_t_text_pool : (N, latent_dim) — 전 레이어 mean pool 후 레이어 평균
    z_t_spk_pool  : (N, latent_dim)
    spk_ids       : (N,)
    """
    model.eval()
    teacher.eval()

    text_feats, spk_feats, labels = [], [], []

    for batch in dataloader:
        if len(batch) != 5:
            continue
        signal, sig_len, _, _, sample_id = batch
        signal  = signal.to(device)
        sig_len = sig_len.to(device)
        spk_id  = spk_table.to(device)[sample_id.long()]  # (B,)

        # Teacher forward → tch_feats list 채우기
        model.tch_feats.clear()
        proc_t, len_t = teacher.preprocessor(
            input_signal=signal, length=sig_len,
        )
        _ = teacher.encoder(audio_signal=proc_t, length=len_t)

        # 모든 레이어에서 z_t_text / z_t_spk 추출 → mean pool
        batch_text, batch_spk = [], []
        for t_bht in model.tch_feats:
            t_bct    = t_bht.transpose(1, 2)         # (B, teacher_dim, T)
            z_text   = model.enc_text_t(t_bct)       # (B, latent_dim, T)
            z_spk    = model.enc_spk_t(t_bct)        # (B, latent_dim, T)
            batch_text.append(z_text.mean(dim=2))    # (B, latent_dim)
            batch_spk.append(z_spk.mean(dim=2))

        # 레이어 평균
        feat_text = torch.stack(batch_text).mean(dim=0)  # (B, latent_dim)
        feat_spk  = torch.stack(batch_spk).mean(dim=0)   # (B, latent_dim)

        text_feats.append(feat_text.cpu())
        spk_feats.append(feat_spk.cpu())
        labels.append(spk_id.cpu())

    return (
        torch.cat(text_feats),
        torch.cat(spk_feats),
        torch.cat(labels),
    )


# ── Linear Probe ─────────────────────────────────────────────────────────────

def train_probe(train_feats, train_labels, num_spk, epochs=30, lr=1e-2, device="cuda"):
    """frozen feature 위에 linear probe 학습"""
    ds = TensorDataset(train_feats.to(device), train_labels.to(device))
    dl = DataLoader(ds, batch_size=512, shuffle=True, drop_last=False)
    probe = nn.Linear(train_feats.size(1), num_spk).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(epochs):
        probe.train()
        correct = total = 0
        for x, y in dl:
            logits = probe(x)
            loss   = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == y).sum().item()
            total   += y.size(0)
        if (ep + 1) % 10 == 0:
            print(f"    probe epoch {ep+1:3d}/{epochs}: train_acc={correct/total:.4f}")

    return probe


@torch.no_grad()
def eval_probe(probe, feats, labels, device):
    probe.eval()
    ds = TensorDataset(feats.to(device), labels.to(device))
    dl = DataLoader(ds, batch_size=512, shuffle=False)
    correct = total = 0
    for x, y in dl:
        preds    = probe(x).argmax(1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / total


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser("Speaker accuracy probe for enc_text_t / enc_spk_t")

    # checkpoint
    p.add_argument("--ckpt",           type=str,      required=True)
    p.add_argument("--out",            type=str,      default="probe_results/exp")
    p.add_argument("--disen_mode",     type=int,      default=1,
                   help="1=E2(orth), 3=E4(orth+GRL)")

    # data (train_grp_kd.py와 동일)
    p.add_argument("--data_dir",       type=str,      default="data")
    p.add_argument("--data_script",    type=str,      default="./librispeech_asr.py")
    p.add_argument("--data_cfg",       type=str,      default="train_100")
    p.add_argument("--train_split",    type=str,      default="train.clean.100")
    p.add_argument("--val_split",      type=str,      default="dev.clean")
    p.add_argument("--test_split",     type=str,      default="test.clean")
    p.add_argument("--sample_rate",    type=int,      default=16000)
    p.add_argument("--batch_size",     type=int,      default=32)
    p.add_argument("--teacher_name",   type=str,      default="stt_en_conformer_ctc_small")

    # model (train_grp_kd.py와 동일 기본값)
    p.add_argument("--model_version",  type=int,      default=4)
    p.add_argument("--latent_dim",     type=int,      default=96)
    p.add_argument("--diffusion_steps",type=int,      default=9)
    p.add_argument("--flow_steps",     type=int,      default=8)
    p.add_argument("--kd_loss_type",   type=str,      default="mse")

    # probe
    p.add_argument("--probe_epochs",   type=int,      default=30)
    p.add_argument("--probe_lr",       type=float,    default=1e-2)

    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 데이터 (train_grp_kd.py 동일) ────────────────────────────────────────
    manifest_dir = os.path.join(args.data_dir, args.data_cfg, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    cache_dir = os.path.join(args.data_dir, args.data_cfg, "cache")

    dl_cfg = DownloadConfig(
        cache_dir=cache_dir, force_download=False, resume_download=True,
        max_retries=10, disable_tqdm=False,
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=72000)}},
        delete_extracted=False, extract_compressed_file=True, force_extract=True,
    )

    train_ds = load_dataset(args.data_script, args.data_cfg, split=args.train_split,
                            trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir)
    val_ds   = load_dataset(args.data_script, args.data_cfg, split=args.val_split,
                            trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir)
    test_ds  = load_dataset(args.data_script, args.data_cfg, split=args.test_split,
                            trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir)

    extra_splits = {}
    for split_name in ["dev.other", "test.other"]:
        try:
            extra_splits[split_name] = load_dataset(
                args.data_script, args.data_cfg, split=split_name,
                trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir,
            )
        except Exception:
            pass

    spk_map_path        = os.path.join(manifest_dir, "speaker_id_mapping.json")
    train_manifest      = os.path.join(manifest_dir, "train.json")
    dev_clean_manifest  = os.path.join(manifest_dir, "dev_clean.json")
    dev_other_manifest  = os.path.join(manifest_dir, "dev_other.json")
    test_clean_manifest = os.path.join(manifest_dir, "test_clean.json")
    test_other_manifest = os.path.join(manifest_dir, "test_other.json")

    spk2idx, idx2spk = scan_speakers(train_ds)
    num_spk = len(spk2idx)
    phys_cache_root = Path(manifest_dir) / "phys_cache"

    if not os.path.isfile(train_manifest):
        build_manifest_from_hf_with_meta(train_ds, train_manifest, cache_dir, spk2idx, "train", phys_cache_root)
    if not os.path.isfile(dev_clean_manifest):
        build_manifest_from_hf_with_meta(val_ds, dev_clean_manifest, cache_dir, spk2idx, "dev_clean", phys_cache_root)
    if not os.path.isfile(test_clean_manifest):
        build_manifest_from_hf_with_meta(test_ds, test_clean_manifest, cache_dir, spk2idx, "test_clean", phys_cache_root)
    if "dev.other" in extra_splits and not os.path.isfile(dev_other_manifest):
        build_manifest_from_hf_with_meta(extra_splits["dev.other"], dev_other_manifest, cache_dir, spk2idx, "dev_other", phys_cache_root)
    if "test.other" in extra_splits and not os.path.isfile(test_other_manifest):
        build_manifest_from_hf_with_meta(extra_splits["test.other"], test_other_manifest, cache_dir, spk2idx, "test_other", phys_cache_root)

    # ── Teacher & Student 모델 준비 ───────────────────────────────────────────
    trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=1,
                         default_root_dir=args.out, enable_checkpointing=False,
                         logger=False)

    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name=args.teacher_name,
        map_location=device, trainer=trainer,
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    release_nemoAPI(teacher, out_folder=os.path.join(args.out, "nemo_archive"))

    stu_cfg = deepcopy(teacher.cfg)
    stu_cfg.train_ds.is_tarred         = False
    stu_cfg.train_ds.manifest_filepath = train_manifest
    stu_cfg.train_ds.sample_rate       = args.sample_rate
    stu_cfg.train_ds.batch_size        = args.batch_size
    stu_cfg.validation_ds.is_tarred         = False
    stu_cfg.validation_ds.manifest_filepath = dev_clean_manifest
    stu_cfg.validation_ds.sample_rate       = args.sample_rate
    stu_cfg.validation_ds.batch_size        = args.batch_size
    stu_cfg.test_ds.is_tarred         = False
    stu_cfg.test_ds.manifest_filepath = test_clean_manifest
    stu_cfg.test_ds.sample_rate       = args.sample_rate
    stu_cfg.test_ds.batch_size        = args.batch_size
    stu_cfg.encoder.d_model = max(8, teacher.cfg.encoder.d_model // 2)
    stu_cfg.encoder.n_heads = max(1, teacher.cfg.encoder.n_heads // 2)
    stu_cfg.decoder.feat_in = max(8, teacher.cfg.decoder.feat_in  // 2)

    OmegaConf.set_struct(stu_cfg, False)
    for ds_key in ("train_ds", "validation_ds", "test_ds"):
        OmegaConf.set_struct(getattr(stu_cfg, ds_key), False)
    stu_cfg.train_ds.return_sample_id      = True
    stu_cfg.validation_ds.return_sample_id = False
    stu_cfg.test_ds.return_sample_id       = False

    dim_s = stu_cfg.encoder.d_model
    dim_t = teacher.cfg.encoder.d_model

    flow_cfg = {
        "meta_encoder_type": "mlp",
        "hidden_dim":        args.latent_dim,
        "time_embed_dim":    32,
        "training_sampling": args.flow_steps,
        "weight":            1.0,
        "noise_schedule":    "rectified",
        "shape_transform":   "identity",
        "loss":              "mse",
    }

    model = DistilFlowMatchingCTCModelBPE(
        cfg=stu_cfg, trainer=trainer, teacher_model=teacher,
        version=args.model_version, kd_loss_type=args.kd_loss_type,
        student_dim=dim_s, teacher_dim=dim_t, latent_dim=args.latent_dim,
        diffusion_steps=args.diffusion_steps, flow_cfg=flow_cfg,
        disen_mode=args.disen_mode, num_spk=num_spk,
    )

    # ── 체크포인트 로드 ───────────────────────────────────────────────────────
    print(f"\n[INFO] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # spk_table 주입
    spk_table = load_speaker_table_from_manifest(train_manifest)
    print(f"[INFO] spk_table: {len(spk_table)} samples, {num_spk} speakers")

    # ── 데이터로더 (return_sample_id=True 필요) ───────────────────────────────
    def make_dl(manifest):
        cfg = deepcopy(model.cfg.train_ds)  # return_sample_id=True
        cfg.manifest_filepath = manifest
        cfg.shuffle = False
        model.setup_training_data(cfg)
        return model.train_dataloader()

    def make_eval_dl(manifest):
        cfg = deepcopy(model.cfg.train_ds)  # return_sample_id=True (eval도 spk_id 필요)
        cfg.manifest_filepath = manifest
        cfg.shuffle = False
        model.setup_training_data(cfg)
        return model.train_dataloader()

    # ── Feature 추출 ─────────────────────────────────────────────────────────
    print("\n[1/4] Extracting features from train set (for probe training)...")
    train_dl = make_dl(train_manifest)
    train_text, train_spk, train_labels = extract_features(
        model, teacher, train_dl, spk_table, device,
    )
    print(f"      train: {train_text.shape[0]} samples, latent_dim={train_text.shape[1]}")

    results = {}

    for split_name, manifest in [
        ("dev_clean",  dev_clean_manifest),
        ("dev_other",  dev_other_manifest),
        ("test_clean", test_clean_manifest),
        ("test_other", test_other_manifest),
    ]:
        if not os.path.isfile(manifest):
            print(f"[WARN] {split_name} manifest not found, skip")
            continue

        print(f"\n[INFO] Extracting features from {split_name}...")
        eval_dl = make_eval_dl(manifest)
        eval_text, eval_spk, eval_labels = extract_features(
            model, teacher, eval_dl, spk_table, device,
        )
        print(f"      {split_name}: {eval_text.shape[0]} samples")

        # ── z_t_spk probe (sanity check) ─────────────────────────────────────
        print(f"\n  [z_t_spk probe — {split_name}]")
        spk_probe = train_probe(train_spk, train_labels, num_spk,
                                epochs=args.probe_epochs, lr=args.probe_lr, device=device)
        spk_acc = eval_probe(spk_probe, eval_spk, eval_labels, device)
        print(f"  z_t_spk  speaker acc ({split_name}): {spk_acc:.4f} ({spk_acc*100:.2f}%)")

        # ── z_t_text probe (main metric) ─────────────────────────────────────
        print(f"\n  [z_t_text probe — {split_name}]")
        text_probe = train_probe(train_text, train_labels, num_spk,
                                 epochs=args.probe_epochs, lr=args.probe_lr, device=device)
        text_acc = eval_probe(text_probe, eval_text, eval_labels, device)
        print(f"  z_t_text speaker acc ({split_name}): {text_acc:.4f} ({text_acc*100:.2f}%)")

        results[split_name] = {
            "z_t_spk_acc":  round(spk_acc,  4),
            "z_t_text_acc": round(text_acc, 4),
            "n_samples":    eval_text.shape[0],
        }

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    result_path = os.path.join(args.out, "probe_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to {result_path}")

    # ── 요약 출력 ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"{'Split':<12} {'z_t_spk acc':>14} {'z_t_text acc':>14}")
    print("="*60)
    for split, r in results.items():
        print(f"{split:<12} {r['z_t_spk_acc']*100:>13.2f}% {r['z_t_text_acc']*100:>13.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
