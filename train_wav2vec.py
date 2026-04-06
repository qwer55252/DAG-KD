#!/usr/bin/env python3
"""
DAG-KD wav2vec2.0 학습 스크립트
- train.py의 데이터 파이프라인(manifest, phys_cache) 재사용
- NeMo DataLoader 대신 순수 PyTorch DataLoader + HuggingFace Processor 사용
- DistilDAGKDWav2Vec2 (models_wav2vec.py) 로 학습
"""

import os
import json
import torch
import aiohttp
import argparse
import statistics
import numpy as np
import soundfile as sf
import lightning as pl
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Dict
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from datasets import load_dataset, DownloadConfig, config as hf_config
from transformers import Wav2Vec2Processor

from models_wav2vec import DistilDAGKDWav2Vec2
from utils import (
    scan_speakers,
    build_manifest_from_hf_with_meta,
    str2bool,
    compute_sample_wers,
    compute_speaker_durations,
    int_list_arg,
    save_speaker_mapping,
    build_phys_cache_for_manifest,
    snapshot_sources,
)


# ============================================================
# Dataset
# ============================================================

class ManifestDataset(torch.utils.data.Dataset):
    """
    NeMo manifest(jsonl) 포맷을 읽어 wav2vec2 입력으로 변환하는 Dataset.
    각 샘플: {"audio_filepath", "text", "spk_idx", "manifest_id", ...}
    """

    def __init__(self, manifest_path: str, processor: Wav2Vec2Processor, sample_rate: int = 16000):
        self.processor = processor
        self.sample_rate = sample_rate
        self.data: List[Dict] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                obj = json.loads(line.strip())
                obj["_dataset_idx"] = idx
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]

        # 오디오 로드
        wav, sr = sf.read(item["audio_filepath"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)  # stereo → mono

        # 샘플레이트 불일치 시 resampling
        if sr != self.sample_rate:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)

        # Wav2Vec2FeatureExtractor: normalize → input_values (T_wav,)
        input_values = self.processor.feature_extractor(
            wav, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_values[0]  # (T_wav,)

        return {
            "input_values": input_values,          # (T_wav,) float32
            "text": item.get("text", ""),
            "spk_idx": int(item.get("spk_idx", -1)),
            "manifest_id": int(item.get("manifest_id", idx + 1)),  # 1-based
        }


# ============================================================
# DataCollator
# ============================================================

@dataclass
class DataCollatorCTCPadding:
    """
    가변 길이 오디오를 패딩하고 레이블(CTC)도 패딩하는 collator.
    - input_values: 오른쪽 0-패딩
    - attention_mask: 유효 구간 1
    - labels: 오른쪽 -100 패딩 (CTC loss에서 ignore)
    """
    processor: Any

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # ---- 오디오 패딩 ----
        max_wav_len = max(f["input_values"].size(0) for f in features)
        B = len(features)

        input_values = torch.zeros(B, max_wav_len, dtype=torch.float32)
        attention_mask = torch.zeros(B, max_wav_len, dtype=torch.long)
        for i, f in enumerate(features):
            L = f["input_values"].size(0)
            input_values[i, :L] = f["input_values"]
            attention_mask[i, :L] = 1

        # ---- 텍스트 토크나이징 + 레이블 패딩 ----
        texts = [f["text"] for f in features]
        label_batch = self.processor.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
        )
        # padding 위치를 -100으로 (CTC ignore index)
        labels = label_batch["input_ids"].masked_fill(
            label_batch["attention_mask"].ne(1), -100
        )

        return {
            "input_values": input_values,           # (B, T_wav)
            "attention_mask": attention_mask,        # (B, T_wav)
            "labels": labels,                        # (B, T_label)
            "speaker_ids": torch.tensor(
                [f["spk_idx"] for f in features], dtype=torch.long
            ),                                       # (B,)
            "manifest_ids": torch.tensor(
                [f["manifest_id"] for f in features], dtype=torch.long
            ),                                       # (B,)
        }


# ============================================================
# Lightning DataModule
# ============================================================

class Wav2VecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_manifest: str,
        val_manifest: str,
        processor: Wav2Vec2Processor,
        batch_size: int = 8,
        num_workers: int = 4,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self._collator = DataCollatorCTCPadding(processor=processor)

    def setup(self, stage=None):
        self.train_ds = ManifestDataset(self.train_manifest, self.processor, self.sample_rate)
        self.val_ds = ManifestDataset(self.val_manifest, self.processor, self.sample_rate)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collator,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collator,
            pin_memory=True,
        )


# ============================================================
# 평가 유틸
# ============================================================

def evaluate_split(
    model: DistilDAGKDWav2Vec2,
    manifest_path: str,
    processor: Wav2Vec2Processor,
    batch_size: int,
    split_name: str,
    wandb_logger,
    trainer,
    plot_dir: str,
):
    """단일 split에 대해 WER 계산 + 히스토그램 저장."""
    if not os.path.isfile(manifest_path):
        print(f"[WARN] manifest not found for {split_name}: {manifest_path}, skip")
        return

    print(f"\n===== Evaluating on {split_name} =====")

    collator = DataCollatorCTCPadding(processor=processor)
    ds = ManifestDataset(manifest_path, processor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=4, collate_fn=collator, pin_memory=True)

    model.eval()
    device = next(model.parameters()).device

    all_refs: List[str] = []
    all_hyps: List[str] = []

    with torch.no_grad():
        for batch in dl:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            model.stu_feats.clear()
            outputs = model.student(
                input_values=input_values,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # (B, T, V)
            pred_ids = logits.argmax(dim=-1).cpu()  # (B, T)

            # Greedy CTC decode
            for i in range(pred_ids.size(0)):
                # 예측
                hyp_ids = pred_ids[i].tolist()
                hyp_ids_collapsed = []
                prev = None
                for t in hyp_ids:
                    if t == model._blank_id:
                        prev = None
                        continue
                    if t != prev:
                        hyp_ids_collapsed.append(t)
                    prev = t
                hyp_str = processor.decode(hyp_ids_collapsed)

                # 레퍼런스
                ref_ids = labels[i].tolist()
                ref_ids = [t for t in ref_ids if t != -100 and t != model._blank_id]
                ref_str = processor.decode(ref_ids)

                all_hyps.append(hyp_str.lower().strip())
                all_refs.append(ref_str.lower().strip())

    sample_wers = compute_sample_wers(all_refs, all_hyps)
    sample_wers_pct = [w * 100.0 for w in sample_wers]
    wer_mean = float(statistics.mean(sample_wers_pct)) if sample_wers_pct else 0.0
    wer_std = float(statistics.stdev(sample_wers_pct)) if len(sample_wers_pct) > 1 else 0.0

    print(f"→ {split_name}: per-sample WER = {wer_mean:.2f}% ± {wer_std:.2f}%")

    # WER 분포 그림 저장
    os.makedirs(plot_dir, exist_ok=True)
    wers_np = np.array(sample_wers_pct, dtype=float)

    plt.figure()
    bins = [0, 10, 20, 30, 50, 100, 200]
    plt.hist(wers_np, bins=bins, edgecolor="black")
    plt.xlabel("Per-sample WER (%)")
    plt.ylabel("Count")
    plt.title(f"WER Histogram - {split_name}\nmean={wer_mean:.2f}%, std={wer_std:.2f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"wer_hist_{split_name}.png"))
    plt.close()

    plt.figure()
    plt.boxplot(wers_np, vert=True, showfliers=True)
    plt.ylabel("Per-sample WER (%)")
    plt.title(f"WER Boxplot - {split_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"wer_box_{split_name}.png"))
    plt.close()

    wandb_logger.log_metrics(
        {f"{split_name}/wer_mean": wer_mean, f"{split_name}/wer_std": wer_std},
        step=trainer.current_epoch,
    )


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser("DAG-KD wav2vec2 train script")

    # Data
    p.add_argument("--data_dir",       type=str,      default="data")
    p.add_argument("--data_script",    type=str,      default="./librispeech_asr.py")
    p.add_argument("--data_cfg",       type=str,      default="train_100")
    p.add_argument("--train_split",    type=str,      default="train.clean.100")
    p.add_argument("--val_split",      type=str,      default="dev.clean")
    p.add_argument("--test_split",     type=str,      default="test.clean")
    p.add_argument("--sample_rate",    type=int,      default=16000)
    p.add_argument("--batch_size",     type=int,      default=8)
    p.add_argument("--num_workers",    type=int,      default=4)
    p.add_argument("--test_mode",      type=str2bool, default=False)

    # Logging/ckpt
    p.add_argument("--epochs",         type=int,      default=100)
    p.add_argument("--gpus",           type=int,      default=1)
    p.add_argument("--out",            type=str,      default="outputs")
    p.add_argument("--resume_ckpt_path", type=str,    default="")

    # Teacher / Student
    p.add_argument("--teacher_name",   type=str,      default="facebook/wav2vec2-large-960h")
    p.add_argument("--student_name",   type=str,      default="facebook/wav2vec2-base-960h")
    p.add_argument("--processor_name", type=str,      default="",
                   help="Processor HF ID. 기본값: student_name과 동일.")

    # KD config
    p.add_argument("--use_ctc",          type=str2bool, default=True)
    p.add_argument("--use_logit_kd",     type=str2bool, default=True)
    p.add_argument("--kd_alpha",         type=float,    default=0.5)
    p.add_argument("--kd_temperature",   type=float,    default=1.0)
    p.add_argument("--use_layer_kd",     type=str2bool, default=False)
    p.add_argument("--layer_kd_alpha",   type=float,    default=0.5)

    # Generative KD
    p.add_argument("--use_flow",         type=str2bool, default=False)
    p.add_argument("--flow_steps",       type=int,      default=8)
    p.add_argument("--flow_weight",      type=float,    default=1.0)
    p.add_argument("--use_diffkd",       type=str2bool, default=False)
    p.add_argument("--diffkd_steps",     type=int,      default=5)

    # Disentanglement
    p.add_argument("--use_disent",        type=str2bool,   default=True)
    # Teacher 레이어 선택 (1-based, Factorization용)
    p.add_argument("--tch_spk_layers",   type=int_list_arg, default=[1, 2],
                   help="Teacher에서 speaker rep 뽑을 레이어 (1-based, e.g. '1,2'). 24L 기준 하위 추천.")
    p.add_argument("--tch_txt_layers",   type=int_list_arg, default=[23, 24],
                   help="Teacher에서 text rep 뽑을 레이어 (1-based, e.g. '23,24'). 24L 기준 상위 추천.")
    # Student 레이어 선택 (1-based, S-DisKD용)
    p.add_argument("--stu_spk_layers",   type=int_list_arg, default=[1, 2],
                   help="Student에서 speaker rep 뽑을 레이어 (1-based, e.g. '1,2'). 12L 기준 하위 추천.")
    p.add_argument("--stu_txt_layers",   type=int_list_arg, default=[11, 12],
                   help="Student에서 text rep 뽑을 레이어 (1-based, e.g. '11,12'). 12L 기준 상위 추천.")
    p.add_argument("--disen_mi_pairs",       type=str,        default="ts,tp,ps")
    p.add_argument("--disen_lll_weight",     type=float,      default=1.0)
    p.add_argument("--disen_mi_weight",      type=float,      default=1e-3)

    # W&B
    p.add_argument("--wandb_project",    type=str,  default=os.getenv("PRJ_NAME", "DAG-KD-wav2vec"))
    p.add_argument("--wandb_run",        type=str,  default=os.getenv("EXP_NAME", "wav2vec_run"))
    p.add_argument("--disen_vis_enable", type=str2bool, default=False)

    # Text speaker probe
    p.add_argument("--use_txt_spk_probe", type=str2bool, default=True)
    p.add_argument("--txt_probe_lambda",  type=float,    default=1.0)

    # MI CLUB
    p.add_argument("--neg_K",            type=int,   default=8)
    p.add_argument("--mi_warmup_steps",  type=int,   default=5000)
    p.add_argument("--mi_ramp_steps",    type=int,   default=20000)
    p.add_argument("--mi_lambda_max",    type=float, default=0.01)
    p.add_argument("--lll_lambda_max",   type=float, default=0.01)
    p.add_argument("--mi_clamp_min0",    type=str2bool, default=True)

    # S-DisKD
    p.add_argument("--use_stu_txt_kd",    type=str2bool, default=False)
    p.add_argument("--use_stu_spk_kd",    type=str2bool, default=False)
    p.add_argument("--use_stu_club",      type=str2bool, default=False)
    p.add_argument("--stu_txt_kd_weight", type=float,    default=1.0)
    p.add_argument("--stu_spk_kd_weight", type=float,    default=1.0)
    p.add_argument("--stu_club_weight",   type=float,    default=1e-3)

    # Optimizer
    p.add_argument("--learning_rate",    type=float, default=3e-4)
    p.add_argument("--warmup_epochs",    type=int,   default=0,
                   help="Linear warmup epoch 수 (이후 CosineAnnealingLR). large 모델 fine-tuning 시 5~10 권장")
    p.add_argument("--kd_warmup_epochs", type=int,   default=0,
                   help="KD 시작 전 CTC only로 학습할 epoch 수. random init student 학습 시 권장 (e.g. 10)")
    p.add_argument("--freeze_feature_extractor", type=str2bool, default=False,
                   help="Student CNN feature extractor를 freeze (large 모델 fine-tuning 시 권장)")
    p.add_argument("--random_init_student", type=str2bool, default=False,
                   help="Student를 random initialization으로 시작 (KD 순수 효과 측정용)")

    args = p.parse_args()

    # ---- Output dir ----
    os.makedirs(args.out, exist_ok=True)
    snapshot_sources(args.out)

    manifest_dir = os.path.join(args.data_dir, args.data_cfg, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)

    # ---- HuggingFace datasets ----
    cache_dir = os.path.join(args.data_dir, args.data_cfg, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    hf_config.HF_DATASETS_CACHE = cache_dir
    dl_cfg = DownloadConfig(
        cache_dir=cache_dir,
        force_download=False,
        resume_download=True,
        max_retries=8,
        disable_tqdm=False,
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=72000)}},
        delete_extracted=False,
        extract_compressed_file=True,
        force_extract=True,
    )

    train_ds = load_dataset(args.data_script, args.data_cfg, split=args.train_split,
                            trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir)
    val_ds   = load_dataset(args.data_script, args.data_cfg, split=args.val_split,
                            trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir)
    test_ds  = load_dataset(args.data_script, args.data_cfg, split=args.test_split,
                            trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir)
    print(f"[INFO] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    extra_splits = {}
    for split_name in ["dev.other", "test.other"]:
        try:
            ds = load_dataset(args.data_script, args.data_cfg, split=split_name,
                              trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir)
            extra_splits[split_name] = ds
            print(f"[INFO] extra split: {split_name}, n={len(ds)}")
        except Exception as e:
            print(f"[WARN] could not load split '{split_name}': {e}")

    # ---- Speaker scan ----
    print("[INFO] scanning speakers from train split...")
    spk2idx, idx2spk = scan_speakers(train_ds)
    num_spk = len(spk2idx)
    print(f"[SCAN] num_speakers={num_spk}")
    spk_map_path = os.path.join(manifest_dir, "speaker_id_mapping.json")
    if not os.path.isfile(spk_map_path):
        save_speaker_mapping(spk2idx, idx2spk, spk_map_path)

    # ---- Manifest 생성 ----
    phys_cache_root = Path(manifest_dir) / "phys_cache"
    phys_cache_root.mkdir(parents=True, exist_ok=True)

    train_manifest      = os.path.join(manifest_dir, "train.json")
    dev_clean_manifest  = os.path.join(manifest_dir, "dev_clean.json")
    dev_other_manifest  = os.path.join(manifest_dir, "dev_other.json")
    test_clean_manifest = os.path.join(manifest_dir, "test_clean.json")
    test_other_manifest = os.path.join(manifest_dir, "test_other.json")

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

    spk_dur_out = os.path.join(manifest_dir, "speaker_durations_train.json")
    compute_speaker_durations(train_manifest, spk_dur_out)

    # ---- test_mode: 데이터 축소 ----
    if args.test_mode:
        train_ds = train_ds.select(range(200))
        spk2idx, idx2spk = scan_speakers(train_ds)
        num_spk = len(spk2idx)
        val_ds   = val_ds.select(range(200))
        test_ds  = test_ds.select(range(200))
        for k in list(extra_splits.keys()):
            extra_splits[k] = extra_splits[k].select(range(min(200, len(extra_splits[k]))))

        tm_train = os.path.join(manifest_dir, "test_mode_train.json")
        tm_val   = os.path.join(manifest_dir, "test_mode_val.json")
        tm_test  = os.path.join(manifest_dir, "test_mode_test.json")
        build_manifest_from_hf_with_meta(train_ds, tm_train, cache_dir, spk2idx, "test_mode_train", phys_cache_root)
        build_manifest_from_hf_with_meta(val_ds,   tm_val,   cache_dir, spk2idx, "test_mode_val",   phys_cache_root)
        build_manifest_from_hf_with_meta(test_ds,  tm_test,  cache_dir, spk2idx, "test_mode_test",  phys_cache_root)
        train_manifest     = tm_train
        dev_clean_manifest = tm_val
        test_clean_manifest = tm_test

    # ---- Phys cache 생성 ----
    HOP_MS = 10.0
    WIN_MS = 25.0
    SR = args.sample_rate
    print(f"[INFO] building phys_cache in {phys_cache_root}")
    if args.test_mode:
        build_phys_cache_for_manifest(train_manifest, "test_mode_train", phys_cache_root=phys_cache_root, HOP_MS=HOP_MS, WIN_MS=WIN_MS, SR=SR)
        build_phys_cache_for_manifest(dev_clean_manifest, "test_mode_val", phys_cache_root=phys_cache_root, HOP_MS=HOP_MS, WIN_MS=WIN_MS, SR=SR)
        build_phys_cache_for_manifest(test_clean_manifest, "test_mode_test", phys_cache_root=phys_cache_root, HOP_MS=HOP_MS, WIN_MS=WIN_MS, SR=SR)
    else:
        build_phys_cache_for_manifest(train_manifest, "train", phys_cache_root=phys_cache_root, HOP_MS=HOP_MS, WIN_MS=WIN_MS, SR=SR)
        build_phys_cache_for_manifest(dev_clean_manifest, "dev_clean", phys_cache_root=phys_cache_root, HOP_MS=HOP_MS, WIN_MS=WIN_MS, SR=SR)
        build_phys_cache_for_manifest(test_clean_manifest, "test_clean", phys_cache_root=phys_cache_root, HOP_MS=HOP_MS, WIN_MS=WIN_MS, SR=SR)

    # ---- Processor (tokenizer + feature extractor) ----
    processor_name = args.processor_name if args.processor_name else args.student_name
    processor = Wav2Vec2Processor.from_pretrained(processor_name)
    blank_id = processor.tokenizer.pad_token_id  # wav2vec2에서 PAD = CTC blank (=0)
    print(f"[INFO] processor loaded from {processor_name}, blank_id={blank_id}")

    # ---- Lightning Trainer ----
    wandb = WandbLogger(project=args.wandb_project, name=args.wandb_run, save_dir=args.out)
    ckpt_dir = os.path.join(args.out, "checkpoints")
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="last",
        save_top_k=0,
        save_last=True,
    )

    trainer = pl.Trainer(
        devices=args.gpus,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True) if args.gpus > 1 else "auto",
        max_epochs=args.epochs,
        default_root_dir=args.out,
        logger=wandb,
        callbacks=[ckpt_cb],
        gradient_clip_val=1.0,
    )

    # ---- 모델 ----
    model = DistilDAGKDWav2Vec2(
        teacher_name=args.teacher_name,
        student_name=args.student_name,
        num_spk=num_spk,
        phys_cache_root=str(phys_cache_root),
        out_dir=args.out,
        train_manifest=train_manifest,
        blank_id=blank_id,
        # KD
        use_ctc=args.use_ctc,
        use_logit_kd=args.use_logit_kd,
        kd_alpha=args.kd_alpha,
        kd_temperature=args.kd_temperature,
        use_layer_kd=args.use_layer_kd,
        layer_kd_alpha=args.layer_kd_alpha,
        # Generative KD
        use_flow=args.use_flow,
        flow_steps=args.flow_steps,
        flow_weight=args.flow_weight,
        use_diffkd=args.use_diffkd,
        diffkd_steps=args.diffkd_steps,
        # Disentanglement
        use_disent=args.use_disent,
        tch_spk_layers=args.tch_spk_layers,
        tch_txt_layers=args.tch_txt_layers,
        stu_spk_layers=args.stu_spk_layers,
        stu_txt_layers=args.stu_txt_layers,
        disen_mi_pairs=args.disen_mi_pairs,
        disen_lll_weight=args.disen_lll_weight,
        disen_mi_weight=args.disen_mi_weight,
        # Probe
        use_txt_spk_probe=args.use_txt_spk_probe,
        txt_probe_lambda=args.txt_probe_lambda,
        # MI
        neg_K=args.neg_K,
        mi_warmup_steps=args.mi_warmup_steps,
        mi_ramp_steps=args.mi_ramp_steps,
        mi_lambda_max=args.mi_lambda_max,
        lll_lambda_max=args.lll_lambda_max,
        mi_clamp_min0=args.mi_clamp_min0,
        # S-DisKD
        use_stu_txt_kd=args.use_stu_txt_kd,
        use_stu_spk_kd=args.use_stu_spk_kd,
        use_stu_club=args.use_stu_club,
        stu_txt_kd_weight=args.stu_txt_kd_weight,
        stu_spk_kd_weight=args.stu_spk_kd_weight,
        stu_club_weight=args.stu_club_weight,
        # Vis
        disen_vis_enable=args.disen_vis_enable,
        # Optimizer
        learning_rate=args.learning_rate,
        warmup_epochs=args.warmup_epochs,
        kd_warmup_epochs=args.kd_warmup_epochs,
        freeze_feature_extractor=args.freeze_feature_extractor,
        random_init_student=args.random_init_student,
        # Audio
        sample_rate=args.sample_rate,
    )

    # processor를 모델에 주입 (validation_step의 WER 디코딩에서 사용)
    model.processor = processor

    # ---- DataModule ----
    dm = Wav2VecDataModule(
        train_manifest=train_manifest,
        val_manifest=dev_clean_manifest,
        processor=processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=args.sample_rate,
    )

    # ---- 학습 ----
    ckpt_path = args.resume_ckpt_path
    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"[INFO] Resuming from: {ckpt_path}")
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    else:
        if ckpt_path:
            print(f"[WARN] checkpoint not found at {ckpt_path}, training from scratch.")
        else:
            print("[INFO] training from scratch.")
        trainer.fit(model, datamodule=dm)

    # ---- 평가 ----
    plot_dir = os.path.join(args.out, "xai/wer_plots")
    eval_targets = [
        ("dev_clean",  dev_clean_manifest),
        ("dev_other",  dev_other_manifest),
        ("test_clean", test_clean_manifest),
        ("test_other", test_other_manifest),
    ]
    for split_name, manifest in eval_targets:
        evaluate_split(
            model=model,
            manifest_path=manifest,
            processor=processor,
            batch_size=args.batch_size,
            split_name=split_name,
            wandb_logger=wandb,
            trainer=trainer,
            plot_dir=plot_dir,
        )


if __name__ == "__main__":
    main()
