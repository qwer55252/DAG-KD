#!/usr/bin/env python3
import os
import json
import torch
import aiohttp
import argparse
import statistics
import numpy as np
import lightning as pl
import matplotlib.pyplot as plt
from copy import deepcopy
from omegaconf import OmegaConf
from nemo.collections import asr as nemo_asr
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from datasets import load_dataset, DownloadConfig

from models_ver2 import TeacherASRWithDisent, StudentASRWithDisentKD
from utils import (
    scan_speakers, build_manifest_from_hf_with_meta, str2bool,
    release_nemoAPI, compute_sample_wers, compute_speaker_durations,
    save_speaker_mapping, save_mel_examples_from_manifest,
    materialize_nemo_artifacts_in_cfg, rotate_last_ckpts, int_list_arg
)

def prepare_datasets(args):
    """Stage 0: 데이터셋 로드 및 Manifest 생성"""
    print("\n===== Stage 0: Load Dataset =====")
    os.makedirs(args.out, exist_ok=True)
    manifest_dir = os.path.join(args.data_dir, args.data_cfg, "manifests")
    cache_dir = os.path.join(args.data_dir, args.data_cfg, "cache")
    os.makedirs(manifest_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    dl_cfg = DownloadConfig(cache_dir=cache_dir, resume_download=True, extract_compressed_file=True)
    
    # Dataset 로드
    splits = {
        "train": args.train_split, "val": args.val_split, "test": args.test_split,
        "dev_other": "dev.other", "test_other": "test.other"
    }
    datasets = {}
    for name, split in splits.items():
        try:
            datasets[name] = load_dataset(args.data_script, args.data_cfg, split=split, download_config=dl_cfg, cache_dir=cache_dir)
            if args.test_mode: datasets[name] = datasets[name].select(range(min(200, len(datasets[name]))))
        except Exception as e:
            print(f"[WARN] Could not load split '{split}': {e}")

    # Speaker 스캔 & 맵핑
    spk2idx, _ = scan_speakers(datasets["train"])
    save_speaker_mapping(spk2idx, {v: k for k, v in spk2idx.items()}, os.path.join(manifest_dir, "speaker_id_mapping.json"))

    # Manifest 경로 지정
    manifests = {k: os.path.join(manifest_dir, f"{k}{'_test' if args.test_mode else ''}.json") for k in datasets.keys()}
    for name, ds in datasets.items():
        if not os.path.isfile(manifests[name]):
            build_manifest_from_hf_with_meta(ds, manifests[name], cache_dir, spk2idx)

    return manifests, len(spk2idx)

def evaluate_model(model, trainer, manifests, args, stage_name="Student"):
    """모델 평가 및 시각화 공통 함수 (Teacher/Student 모두 재사용)"""
    print(f"\n===== Evaluating {stage_name} =====")
    model.eval()
    plot_dir = os.path.join(args.out, f"xai/wer_plots_{stage_name.lower()}")
    os.makedirs(plot_dir, exist_ok=True)

    eval_targets = ["val", "dev_other", "test", "test_other"]
    
    for split_key in eval_targets:
        if split_key not in manifests or not os.path.isfile(manifests[split_key]):
            continue
            
        manifest = manifests[split_key]
        print(f"--- On {split_key} ---")
        
        # 1) Lightning Test
        test_cfg = deepcopy(model.cfg.test_ds)
        test_cfg.manifest_filepath = manifest
        test_cfg.shuffle = False
        model.setup_test_data(test_cfg)
        
        results = trainer.test(model=model, dataloaders=[model.test_dataloader()], verbose=True)
        if results:
            trainer.logger.log_metrics({f"{stage_name.lower()}/{split_key}/wer": results[0].get("test_wer", 0)}, step=trainer.current_epoch)

        # 2) Per-sample WER 계산 및 Plot
        with open(manifest, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f]

        hyps = model.transcribe(audio=[e["audio_filepath"] for e in entries], batch_size=args.batch_size, verbose=False)
        sample_wers = [w * 100.0 for w in compute_sample_wers([e["text"] for e in entries], hyps)]
        
        if sample_wers:
            wer_mean = float(statistics.mean(sample_wers))
            wer_std = float(statistics.stdev(sample_wers)) if len(sample_wers) > 1 else 0.0
            
            plt.figure()
            plt.hist(np.array(sample_wers, dtype=float), bins=[0, 10, 20, 30, 50, 100, 200], edgecolor="black")
            plt.title(f"{stage_name} WER - {split_key}\nMean: {wer_mean:.2f}%, Std: {wer_std:.2f}%")
            plt.savefig(os.path.join(plot_dir, f"wer_hist_{split_key}.png"))
            plt.close()

def main():
    p = argparse.ArgumentParser("DAG-KD train script")
    # Data & Logging
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--data_script", type=str, default="./librispeech_asr.py")
    p.add_argument("--data_cfg", type=str, default="train_100")
    p.add_argument("--train_split", type=str, default="train.clean.100")
    p.add_argument("--val_split", type=str, default="dev.clean")
    p.add_argument("--test_split", type=str, default="test.clean")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--test_mode", type=str2bool, default=False)
    p.add_argument("--out", type=str, default="outputs")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--wandb_project", type=str, default="DAG-KD")
    
    # Model Configs
    p.add_argument("--teacher_name", type=str, default="stt_en_conformer_ctc_small")
    p.add_argument("--train_teacher", type=str2bool, default=False)
    p.add_argument("--teacher_ckpt_path", type=str, default="")
    p.add_argument("--disen_mi_weight", type=float, default=1e-3)
    p.add_argument("--disen_lll_weight", type=float, default=1.0)
    
    # KD Configs
    p.add_argument("--use_logit_kd", type=str2bool, default=True)
    p.add_argument("--kd_alpha", type=float, default=0.5)
    p.add_argument("--kd_temperature", type=float, default=1.0)
    p.add_argument("--gen_kd_type", type=str, default="flow", choices=["mse", "flow", "diff"])
    p.add_argument("--gen_kd_weight", type=float, default=1.0)
    
    args = p.parse_args()
    wandb = WandbLogger(project=args.wandb_project, save_dir=args.out)

    # 1. Dataset 로드
    manifests, num_spk = prepare_datasets(args)

    # 2. Teacher 준비 및 학습
    print("\n===== Stage 1: Teacher Setup =====")
    nemo_teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.teacher_name)
    teacher_cfg = deepcopy(nemo_teacher.cfg)
    OmegaConf.set_struct(teacher_cfg, False)
    
    for split, path in zip(["train", "validation", "test"], [manifests["train"], manifests["val"], manifests["test"]]):
        getattr(teacher_cfg, f"{split}_ds").manifest_filepath = path
        getattr(teacher_cfg, f"{split}_ds").batch_size = args.batch_size
        getattr(teacher_cfg, f"{split}_ds").is_tarred = False
    
    teacher_cfg.num_spk = num_spk
    teacher = TeacherASRWithDisent(cfg=teacher_cfg, disen_mi_weight=args.disen_mi_weight)
    teacher.load_from_pretrained_nemo(nemo_teacher, strict=False)

    if args.train_teacher:
        trainer = pl.Trainer(devices=args.gpus, accelerator="gpu", max_epochs=args.epochs, logger=wandb)
        trainer.fit(teacher)
        teacher_ckpt = os.path.join(args.out, "checkpoints", "last.ckpt")
    else:
        teacher_ckpt = args.teacher_ckpt_path

    # 선택 사항: Teacher 평가
    # evaluate_model(teacher, pl.Trainer(devices=args.gpus, accelerator="gpu", logger=wandb), manifests, args, "Teacher")

    # 3. Student 준비 및 학습
    print("\n===== Stage 3: Student Training =====")
    stu_cfg = deepcopy(teacher_cfg)
    stu_cfg.encoder.d_model = max(8, teacher_cfg.encoder.d_model // 2)
    stu_cfg.encoder.n_heads = max(1, teacher_cfg.encoder.n_heads // 2)
    stu_cfg.train_ds.return_sample_id = True
    
    teacher_loaded = TeacherASRWithDisent.load_from_checkpoint(teacher_ckpt, cfg=teacher_cfg)
    teacher_loaded.eval()

    student_model = StudentASRWithDisentKD(
        cfg=stu_cfg, teacher=teacher_loaded, 
        use_logit_kd=args.use_logit_kd, kd_alpha=args.kd_alpha, kd_temperature=args.kd_temperature,
        gen_kd_type=args.gen_kd_type, gen_kd_weight=args.gen_kd_weight
    )

    student_trainer = pl.Trainer(devices=args.gpus, accelerator="gpu", max_epochs=args.epochs, logger=wandb)
    student_trainer.fit(student_model)

    # 4. Student 평가
    evaluate_model(student_model, student_trainer, manifests, args, "Student")

if __name__ == "__main__":
    main()