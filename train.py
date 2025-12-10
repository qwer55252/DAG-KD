#!/usr/bin/env python3
"""
목표
- Teacher/Student는 동일한 layer 수를 갖되 Student의 head 수와 hidden dim이 절반.
- ASR에서 상/하위층 표현의 성질(하위: acoustic/speaker/prosody, 상위: linguistic)을 고려하여
  1) Generative KD (Flow Matching 또는 DiffKD)로 layer feature를 정렬
  2) Logit KD (CTC 로짓 KL)
  3) Layerwise Metric KD (MSE)
  4) Disentanglement (언어/화자 적대 분류기 + GRL)로 content 보존 유도
을 결합.

데이터 가정
- NeMo manifest(JSONL) 항목에 최소 {audio_filepath, duration, text}가 있고,
  가능하면 {lang, speaker} 필드를 추가하면 Disentanglement loss가 활성화됩니다.
- lang/speaker가 없으면 해당 loss는 0으로 처리됩니다.

주의
- 본 코드는 연구용 레퍼런스 구현이며, 실제 실험에서는 학습 안정화를 위해 loss 가중치, 샘플링 스케줄,
  라우터/GRL 스케줄, augmentation 등을 조정하세요.
"""

import os
import json
import torch
import aiohttp
import argparse
import statistics
import numpy as np
import lightning as pl
import soundfile as sf
from copy import deepcopy
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
# import Nemo.nemo.collections.asr as nemo_asr
from nemo.collections import asr as nemo_asr
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from datasets import load_dataset, DownloadConfig, config as hf_config

from models import DistilDAGKDCTCModelBPE
from utils import (
    scan_speakers, 
    build_manifest_from_hf_with_meta,
    str2bool,
    release_nemoAPI,
    compute_sample_wers,
    compute_speaker_durations,
    int_list_arg,
    save_speaker_mapping,
    save_mel_examples_from_manifest
)

def main():
    p = argparse.ArgumentParser("DAG-KD train script")
    # Data
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--data_script", type=str, default="./librispeech_asr.py")
    p.add_argument("--data_cfg", type=str, default="train_100")
    p.add_argument("--train_split", type=str, default="train.clean.100")
    p.add_argument("--val_split", type=str, default="dev.clean")
    p.add_argument("--test_split", type=str, default="test.clean")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--test_mode", type=str2bool, default=False)
    p.add_argument("--lang_key", type=str, default="language")
    p.add_argument("--spk_key",  type=str, default="speaker")

    # Logging/ckpt
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--out", type=str, default="outputs")
    p.add_argument("--resume_ckpt_path", type=str, default="", help="Checkpoint path to resume training from")

    # Teacher/Student
    p.add_argument("--teacher_name", type=str, default="stt_en_conformer_ctc_small")
    p.add_argument("--train_teacher", type=str2bool, default=False)

    # KD config
    p.add_argument("--use_ctc", type=str2bool, default=True)
    p.add_argument("--use_logit_kd", type=str2bool, default=True)
    p.add_argument("--kd_alpha", type=float, default=0.5)
    p.add_argument("--kd_temperature", type=float, default=1.0)
    p.add_argument("--use_layer_kd", type=str2bool, default=False)
    p.add_argument("--layer_kd_alpha", type=float, default=0.5)

    # Generative KD
    p.add_argument("--use_flow", type=str2bool, default=False)
    p.add_argument("--flow_steps", type=int, default=8)
    p.add_argument("--flow_weight", type=float, default=1.0)
    p.add_argument("--use_diffkd", type=str2bool, default=False)
    p.add_argument("--diffkd_steps", type=int, default=5)

    # Disentanglement
    p.add_argument("--use_disent", type=str2bool, default=True)
    p.add_argument("--disent_spk_layers", type=int_list_arg, default=[1, 2])
    p.add_argument("--disent_txt_layers", type=int_list_arg, default=[15, 16])

    # W&B
    p.add_argument("--wandb_project", type=str, default=os.getenv("PRJ_NAME", "DAG-KD"))
    p.add_argument("--wandb_run", type=str, default=os.getenv("EXP_NAME", "dagkd_run"))
    p.add_argument("--disen_vis_enable", type=str2bool, default=False)

    args = p.parse_args()

    # Output & manifests
    os.makedirs(args.out, exist_ok=True)
    manifest_dir = os.path.join(args.data_dir, args.data_cfg, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)

    # HF datasets
    cache_dir = os.path.join(args.data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    hf_config.HF_DATASETS_CACHE = cache_dir
    dl_cfg = DownloadConfig(
        cache_dir=cache_dir,
        force_download=False,
        resume_download=True,
        max_retries=8,
        disable_tqdm=False,
        download_desc="Downloading dataset",
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=72000)}},
        delete_extracted=False,
        extract_compressed_file=True,
        force_extract=True,
    )

    train_ds = load_dataset(args.data_script, args.data_cfg, split=args.train_split, trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir)
    val_ds   = load_dataset(args.data_script, args.data_cfg, split=args.val_split,   trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir)
    test_ds  = load_dataset(args.data_script, args.data_cfg, split=args.test_split,  trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir)
    print(f"[INFO] Loaded train split: {args.train_split}, n={len(train_ds)}")
    print(f"[INFO] Loaded val split: {args.val_split}, n={len(val_ds)}")
    print(f"[INFO] Loaded test split: {args.test_split}, n={len(test_ds)}")
    
    # 추가 평가용 split (dev.other, test.other)
    extra_splits = {}
    for split_name in ["dev.other", "test.other"]:
        try:
            ds = load_dataset(
                args.data_script, args.data_cfg, split=split_name,
                trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir
            )
            extra_splits[split_name] = ds
            print(f"[INFO] Loaded extra split: {split_name}, n={len(ds)}")
        except Exception as e:
            print(f"[WARN] Could not load split '{split_name}': {e}")
    
    # 스캔
    print(f"[INFO] scanning speakers from train split...")
    spk2idx, idx2spk = scan_speakers(train_ds)
    num_spk = len(spk2idx)
    print(f"[SCAN] num_speakers={num_spk}")
    # id -> idx 매핑 파일 저장
    spk_map_path = os.path.join(manifest_dir, "speaker_id_mapping.json")
    if not os.path.isfile(spk_map_path):
        save_speaker_mapping(spk2idx, idx2spk, spk_map_path)
        print(f"[INFO] saved speaker ID mapping to {spk_map_path}")
    
    
    train_manifest      = os.path.join(manifest_dir, "train.json")
    dev_clean_manifest  = os.path.join(manifest_dir, "dev_clean.json")
    dev_other_manifest  = os.path.join(manifest_dir, "dev_other.json")
    test_clean_manifest = os.path.join(manifest_dir, "test_clean.json")
    test_other_manifest = os.path.join(manifest_dir, "test_other.json")

    # manifest들 생성
    if not os.path.isfile(train_manifest):
        build_manifest_from_hf_with_meta(train_ds, train_manifest, cache_dir, spk2idx)
    if not os.path.isfile(dev_clean_manifest):
        build_manifest_from_hf_with_meta(val_ds, dev_clean_manifest, cache_dir, spk2idx)
    if not os.path.isfile(test_clean_manifest):
        build_manifest_from_hf_with_meta(test_ds, test_clean_manifest, cache_dir, spk2idx)
    if "dev.other" in extra_splits and not os.path.isfile(dev_other_manifest):
        build_manifest_from_hf_with_meta(extra_splits["dev.other"], dev_other_manifest, cache_dir, spk2idx)
    if "test.other" in extra_splits and not os.path.isfile(test_other_manifest):
        build_manifest_from_hf_with_meta(extra_splits["test.other"], test_other_manifest, cache_dir, spk2idx)

    # speaker별 발화 시간 총합 계산 (train split 기준)
    print(f"[INFO] calculating speaker durations from train manifest...")
    train_manifest_for_stat = train_manifest
    spk_dur_out = os.path.join(manifest_dir, "speaker_durations_train.json")
    compute_speaker_durations(train_manifest_for_stat, spk_dur_out)
    print(f"[INFO] saved speaker durations to {spk_dur_out}")
    
    
    if args.test_mode:
        train_ds = train_ds.select(range(200))
        val_ds   = val_ds.select(range(200))
        test_ds  = test_ds.select(range(200))
        for k in list(extra_splits.keys()):
            extra_splits[k] = extra_splits[k].select(range(min(200, len(extra_splits[k]))))
        
        test_mode_train_manifest = os.path.join(manifest_dir, "test_mode_train.json")
        test_mode_val_manifest = os.path.join(manifest_dir, "test_mode_val.json")
        test_mode_test_manifest = os.path.join(manifest_dir, "test_mode_test.json")
        build_manifest_from_hf_with_meta(train_ds, test_mode_train_manifest, cache_dir, spk2idx)
        build_manifest_from_hf_with_meta(val_ds, test_mode_val_manifest, cache_dir, spk2idx)
        build_manifest_from_hf_with_meta(test_ds, test_mode_test_manifest, cache_dir, spk2idx)
    
    
    # W&B
    wandb = WandbLogger(project=args.wandb_project, name=args.wandb_run, save_dir=args.out)
    ckpt_dir = os.path.join(args.out, "checkpoints")
    ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir, filename="last", save_top_k=0, save_last=True)

    trainer = pl.Trainer(
        devices=args.gpus,
        accelerator="gpu",
        max_epochs=args.epochs,
        default_root_dir=args.out,
        logger=wandb,
        callbacks=[ckpt_cb],
    )

    # Teacher
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name=args.teacher_name,
        map_location="cuda:0" if torch.cuda.is_available() else "cpu",
        trainer=trainer,
    )
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad = False

    # .nemo unpack (for safety with ASR DL construction)
    release_nemoAPI(teacher, out_folder=os.path.join(args.out, "nemo_archive"))

    # Student cfg: teacher cfg 복제 후 hidden/head 절반
    stu_cfg = deepcopy(teacher.cfg)
    stu_cfg.train_ds.is_tarred = False
    stu_cfg.train_ds.manifest_filepath = train_manifest if not args.test_mode else test_mode_train_manifest
    stu_cfg.train_ds.sample_rate = args.sample_rate
    stu_cfg.train_ds.batch_size = args.batch_size

    stu_cfg.validation_ds.is_tarred = False
    stu_cfg.validation_ds.manifest_filepath = dev_clean_manifest  if not args.test_mode else test_mode_val_manifest
    stu_cfg.validation_ds.sample_rate = args.sample_rate
    stu_cfg.validation_ds.batch_size = args.batch_size

    stu_cfg.test_ds.is_tarred = False
    stu_cfg.test_ds.manifest_filepath = test_clean_manifest  if not args.test_mode else test_mode_test_manifest
    stu_cfg.test_ds.sample_rate = args.sample_rate
    stu_cfg.test_ds.batch_size = args.batch_size

    # 절반 스케일
    stu_cfg.encoder.d_model = max(8, teacher.cfg.encoder.d_model // 2)
    stu_cfg.encoder.n_heads = max(1, teacher.cfg.encoder.n_heads // 2)
    stu_cfg.decoder.feat_in = max(8, teacher.cfg.decoder.feat_in // 2)
    
    # NeMo dataconfig에 sample_id 반환 옵션 켜기
    OmegaConf.set_struct(stu_cfg, False)
    OmegaConf.set_struct(stu_cfg.train_ds, False)
    OmegaConf.set_struct(stu_cfg.validation_ds, False)
    OmegaConf.set_struct(stu_cfg.test_ds, False)
    stu_cfg.train_ds.return_sample_id = True
    stu_cfg.validation_ds.return_sample_id = False
    stu_cfg.test_ds.return_sample_id = False
    
    stu_cfg.latent_dim = 96
    # stu_cfg에 다음 키를 추가하면 튜닝 용이
    stu_cfg.disen_mi_weight = 1e-3              # λ_MI
    stu_cfg.rec_txt_lambda = 0.1
    stu_cfg.rec_spk_lambda = 0.1              
    stu_cfg.disen_mi_pairs  = "ts,tp,ps"        # 사용 쌍
    stu_cfg.disen_gst_tokens = 10
    stu_cfg.disen_gst_heads  = 4
    stu_cfg.disen_gst_token_dim = 96
    stu_cfg.disen_gst_ref_dim = 96
    stu_cfg.disen_spk_ce_lambda = 1.0
    stu_cfg.disen_global_style = True
    
    # cfg 주입
    stu_cfg.num_spk = num_spk
    stu_cfg.num_lang = 0  # 모노링구얼: 언어 헤드 비활성
    stu_cfg.out_dir = args.out
    stu_cfg.disen_vis_enable = args.disen_vis_enable

    model = DistilDAGKDCTCModelBPE(
        cfg=stu_cfg,
        trainer=trainer,
        teacher_model=teacher,
        use_ctc=args.use_ctc,
        use_logit_kd=args.use_logit_kd,
        kd_alpha=args.kd_alpha,
        kd_temperature=args.kd_temperature,
        use_layer_kd=args.use_layer_kd,
        layer_kd_alpha=args.layer_kd_alpha,
        use_flow=args.use_flow,
        flow_steps=args.flow_steps,
        flow_weight=args.flow_weight,
        use_diffkd=args.use_diffkd,
        diffkd_steps=args.diffkd_steps,
        use_disent=args.use_disent,
        disent_spk_layers=args.disent_spk_layers,
        disent_txt_layers=args.disent_txt_layers,
    )
    
    # ====== 멜 스펙트로그램 예시 저장 ======
    mel_dir = os.path.join(args.out, "xai/mel_examples")
    save_mel_examples_from_manifest(
        manifest_path=train_manifest,
        model=model,
        out_dir=mel_dir,
        num_examples=4,
        split_name="train",
    )

    # 사용자 인자 처리
    ckpt_path = args.resume_ckpt_path
    # 편의상 --resume_from last 라고 주면 자동으로 last.ckpt 사용

    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"[INFO] Resuming training from checkpoint: {ckpt_path}")
        trainer.fit(model, ckpt_path=ckpt_path)
    else:
        if ckpt_path:
            print(f"[WARN] Checkpoint not found at {ckpt_path}, start training from scratch.")
        else:
            print("[INFO] No resume_from specified, start training from scratch.")
        trainer.fit(model)

    # 평가
    eval_targets = [
        ("dev_clean",  dev_clean_manifest),
        ("dev_other",  dev_other_manifest),
        ("test_clean", test_clean_manifest),
        ("test_other", test_other_manifest),
    ]
    for split_name, manifest in eval_targets:
        if not os.path.isfile(manifest):
            print(f"[WARN] manifest not found for {split_name}: {manifest}, skip")
            continue

        print(f"\n===== Evaluating on {split_name} =====")

        # 1) Lightning test (loss / 평균 WER)
        test_cfg = deepcopy(model.cfg.test_ds)
        test_cfg.manifest_filepath = manifest
        test_cfg.shuffle = False
        model.setup_test_data(test_cfg)
        dl = model.test_dataloader()
        results = trainer.test(model=model, dataloaders=[dl], verbose=True)

        if results and isinstance(results, list):
            res = results[0]
            wer = res.get("test_wer", res.get("wer", None))
            loss = res.get("test_loss", res.get("loss", None))
            print(f"→ {split_name}: loss={loss:.4f} | wer={wer:.2%}")
            key = f"{split_name}/wer".replace('.', '_')
            wandb.log_metrics({key: wer}, step=trainer.current_epoch)

        # 2) per-sample WER mean ± std
        with open(manifest, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f]

        audio_files = [e["audio_filepath"] for e in entries]
        ref_texts   = [e["text"] for e in entries]

        hyps = model.transcribe(
            audio=audio_files,
            batch_size=args.batch_size,
            return_hypotheses=False,
            num_workers=0,
            verbose=False,
        )

        sample_wers = compute_sample_wers(ref_texts, hyps)
        sample_wers_pct = [w * 100.0 for w in sample_wers]

        wer_mean = float(statistics.mean(sample_wers_pct))
        wer_std  = float(statistics.stdev(sample_wers_pct)) if len(sample_wers_pct) > 1 else 0.0

        print(f"→ {split_name}: per-sample WER = {wer_mean:.2f}% ± {wer_std:.2f}%")
                
        # ===================== WER 분포 그림 저장 =====================
        # 결과 그림 저장 디렉토리: <out>/wer_plots
        plot_dir = os.path.join(args.out, "xai/wer_plots")
        os.makedirs(plot_dir, exist_ok=True)

        wers_np = np.array(sample_wers_pct, dtype=float)

        # (1) 히스토그램
        plt.figure()
        bins = [0, 10, 20, 30, 50, 100, 200]
        plt.hist(wers_np, bins=bins, edgecolor="black")
        plt.xlabel("Per-sample WER (%)")
        plt.ylabel("Count")
        plt.title(
            f"WER Histogram - {split_name}\n"
            f"mean={wer_mean:.2f}%, std={wer_std:.2f}%"
        )
        plt.tight_layout()
        hist_path = os.path.join(plot_dir, f"wer_hist_{split_name}.png")
        plt.savefig(hist_path)
        plt.close()

        # (2) Boxplot (선택사항, 분포 한 번에 보기 좋음)
        plt.figure()
        plt.boxplot(wers_np, vert=True, showfliers=True)
        plt.ylabel("Per-sample WER (%)")
        plt.title(f"WER Boxplot - {split_name}")
        plt.tight_layout()
        box_path = os.path.join(plot_dir, f"wer_box_{split_name}.png")
        plt.savefig(box_path)
        plt.close()
        # ============================================================

        wandb.log_metrics(
            {
                f"{split_name}/wer_mean": wer_mean,
                f"{split_name}/wer_std": wer_std,
            },
            step=trainer.current_epoch,
        )

if __name__ == "__main__":
    main()
