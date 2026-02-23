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
from nemo.utils.app_state import AppState
from datasets import load_dataset, DownloadConfig, config as hf_config

from models_ver2 import (
    TeacherASRWithDisent,
    StudentASRWithDisentKD
)
from utils import (
    scan_speakers, 
    build_manifest_from_hf_with_meta,
    str2bool,
    release_nemoAPI,
    compute_sample_wers,
    compute_speaker_durations,
    int_list_arg,
    save_speaker_mapping,
    save_mel_examples_from_manifest,
    materialize_nemo_artifacts_in_cfg,
    load_speaker_table_from_manifest,
    rotate_last_ckpts,
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
    p.add_argument("--disen_mi_pairs", type=str, default="ts,tp,ps")
    p.add_argument("--disen_lll_weight", type=float, default=1.0)
    p.add_argument("--disen_mi_weight", type=float, default=1e-3)

    # W&B
    p.add_argument("--wandb_project", type=str, default=os.getenv("PRJ_NAME", "DAG-KD"))
    p.add_argument("--wandb_run", type=str, default=os.getenv("EXP_NAME", "dagkd_run"))
    p.add_argument("--disen_vis_enable", type=str2bool, default=False)
    
    # text speaker probe
    p.add_argument("--use_txt_spk_probe", type=str2bool, default=True)
    
    p.add_argument("--use_stu_spk_adv", type=str2bool, default=False)
    p.add_argument("--stu_spk_adv_lambda_max", type=float, default=0.1)
    p.add_argument("--stu_spk_adv_warmup_steps", type=int, default=2000)
    
    p.add_argument("--use_layerwise_disent", type=str2bool, default=False)
    p.add_argument("--use_layerwise_flow", type=str2bool, default=False)
    p.add_argument("--use_layerwise_diffkd", type=str2bool, default=False)
    p.add_argument("--layer_list_for_disent", type=int_list_arg, default=[4,8,12,16])
    p.add_argument("--neg_K", type=int, default=8)
    
    p.add_argument("--mi_warmup_steps", type=int, default=5000)      # CLUB만 학습하는 구간
    p.add_argument("--mi_ramp_steps", type=int, default=20000)       # penalty를 올리는 구간
    p.add_argument("--mi_lambda_max", type=float, default=0.01)      # 최종 MI 가중치
    p.add_argument("--lll_lambda_max", type=float, default=0.01)     # 최종 LLL 가중치 (원하면 분리)
    p.add_argument("--mi_clamp_min0", type=str2bool, default=True)   # mi_upper <0 클램프

    p.add_argument("--teacher_ckpt_path", type=str, default="", help="학습된 teacher ckpt 경로(버전2)")
    p.add_argument("--freeze_teacher_encdec", type=str2bool, default=False)
    p.add_argument("--freeze_student_encdec", type=str2bool, default=False)

    p.add_argument("--gen_kd_type", type=str, default="flow", choices=["mse", "flow", "diff"]) ### flow, diff 같이 사용할 수도 있어야 함
    p.add_argument("--gen_kd_weight", type=float, default=1.0)
    
    args = p.parse_args()

    # -------------------- Stage 0: Load Dataset --------------------
    print("\n===== Stage 0: Load Dataset =====")
    # Output & manifests
    os.makedirs(args.out, exist_ok=True)
    manifest_dir = os.path.join(args.data_dir, args.data_cfg, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)

    # HF datasets
    cache_dir = os.path.join(args.data_dir, args.data_cfg, "cache")
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
        spk2idx, idx2spk = scan_speakers(train_ds)
        num_spk = len(spk2idx)
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
    print(f"[SCAN] num_speakers={num_spk}")
    
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

    # -------------------- Stage 1: Teacher training --------------------
    print("\n===== Stage 1: Teacher training =====")
    # 1) pretrained NeMo teacher (weights source)
    # .nemo unpack (for safety with ASR DL construction)
    nemo_teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name=args.teacher_name,
        map_location="cuda:0" if torch.cuda.is_available() else "cpu",
        trainer=trainer,
    )
    release_nemoAPI(nemo_teacher, out_folder=os.path.join(args.out, "nemo_archive"))
    
    # 2) teacher_cfg는 nemo_teacher.cfg 복제 후 dataloader/옵션 주입
    teacher_cfg = deepcopy(nemo_teacher.cfg)
    archive_dir = os.path.abspath(os.path.join(args.out, "nemo_archive"))
    teacher_cfg = materialize_nemo_artifacts_in_cfg(teacher_cfg, archive_dir)

    OmegaConf.set_struct(teacher_cfg, False)
    teacher_cfg.train_ds.is_tarred = False
    teacher_cfg.train_ds.manifest_filepath = train_manifest if not args.test_mode else test_mode_train_manifest
    teacher_cfg.train_ds.sample_rate = args.sample_rate
    teacher_cfg.train_ds.batch_size = args.batch_size
    teacher_cfg.validation_ds.is_tarred = False
    teacher_cfg.validation_ds.manifest_filepath = dev_clean_manifest if not args.test_mode else test_mode_val_manifest
    teacher_cfg.validation_ds.sample_rate = args.sample_rate
    teacher_cfg.validation_ds.batch_size = args.batch_size
    teacher_cfg.test_ds.is_tarred = False
    teacher_cfg.test_ds.manifest_filepath = test_clean_manifest  if not args.test_mode else test_mode_test_manifest
    teacher_cfg.test_ds.sample_rate = args.sample_rate
    teacher_cfg.test_ds.batch_size = args.batch_size
    teacher_cfg.train_ds.return_sample_id = True
    teacher_cfg.validation_ds.return_sample_id = False
    teacher_cfg.test_ds.return_sample_id = False

    # disent cfg 주입
    teacher_cfg.latent_dim = 96
    teacher_cfg.num_spk = num_spk
    teacher_cfg.disen_mi_pairs = args.disen_mi_pairs
    teacher_cfg.use_txt_spk_probe = args.use_txt_spk_probe
    teacher_cfg.txt_probe_lambda = args.txt_probe_lambda
    teacher_cfg.mi_clamp_min0 = args.mi_clamp_min0

    # 3) teacher는 DistilDAGKDCTCModelBPE로 선언
    teacher = TeacherASRWithDisent(
        cfg=teacher_cfg,
        trainer=trainer,
        disen_mi_weight=args.disen_mi_weight,
        disen_lll_weight=args.disen_lll_weight,
        freeze_pretrained_encoder=args.freeze_teacher_encdec,
        freeze_pretrained_decoder=args.freeze_teacher_encdec,
    )
    release_nemoAPI(teacher, out_folder=os.path.join(args.out, "teacher_archive")) ### 이거 그냥 두면 된나?
    
    # 4) pretrained weights 로드 (encoder/decoder/ctc head 등)
    missing, unexpected = teacher.load_from_pretrained_nemo(nemo_teacher, strict=False)
    print("[Teacher init] missing keys:", len(missing), "unexpected keys:", len(unexpected))
    
    teacher_ckpt_dir = os.path.join(args.out, "checkpoints_teacher")
    teacher_ckpt_cb = ModelCheckpoint(
        dirpath=teacher_ckpt_dir,
        filename="teacher-{epoch}-{step}",
        save_last="link",   # last.ckpt를 최신 ckpt로 심볼릭 링크
        save_top_k=-1,
    )

    teacher_trainer = pl.Trainer(
        devices=args.gpus,
        accelerator="gpu",
        max_epochs=args.epochs,
        default_root_dir=args.out,
        logger=wandb,
        callbacks=[teacher_ckpt_cb],
    )    
    
    if args.train_teacher:
        teacher_trainer.fit(teacher)
        rotate_last_ckpts(teacher_ckpt_dir, keep=30)
        teacher_ckpt_path = os.path.join(teacher_ckpt_dir, "last.ckpt")
    else:
        teacher_ckpt_path = args.teacher_ckpt_path
        if not teacher_ckpt_path:
            raise ValueError("train_teacher=False면 --teacher_ckpt_path를 줘야 합니다.")
        
    # -------------------- Stage 2: Teacher Evaluation --------------------
    # if args.train_teacher:
    #     print("\n===== Stage 2: Teacher Evaluation =====")
    #     teacher_eval = teacher.eval()
    #     for p_ in teacher_eval.parameters():
    #         p_.requires_grad = False

    #     # 평가할 split들 (manifest 존재하는 것만)
    #     eval_targets = [
    #         ("dev_clean",  dev_clean_manifest),
    #         ("dev_other",  dev_other_manifest),
    #         ("test_clean", test_clean_manifest),
    #         ("test_other", test_other_manifest),
    #     ]

    #     for split_name, manifest in eval_targets:
    #         if not os.path.isfile(manifest):
    #             print(f"[WARN] manifest not found for {split_name}: {manifest}, skip")
    #             continue

    #         print(f"\n===== [Teacher] Evaluating on {split_name} =====")

    #         # 1) Lightning test (loss / 평균 WER)
    #         test_cfg = deepcopy(teacher_eval.cfg.test_ds)
    #         OmegaConf.set_struct(test_cfg, False)
    #         test_cfg.manifest_filepath = manifest
    #         test_cfg.shuffle = False
    #         test_cfg.batch_size = args.batch_size
    #         # sample_id는 평가에 필요 없으니 꺼두는 게 안전
    #         if hasattr(test_cfg, "return_sample_id"):
    #             test_cfg.return_sample_id = False

    #         teacher_eval.setup_test_data(test_cfg)
    #         dl = teacher_eval.test_dataloader()

    #         results = teacher_trainer.test(model=teacher_eval, dataloaders=[dl], verbose=True)
    #         if results and isinstance(results, list):
    #             res = results[0]
    #             wer = res.get("test_wer", res.get("wer", None))
    #             loss = res.get("test_loss", res.get("loss", None))
    #             if loss is not None and wer is not None:
    #                 print(f"→ [Teacher] {split_name}: loss={loss:.4f} | wer={wer:.2%}")
    #                 key = f"teacher/{split_name}/wer".replace('.', '_')
    #                 wandb.log_metrics({key: float(wer)}, step=teacher_trainer.current_epoch)

    #         # 2) per-sample WER mean ± std + 분포 플롯 저장
    #         with open(manifest, "r", encoding="utf-8") as f:
    #             entries = [json.loads(line) for line in f]

    #         audio_files = [e["audio_filepath"] for e in entries]
    #         ref_texts   = [e["text"] for e in entries]

    #         hyps = teacher_eval.transcribe(
    #             audio=audio_files,
    #             batch_size=args.batch_size,
    #             return_hypotheses=False,
    #             num_workers=0,
    #             verbose=False,
    #         )

    #         sample_wers = compute_sample_wers(ref_texts, hyps)
    #         sample_wers_pct = [w * 100.0 for w in sample_wers]

    #         wer_mean = float(statistics.mean(sample_wers_pct)) if len(sample_wers_pct) > 0 else 0.0
    #         wer_std  = float(statistics.stdev(sample_wers_pct)) if len(sample_wers_pct) > 1 else 0.0
    #         print(f"→ [Teacher] {split_name}: per-sample WER = {wer_mean:.2f}% ± {wer_std:.2f}%")

    #         # plots 저장: <out>/xai/wer_plots_teacher
    #         plot_dir = os.path.join(args.out, "xai/wer_plots_teacher")
    #         os.makedirs(plot_dir, exist_ok=True)

    #         wers_np = np.array(sample_wers_pct, dtype=float)

    #         # (1) 히스토그램
    #         plt.figure()
    #         bins = [0, 10, 20, 30, 50, 100, 200]
    #         plt.hist(wers_np, bins=bins, edgecolor="black")
    #         plt.xlabel("Per-sample WER (%)")
    #         plt.ylabel("Count")
    #         plt.title(f"[Teacher] WER Histogram - {split_name}\nmean={wer_mean:.2f}%, std={wer_std:.2f}%")
    #         plt.tight_layout()
    #         hist_path = os.path.join(plot_dir, f"teacher_wer_hist_{split_name}.png")
    #         plt.savefig(hist_path)
    #         plt.close()

    #         # (2) Boxplot
    #         plt.figure()
    #         plt.boxplot(wers_np, vert=True, showfliers=True)
    #         plt.ylabel("Per-sample WER (%)")
    #         plt.title(f"[Teacher] WER Boxplot - {split_name}")
    #         plt.tight_layout()
    #         box_path = os.path.join(plot_dir, f"teacher_wer_box_{split_name}.png")
    #         plt.savefig(box_path)
    #         plt.close()

    #         # W&B 로깅
    #         wandb.log_metrics(
    #             {
    #                 f"teacher/{split_name}/wer_mean": wer_mean,
    #                 f"teacher/{split_name}/wer_std": wer_std,
    #             },
    #             step=teacher_trainer.current_epoch,
    #         )

    # -------------------- Stage 3: Student training --------------------
    print("\n===== Stage 3: Student training =====")
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
    
    # stu_cfg에 다음 키를 추가하면 튜닝 용이
    stu_cfg.latent_dim = 96
    stu_cfg.disen_mi_weight = args.disen_mi_weight              # λ_MI
    stu_cfg.rec_txt_lambda = 0.1
    stu_cfg.rec_spk_lambda = 0.1              
    stu_cfg.disen_mi_pairs  = args.disen_mi_pairs        # 사용 쌍
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
    
    # text porbe 설정
    stu_cfg.use_txt_spk_probe = args.use_txt_spk_probe if hasattr(args, "use_txt_spk_probe") else True
    stu_cfg.txt_probe_lambda = args.txt_probe_lambda if hasattr(args, "txt_probe_lambda") else 1.0
    
    stu_cfg.use_stu_spk_adv = args.use_stu_spk_adv if hasattr(args, "use_stu_spk_adv") else True
    stu_cfg.stu_spk_adv_lambda_max = args.stu_spk_adv_lambda_max if hasattr(args, "stu_spk_adv_lambda_max") else 0.1
    stu_cfg.stu_spk_adv_warmup_steps = args.stu_spk_adv_warmup_steps if hasattr(args, "stu_spk_adv_warmup_steps") else 2000
    stu_cfg.stu_spk_adv_hidden = 96
    stu_cfg.stu_spk_adv_dropout = 0.2
    stu_cfg.use_layerwise_disent = args.use_layerwise_disent
    stu_cfg.use_layerwise_flow = args.use_layerwise_flow  # flow/disdiffkd도 동일하게 맞춤
    stu_cfg.use_layerwise_diffkd = args.use_layerwise_diffkd
    stu_cfg.layer_list_for_disent = args.layer_list_for_disent  # 1-based index

    stu_cfg.mi_warmup_steps = args.mi_warmup_steps
    stu_cfg.mi_ramp_steps   = args.mi_ramp_steps
    stu_cfg.mi_lambda_max   = args.mi_lambda_max
    stu_cfg.lll_lambda_max  = args.lll_lambda_max
    stu_cfg.mi_clamp_min0   = args.mi_clamp_min0
    
    # teacher 로드 + freeze
    teacher_cfg.num_spk = num_spk
    teacher_loaded = TeacherASRWithDisent.load_from_checkpoint(
        teacher_ckpt_path,
        cfg=teacher_cfg,
        disen_mi_weight=args.disen_mi_weight,
        disen_lll_weight=args.disen_lll_weight,
    )
    teacher_loaded.eval()
    for p_ in teacher_loaded.parameters():
        p_.requires_grad = False
    
    student_model = StudentASRWithDisentKD(
        cfg=stu_cfg,
        trainer=trainer,
        teacher=teacher_loaded,
        use_logit_kd=args.use_logit_kd,
        kd_alpha=args.kd_alpha,
        kd_temperature=args.kd_temperature,
        use_gen_kd=True,
        gen_kd_type=args.gen_kd_type,
        gen_kd_weight=args.gen_kd_weight,
        disen_mi_weight=args.disen_mi_weight,
        disen_lll_weight=args.disen_lll_weight,
        freeze_pretrained_encoder=args.freeze_student_encdec,
        freeze_pretrained_decoder=args.freeze_student_encdec,
    )
    
    # ====== 멜 스펙트로그램 예시 저장 ======
    mel_dir = os.path.join(args.out, "xai/mel_examples")
    save_mel_examples_from_manifest(
        manifest_path=train_manifest,
        model=student_model,
        out_dir=mel_dir,
        num_examples=4,
        split_name="train",
    )
    
    # 사용자 인자 처리
    student_ckpt_dir = os.path.join(args.out, "checkpoints_student")
    student_ckpt_cb = ModelCheckpoint(dirpath=student_ckpt_dir, filename="student_last", save_top_k=0, save_last=True)

    student_trainer = pl.Trainer(
        devices=args.gpus,
        accelerator="gpu",
        max_epochs=args.epochs,
        default_root_dir=args.out,
        logger=wandb,
        callbacks=[student_ckpt_cb],
    )
    student_trainer.fit(student_model)

    # -------------------- Stage 4: Student Evaluation --------------------
    print("\n===== Stage 4: Student Evaluation =====") 
    ### TODO: librispeech 뿐만 아니라, GigaSpeech testset도 추가적으로 평가할 수 있도록
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
        test_cfg = deepcopy(student_model.cfg.test_ds)
        test_cfg.manifest_filepath = manifest
        test_cfg.shuffle = False
        student_model.setup_test_data(test_cfg)
        dl = student_model.test_dataloader()
        results = student_trainer.test(model=student_model, dataloaders=[dl], verbose=True)

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

        hyps = student_model.transcribe(
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
