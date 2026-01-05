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
from datasets import load_dataset, DownloadConfig, config as hf_config

from models_ver2 import TeacherASRWithDisent
from utils import (
    scan_speakers,
    build_manifest_from_hf_with_meta,
    build_manifest_from_hf_gigaspeech,
    str2bool,
    release_nemoAPI,
    compute_sample_wers,
    compute_speaker_durations,
    save_speaker_mapping,
    materialize_nemo_artifacts_in_cfg,
    rotate_last_ckpts,
    head_manifest,
)

def main():
    p = argparse.ArgumentParser("Teacher-only train+eval")

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

    # Logging/ckpt
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--out", type=str, default="outputs_teacher_only")
    p.add_argument("--resume_ckpt_path", type=str, default="", help="teacher checkpoint to resume from (optional)")

    # Teacher
    p.add_argument("--teacher_name", type=str, default="stt_en_conformer_ctc_small")
    p.add_argument("--freeze_teacher_encdec", type=str2bool, default=False)

    # Disentanglement
    p.add_argument("--disen_mi_pairs", type=str, default="ts,tp,ps")
    p.add_argument("--disen_lll_weight", type=float, default=1.0)
    p.add_argument("--disen_mi_weight", type=float, default=1e-3)

    p.add_argument("--use_txt_spk_probe", type=str2bool, default=True)
    p.add_argument("--mi_clamp_min0", type=str2bool, default=True)

    # W&B
    p.add_argument("--wandb_project", type=str, default=os.getenv("PRJ_NAME", "DAG-KD"))
    p.add_argument("--wandb_run", type=str, default=os.getenv("EXP_NAME", "teacher_only"))
    
    # Extra eval data
    p.add_argument(
        "--extra_eval_data",
        type=str,
        default="",
        choices=["", "librispeech", "tedlium2", "commonvoice", "gigaspeech"],
        help="추가 평가 데이터셋 (빈 문자열이면 Stage3 스킵)",
    )

    args = p.parse_args()

    # -------------------- Stage 0: Load Dataset --------------------
    print("\n===== Stage 0: Load Dataset =====")
    os.makedirs(args.out, exist_ok=True)
    manifest_dir = os.path.join(args.data_dir, args.data_cfg, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)

    cache_dir = os.path.join(args.data_dir, args.data_cfg, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    hf_config.HF_DATASETS_CACHE = cache_dir

    dl_cfg = DownloadConfig(
        cache_dir=cache_dir,
        force_download=False,
        resume_download=True,
        max_retries=8,
        disable_tqdm=False,
        download_desc="Downloading LibriSpeech dataset",
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
    print(f"[INFO] Loaded train split: {args.train_split}, n={len(train_ds)}")
    print(f"[INFO] Loaded val split: {args.val_split}, n={len(val_ds)}")
    print(f"[INFO] Loaded test split: {args.test_split}, n={len(test_ds)}")

    # extra eval splits
    extra_splits = {}
    for split_name in ["dev.other", "test.other"]:
        try:
            ds = load_dataset(args.data_script, args.data_cfg, split=split_name,
                              trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir)
            extra_splits[split_name] = ds
            print(f"[INFO] Loaded extra split: {split_name}, n={len(ds)}")
        except Exception as e:
            print(f"[WARN] Could not load split '{split_name}': {e}")

    print(f"[INFO] scanning speakers from train split...")
    spk2idx, idx2spk = scan_speakers(train_ds)
    num_spk = len(spk2idx)
    print(f"[SCAN] num_speakers={num_spk}")

    spk_map_path = os.path.join(manifest_dir, "speaker_id_mapping.json")
    if not os.path.isfile(spk_map_path):
        save_speaker_mapping(spk2idx, idx2spk, spk_map_path)
        print(f"[INFO] saved speaker ID mapping to {spk_map_path}")

    train_manifest      = os.path.join(manifest_dir, "train.json")
    dev_clean_manifest  = os.path.join(manifest_dir, "dev_clean.json")
    dev_other_manifest  = os.path.join(manifest_dir, "dev_other.json")
    test_clean_manifest = os.path.join(manifest_dir, "test_clean.json")
    test_other_manifest = os.path.join(manifest_dir, "test_other.json")

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

    spk_dur_out = os.path.join(manifest_dir, "speaker_durations_train.json")
    if not os.path.isfile(spk_dur_out):
        compute_speaker_durations(train_manifest, spk_dur_out)
        print(f"[INFO] saved speaker durations to {spk_dur_out}")

    if args.test_mode:
        train_ds = train_ds.select(range(50))
        val_ds   = val_ds.select(range(50))
        test_ds  = test_ds.select(range(50))
        for k in list(extra_splits.keys()):
            extra_splits[k] = extra_splits[k].select(range(min(50, len(extra_splits[k]))))

        test_mode_train_manifest = os.path.join(manifest_dir, "test_mode_train.json")
        test_mode_val_manifest   = os.path.join(manifest_dir, "test_mode_val.json")
        test_mode_test_manifest  = os.path.join(manifest_dir, "test_mode_test.json")

        spk2idx, idx2spk = scan_speakers(train_ds)
        num_spk = len(spk2idx)

        build_manifest_from_hf_with_meta(train_ds, test_mode_train_manifest, cache_dir, spk2idx)
        build_manifest_from_hf_with_meta(val_ds,   test_mode_val_manifest,   cache_dir, spk2idx)
        build_manifest_from_hf_with_meta(test_ds,  test_mode_test_manifest,  cache_dir, spk2idx)

        train_manifest = test_mode_train_manifest
        dev_clean_manifest = test_mode_val_manifest
        test_clean_manifest = test_mode_test_manifest

    # -------------------- Logger/Trainer --------------------
    wandb = WandbLogger(project=args.wandb_project, name=args.wandb_run, save_dir=args.out)

    teacher_ckpt_dir = os.path.join(args.out, "checkpoints_teacher")
    os.makedirs(teacher_ckpt_dir, exist_ok=True)

    teacher_ckpt_cb = ModelCheckpoint(
        dirpath=teacher_ckpt_dir,
        filename="teacher-{epoch}-{step}",
        save_last="link",
        save_top_k=-1,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = args.gpus if accelerator == "gpu" else 1

    trainer = pl.Trainer(
        devices=devices,
        accelerator=accelerator,
        max_epochs=args.epochs,
        default_root_dir=args.out,
        logger=wandb,
        callbacks=[teacher_ckpt_cb],
        # 멀티 GPU면 보통 ddp가 wandb/스폰 이슈 줄임
        # strategy="ddp" if (accelerator == "gpu" and int(devices) > 1) else "auto",
    )

    # -------------------- Stage 1: Build teacher from pretrained NeMo --------------------
    print("\n===== Stage 1: Teacher training =====")
    nemo_teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name=args.teacher_name,
        map_location="cuda:0" if torch.cuda.is_available() else "cpu",
        trainer=trainer,
    )
    # cfg artifact materialize 용도. (필요 없으면 빼도 됨)
    release_nemoAPI(nemo_teacher, out_folder=os.path.join(args.out, "nemo_archive"))

    teacher_cfg = deepcopy(nemo_teacher.cfg)
    archive_dir = os.path.abspath(os.path.join(args.out, "nemo_archive"))
    teacher_cfg = materialize_nemo_artifacts_in_cfg(teacher_cfg, archive_dir)

    OmegaConf.set_struct(teacher_cfg, False)
    teacher_cfg.train_ds.is_tarred = False
    teacher_cfg.validation_ds.is_tarred = False
    teacher_cfg.test_ds.is_tarred = False

    teacher_cfg.train_ds.manifest_filepath = train_manifest
    teacher_cfg.validation_ds.manifest_filepath = dev_clean_manifest
    teacher_cfg.test_ds.manifest_filepath = test_clean_manifest

    teacher_cfg.train_ds.sample_rate = args.sample_rate
    teacher_cfg.validation_ds.sample_rate = args.sample_rate
    teacher_cfg.test_ds.sample_rate = args.sample_rate

    teacher_cfg.train_ds.batch_size = args.batch_size
    teacher_cfg.validation_ds.batch_size = args.batch_size
    teacher_cfg.test_ds.batch_size = args.batch_size

    teacher_cfg.train_ds.return_sample_id = True
    teacher_cfg.validation_ds.return_sample_id = False
    teacher_cfg.test_ds.return_sample_id = False

    # disent cfg 주입
    teacher_cfg.latent_dim = 96
    teacher_cfg.num_spk = num_spk
    teacher_cfg.disen_mi_pairs = args.disen_mi_pairs
    teacher_cfg.use_txt_spk_probe = args.use_txt_spk_probe
    teacher_cfg.mi_clamp_min0 = args.mi_clamp_min0

    teacher = TeacherASRWithDisent(
        cfg=teacher_cfg,
        trainer=trainer,
        disen_mi_weight=args.disen_mi_weight,
        disen_lll_weight=args.disen_lll_weight,
        freeze_pretrained_encoder=args.freeze_teacher_encdec,
        freeze_pretrained_decoder=args.freeze_teacher_encdec,
    )

    missing, unexpected = teacher.load_from_pretrained_nemo(nemo_teacher, strict=False)
    print("[Teacher init] missing keys:", len(missing), "unexpected keys:", len(unexpected))

    # -------------------- Train --------------------
    ckpt_path = args.resume_ckpt_path if args.resume_ckpt_path else None
    trainer.fit(teacher, ckpt_path=ckpt_path)

    rotate_last_ckpts(teacher_ckpt_dir, keep=30)
    teacher_ckpt_path = os.path.join(teacher_ckpt_dir, "last.ckpt")
    print(f"[INFO] teacher_ckpt_path={teacher_ckpt_path}")

    # -------------------- Stage 2: Teacher Evaluation on LibriSpeech--------------------
    print("\n===== Stage 2: Teacher Evaluation on LibriSpeech =====")
    teacher_eval = teacher.eval()
    for p_ in teacher_eval.parameters():
        p_.requires_grad = False

    eval_targets = [
        ("dev_clean",  dev_clean_manifest),
        ("dev_other",  dev_other_manifest),
        ("test_clean", test_clean_manifest),
        ("test_other", test_other_manifest),
    ]

    plot_dir = os.path.join(args.out, "xai/wer_plots_teacher")
    os.makedirs(plot_dir, exist_ok=True)
    for split_name, manifest in eval_targets:
        if args.test_mode:
            # 50개만 평가: manifest를 앞 N줄만 가진 "eval 전용 manifest"로 교체
            N_EVAL = 50
            # 원본 manifest가 없으면 아래에서 skip되도록 그대로 둠
            if manifest and os.path.isfile(manifest):
                tm_manifest = os.path.join(manifest_dir, f"test_mode_eval_{split_name}.json")
                tm_manifest, n_written = head_manifest(manifest, tm_manifest, N_EVAL)
                if n_written == 0:
                    print(f"[WARN] test_mode eval manifest is empty for {split_name}: {tm_manifest}, skip")
                    continue
                print(f"[Stage2:test_mode] {split_name}: using first {n_written} samples -> {tm_manifest}")
                manifest = tm_manifest
        if not manifest or (not os.path.isfile(manifest)):
            print(f"[WARN] manifest not found for {split_name}: {manifest}, skip")
            continue
        
        print(f"\n===== [Teacher] Evaluating on {split_name} =====")

        # 1) Lightning test
        test_cfg = deepcopy(teacher_eval.cfg.test_ds)
        OmegaConf.set_struct(test_cfg, False)
        test_cfg.manifest_filepath = manifest
        test_cfg.shuffle = False
        test_cfg.batch_size = args.batch_size
        if hasattr(test_cfg, "return_sample_id"):
            test_cfg.return_sample_id = False

        teacher_eval.setup_test_data(test_cfg)
        dl = teacher_eval.test_dataloader()

        results = trainer.test(model=teacher_eval, dataloaders=[dl], verbose=True)
        if results and isinstance(results, list):
            res = results[0]
            wer = res.get("test_wer", res.get("wer", None))
            loss = res.get("test_loss", res.get("loss", None))
            if loss is not None and wer is not None:
                print(f"→ [Teacher] {split_name}: loss={loss:.4f} | wer={wer:.2%}")
                wandb.log_metrics({f"teacher/{split_name}/wer": float(wer)}, step=trainer.current_epoch)

        # 2) per-sample WER
        with open(manifest, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f]
        audio_files = [e["audio_filepath"] for e in entries]
        ref_texts   = [e["text"] for e in entries]

        hyps = teacher_eval.transcribe(
            audio=audio_files,
            batch_size=args.batch_size,
            return_hypotheses=False,
            num_workers=0,
            verbose=False,
        )

        sample_wers = compute_sample_wers(ref_texts, hyps)
        sample_wers_pct = [w * 100.0 for w in sample_wers]

        wer_mean = float(statistics.mean(sample_wers_pct)) if len(sample_wers_pct) else 0.0
        wer_std  = float(statistics.stdev(sample_wers_pct)) if len(sample_wers_pct) > 1 else 0.0
        print(f"→ [Teacher] {split_name}: per-sample WER = {wer_mean:.2f}% ± {wer_std:.2f}%")

        wers_np = np.array(sample_wers_pct, dtype=float)

        # histogram
        plt.figure()
        bins = [0, 10, 20, 30, 50, 100, 200]
        plt.hist(wers_np, bins=bins, edgecolor="black")
        plt.xlabel("Per-sample WER (%)")
        plt.ylabel("Count")
        plt.title(f"[Teacher] WER Histogram - {split_name}\nmean={wer_mean:.2f}%, std={wer_std:.2f}%")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"teacher_wer_hist_{split_name}.png"))
        plt.close()

        # boxplot
        plt.figure()
        plt.boxplot(wers_np, vert=True, showfliers=True)
        plt.ylabel("Per-sample WER (%)")
        plt.title(f"[Teacher] WER Boxplot - {split_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"teacher_wer_box_{split_name}.png"))
        plt.close()

        wandb.log_metrics(
            {
                f"teacher/{split_name}/wer_mean": wer_mean,
                f"teacher/{split_name}/wer_std": wer_std,
            },
            step=trainer.current_epoch,
        )
    # -------------------- Stage 3: Teacher Evaluation on GigaSpeech--------------------
    print("\n===== Stage 3: Teacher Evaluation on GigaSpeech =====")

    # (A) gigaspeech 평가용 HF 로드 설정
    gigaspeech_script = "./gigaspeech_asr.py"
    gigaspeech_cfg = "dev"
    gigaspeech_splits = ["validation", "test"]
    gigaspeech_cache_dir = os.path.join(args.data_dir, gigaspeech_cfg, "cache")
    
    # Stage0에서 쓰던 cache_dir 그대로 써도 됨 (extracted 폴더도 거기 생김)
    dl_cfg_gs = DownloadConfig(
        cache_dir=gigaspeech_cache_dir,
        force_download=False,
        resume_download=True,
        max_retries=8,
        disable_tqdm=False,
        download_desc="Downloading GigaSpeech Eval dataset",
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=72000)}},
        delete_extracted=False,
        extract_compressed_file=True,
        force_extract=True,
    )

    # (B) Stage3 전용 manifest 저장 위치 (Libri manifest랑 섞지 않게 분리 추천)
    extra_manifest_dir = os.path.join(args.data_dir, gigaspeech_cfg, "manifests")
    os.makedirs(extra_manifest_dir, exist_ok=True)

    # (C) split별 평가
    for split in gigaspeech_splits:
        print(f"\n===== [Teacher] GigaSpeech split={split} =====")

        ds = load_dataset(
            gigaspeech_script,
            gigaspeech_cfg,
            split=split,
            trust_remote_code=True,
            download_config=dl_cfg_gs,
            cache_dir=cache_dir,
        )

        # test_mode면 일부만
        # if args.test_mode:
        #     n = min(50, len(ds))
        #     ds = ds.select(range(n))
        #     print(f"[Stage3] test_mode=True -> using n={len(ds)}")
        # 1) gigaspeech manifest 생성 (너가 주는 함수 그대로 사용)
        manifest_i = os.path.join(extra_manifest_dir, f"gigaspeech_{split}.json")
        build_manifest_from_hf_gigaspeech(ds, manifest_i, cache_dir)

        # (선택) 비어있는 manifest면 즉시 에러로 잡기 (이전 프로젝트랑 동일)
        n_lines = sum(1 for line in open(manifest_i, "r", encoding="utf-8") if line.strip())
        if n_lines == 0:
            raise RuntimeError(f"[Stage3] GigaSpeech manifest has 0 lines: {manifest_i}")

        # 2) NeMo test config 세팅 + dataloader 구성
        test_cfg = deepcopy(teacher_eval.cfg.test_ds)
        OmegaConf.set_struct(test_cfg, False)
        test_cfg.manifest_filepath = manifest_i
        test_cfg.shuffle = False
        test_cfg.batch_size = args.batch_size
        if hasattr(test_cfg, "return_sample_id"):
            test_cfg.return_sample_id = False

        teacher_eval.setup_test_data(test_cfg)
        dl = teacher_eval.test_dataloader()

        # 3) Lightning test
        results = trainer.test(model=teacher_eval, dataloaders=[dl], verbose=True)
        res = results[0] if results else {}

        wer = res.get("test_wer", res.get("wer", None))
        loss = res.get("test_loss", res.get("loss", None))

        print(f"→ [Teacher][GigaSpeech] {split}: loss={loss} | wer={wer}")

        # 4) W&B 로깅 (키는 너의 Stage2 스타일에 맞춰줌)
        log_dict = {}
        if wer is not None:
            log_dict[f"teacher/gigaspeech/{split}/wer"] = float(wer)
        if loss is not None:
            log_dict[f"teacher/gigaspeech/{split}/loss"] = float(loss)
        if log_dict:
            wandb.log_metrics(log_dict, step=trainer.current_epoch)
    
    
if __name__ == "__main__":
    main()
