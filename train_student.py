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

from models_ver2 import TeacherASRWithDisent, StudentASRWithDisentKD
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
)

def main():
    p = argparse.ArgumentParser("Student-only train+eval (teacher frozen for KD)")

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
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--out", type=str, default="outputs_student_only")
    p.add_argument("--resume_ckpt_path", type=str, default="", help="resume student ckpt (optional)")

    # Teacher source (cfg + ckpt)
    p.add_argument("--teacher_name", type=str, default="stt_en_conformer_ctc_small")
    p.add_argument("--teacher_ckpt_path", type=str, required=True, help="학습된 teacher ckpt 경로")
    p.add_argument("--freeze_student_encdec", type=str2bool, default=False)

    # KD
    p.add_argument("--use_logit_kd", type=str2bool, default=True)
    p.add_argument("--use_layer_kd", type=str2bool, default=False)
    p.add_argument("--kd_alpha", type=float, default=0.5)
    p.add_argument("--kd_temperature", type=float, default=1.0)
    p.add_argument("--gen_kd_type", type=str, default="flow", choices=["mse", "flow", "diff"])
    p.add_argument("--gen_kd_weight", type=float, default=1.0)

    # Disent
    p.add_argument("--use_txt_spk_probe", type=str2bool, default=True)
    p.add_argument("--txt_probe_lambda", type=float, default=1.0)
    p.add_argument("--disen_mi_pairs", type=str, default="ts,tp,ps")
    p.add_argument("--disen_lll_weight", type=float, default=1.0)
    p.add_argument("--disen_mi_weight", type=float, default=1e-3)
    p.add_argument("--mi_clamp_min0", type=str2bool, default=True)

    p.add_argument("--mi_warmup_steps", type=int, default=5000)
    p.add_argument("--mi_ramp_steps", type=int, default=20000)
    p.add_argument("--mi_lambda_max", type=float, default=0.01)
    p.add_argument("--lll_lambda_max", type=float, default=0.01)

    # W&B
    p.add_argument("--wandb_project", type=str, default=os.getenv("PRJ_NAME", "DAG-KD"))
    p.add_argument("--wandb_run", type=str, default=os.getenv("EXP_NAME", "student_only"))
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # -------------------- Stage 0: Dataset + Manifests --------------------
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

    extra_splits = {}
    for split_name in ["dev.other", "test.other"]:
        try:
            ds = load_dataset(args.data_script, args.data_cfg, split=split_name,
                              trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir)
            extra_splits[split_name] = ds
        except Exception as e:
            print(f"[WARN] Could not load split '{split_name}': {e}")

    print(f"[INFO] train n={len(train_ds)} | val n={len(val_ds)} | test n={len(test_ds)}")

    spk2idx, idx2spk = scan_speakers(train_ds)
    num_spk = len(spk2idx)
    print(f"[SCAN] num_speakers={num_spk}")

    spk_map_path = os.path.join(manifest_dir, "speaker_id_mapping.json")
    if not os.path.isfile(spk_map_path):
        save_speaker_mapping(spk2idx, idx2spk, spk_map_path)

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

    # -------------------- Stage 1: Logger/Trainer (Student only) --------------------
    wandb = WandbLogger(project=args.wandb_project, name=args.wandb_run, save_dir=args.out)

    student_ckpt_dir = os.path.join(args.out, "checkpoints_student")
    os.makedirs(student_ckpt_dir, exist_ok=True)
    student_ckpt_cb = ModelCheckpoint(dirpath=student_ckpt_dir, filename="student_last", save_top_k=0, save_last=True)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = args.gpus if accelerator == "gpu" else 1

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=args.epochs,
        default_root_dir=args.out,
        logger=wandb,
        callbacks=[student_ckpt_cb],
        # 멀티GPU면 아래 한 줄이 wandb/스폰 문제 줄이는 데 도움될 때가 많음
        # strategy="ddp" if devices and int(devices) > 1 else "auto",
    )

    # -------------------- Stage 2: Load teacher (frozen, for KD) --------------------
    # cfg를 안전하게 얻기 위해 pretrained nemo 모델에서 cfg만 가져오고, ds/옵션 주입
    nemo_teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name=args.teacher_name,
        map_location="cuda:0" if torch.cuda.is_available() else "cpu",
        trainer=trainer,
    )
    release_nemoAPI(nemo_teacher, out_folder=os.path.join(args.out, "nemo_archive_teacher_cfg"))
    teacher_cfg = deepcopy(nemo_teacher.cfg)

    archive_dir = os.path.abspath(os.path.join(args.out, "nemo_archive_teacher_cfg"))
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

    # disent cfg
    teacher_cfg.latent_dim = 96
    teacher_cfg.num_spk = num_spk
    teacher_cfg.disen_mi_pairs = args.disen_mi_pairs
    teacher_cfg.use_txt_spk_probe = args.use_txt_spk_probe
    teacher_cfg.mi_clamp_min0 = args.mi_clamp_min0

    teacher_loaded = TeacherASRWithDisent.load_from_checkpoint(
        args.teacher_ckpt_path,
        cfg=teacher_cfg,
        disen_mi_weight=args.disen_mi_weight,
        disen_lll_weight=args.disen_lll_weight,
    )
    teacher_loaded.eval()
    for p_ in teacher_loaded.parameters():
        p_.requires_grad = False

    # -------------------- Stage 3: Build student cfg + init student --------------------
    stu_cfg = deepcopy(teacher_cfg)
    OmegaConf.set_struct(stu_cfg, False)
    OmegaConf.set_struct(stu_cfg.train_ds, False)
    OmegaConf.set_struct(stu_cfg.validation_ds, False)
    OmegaConf.set_struct(stu_cfg.test_ds, False)

    stu_cfg.train_ds.manifest_filepath = train_manifest
    stu_cfg.validation_ds.manifest_filepath = dev_clean_manifest
    stu_cfg.test_ds.manifest_filepath = test_clean_manifest
    stu_cfg.train_ds.batch_size = args.batch_size
    stu_cfg.validation_ds.batch_size = args.batch_size
    stu_cfg.test_ds.batch_size = args.batch_size
    stu_cfg.train_ds.sample_rate = args.sample_rate
    stu_cfg.validation_ds.sample_rate = args.sample_rate
    stu_cfg.test_ds.sample_rate = args.sample_rate

    stu_cfg.train_ds.return_sample_id = True
    stu_cfg.validation_ds.return_sample_id = False
    stu_cfg.test_ds.return_sample_id = False

    # student capacity downscale
    stu_cfg.encoder.d_model = max(8, int(teacher_cfg.encoder.d_model) // 2)
    stu_cfg.encoder.n_heads = max(1, int(teacher_cfg.encoder.n_heads) // 2)
    stu_cfg.decoder.feat_in = max(8, int(teacher_cfg.decoder.feat_in) // 2)

    # extra knobs
    stu_cfg.latent_dim = 96
    stu_cfg.num_spk = num_spk
    stu_cfg.disen_mi_pairs = args.disen_mi_pairs

    stu_cfg.use_txt_spk_probe = args.use_txt_spk_probe

    stu_cfg.mi_warmup_steps = args.mi_warmup_steps
    stu_cfg.mi_ramp_steps   = args.mi_ramp_steps
    stu_cfg.mi_lambda_max   = args.mi_lambda_max
    stu_cfg.lll_lambda_max  = args.lll_lambda_max
    stu_cfg.mi_clamp_min0   = args.mi_clamp_min0

    student_model = StudentASRWithDisentKD(
        cfg=stu_cfg,
        trainer=trainer,
        teacher=teacher_loaded,
        use_logit_kd=args.use_logit_kd,
        use_layer_kd=args.use_layer_kd,
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

    # optional: mel examples
    mel_dir = os.path.join(args.out, "xai/mel_examples")
    save_mel_examples_from_manifest(
        manifest_path=train_manifest,
        model=student_model,
        out_dir=mel_dir,
        num_examples=4,
        split_name="train",
    )

    # -------------------- Stage 4: Train student --------------------
    fit_ckpt = args.resume_ckpt_path if args.resume_ckpt_path else None
    trainer.fit(student_model, ckpt_path=fit_ckpt)

    # -------------------- Stage 5: Evaluate student --------------------
    eval_targets = [
        ("dev_clean",  dev_clean_manifest),
        ("dev_other",  dev_other_manifest),
        ("test_clean", test_clean_manifest),
        ("test_other", test_other_manifest),
    ]

    plot_dir = os.path.join(args.out, "xai/wer_plots")
    os.makedirs(plot_dir, exist_ok=True)

    for split_name, manifest in eval_targets:
        if not os.path.isfile(manifest):
            print(f"[WARN] manifest not found for {split_name}: {manifest}, skip")
            continue

        print(f"\n===== [Student] Evaluating on {split_name} =====")

        test_cfg = deepcopy(student_model.cfg.test_ds)
        OmegaConf.set_struct(test_cfg, False)
        test_cfg.manifest_filepath = manifest
        test_cfg.shuffle = False
        test_cfg.batch_size = args.batch_size
        if hasattr(test_cfg, "return_sample_id"):
            test_cfg.return_sample_id = False

        student_model.setup_test_data(test_cfg)
        dl = student_model.test_dataloader()
        results = trainer.test(model=student_model, dataloaders=[dl], verbose=True)
        if results and isinstance(results, list):
            res = results[0]
            wer = res.get("test_wer", res.get("wer", None))
            loss = res.get("test_loss", res.get("loss", None))
            if loss is not None and wer is not None:
                print(f"→ {split_name}: loss={loss:.4f} | wer={wer:.2%}")
                wandb.log_metrics({f"{split_name}/wer": float(wer)}, step=trainer.current_epoch)

        # per-sample WER
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
        wer_mean = float(statistics.mean(sample_wers_pct)) if len(sample_wers_pct) else 0.0
        wer_std  = float(statistics.stdev(sample_wers_pct)) if len(sample_wers_pct) > 1 else 0.0
        print(f"→ {split_name}: per-sample WER = {wer_mean:.2f}% ± {wer_std:.2f}%")

        wers_np = np.array(sample_wers_pct, dtype=float)

        # histogram
        plt.figure()
        bins = [0, 10, 20, 30, 50, 100, 200]
        plt.hist(wers_np, bins=bins, edgecolor="black")
        plt.xlabel("Per-sample WER (%)")
        plt.ylabel("Count")
        plt.title(f"WER Histogram - {split_name}\nmean={wer_mean:.2f}%, std={wer_std:.2f}%")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"wer_hist_{split_name}.png"))
        plt.close()

        # boxplot
        plt.figure()
        plt.boxplot(wers_np, vert=True, showfliers=True)
        plt.ylabel("Per-sample WER (%)")
        plt.title(f"WER Boxplot - {split_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"wer_box_{split_name}.png"))
        plt.close()

        wandb.log_metrics(
            {f"{split_name}/wer_mean": wer_mean, f"{split_name}/wer_std": wer_std},
            step=trainer.current_epoch,
        )

if __name__ == "__main__":
    main()
