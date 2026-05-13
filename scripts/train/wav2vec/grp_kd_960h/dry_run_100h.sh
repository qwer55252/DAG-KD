#!/bin/bash
# Dry-run: validate SSL student + 4-GPU DDP + batch=10 memory fit on 100h
# (smaller scan_speakers cost than 960h, manifests already cached)
export WANDB_API_KEY=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
OUT=outputs/wav2vec/grp_kd_960h/_dry_run
rm -rf "$OUT"; mkdir -p "$OUT"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 360 python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec-dry \
  --wandb_run dry_run_ssl_4gpu \
  --out "$OUT" \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name facebook/wav2vec2-large-960h \
  --student_name facebook/wav2vec2-base \
  --processor_name facebook/wav2vec2-base-960h \
  --random_init_student False \
  --freeze_feature_extractor True \
  --use_ctc True \
  --use_logit_kd True \
  --kd_alpha 0.1 \
  --kd_temperature 1.0 \
  --use_grp_kd True \
  --grp_latent_dim 96 \
  --grp_fm_steps 8 \
  --grp_diff_steps 9 \
  --grp_rec_weight 1.0 \
  --grp_gen_weight 1.0 \
  --grp_disen_mode 1 \
  --grp_orth_weight 1.0 \
  --grp_spk_cls_weight 1.0 \
  --batch_size 18 \
  --epochs 1 \
  --gpus 4 \
  --learning_rate 1e-4 \
  --warmup_epochs 1 \
  --kd_warmup_epochs 1 \
  2>&1 | tee "$OUT/train.log"
