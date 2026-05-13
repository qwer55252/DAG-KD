#!/bin/bash
export WANDB_API_KEY=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
# Track A / E1 — wav2vec2 SSL student + GRP-KD baseline (disen_mode=0) on 960h
# Student: facebook/wav2vec2-base (SSL only)
# Adds: logit KD + GRP-KD baseline on top of A-E0
OUT=outputs/wav2vec/grp_kd_960h/A_E1_kd
mkdir -p "$OUT"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec \
  --wandb_run wav2vec_960h_A_E1_kd \
  --out "$OUT" \
  --data_script ./librispeech_asr.py \
  --data_cfg all \
  --train_split "train.clean.100+train.clean.360+train.other.500" \
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
  --use_layer_kd False \
  --use_flow False \
  --use_diffkd False \
  --use_disent False \
  --use_txt_spk_probe False \
  --use_grp_kd True \
  --grp_latent_dim 96 \
  --grp_fm_steps 8 \
  --grp_diff_steps 9 \
  --grp_rec_weight 1.0 \
  --grp_gen_weight 1.0 \
  --grp_disen_mode 0 \
  --batch_size 16 \
  --epochs 10 \
  --gpus 4 \
  --learning_rate 1e-4 \
  --warmup_epochs 1 \
  --kd_warmup_epochs 1 \
  2>&1 | tee "$OUT/train.log"
