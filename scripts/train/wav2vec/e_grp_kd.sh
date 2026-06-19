#!/bin/bash
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY before running}"
# GRP-KD 대조군 실험
# Ref: "Knowledge Distillation via Generative Reconstruction Pathways for E2E ASR" (ICASSP 2026)
#
# 논문 설정 재현:
#   - Shared AE (D_L=96) + Flow Matching (K=8 steps) + Diffusion (9 steps)
#   - L_total = λ1*L_rec + λ2*(L_FM + L_DF) + α*L_CTC + β*L_logitKD
#   - λ1=1.0, λ2=1.0 (grp_rec_weight, grp_gen_weight)
#   - α=1.0, β=0.1  (CTC dominant, logitKD 보조)
#
# 비교 대상:
#   - E2-C Layer KD (18.79%) — 현재 최고 성능
#   - E2-D DAG-KD v2 (phys_loss fix 후 재실험)
OUT=outputs/wav2vec/e_grp_kd
mkdir -p "$OUT"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec \
  --wandb_run wav2vec_grp_kd \
  --out "$OUT" \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name facebook/wav2vec2-large-960h \
  --student_name facebook/wav2vec2-base-960h \
  --random_init_student True \
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
  --batch_size 4 \
  --epochs 100 \
  --gpus 4 \
  --learning_rate 1e-4 \
  --warmup_epochs 5 \
  --kd_warmup_epochs 10 \
  2>&1 | tee "$OUT/train.log"
