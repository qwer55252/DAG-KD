#!/bin/bash
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY before running}"
# wav2vec2 E1-v2: Baseline (조건 재통제)
# 변경 사항 (vs e1_baseline):
#   - learning_rate: 3e-4 → 1e-4  (E0-S와 동일하게 맞춤)
#   - freeze_feature_extractor: (없음) → True  (E0-S와 동일하게 맞춤)
#   - warmup_epochs: (없음) → 5
# 목적: E0-S(5.84%)와 동일 조건에서 Logit KD의 순효과 측정
OUT=outputs/wav2vec/e1_baseline_v2
mkdir -p "$OUT"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec \
  --wandb_run wav2vec_e1_baseline_v2 \
  --out "$OUT" \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name facebook/wav2vec2-large-960h \
  --student_name facebook/wav2vec2-base-960h \
  --use_ctc True \
  --use_logit_kd True \
  --kd_alpha 0.5 \
  --kd_temperature 1.0 \
  --use_layer_kd False \
  --use_flow False \
  --use_diffkd False \
  --use_disent False \
  --use_txt_spk_probe False \
  --batch_size 8 \
  --epochs 100 \
  --gpus 4 \
  --learning_rate 1e-4 \
  --warmup_epochs 5 \
  --freeze_feature_extractor True \
  2>&1 | tee "$OUT/train.log"
