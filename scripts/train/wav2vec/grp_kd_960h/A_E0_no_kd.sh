#!/bin/bash
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY before running}"
# Track A / E0 — wav2vec2 SSL student CTC fine-tuning (no KD) on 960h
# Student: facebook/wav2vec2-base (SSL only, 12L d=768, ~95M)
# Teacher: loaded for parity but not used (use_logit_kd=False, use_grp_kd=False)
# Purpose: single-digit WER baseline, anchor for measuring KD lift in E1/E2
OUT=outputs/wav2vec/grp_kd_960h/A_E0_no_kd
mkdir -p "$OUT"

RESUME_ARGS=()
if [[ -f "$OUT/checkpoints/last.ckpt" ]]; then
  RESUME_ARGS=(--resume_ckpt_path "$OUT/checkpoints/last.ckpt")
fi

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec \
  --wandb_run wav2vec_960h_A_E0_no_kd \
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
  --use_logit_kd False \
  --use_layer_kd False \
  --use_flow False \
  --use_diffkd False \
  --use_disent False \
  --use_txt_spk_probe False \
  --use_grp_kd False \
  --batch_size 16 \
  --epochs 10 \
  --gpus 4 \
  --learning_rate 1e-4 \
  --warmup_epochs 1 \
  --kd_warmup_epochs 1 \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "$OUT/train.log"
