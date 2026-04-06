#!/bin/bash
# Teacher (wav2vec2-large-960h) 단독 fine-tuning — CTC only, no KD
# 비교군: 가장 강력한 상한선 (oracle upper bound)
# Note: large 모델 fine-tuning → LR=3e-5 + warmup 5 epoch + feature extractor freeze
OUT=outputs/wav2vec/e0_teacher_ft
mkdir -p "$OUT"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec \
  --wandb_run wav2vec_e0_teacher_ft \
  --out "$OUT" \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name facebook/wav2vec2-large-960h \
  --student_name facebook/wav2vec2-large-960h \
  --use_ctc True \
  --use_logit_kd False \
  --use_layer_kd False \
  --use_flow False \
  --use_diffkd False \
  --use_disent False \
  --use_txt_spk_probe False \
  --freeze_feature_extractor True \
  --batch_size 4 \
  --epochs 100 \
  --gpus 4 \
  --learning_rate 5e-5 \
  --warmup_epochs 5 \
  2>&1 | tee "$OUT/train.log"
