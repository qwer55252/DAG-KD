#!/bin/bash
# e2_layer_kd: Random init student + CTC + Layerwise KD (MSE on hidden states)
OUT=outputs/wav2vec/e2_layer_kd
mkdir -p "$OUT"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec \
  --wandb_run wav2vec_e2_layer_kd \
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
  --use_logit_kd False \
  --use_layer_kd True \
  --layer_kd_alpha 0.5 \
  --use_flow False \
  --use_diffkd False \
  --use_disent False \
  --use_txt_spk_probe False \
  --batch_size 4 \
  --epochs 100 \
  --gpus 4 \
  --learning_rate 1e-4 \
  --warmup_epochs 5 \
  2>&1 | tee "$OUT/train.log"
