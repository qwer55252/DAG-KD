#!/bin/bash
# E1: Layer KD baseline (GRL 없음)
# 비교군 — 기존 layerwise metric KD (student→teacher projection MSE)

CUDA_VISIBLE_DEVICES=0 python train.py \
  --wandb_run layerwise_grl_E1_baseline \
  --out outputs/layerwise_spk_grl/E1_baseline \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name stt_en_conformer_ctc_small \
  --use_ctc True \
  --use_logit_kd True \
  --use_layer_kd True \
  --layer_kd_alpha 1.0 \
  --use_flow False \
  --use_diffkd False \
  --use_disent False \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1 \
  --use_layerwise_spk_grl False
