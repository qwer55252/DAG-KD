#!/bin/bash
# E2: Layer KD + Layerwise Spk GRL
# 모든 teacher 레이어에 개별 enc_i + shared GRL spk classifier
# λ_adv=0.1, grl_alpha=0.1

CUDA_VISIBLE_DEVICES=0 python train.py \
  --wandb_run layerwise_grl_E2_grl \
  --out outputs/layerwise_spk_grl/E2_layerwise_grl \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name stt_en_conformer_ctc_small \
  --use_ctc True \
  --use_logit_kd True \
  --use_layer_kd False \
  --use_flow False \
  --use_diffkd False \
  --use_disent False \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1 \
  --use_layerwise_spk_grl True \
  --spk_grl_alpha 0.1 \
  --spk_grl_adv_weight 0.1
