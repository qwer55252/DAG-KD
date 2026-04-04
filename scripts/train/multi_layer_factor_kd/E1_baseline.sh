#!/bin/bash
# E1: Baseline — 기존 single-layer DAG-KD (MI_ablation best: ts only)

CUDA_VISIBLE_DEVICES=0 python train.py \
  --wandb_run multi_kd_E1_baseline \
  --out outputs/multi_layer_factor_kd/E1_baseline \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name stt_en_conformer_ctc_small \
  --use_ctc True \
  --use_logit_kd True \
  --use_layer_kd False \
  --use_flow True \
  --use_diffkd True \
  --use_disent True \
  --disent_spk_layers "4" \
  --disent_txt_layers "16" \
  --flow_steps 8 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1 \
  --disen_mi_pairs "ts" \
  --disen_lll_weight 1.0 \
  --disen_mi_weight 1.0 \
  --use_multi_layer_kd False \
  --cka_log_interval 500
