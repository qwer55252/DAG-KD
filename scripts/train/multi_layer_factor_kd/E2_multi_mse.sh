#!/bin/bash
# E2: Multi-Layer Shared AE + MSE KD
# spk=[2,4,6], txt=[12,14,16] 각각 Shared enc/dec → Student 대응 레이어와 1:1 MSE KD

CUDA_VISIBLE_DEVICES=0 python train.py \
  --wandb_run multi_kd_E2_mse \
  --out outputs/multi_layer_factor_kd/E2_multi_mse \
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
  --disent_spk_layers "2,4,6" \
  --disent_txt_layers "12,14,16" \
  --flow_steps 8 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1 \
  --disen_mi_pairs "ts" \
  --disen_lll_weight 1.0 \
  --disen_mi_weight 1.0 \
  --use_multi_layer_kd True \
  --multi_kd_spk_layers "2,4,6" \
  --multi_kd_txt_layers "12,14,16" \
  --multi_layer_kd_type "mse" \
  --multi_layer_kd_weight 1.0 \
  --cka_log_interval 500
