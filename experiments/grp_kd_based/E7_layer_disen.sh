#!/bin/bash
# E7: GRP-KD ver4 + Orth + SpkCls + GRL(teacher) + layer-selective disentanglement
# E4 구조 전체 유지 (disen_mode=3), layer_disen_decay=0.8 추가
# layer_weight = 1.0 - 0.8 * (layer_idx / 15): 하위 레이어 강한 제약, 상위 레이어 약한 제약
# 가설: 하위 레이어(speaker/acoustic 정보 多) → 강한 분리, 상위 레이어(linguistic 정보 多) → 약한 분리
# → other split 화자 다양성에서 더 robust한 표현 학습 기대

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E7_layer_disen \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E7_layer_disen \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name stt_en_conformer_ctc_small \
  --use_ctc True \
  --use_logit_distillation True \
  --kd_alpha 0.1 \
  --kd_temperature 1.0 \
  --model_version 4 \
  --latent_dim 96 \
  --diffusion_steps 9 \
  --flow_steps 8 \
  --kd_loss_type mse \
  --disen_mode 3 \
  --orth_weight 1.0 \
  --spk_cls_weight 1.0 \
  --grl_weight 1.0 \
  --grl_alpha 0.1 \
  --layer_disen_decay 0.8 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
