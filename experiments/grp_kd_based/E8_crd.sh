#!/bin/bash
# E8: GRP-KD ver4 + Orth + SpkCls + GRL(teacher) + CRD
# E4 구조 전체 유지 (disen_mode=3, grl_alpha=0.1), CRD InfoNCE 추가
#
# CRD: z_s_text와 z_t_text를 mean pool 후 L2 normalize → InfoNCE loss
#   positive pair: 같은 발화의 (z_s_text_i, z_t_text_i)
#   negative pair: 같은 배치 내 다른 발화 (batch_size-1개)
#   → student가 teacher의 발화 간 유사도 구조를 학습 (speaker-invariant)
#
# 가설: FM+Diffusion(절대값 매칭) + CRD(상대 구조 보존)가 보완적으로 작용하여
#       student가 더 풍부한 linguistic representation을 학습 → clean/other 개선

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E8_crd \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E8_crd \
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
  --crd_weight 1.0 \
  --crd_temperature 0.07 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
