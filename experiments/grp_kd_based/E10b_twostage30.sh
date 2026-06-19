#!/bin/bash
# E10b: GRP-KD ver4 + Two-Stage Training (stage1=30 epochs)
# E4 구조 전체 유지 (disen_mode=3, grl_alpha=0.1)
# Stage 1 (epoch 0~29): KD + disen losses만 (CTC 제외) → feature pre-initialization
# Stage 2 (epoch 30~99): E4 전체 loss (CTC 포함)
#
# E10a(20ep) 대비 더 긴 pre-init 기간
# Stage 1이 길수록 feature 정렬은 충분하지만 CTC 복구 기간이 줄어드는 trade-off

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E10b_twostage30 \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E10b_twostage30 \
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
  --stage1_epochs 30 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
