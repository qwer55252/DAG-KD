#!/bin/bash
# E10a: GRP-KD ver4 + Two-Stage Training (stage1=20 epochs)
# E4 구조 전체 유지 (disen_mode=3, grl_alpha=0.1)
# Stage 1 (epoch 0~19): KD + disen losses만 (CTC 제외) → feature pre-initialization
# Stage 2 (epoch 20~99): E4 전체 loss (CTC 포함)
#
# 가설: CTC와 KD의 gradient 충돌을 순차 학습으로 해소
#       FitNets (Romero et al., 2015) 이론적 근거
#       student scratch 초기화이므로 Stage 1에서 teacher 방향으로 feature를 먼저 정렬

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E10a_twostage20 \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E10a_twostage20 \
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
  --stage1_epochs 20 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
