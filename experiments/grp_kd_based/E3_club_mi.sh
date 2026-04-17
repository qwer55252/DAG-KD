#!/bin/bash
# E3: GRP-KD ver4 + Latent Disentanglement (CLUB MI + Speaker Classifier)
# E2와 동일한 구조, orth_loss → CLUB MI upper bound로 대체
# CLUB: q(z_t_spk | z_t_text) 학습 후 MI upper bound 최소화 (K=8 negative)

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E3_club_mi \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E3_club_mi \
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
  --disen_mode 2 \
  --orth_weight 1.0 \
  --spk_cls_weight 1.0 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
