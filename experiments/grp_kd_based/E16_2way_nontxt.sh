#!/bin/bash
# E16: GRP-KD ver4 + Two-Stage (stage1=25) + 2-Way Disentanglement (nontxt merged)
#
# E15 대비 변경:
#   disen_mode: 7 → 8
#   enc_spk_t + enc_pros_t (separate) → enc_nontxt_t (merged)
#   3-way orth (text⊥spk / text⊥pros / spk⊥pros) → 1-way orth (text⊥nontxt)
#   spk_cls: z_t_spk → z_t_nontxt
#   pros_sup: z_t_pros → z_t_nontxt (joint supervision)
#
# 가설: spk와 pros는 teacher feature 안에서 highly correlated.
#   E15의 spk⊥pros constraint가 이 상관성 때문에 pros_sup을 무력화시켰음.
#   둘을 하나의 nontxt 표현으로 합치고 text⊥nontxt만 강제하면
#   pros_sup이 살아남을 가능성이 높아짐.
#
# Stage 1 (epoch 0-24, CTC 없음):
#   KD + orth(text⊥nontxt) + pros_sup(frame-level F0/energy) + spk_cls + grl
# Stage 2 (epoch 25-100, full):
#   + CTC

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E16_2way_nontxt \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E16_2way_nontxt \
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
  --disen_mode 8 \
  --orth_weight 1.0 \
  --spk_cls_weight 1.0 \
  --grl_weight 1.0 \
  --grl_alpha 0.1 \
  --pros_sup_weight 1.0 \
  --f0_seq_path data/train_100/manifests/f0_seq_train.pt \
  --stage1_epochs 25 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
