#!/bin/bash
# E23: Cross-Covariance Decorrelation — orth 대체
#
# ── 논문 근거 ──────────────────────────────────────────────────────────────
# Zbontar, J. et al. (2021). "Barlow Twins: Self-Supervised Learning via
#   Redundancy Reduction." ICML 2021. arXiv:2103.03230
#
# Bardes, A. et al. (2022). "VICReg: Variance-Invariance-Covariance
#   Regularization for Self-Supervised Learning." ICLR 2022. arXiv:2105.04906
#
# ── E10c orth와의 차이 ──────────────────────────────────────────────────────
# orth:       E[(z_text · z_spk)²] → 0
#             두 벡터의 내적 (scalar 1개)를 0으로
#
# cross_cov:  C_ij = E[z_text_i × z_spk_j] / (σ_i × σ_j)  ∀i,j
#             ||C||_F² / D → 0
#             D×D = 9216개 pairwise correlation 전부 제거
#             선형 독립성 이론적으로 완전 보장
#
# ── 설정 ───────────────────────────────────────────────────────────────────
# orth_weight=0.0: orth 제거 (cross_cov로 대체)
# cross_cov_weight=1.0: E10c의 orth_weight=1.0과 동일 스케일
# 나머지 E10c와 완전 동일 (GRL+spk_cls+stage1=25)

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E23_cross_cov \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E23_cross_cov \
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
  --orth_weight 0.0 \
  --cross_cov_weight 1.0 \
  --spk_cls_weight 1.0 \
  --grl_weight 1.0 \
  --grl_alpha 0.1 \
  --stage1_epochs 25 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
