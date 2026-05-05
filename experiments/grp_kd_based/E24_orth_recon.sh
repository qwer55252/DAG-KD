#!/bin/bash
# E24: Orthogonal Reconstruction — Gram-Schmidt projection in recon path
#
# ── 핵심 아이디어 ────────────────────────────────────────────────────────────
# recon = dec(z_t_text + z_t_spk) 경로에서 z_t_text의 speaker 방향 성분을
# Gram-Schmidt로 제거한 뒤 복원:
#
#   spk_unit       = z_t_spk / ||z_t_spk||
#   proj           = (z_t_text · spk_unit) * spk_unit
#   z_t_text_orth  = z_t_text - proj
#   recon          = dec(z_t_text_orth + z_t_spk)
#
# 수학적 보장:
#   ∂L_recon/∂z_t_text = (I - û_spk û_spk^T) · ∂L/∂z_t_text_orth
#   → recon gradient가 z_t_text를 speaker 방향으로 절대 밀 수 없음
#
# ── 논문 근거 ──────────────────────────────────────────────────────────────
# Ravfogel et al. ACL 2020, "Null It Out: Guarding Protected Attributes
#   by Iterative Nullspace Projection (INLP)"
#   보호 속성(speaker)의 방향을 null-space projection으로 제거.
#   INLP는 post-hoc + iterative, E24는 학습 중 forward pass에서 동적 적용.
#
# ── E10c 대비 변경점 ────────────────────────────────────────────────────────
# orth_recon=True 추가, 나머지 모든 파라미터 E10c와 동일
# (KD 타깃 z_t_text_d = z_t_text.detach() 는 projection 이전 값 → fm_pre 안전)

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E24_orth_recon \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E24_orth_recon \
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
  --stage1_epochs 25 \
  --orth_recon True \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
