#!/bin/bash
# E22: VIB as disentanglement replacement (orth/GRL 제거, VIB+spk_cls+recon으로 대체)
#
# ── 설계 근거 ───────────────────────────────────────────────────────────────
# E10c (orth+GRL+spk_cls): 외부 gradient로 speaker 정보를 "강제로 밀어냄"
# E22: VIB의 KL penalty가 z_t_text를 압축, recon이 z_t_spk로 speaker를 당김
#
# 핵심 메커니즘 (추가 loss 없이 자동 분리):
#   1. VIB KL penalty → z_t_text의 정보량 압축 (CTC에 필요 없는 것 제거)
#   2. recon = dec(z_t_text_VIB + z_t_spk) → z_t_spk가 z_t_text에서 빠진
#      speaker 통계를 보상해야 함 (complementarity 압력)
#   3. spk_cls → z_t_spk가 speaker-discriminative하도록 유지
#
#   GRL/orth 없이도 recon 압력으로 자연스럽게 분리 유도
#
# ── 논문 근거 ──────────────────────────────────────────────────────────────
# Alemi, A. A., Fischer, I., Dillon, J. V., & Murphy, K. (2017).
#   "Deep Variational Information Bottleneck." ICLR 2017. arXiv:1612.00410
#
#   IB 목표: max I(Z;Y) - β·I(X;Z)
#   Variational bound: I(Z;Y) ≥ E[log p(y|z)]
#                      I(X;Z) ≤ KL(q(z|x) ‖ p(z))
#   → Loss: E[-log p(y|z)] + β · KL(N(μ,σ²) ‖ N(0,I))
#
# vib_beta=0.01: add-on(1e-3) 대비 10배 → primary disentanglement 역할 수행
# orth_weight=0, grl_weight=0: geometric/adversarial constraint 제거
# spk_cls_weight=1.0: z_t_spk의 speaker-discriminativeness 유지

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E22_vib \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E22_vib \
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
  --grl_weight 0.0 \
  --spk_cls_weight 1.0 \
  --stage1_epochs 25 \
  --vib_beta 0.01 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
