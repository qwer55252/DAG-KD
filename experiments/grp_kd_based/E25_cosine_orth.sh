#!/bin/bash
# E25: Cosine Orth — orth loss를 cos²(θ)로 교체
#
# ── 핵심 아이디어 ────────────────────────────────────────────────────────────
# 기존 orth: (z_t_text · z_t_spk)² = ‖z_t_text‖·‖z_t_spk‖·cos²(θ)
#   → 벡터 크기를 줄이면 cos²(θ)=1이어도 loss가 0에 가까워짐 (크기 우회 가능)
#
# E25 cosine orth: cos²(θ) only
#   zt_n = z_t_text / ‖z_t_text‖
#   zs_n = z_t_spk  / ‖z_t_spk‖
#   loss = (zt_n · zs_n)²  → 오직 방향만 제어, 크기 우회 불가
#
# ── 논문 근거 ──────────────────────────────────────────────────────────────
# Wang et al. NeurIPS 2020, "Understanding Contrastive Representation Learning
#   through Alignment and Uniformity on the Hypersphere"
#   unit hypersphere 위에서 각도 거리가 표현 학습의 자연스러운 척도임을 이론적 정립
#
# Schroff et al. CVPR 2015, "FaceNet: A Unified Embedding for Face Recognition"
#   L2 정규화 임베딩에서 cosine/angular distance가 metric learning의 핵심
#
# ── E24 대비 변경점 ─────────────────────────────────────────────────────────
# E24: orth_recon=True  (recon gradient 차단)
# E25: orth_recon=True + cosine_orth=True  (orth 자체도 크기 우회 불가)
# → recon gradient + orth gradient 양쪽 모두 speaker 방향 강화 불가

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E25_cosine_orth \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E25_cosine_orth \
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
  --cosine_orth True \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
