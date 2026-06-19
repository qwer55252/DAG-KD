#!/bin/bash
# E26: Stage-Gated Disentanglement — Stage2 orth/GRL 감쇠
#
# ── 핵심 아이디어 ────────────────────────────────────────────────────────────
# E10c에서 Stage1/Stage2 구조가 핵심 성능 요인임을 확인했으나,
# Stage2에서도 orth(w=1.0)/GRL(w=1.0)이 계속 enc_text_t에 gradient를 준다.
# enc_text_t gradient 구조:
#   Stage1: recon + orth(1.0) + GRL(1.0)  — CTC 없음, disentanglement 집중
#   Stage2: recon + orth(1.0) + GRL(1.0)  — E10c와 동일 (이게 문제일 수 있음)
#
# E26은 Stage1에서 disentanglement를 충분히 확립한 뒤,
# Stage2에서 orth/GRL 가중치를 0.1로 줄여 recon이 z_t_text를 더 자유롭게 학습하게 한다.
#   Stage1: orth=1.0, grl=1.0  (E10c와 동일)
#   Stage2: orth=0.1, grl=0.1  (10배 감소)
#
# ── 논문 근거 ──────────────────────────────────────────────────────────────
# Yu et al., NeurIPS 2020, "Gradient Surgery for Multi-Task Learning"
#   멀티태스크 학습에서 태스크 간 gradient가 충돌할 때 성능 저하.
#   Stage2에서 orth/GRL gradient를 줄이면 recon의 z_t_text 학습 방해를 감소.
#
# Romero et al., ICLR 2015, "FaceNet (FitNets: Hints for Thin Deep Nets)"
#   Stage1에서 구조(hint)를 먼저 확립하고, Stage2에서 task 최적화.
#   Stage1 disentanglement가 "충분히" 확립되면 Stage2에서 유지 비용을 줄일 수 있음.
#
# ── E10c 대비 변경점 ────────────────────────────────────────────────────────
# E10c: orth_weight=1.0, grl_weight=1.0 → Stage1/Stage2 동일
# E26:  stage1=1.0/1.0, stage2=0.1/0.1  → Stage2에서 10배 감쇠

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E26_stage2_disen_decay \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E26_stage2_disen_decay \
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
  --stage2_orth_weight 0.1 \
  --stage2_grl_weight 0.1 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
