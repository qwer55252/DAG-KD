#!/bin/bash
# E18a: E10c 기반 hyperparameter 튜닝 — kd_alpha 강화
#
# E10c 분석 결과:
#   - GRL: per-layer CE=5.5 ≈ random(5.53) → 이미 목표 달성, 변경 불필요
#   - Logit KD: gradient 예산의 3.7%만 차지 (kd_alpha=0.1 × loss=118)
#   - FM+DiffKD: 이미 0.2%로 수렴 완료
#
# 변경: kd_alpha 0.1 → 0.3
#   - Logit KD 기여도: 3.7% → 11.1%
#   - Teacher 소프트 레이블(출력 분포)이 hard CTC 레이블보다
#     클래스 간 유사도 정보를 담고 있어 더 풍부한 ASR 신호 제공
#   - 나머지 파라미터(orth, grl, spk_cls, stage1_epochs) E10c와 완전 동일
#
# Stage 1 (epoch 0-24): KD + disen (GRL+orth+spk_cls), CTC 제외
# Stage 2 (epoch 25-99): 전체 loss (CTC 포함)

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E18a_kd_alpha03 \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E18a_kd_alpha03 \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name stt_en_conformer_ctc_small \
  --use_ctc True \
  --use_logit_distillation True \
  --kd_alpha 0.3 \
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
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
