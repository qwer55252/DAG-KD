#!/bin/bash
# E20: E10c (orth+GRL+spk_cls) + CLUB MI 조합
#
# 분석 배경:
#   - E10c (orth+GRL+spk_cls): dev_clean 10.59% — 기하학적 hard constraint 효과적
#   - E19 (pure CLUB MI, stage1 fix): dev_clean 11.26% — soft constraint만으론 부족
#
# 핵심 아이디어:
#   - orth: 선형 의존성을 강하게 제거 (내적=0 강제)
#   - CLUB MI: orth가 못 잡는 비선형 의존성까지 추가 억제
#   - 두 constraint가 상호보완: orth가 탐색 공간을 좁혀줘서 CLUB q 추정이 더 정확해짐
#
# CLUB Stage 1/2 분리 (E19에서 검증된 방식):
#   - Stage 1 (0-24): ll_loss ONLY → q 워밍업 (mi_upper 없음)
#   - Stage 2 (25-99): mi_upper 추가 (orth+GRL은 Stage 1부터 항상 활성)
#
# 논문 근거:
#   - CLUB (Cheng et al., NeurIPS 2020): MI upper bound minimization
#   - DRIT++ (Lee et al., IJCV 2020): combining multiple disentanglement constraints

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E20_orth_club_mi \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E20_orth_club_mi \
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
  --club_mi_weight 0.1 \
  --spk_cls_weight 1.0 \
  --grl_weight 1.0 \
  --grl_alpha 0.1 \
  --stage1_epochs 25 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
