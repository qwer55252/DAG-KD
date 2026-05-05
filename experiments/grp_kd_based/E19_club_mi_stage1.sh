#!/bin/bash
# E19: CLUB MI with Stage 1 variational network pre-training
#
# E3 실패 원인 분석:
#   - ll_loss(variational net q 학습)와 mi_upper(MI 최소화)가 동시에 실행
#   - q가 p(z_spk|z_text)를 추정하는 동안 encoder도 변화 → moving target
#   - 결과: club_mi=-15.6 (불가능한 값), WER 13.1%
#
# E19 핵심 변경:
#   - Stage 1 (epoch 0-24): ll_loss ONLY → q가 현재 encoder 분포에 수렴
#   - Stage 2 (epoch 25-99): mi_upper 추가 → q 안정화된 후 MI 최소화
#   - club_mi_weight=0.1 (E3의 orth_weight=1.0 대비 10배 낮춤)
#   - disen_mode=2 (orth/GRL/spk_cls 없음, pure CLUB MI)
#
# 논문 근거:
#   - CLUB (Cheng et al., 2020): MI upper bound, but requires stable q
#   - Two-stage training (E10c): Stage 1 feature pre-init이 최종 WER에 결정적
#
# 비교 대상: E3 (club_mi=-15.6, WER 13.1%), E10c (best, WER 10.59%)

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E19_club_mi_stage1 \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E19_club_mi_stage1 \
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
  --club_mi_weight 0.1 \
  --orth_weight 0.0 \
  --spk_cls_weight 0.0 \
  --grl_weight 0.0 \
  --stage1_epochs 25 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
