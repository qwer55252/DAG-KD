#!/bin/bash
# E9: GRP-KD ver4 + Orth + SpkCls + GRL(teacher) + Top-K Layer KD
# E4 구조 전체 유지 (disen_mode=3, grl_alpha=0.1), 상위 4개 레이어만 KD
#
# 현재 (E1~E8): 16개 레이어 전체 KD (layer 0~15)
# E9:           상위 4개 레이어만 KD (layer 12~15)
#   layer_idx는 전체 기준 유지 (12,13,14,15) → layer_disen_w 계산 정상 동작
#
# 근거: E7 결과에서 상위 레이어 제약을 줄였더니 오히려 WER 악화 →
#       상위 레이어가 ASR에 가장 중요한 정보를 담고 있음을 역설적으로 확인.
#       student(88-dim) capacity를 상위 4개 레이어에 집중시키면
#       KD 신호 quality가 높아져 WER 개선 기대.

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E9_topk_kd \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E9_topk_kd \
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
  --kd_top_k 8 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
