#!/bin/bash
# E14: GRP-KD ver4 + Two-Stage (stage1=25) + 3-Way Disentanglement (GPRE prosody anchor)
#
# E10c 대비 변경:
#   disen_mode=6  → enc_pros_t = GlobalProsodyReferenceEncoder(mel→Conv2d×3+GRU)
#   pros_proj     → Linear(96→2), z_t_pros → [pred_f0, pred_energy] 프레임별 예측
#   f0_seq_path   → 프레임별 [f0_hz, energy] 시퀀스 텐서 (N, T_max, 2)
#
# 사전 실행 필요:
#   python precompute_f0.py \
#     --manifest data/train_100/manifests/train.json \
#     --out_seq  data/train_100/manifests/f0_seq_train.pt \
#     --workers  8
#
# Stage 1 (epoch 0-24, CTC 없음):
#   KD + orth + spk_cls + grl + pros_orth + pros_sup  ← GPRE 안정화
# Stage 2 (epoch 25-100, full):
#   + CTC
#
# 나머지 (stage1=25, grl_alpha=0.1, kd_alpha=0.1 등) E10c 완전 동일

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E14_3way_gpre \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E14_3way_gpre \
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
  --disen_mode 6 \
  --orth_weight 1.0 \
  --spk_cls_weight 1.0 \
  --grl_weight 1.0 \
  --grl_alpha 0.1 \
  --pros_orth_weight 1.0 \
  --pros_sup_weight 1.0 \
  --f0_seq_path data/train_100/manifests/f0_seq_train.pt \
  --stage1_epochs 25 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
