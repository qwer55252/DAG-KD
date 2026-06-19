#!/bin/bash
# E17: GRP-KD ver4 + Two-Stage (stage1=25) + 2-Way Disentanglement
#       + A: F0-GRL on z_t_text
#       + B: Speaker-normalized F0 targets
#
# E16 대비 변경:
#   disen_mode: 8 → 9
#   A: f0_pred_text(grl(z_t_text)) — adversarially push F0 out of text encoder
#      (speaker GRL과 동일 GRL layer 재사용, 동일 grl_weight 적용)
#   B: pros_sup 타겟 = (F0 - speaker_mean_F0) / speaker_std_F0
#      (within-speaker pitch variation만 남김; spk_cls와 pros_sup이 다른 정보 학습)
#
# 가설:
#   E16 실패 원인: enc_text_t가 우연히 F0 방향을 선점 → enc_nontxt_t의 pros_sup이 죽음
#   A: GRL로 text에서 F0를 명시적으로 제거 → nontxt가 F0 방향을 확보
#   B: 화자별 pitch range 제거 → spk_cls와 pros_sup의 gradient 충돌 감소
#
# Stage 1 (epoch 0-24, CTC 없음):
#   KD + orth(text⊥nontxt) + pros_sup(spk-norm F0) + spk_cls + spk-GRL + f0-GRL
# Stage 2 (epoch 25-100, full):
#   + CTC

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E17_f0grl_spknorm \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E17_f0grl_spknorm \
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
  --disen_mode 9 \
  --orth_weight 1.0 \
  --spk_cls_weight 1.0 \
  --grl_weight 1.0 \
  --grl_alpha 0.1 \
  --pros_sup_weight 1.0 \
  --f0_seq_path data/train_100/manifests/f0_seq_train.pt \
  --stage1_epochs 25 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
