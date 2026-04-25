#!/bin/bash
# E15: GRP-KD ver4 + Two-Stage (stage1=25) + 3-Way Disentanglement
#       Teacher-Feature pros encoder + Frame-level F0/energy supervision
#
# E13 대비 변경:
#   pros_sup: 발화 단위 mean F0 → 프레임 단위 F0/energy (gradient collapse 방지)
# E14 대비 변경:
#   enc_pros_t: GPRE(mel) → Conv1d(teacher_dim, latent_dim, k=1) (비자명 orthogonality)
# E13/E14 공통 대비 변경:
#   pros_orth 분리: text⊥pros (pros_orth_loss) + spk⊥pros (spk_pros_orth_loss) 별도 로깅
#
# Stage 1 (epoch 0-24, CTC 없음):
#   KD + orth(text⊥spk) + pros_orth(text⊥pros) + spk_pros_orth(spk⊥pros)
#   + pros_sup(frame-level F0/energy) + spk_cls + grl
# Stage 2 (epoch 25-100, full):
#   + CTC

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E15_3way_tch_frm \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E15_3way_tch_frm \
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
  --disen_mode 7 \
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
