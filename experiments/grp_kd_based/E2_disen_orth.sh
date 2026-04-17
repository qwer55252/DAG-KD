#!/bin/bash
# E2: GRP-KD ver4 + Latent Disentanglement (Orthogonal constraint + Speaker Classifier)
# Teacher/Student raw feature에서 병렬 인코더 (enc_text, enc_spk) 사용
# Orthogonality loss: (z_text * z_spk).sum().pow(2) 최소화
# Speaker Classifier: z_t_spk → CE loss
# FM(pre) + Diffusion: text subspace에만 적용 (z_s_text ↔ z_t_text)

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E2_disen_orth \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E2_disen_orth \
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
  --disen_mode 1 \
  --orth_weight 1.0 \
  --spk_cls_weight 1.0 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
