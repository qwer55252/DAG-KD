#!/bin/bash
# E1: GRP-KD ver4 baseline (AE + FM(pre) + NoiseAdapter + Diffusion + KD(post))
# asr_train_diffm.py ver4 그대로 재현
# Loss: CTC + kd_alpha * logit_KD + Σ(recon + fm_pre + kd_post) per layer

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E1_ver4 \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E1_ver4 \
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
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
