#!/bin/bash
# E5: GRP-KD ver4 + Orth + SpkCls + GRL(teacher) with DANN alpha annealing
# E4 구조 그대로, grl_alpha를 0→grl_alpha_max로 DANN-style로 annealing
# α(p) = grl_alpha_max × (2/(1+exp(-10p))-1), p = global_step / total_steps
# 초반엔 KD 신호 안정화, 후반에 adversarial pressure 증가

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E5_grl_anneal \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E5_grl_anneal \
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
  --grl_anneal True \
  --grl_alpha_max 1.0 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
