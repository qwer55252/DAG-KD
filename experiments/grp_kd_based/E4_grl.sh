#!/bin/bash
# E4: GRP-KD ver4 + Orth + SpkCls(spk) + GRL on z_t_text
# E2 구조 그대로 유지 (enc_text_t + enc_spk_t + orth + SpkCls_spk)
# 추가: z_t_text → GRL → SpkCls_text → CE loss (speaker 못 맞추게)
# GRL이 enc_text_t에 역전된 gradient를 전달 → z_t_text에서 speaker 정보 적극 제거

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E4_grl \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E4_grl \
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
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
