#!/bin/bash
# E6: GRP-KD ver4 + Orth + SpkCls + GRL(teacher) + GRL(student)
# E4 구조 전체 유지 + z_s_text에도 GRL 추가 (disen_mode=4)
# z_s_text → GRL(α=0.1) → SpkCls_s → CE loss
# teacher와 student 양쪽 모두 speaker-free 표현 유도

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E6_student_grl \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E6_student_grl \
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
  --disen_mode 4 \
  --orth_weight 1.0 \
  --spk_cls_weight 1.0 \
  --grl_weight 1.0 \
  --grl_alpha 0.1 \
  --grl_s_weight 1.0 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
