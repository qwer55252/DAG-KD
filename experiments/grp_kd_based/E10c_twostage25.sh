#!/bin/bash
# E10c: GRP-KD ver4 + Two-Stage Training (stage1=25 epochs)
# E10a(20ep)와 E10b(30ep)의 중간값 탐색
# E10a: clean 좋음(10.88) but other 악화 / E10b: other 좋음(28.21) but clean 손해
# Stage 1 (epoch 0~24): KD + disen losses만 (CTC 제외)
# Stage 2 (epoch 25~99): E4 전체 loss (CTC 포함)

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E10c_twostage25 \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E10c_twostage25 \
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
  --stage1_epochs 25 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
