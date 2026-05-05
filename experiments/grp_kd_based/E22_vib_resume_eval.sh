#!/bin/bash
# E22 resume — epoch 99 완료 후 inference segfault로 WER 미수집
# --resume_ckpt로 last.ckpt 로드, max_epochs=100 이미 달성 → training skip, eval만 실행

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E22_vib \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E22_vib \
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
  --orth_weight 0.0 \
  --grl_weight 0.0 \
  --spk_cls_weight 1.0 \
  --stage1_epochs 25 \
  --vib_beta 0.01 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1 \
  --resume_ckpt outputs/grp_kd_based/E22_vib/checkpoints/last.ckpt
