#!/bin/bash
# E21: Instance Normalization on z_t_text
#
# 핵심 아이디어:
#   z_t_text (B, latent_dim, T)에 InstanceNorm1d 적용
#   → 각 채널의 utterance-level mean/var 제거
#   → 화자별 채널 활성화 수준 차이 소거 → speaker style 제거
#
# 부가 효과:
#   recon_loss = dec(z_t_text_IN + z_t_spk) → teacher_feat 재구성 시
#   z_t_text에서 제거된 mean/var를 z_t_spk가 보상해야 함
#   → z_t_spk가 자연스럽게 speaker style을 더 많이 담게 유도됨
#   → disentanglement이 reconstruction 압력으로 자동 강화
#
# 논문 근거:
#   Huang & Belongie, ICCV 2017 (AdaIN):
#     "Style = channel-wise mean & variance of feature maps"
#     IN removes style while preserving content structure
#   StarGAN-VC2 (Kaneko et al., Interspeech 2019):
#     IN on encoder features → speaker-independent content representation
#
# E10c 기반 + use_text_in=True만 추가 (다른 파라미터 동일)

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E21_text_in \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E21_text_in \
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
  --use_text_in True \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
