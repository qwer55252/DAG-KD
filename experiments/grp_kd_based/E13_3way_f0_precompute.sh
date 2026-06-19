#!/bin/bash
# E13: GRP-KD ver4 + Two-Stage (stage1=25) + 3-Way Orthogonal Disentanglement
#       prosody anchor = 오프라인 precompute F0/energy (librosa.pyin)
#
# 사전 실행 필요:
#   python precompute_f0.py \
#     --manifest data/train_100/manifests/train.json \
#     --out      data/train_100/manifests/f0_stats_train.pt \
#     --workers  8
#
# E10c 대비 변경:
#   disen_mode=5  → enc_pros_t 추가, 3-way orth (text⊥spk⊥pros)
#   f0_stats_path → 발화별 [mean_f0, mean_energy] 텐서
#   pros_proj     → Linear(2, 96), jointly 학습
#   recon         → z_t_text + z_t_spk + z_t_pros → lat_dec
#
# 나머지 (stage1=25, grl_alpha=0.1, kd_alpha=0.1 등) E10c 완전 동일

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E13_3way_f0_precompute \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E13_3way_f0_precompute \
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
  --disen_mode 5 \
  --orth_weight 1.0 \
  --spk_cls_weight 1.0 \
  --grl_weight 1.0 \
  --grl_alpha 0.1 \
  --pros_orth_weight 1.0 \
  --pros_sup_weight 1.0 \
  --f0_stats_path data/train_100/manifests/f0_stats_train.pt \
  --stage1_epochs 25 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
