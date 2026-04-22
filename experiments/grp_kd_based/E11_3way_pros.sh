#!/bin/bash
# E11: GRP-KD ver4 + Two-Stage (stage1=25) + 3-Way Orthogonal Disentanglement
# E10c 구조 전체 유지 + disen_mode=5 (Text / Speaker / Prosody 3-way orth)
#
# 신규:
#   enc_pros_t: teacher feature → z_t_pros (96-dim)
#   pros_ref: mel → GlobalProsodyReferenceEncoder → pros_ref_emb (96-dim)
#   orth 3쌍: text⊥spk + text⊥pros + spk⊥pros
#   GST supervision: MSE(z_t_pros.mean(-1), pros_ref_emb)
#   재구성: z_t_text + z_t_spk + z_t_pros → lat_dec (완전 3-way 분해)
#
# 가설: teacher feature를 3-way로 분리하면 z_t_text가 더 순수한 linguistic
#       representation이 되어 E10c 대비 WER 추가 개선

CUDA_VISIBLE_DEVICES=0 python train_grp_kd.py \
  --wandb_run grp_kd_E11_3way_pros \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E11_3way_pros \
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
  --stage1_epochs 25 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
