#!/bin/bash
# E3: Physical Normalization only (cross_recon=False, phys_norm=True)
# 가설: F0/energy z-score 정규화로 MSE scale 불균형 해소, voicing BCE로 이진 분류 개선
python train.py \
  --wandb_run exp_cross_recon_phys_norm-E3_phys_norm \
  --out outputs/exp_cross_recon_phys_norm/E3_phys_norm \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name stt_en_conformer_ctc_small \
  --use_ctc True \
  --use_logit_kd True \
  --use_layer_kd False \
  --use_flow True \
  --use_diffkd True \
  --use_disent True \
  --disent_spk_layers "4" \
  --disent_txt_layers "16" \
  --flow_steps 8 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1 \
  --use_txt_spk_probe True \
  --txt_probe_lambda 1.0 \
  --txt_probe_lr 0.001 \
  --disen_mi_pairs "ts,tp,ps" \
  --disen_lll_weight 1.0 \
  --disen_mi_weight 1.0 \
  --use_cross_recon False \
  --use_phys_norm True
