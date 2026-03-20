#!/bin/bash
# E2: Cross-Reconstruction only (cross_recon=True, phys_norm=False)
# 가설: txt_emb+spk_emb로 teacher layer를 복원하면 두 표현의 분리가 더 의미 있어짐
python train.py \
  --wandb_run exp_cross_recon_phys_norm-E2_cross_recon \
  --out outputs/exp_cross_recon_phys_norm/E2_cross_recon \
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
  --use_cross_recon True \
  --use_phys_norm False
