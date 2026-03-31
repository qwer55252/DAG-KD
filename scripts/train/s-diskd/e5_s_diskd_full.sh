#!/bin/bash
# E5: S-DisKD Full — E4 + Student-side CLUB MI 최소화 (txt ↔ spk)
# Student 내부에서도 text/speaker factor 독립성 강제
python train.py \
  --wandb_run s-diskd_e5_full \
  --out outputs/s-diskd/e5_s_diskd_full \
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
  --use_stu_txt_kd True \
  --stu_txt_kd_weight 1.0 \
  --use_stu_spk_kd True \
  --stu_spk_kd_weight 1.0 \
  --use_stu_club True \
  --stu_club_weight 1e-3
