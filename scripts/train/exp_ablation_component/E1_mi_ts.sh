#!/bin/bash
# E1: MI(txt↔spk) only — ts 쌍만 사용
# [Table 1] MI ablation: txt↔spk만
# [Table 3] Rec(txt) on baseline (E7과 비교)
# 가정: txt와 spk 표현을 독립적으로 분리하는 ts MI만으로도 충분한가?
python train.py \
  --wandb_run ablation-E1_mi_ts \
  --out outputs/exp_ablation_component/E1_mi_ts \
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
  --use_txt_spk_probe False \
  --txt_probe_lambda 1.0 \
  --txt_probe_lr 0.001 \
  --disen_mi_pairs "ts" \
  --disen_lll_weight 1.0 \
  --disen_mi_weight 1.0 \
  --use_pros True \
  --use_mi True \
  --use_rec_loss True \
  --use_txt_rec_loss True \
  --use_phys_loss True \
  --use_mse_kd False
