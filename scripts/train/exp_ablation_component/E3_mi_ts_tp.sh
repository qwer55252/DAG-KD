#!/bin/bash
# E3: MI(txt↔spk, txt↔pros) — ts+tp 쌍 사용
# [Table 1] MI ablation: txt↔spk + txt↔pros
# [Table 3] Rec(txt) on baseline (E9와 비교)
# 가정: pros-spk MI 없이 ts+tp만으로도 삼각 독립성이 충분한가?
python train.py \
  --wandb_run ablation-E3_mi_ts_tp \
  --out outputs/exp_ablation_component/E3_mi_ts_tp \
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
  --disen_mi_pairs "ts,tp" \
  --disen_lll_weight 1.0 \
  --disen_mi_weight 1.0 \
  --use_pros True \
  --use_mi True \
  --use_rec_loss True \
  --use_txt_rec_loss True \
  --use_phys_loss True \
  --use_mse_kd False
