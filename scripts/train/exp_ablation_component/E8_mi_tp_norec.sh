#!/bin/bash
# E8: MI(txt↔pros) — Rec(txt) 없음, k=5 Conv1D projection 사용 (E2에서 txt rec 제거)
# [Table 3] Rec(txt) ablation: E2 vs E8
# 가정: tp MI에서 txt AE reconstruction이 없어도 k=5 projection만으로 동등한 성능이 나오는가?
python train.py \
  --wandb_run ablation-E8_mi_tp_norec \
  --out outputs/exp_ablation_component/E8_mi_tp_norec \
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
  --disen_mi_pairs "tp" \
  --disen_lll_weight 1.0 \
  --disen_mi_weight 1.0 \
  --use_pros True \
  --use_mi True \
  --use_rec_loss True \
  --use_txt_rec_loss False \
  --use_phys_loss True \
  --use_mse_kd False
