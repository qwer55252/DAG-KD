#!/bin/bash
# E2: No MI — CLUB MI 손실 제거, Reconstruction + Speaker CE + Flow/DiffKD 유지
# 가정: 독립성 강제 없이 재구성 손실만으로도 factor 분리가 되는가?
python train.py \
  --wandb_run ablation-E2_no_mi \
  --out outputs/exp_ablation_component/E2_no_mi \
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
  --use_pros True \
  --use_mi False \
  --use_rec_loss True \
  --use_phys_loss True \
  --use_mse_kd False
