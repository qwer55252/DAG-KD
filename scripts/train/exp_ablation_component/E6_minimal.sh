#!/bin/bash
# E6: Minimal — Speaker emb + Speaker classifier + Text emb + MSE KD만 남김
# Prosody 없음, MI 없음, Reconstruction 없음, Flow/DiffKD 없음, Phys 없음
# 가장 단순한 구조: txt_emb(linear proj) → student last layer MSE + spk_ce + CTC + LogitKD
# 가정: 복잡한 손실 없이 이 최소 구성만으로도 의미있는 성능이 나오는가?
python train.py \
  --wandb_run ablation-E6_minimal \
  --out outputs/exp_ablation_component/E6_minimal \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name stt_en_conformer_ctc_small \
  --use_ctc True \
  --use_logit_kd True \
  --use_layer_kd False \
  --use_flow False \
  --use_diffkd False \
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
  --disen_mi_pairs "ts,tp,ps" \
  --disen_lll_weight 1.0 \
  --disen_mi_weight 1.0 \
  --use_pros False \
  --use_mi False \
  --use_rec_loss False \
  --use_phys_loss False \
  --use_mse_kd True
