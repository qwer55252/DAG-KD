#!/bin/bash
# E5: MSE KD Only — Flow/DiffKD 제거, txt_emb vs student last layer 단순 MSE
# 가정: Generative KD(Flow/Diff)가 단순 MSE보다 실제로 효과가 있는가?
python train.py \
  --wandb_run ablation-E5_mse_kd \
  --out outputs/exp_ablation_component/E5_mse_kd \
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
  --use_txt_spk_probe True \
  --txt_probe_lambda 1.0 \
  --txt_probe_lr 0.001 \
  --disen_mi_pairs "ts,tp,ps" \
  --disen_lll_weight 1.0 \
  --disen_mi_weight 1.0 \
  --use_pros True \
  --use_mi True \
  --use_rec_loss True \
  --use_phys_loss True \
  --use_mse_kd True
