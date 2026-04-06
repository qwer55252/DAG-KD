#!/bin/bash
# E5: Multi-Layer Cyclic Reconstruction (GRL, 3쌍) + txt/spk KD (generative)
# MI 항 전부 비활성 → cyclic reconstruction으로 대체
# spk KD 유지 (multi_kd_use_spk True)

CUDA_VISIBLE_DEVICES=0 python train.py \
  --wandb_run multi_kd_E5_cyclic_spk_txt \
  --out outputs/multi_layer_factor_kd/E5_cyclic_spk_txt \
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
  --disent_spk_layers "2,4,6" \
  --disent_txt_layers "12,14,16" \
  --flow_steps 8 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1 \
  --disen_mi_pairs "" \
  --disen_lll_weight 0.0 \
  --disen_mi_weight 0.0 \
  --use_cyclic True \
  --cyclic_pairs "ts" \
  --cyclic_weight 5e-2 \
  --cyclic_grl_alpha 0.1 \
  --cyclic_hidden_dim 128 \
  --use_multi_layer_kd True \
  --multi_kd_spk_layers "2,4,6" \
  --multi_kd_txt_layers "12,14,16" \
  --multi_layer_kd_type "generative" \
  --multi_layer_kd_weight 1.0 \
  --multi_kd_use_spk True \
  --cka_log_interval 500
