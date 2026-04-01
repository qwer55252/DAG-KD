#!/bin/bash
# E2: Cyclic Reconstruction (tp pair only)
# txt_emb(content) <-> pros_emb(noncontent) disentangle via GRL + L2 cyclic loss

CUDA_VISIBLE_DEVICES=0 python train.py \
  --wandb_run cyclic_tp \
  --out outputs/cyclic_disen/E2_cyclic_tp \
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
  --disen_mi_pairs "" \
  --disen_lll_weight 0.0 \
  --disen_mi_weight 0.0 \
  --use_cyclic True \
  --cyclic_pairs "tp" \
  --cyclic_weight 1e-2 \
  --cyclic_grl_alpha 0.1 \
  --cyclic_hidden_dim 128 \
  --cka_log_interval 500 \
  --tsne_log_interval 10
