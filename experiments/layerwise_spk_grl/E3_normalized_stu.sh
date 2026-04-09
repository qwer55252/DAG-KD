#!/bin/bash
# E3: Layerwise Spk GRL + Normalized Student KD + λ_rec 감소
# E2 문제: spk_grl_stu=50 (scale mismatch) → F.normalize로 방향 정렬
# λ_rec=0.1로 rec_loss 비중 줄여 enc_i가 adv signal에 더 집중

CUDA_VISIBLE_DEVICES=0 python train.py \
  --wandb_run layerwise_grl_E3_norm_stu \
  --out outputs/layerwise_spk_grl/E3_normalized_stu \
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
  --use_disent False \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1 \
  --use_layerwise_spk_grl True \
  --spk_grl_alpha 0.1 \
  --spk_grl_adv_weight 0.1 \
  --spk_grl_rec_weight 0.1 \
  --spk_grl_normalize_stu True
