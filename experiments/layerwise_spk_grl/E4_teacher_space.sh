#!/bin/bash
# E4: Layerwise Spk GRL — Teacher Space KD
# E2/E3 문제: enc_i (176→88) 압축으로 정보 손실 → KD 타겟 품질 저하
# E4 수정:
#   - enc_i: 176→176 (동일 차원, 정보 손실 없음)
#   - decoder 제거: MSE(enc_i(t), t) 로 content 보존
#   - stu_loss: shared_proj (88→176) 후 teacher space에서 MSE
#   → student가 정보 손실 없는 spk-free teacher 표현을 teacher space에서 학습

CUDA_VISIBLE_DEVICES=0 python train.py \
  --wandb_run layerwise_grl_E4_teacher_space \
  --out outputs/layerwise_spk_grl/E4_teacher_space \
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
  --spk_grl_rec_weight 1.0
