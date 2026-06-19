#!/bin/bash
# C_E1 — GRP-KD baseline (Track A, large→base)
# 정보 분리 없음. A_E1 재현 → 14.2% 목표.
# GPU: $CUDA_VISIBLE_DEVICES (default 0,1)
export WANDB_API_KEY=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

OUT=outputs/wav2vec/grp_kd_disen_v1/C_E1_baseline
mkdir -p "$OUT"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec \
  --wandb_run grp_kd_disen_v1_C_E1 \
  --out "$OUT" \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name facebook/wav2vec2-large-960h \
  --student_name facebook/wav2vec2-base-960h \
  --random_init_student True \
  --load_pretrained_feature_extractor True \
  --freeze_feature_extractor True \
  --use_ctc True \
  --use_logit_kd True \
  --kd_alpha 0.1 \
  --kd_temperature 1.0 \
  --use_layer_kd False \
  --use_flow False \
  --use_diffkd False \
  --use_disent False \
  --use_txt_spk_probe False \
  --use_grp_kd True \
  --grp_latent_dim 96 \
  --grp_fm_steps 8 \
  --grp_diff_steps 9 \
  --grp_rec_weight 1.0 \
  --grp_gen_weight 1.0 \
  --use_speaker_adv False \
  --use_speaker_cond False \
  --batch_size 8 \
  --epochs 100 \
  --gpus 2 \
  --learning_rate 1.5e-4 \
  --warmup_epochs 5 \
  --kd_warmup_epochs 10 \
  2>&1 | tee "$OUT/train.log"
