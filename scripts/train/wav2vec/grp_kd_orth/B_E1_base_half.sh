#!/bin/bash
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY before running}"
# Track B / E1 — wav2vec2 GRP-KD baseline (disen_mode=0)
# Teacher: wav2vec2-base-960h (12L, d=768, heads=12, ffn=3072)
# Student: wav2vec2-base-960h arch with half dims (12L, d=384, heads=6, ffn=1536)
# Layer alignment: teacher {1..12} ↔ student {1..12} (1:1)
# Conformer 대응: experiments/grp_kd_based/E1_ver4_baseline.sh
OUT=outputs/wav2vec/grp_kd_orth/B_E1_base_half
mkdir -p "$OUT"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec \
  --wandb_run wav2vec_grp_kd_B_E1 \
  --out "$OUT" \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name facebook/wav2vec2-base-960h \
  --student_name facebook/wav2vec2-base-960h \
  --random_init_student True \
  --load_pretrained_feature_extractor True \
  --freeze_feature_extractor True \
  --student_hidden_size 384 \
  --student_num_heads 6 \
  --student_intermediate_size 1536 \
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
  --grp_disen_mode 0 \
  --batch_size 8 \
  --epochs 100 \
  --gpus 2 \
  --learning_rate 1.5e-4 \
  --warmup_epochs 5 \
  --kd_warmup_epochs 10 \
  2>&1 | tee "$OUT/train.log"
