#!/bin/bash
# base2small / E-A: CTC only (student-only baseline, no KD)
# Teacher: wav2vec2-base-960h (12L, hidden=768)
# Student: wav2vec2-base config, random init, hidden=384, heads=6, ffn=1536 (~24M)
# GPU: 0

export WANDB_API_KEY=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

OUT=outputs/wav2vec/base2small/e_ctc_only
mkdir -p "$OUT"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec \
  --wandb_run b2s_ctc_only \
  --out "$OUT" \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name facebook/wav2vec2-base-960h \
  --student_name facebook/wav2vec2-base-960h \
  --random_init_student True \
  --student_hidden_size 384 \
  --student_num_heads 6 \
  --student_intermediate_size 1536 \
  --use_ctc True \
  --use_logit_kd False \
  --use_layer_kd False \
  --use_flow False \
  --use_diffkd False \
  --use_disent False \
  --use_grp_kd False \
  --use_txt_spk_probe False \
  --batch_size 8 \
  --num_workers 2 \
  --epochs 100 \
  --gpus 1 \
  --learning_rate 1e-4 \
  --warmup_epochs 5 \
  2>&1 | tee "$OUT/train.log"
