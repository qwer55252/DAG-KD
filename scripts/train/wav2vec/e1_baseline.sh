#!/bin/bash
# wav2vec2 E1: Baseline
# CTC + Logit KD만 사용 (Disentanglement / Generative KD 없음)
# 비교군: wav2vec2-large-960h → wav2vec2-base-960h 단순 KD
python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec \
  --wandb_run wav2vec_e1_baseline \
  --out outputs/wav2vec/e1_baseline \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name facebook/wav2vec2-large-960h \
  --student_name facebook/wav2vec2-base-960h \
  --use_ctc True \
  --use_logit_kd True \
  --use_layer_kd False \
  --use_flow False \
  --use_diffkd False \
  --use_disent False \
  --tch_spk_layers "1,2" \
  --tch_txt_layers "23,24" \
  --stu_spk_layers "1,2" \
  --stu_txt_layers "11,12" \
  --batch_size 8 \
  --epochs 100 \
  --gpus 1 \
  --learning_rate 3e-4
