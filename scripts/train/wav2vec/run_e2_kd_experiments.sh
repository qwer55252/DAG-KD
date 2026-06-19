#!/bin/bash
# Sequential launcher for e2 KD experiments
# Runs: logit_kd → layer_kd → dag_kd one at a time

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY before running}"

set -e  # exit on error

cd /workspace/DAG-KD

echo "============================================================"
echo "[$(date)] Starting e2 KD experiments sequentially"
echo "============================================================"

# Verify no other training processes are running
RUNNING=$(ps aux | grep train_wav2vec | grep -v grep | wc -l)
if [ "$RUNNING" -gt 0 ]; then
    echo "ERROR: $RUNNING train_wav2vec processes already running! Kill them first."
    ps aux | grep train_wav2vec | grep -v grep
    exit 1
fi

# --- Experiment 1: CTC + Logit KD ---
echo ""
echo "[$(date)] === Starting e2_logit_kd ==="
OUT=outputs/wav2vec/e2_logit_kd
mkdir -p "$OUT"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec \
  --wandb_run wav2vec_e2_logit_kd \
  --out "$OUT" \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name facebook/wav2vec2-large-960h \
  --student_name facebook/wav2vec2-base-960h \
  --random_init_student True \
  --use_ctc True \
  --use_logit_kd True \
  --kd_alpha 0.5 \
  --kd_temperature 1.0 \
  --use_layer_kd False \
  --use_flow False \
  --use_diffkd False \
  --use_disent False \
  --use_txt_spk_probe False \
  --batch_size 4 \
  --epochs 100 \
  --gpus 4 \
  --learning_rate 1e-4 \
  --warmup_epochs 5 \
  --kd_warmup_epochs 10 \
  2>&1 | tee "$OUT/train.log"

echo "[$(date)] === e2_logit_kd DONE ==="

# --- Experiment 2: CTC + Layerwise KD ---
echo ""
echo "[$(date)] === Starting e2_layer_kd ==="
OUT=outputs/wav2vec/e2_layer_kd
mkdir -p "$OUT"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec \
  --wandb_run wav2vec_e2_layer_kd \
  --out "$OUT" \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name facebook/wav2vec2-large-960h \
  --student_name facebook/wav2vec2-base-960h \
  --random_init_student True \
  --use_ctc True \
  --use_logit_kd False \
  --use_layer_kd True \
  --layer_kd_alpha 0.5 \
  --use_flow False \
  --use_diffkd False \
  --use_disent False \
  --use_txt_spk_probe False \
  --batch_size 4 \
  --epochs 100 \
  --gpus 4 \
  --learning_rate 1e-4 \
  --warmup_epochs 5 \
  --kd_warmup_epochs 10 \
  2>&1 | tee "$OUT/train.log"

echo "[$(date)] === e2_layer_kd DONE ==="

# --- Experiment 3: CTC + DAG-KD ---
echo ""
echo "[$(date)] === Starting e2_dag_kd ==="
OUT=outputs/wav2vec/e2_dag_kd
mkdir -p "$OUT"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec \
  --wandb_run wav2vec_e2_dag_kd \
  --out "$OUT" \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name facebook/wav2vec2-large-960h \
  --student_name facebook/wav2vec2-base-960h \
  --random_init_student True \
  --use_ctc True \
  --use_logit_kd True \
  --kd_alpha 0.5 \
  --kd_temperature 1.0 \
  --use_layer_kd False \
  --use_flow False \
  --use_diffkd False \
  --use_disent True \
  --tch_spk_layers "8" \
  --tch_txt_layers "24" \
  --stu_spk_layers "4" \
  --stu_txt_layers "12" \
  --use_txt_spk_probe True \
  --batch_size 4 \
  --epochs 100 \
  --gpus 4 \
  --learning_rate 1e-4 \
  --warmup_epochs 5 \
  --kd_warmup_epochs 10 \
  2>&1 | tee "$OUT/train.log"

echo "[$(date)] === e2_dag_kd DONE ==="
echo ""
echo "============================================================"
echo "[$(date)] All e2 KD experiments completed!"
echo "============================================================"
