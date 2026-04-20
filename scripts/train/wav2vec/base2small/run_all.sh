#!/bin/bash
# base2small 4개 실험 동시 실행 (GPU 0,1,2,3 각 1개)
#
# 실험 구성:
#   GPU 0 — E-A: CTC only       (student-only baseline)
#   GPU 1 — E-B: Logit KD       (output-level KD)
#   GPU 2 — E-C: GRP-KD         (AE+FM+Diffusion, ICASSP 2026 대조군)
#   GPU 3 — E-D: DAG-KD         (Layer KD + Disentanglement, 제안 방법)
#
# Teacher: facebook/wav2vec2-base-960h (12L, hidden=768)
# Student: half-base, random init     (12L, hidden=384, heads=6, ffn=1536, ~24M)
#
# Usage:
#   cd /workspace/DAG-KD
#   bash scripts/train/wav2vec/base2small/run_all.sh

export WANDB_API_KEY=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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /workspace/DAG-KD

echo "============================================================"
echo "[$(date)] base2small: 4개 실험 동시 시작"
echo "============================================================"

# GPU 점유 체크
RUNNING=$(ps aux | grep train_wav2vec | grep -v grep | wc -l)
if [ "$RUNNING" -gt 0 ]; then
    echo "ERROR: train_wav2vec 프로세스 $RUNNING 개가 이미 실행 중입니다."
    ps aux | grep train_wav2vec | grep -v grep
    exit 1
fi

mkdir -p outputs/wav2vec/base2small/e_ctc_only
mkdir -p outputs/wav2vec/base2small/e_logit_kd
mkdir -p outputs/wav2vec/base2small/e_grp_kd
mkdir -p outputs/wav2vec/base2small/e_dag_kd

# GPU 0: CTC only
echo "[$(date)] GPU 0: E-A CTC only 시작"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec --wandb_run b2s_ctc_only \
  --out outputs/wav2vec/base2small/e_ctc_only \
  --data_script ./librispeech_asr.py --data_cfg train_100 \
  --train_split train.clean.100 --val_split dev.clean --test_split test.clean \
  --teacher_name facebook/wav2vec2-base-960h \
  --student_name facebook/wav2vec2-base-960h \
  --random_init_student True \
  --student_hidden_size 384 --student_num_heads 6 --student_intermediate_size 1536 \
  --use_ctc True --use_logit_kd False --use_layer_kd False \
  --use_flow False --use_diffkd False --use_disent False \
  --use_grp_kd False --use_txt_spk_probe False \
  --batch_size 8 --num_workers 2 --epochs 100 --gpus 1 \
  --learning_rate 1e-4 --warmup_epochs 5 \
  2>&1 | tee outputs/wav2vec/base2small/e_ctc_only/train.log &

PID_A=$!

# GPU 1: Logit KD
echo "[$(date)] GPU 1: E-B Logit KD 시작"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec --wandb_run b2s_logit_kd \
  --out outputs/wav2vec/base2small/e_logit_kd \
  --data_script ./librispeech_asr.py --data_cfg train_100 \
  --train_split train.clean.100 --val_split dev.clean --test_split test.clean \
  --teacher_name facebook/wav2vec2-base-960h \
  --student_name facebook/wav2vec2-base-960h \
  --random_init_student True \
  --student_hidden_size 384 --student_num_heads 6 --student_intermediate_size 1536 \
  --use_ctc True --use_logit_kd True --kd_alpha 0.5 --kd_temperature 1.0 \
  --use_layer_kd False --use_flow False --use_diffkd False --use_disent False \
  --use_grp_kd False --use_txt_spk_probe False \
  --batch_size 8 --num_workers 2 --epochs 100 --gpus 1 \
  --learning_rate 1e-4 --warmup_epochs 5 --kd_warmup_epochs 10 \
  2>&1 | tee outputs/wav2vec/base2small/e_logit_kd/train.log &

PID_B=$!

# GPU 2: GRP-KD
echo "[$(date)] GPU 2: E-C GRP-KD 시작"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2 python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec --wandb_run b2s_grp_kd \
  --out outputs/wav2vec/base2small/e_grp_kd \
  --data_script ./librispeech_asr.py --data_cfg train_100 \
  --train_split train.clean.100 --val_split dev.clean --test_split test.clean \
  --teacher_name facebook/wav2vec2-base-960h \
  --student_name facebook/wav2vec2-base-960h \
  --random_init_student True \
  --student_hidden_size 384 --student_num_heads 6 --student_intermediate_size 1536 \
  --use_ctc True --use_logit_kd True --kd_alpha 0.1 --kd_temperature 1.0 \
  --use_layer_kd False --use_flow False --use_diffkd False --use_disent False \
  --use_grp_kd True \
  --grp_latent_dim 96 --grp_fm_steps 8 --grp_diff_steps 9 \
  --grp_rec_weight 1.0 --grp_gen_weight 1.0 \
  --use_txt_spk_probe False \
  --batch_size 8 --num_workers 2 --epochs 100 --gpus 1 \
  --learning_rate 1e-4 --warmup_epochs 5 --kd_warmup_epochs 10 \
  2>&1 | tee outputs/wav2vec/base2small/e_grp_kd/train.log &

PID_C=$!

# GPU 3: DAG-KD
echo "[$(date)] GPU 3: E-D DAG-KD 시작"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=3 python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec --wandb_run b2s_dag_kd \
  --out outputs/wav2vec/base2small/e_dag_kd \
  --data_script ./librispeech_asr.py --data_cfg train_100 \
  --train_split train.clean.100 --val_split dev.clean --test_split test.clean \
  --teacher_name facebook/wav2vec2-base-960h \
  --student_name facebook/wav2vec2-base-960h \
  --random_init_student True \
  --student_hidden_size 384 --student_num_heads 6 --student_intermediate_size 1536 \
  --use_ctc True --use_logit_kd False \
  --use_layer_kd True --layer_kd_alpha 0.5 \
  --use_flow False --use_diffkd False \
  --use_disent True \
  --tch_spk_layers "1,2" --tch_txt_layers "11,12" \
  --stu_spk_layers "1,2" --stu_txt_layers "11,12" \
  --use_txt_spk_probe True --phys_loss_lambda 1e-3 \
  --use_grp_kd False \
  --batch_size 8 --num_workers 2 --epochs 100 --gpus 1 \
  --learning_rate 1e-4 --warmup_epochs 5 --kd_warmup_epochs 10 \
  2>&1 | tee outputs/wav2vec/base2small/e_dag_kd/train.log &

PID_D=$!

echo ""
echo "============================================================"
echo "모든 실험 시작됨:"
echo "  PID $PID_A → GPU 0: E-A CTC only"
echo "  PID $PID_B → GPU 1: E-B Logit KD"
echo "  PID $PID_C → GPU 2: E-C GRP-KD"
echo "  PID $PID_D → GPU 3: E-D DAG-KD"
echo "============================================================"
echo "로그 확인:"
echo "  tail -f outputs/wav2vec/base2small/e_ctc_only/train.log"
echo "  tail -f outputs/wav2vec/base2small/e_logit_kd/train.log"
echo "  tail -f outputs/wav2vec/base2small/e_grp_kd/train.log"
echo "  tail -f outputs/wav2vec/base2small/e_dag_kd/train.log"
echo ""

# 모든 프로세스 종료 대기
wait $PID_A $PID_B $PID_C $PID_D
echo "[$(date)] 모든 실험 완료."
