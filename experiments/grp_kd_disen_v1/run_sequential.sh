#!/bin/bash
# grp_kd_disen_v1 — 4개 실험 순차/병렬 실행
#
# 4 GPUs 가용 시 (2 GPUs/실험 × 2 동시):
#   Phase 1: C_E1 (GPU 0,1)  +  C_E2 (GPU 2,3)
#   Phase 2: C_E3 (GPU 0,1)  +  C_E4 (GPU 2,3)
#
# Usage:
#   cd /workspace/DAG-KD
#   nohup bash experiments/grp_kd_disen_v1/run_sequential.sh \
#     > outputs/wav2vec/grp_kd_disen_v1/run_sequential.log 2>&1 &

set -u

cd /workspace/DAG-KD

mkdir -p outputs/wav2vec/grp_kd_disen_v1

echo "============================================================"
echo "[$(date)] grp_kd_disen_v1: 4개 실험 (Track A, large→base)"
echo "============================================================"

# 사전 점유 체크
RUNNING=$(ps aux | grep "train_wav2vec.py" | grep -v grep | wc -l)
if [ "$RUNNING" -gt 0 ]; then
    echo "ERROR: train_wav2vec 프로세스 ${RUNNING}개 실행 중"
    ps aux | grep train_wav2vec | grep -v grep
    exit 1
fi

# ────────────────────────────────────────────────────────
# Phase 1: C_E1 (baseline) + C_E2 (speaker adv)
# ────────────────────────────────────────────────────────
echo ""
echo "[$(date)] Phase 1: C_E1 (GPU 0,1) + C_E2 (GPU 2,3)"

CUDA_VISIBLE_DEVICES=0,1 bash experiments/grp_kd_disen_v1/C_E1_baseline.sh    &
PID_E1=$!
CUDA_VISIBLE_DEVICES=2,3 bash experiments/grp_kd_disen_v1/C_E2_speaker_adv.sh &
PID_E2=$!

echo "  PID $PID_E1 → C_E1 baseline (GPU 0,1)"
echo "  PID $PID_E2 → C_E2 speaker adv (GPU 2,3)"

wait $PID_E1 $PID_E2
echo "[$(date)] Phase 1 완료."

# ────────────────────────────────────────────────────────
# Phase 2: C_E3 (speaker cond) + C_E4 (full)
# ────────────────────────────────────────────────────────
echo ""
echo "[$(date)] Phase 2: C_E3 (GPU 0,1) + C_E4 (GPU 2,3)"

CUDA_VISIBLE_DEVICES=0,1 bash experiments/grp_kd_disen_v1/C_E3_speaker_cond.sh &
PID_E3=$!
CUDA_VISIBLE_DEVICES=2,3 bash experiments/grp_kd_disen_v1/C_E4_full.sh         &
PID_E4=$!

echo "  PID $PID_E3 → C_E3 speaker cond (GPU 0,1)"
echo "  PID $PID_E4 → C_E4 full (GPU 2,3)"

wait $PID_E3 $PID_E4
echo "[$(date)] Phase 2 완료."

echo ""
echo "============================================================"
echo "[$(date)] grp_kd_disen_v1 전체 4개 실험 완료."
echo "결과 경로: outputs/wav2vec/grp_kd_disen_v1/"
echo "============================================================"
