#!/bin/bash
# base2small: random_init → pretrained 순차 실행 (총 8개 실험)
#
# [Batch 1] random_init=True  (GPU 0~3 동시, half-base student ~24M)
#   GPU 0 — E-A: CTC only
#   GPU 1 — E-B: Logit KD
#   GPU 2 — E-C: GRP-KD
#   GPU 3 — E-D: DAG-KD
#
# [Batch 2] random_init=False (GPU 0~3 동시, pretrained base student ~94M)
#   GPU 0 — E-A: CTC only
#   GPU 1 — E-B: Logit KD
#   GPU 2 — E-C: GRP-KD
#   GPU 3 — E-D: DAG-KD
#
# Usage:
#   cd /workspace/DAG-KD
#   nohup bash scripts/train/wav2vec/base2small/run_sequential.sh \
#     > outputs/wav2vec/base2small/run_sequential.log 2>&1 &

export WANDB_API_KEY=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

cd /workspace/DAG-KD

# ────────────────────────────────────────────────────────────
# 헬퍼: 4개 동시 실행 후 wait
# run_batch <rand_init: True|False> <suffix: random|pretrained>
# ────────────────────────────────────────────────────────────
run_batch() {
    local RAND=$1   # True / False
    local SUFFIX=$2 # random / pretrained

    # student arch: random init일 때만 절반 크기 적용
    local HIDDEN="-1"
    local HEADS="-1"
    local FFN="-1"
    if [ "$RAND" = "True" ]; then
        HIDDEN="384"
        HEADS="6"
        FFN="1536"
    fi

    echo ""
    echo "============================================================"
    echo "[$(date)] Batch: $SUFFIX (random_init=$RAND)"
    echo "============================================================"

    mkdir -p outputs/wav2vec/base2small/${SUFFIX}/e_ctc_only
    mkdir -p outputs/wav2vec/base2small/${SUFFIX}/e_logit_kd
    mkdir -p outputs/wav2vec/base2small/${SUFFIX}/e_grp_kd
    mkdir -p outputs/wav2vec/base2small/${SUFFIX}/e_dag_kd

    # --- GPU 0: CTC only ---
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python train_wav2vec.py \
      --wandb_project DAG-KD-wav2vec \
      --wandb_run b2s_${SUFFIX}_ctc_only \
      --out outputs/wav2vec/base2small/${SUFFIX}/e_ctc_only \
      --data_script ./librispeech_asr.py --data_cfg train_100 \
      --train_split train.clean.100 --val_split dev.clean --test_split test.clean \
      --teacher_name facebook/wav2vec2-base-960h \
      --student_name facebook/wav2vec2-base-960h \
      --random_init_student $RAND \
      --student_hidden_size $HIDDEN \
      --student_num_heads $HEADS \
      --student_intermediate_size $FFN \
      --use_ctc True --use_logit_kd False --use_layer_kd False \
      --use_flow False --use_diffkd False --use_disent False \
      --use_grp_kd False --use_txt_spk_probe False \
      --batch_size 8 --num_workers 2 --epochs 100 --gpus 1 \
      --learning_rate 1e-4 --warmup_epochs 5 \
      2>&1 | tee outputs/wav2vec/base2small/${SUFFIX}/e_ctc_only/train.log &
    local PID_A=$!

    # --- GPU 1: Logit KD ---
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 python train_wav2vec.py \
      --wandb_project DAG-KD-wav2vec \
      --wandb_run b2s_${SUFFIX}_logit_kd \
      --out outputs/wav2vec/base2small/${SUFFIX}/e_logit_kd \
      --data_script ./librispeech_asr.py --data_cfg train_100 \
      --train_split train.clean.100 --val_split dev.clean --test_split test.clean \
      --teacher_name facebook/wav2vec2-base-960h \
      --student_name facebook/wav2vec2-base-960h \
      --random_init_student $RAND \
      --student_hidden_size $HIDDEN \
      --student_num_heads $HEADS \
      --student_intermediate_size $FFN \
      --use_ctc True --use_logit_kd True --kd_alpha 0.5 --kd_temperature 1.0 \
      --use_layer_kd False --use_flow False --use_diffkd False --use_disent False \
      --use_grp_kd False --use_txt_spk_probe False \
      --batch_size 8 --num_workers 2 --epochs 100 --gpus 1 \
      --learning_rate 1e-4 --warmup_epochs 5 --kd_warmup_epochs 10 \
      2>&1 | tee outputs/wav2vec/base2small/${SUFFIX}/e_logit_kd/train.log &
    local PID_B=$!

    # --- GPU 2: GRP-KD ---
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2 python train_wav2vec.py \
      --wandb_project DAG-KD-wav2vec \
      --wandb_run b2s_${SUFFIX}_grp_kd \
      --out outputs/wav2vec/base2small/${SUFFIX}/e_grp_kd \
      --data_script ./librispeech_asr.py --data_cfg train_100 \
      --train_split train.clean.100 --val_split dev.clean --test_split test.clean \
      --teacher_name facebook/wav2vec2-base-960h \
      --student_name facebook/wav2vec2-base-960h \
      --random_init_student $RAND \
      --student_hidden_size $HIDDEN \
      --student_num_heads $HEADS \
      --student_intermediate_size $FFN \
      --use_ctc True --use_logit_kd True --kd_alpha 0.1 --kd_temperature 1.0 \
      --use_layer_kd False --use_flow False --use_diffkd False --use_disent False \
      --use_grp_kd True \
      --grp_latent_dim 96 --grp_fm_steps 8 --grp_diff_steps 9 \
      --grp_rec_weight 1.0 --grp_gen_weight 1.0 \
      --use_txt_spk_probe False \
      --batch_size 8 --num_workers 2 --epochs 100 --gpus 1 \
      --learning_rate 1e-4 --warmup_epochs 5 --kd_warmup_epochs 10 \
      2>&1 | tee outputs/wav2vec/base2small/${SUFFIX}/e_grp_kd/train.log &
    local PID_C=$!

    # --- GPU 3: DAG-KD ---
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=3 python train_wav2vec.py \
      --wandb_project DAG-KD-wav2vec \
      --wandb_run b2s_${SUFFIX}_dag_kd \
      --out outputs/wav2vec/base2small/${SUFFIX}/e_dag_kd \
      --data_script ./librispeech_asr.py --data_cfg train_100 \
      --train_split train.clean.100 --val_split dev.clean --test_split test.clean \
      --teacher_name facebook/wav2vec2-base-960h \
      --student_name facebook/wav2vec2-base-960h \
      --random_init_student $RAND \
      --student_hidden_size $HIDDEN \
      --student_num_heads $HEADS \
      --student_intermediate_size $FFN \
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
      2>&1 | tee outputs/wav2vec/base2small/${SUFFIX}/e_dag_kd/train.log &
    local PID_D=$!

    echo "실험 시작됨 (suffix=$SUFFIX):"
    echo "  PID $PID_A → GPU 0: CTC only"
    echo "  PID $PID_B → GPU 1: Logit KD"
    echo "  PID $PID_C → GPU 2: GRP-KD"
    echo "  PID $PID_D → GPU 3: DAG-KD"

    wait $PID_A $PID_B $PID_C $PID_D
    echo "[$(date)] Batch $SUFFIX 완료."
}

# ────────────────────────────────────────────────────────────
# Batch 1: random init (half-base, ~24M)
# ────────────────────────────────────────────────────────────
run_batch "True" "random"

# ────────────────────────────────────────────────────────────
# Batch 2: pretrained (full base, ~94M)
# ────────────────────────────────────────────────────────────
run_batch "False" "pretrained"

echo ""
echo "============================================================"
echo "[$(date)] 전체 8개 실험 완료."
echo "결과 경로:"
echo "  outputs/wav2vec/base2small/random/"
echo "  outputs/wav2vec/base2small/pretrained/"
echo "============================================================"
