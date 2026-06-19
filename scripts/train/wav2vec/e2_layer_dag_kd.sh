#!/bin/bash
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY before running}"
# e2_layer_dag_kd: Random init student + CTC + Layer KD + Disentanglement (Fix #3 제안 방법)
# 근거:
#   - E2-C (Layer KD) = 18.79%  → E2-B (Logit KD) = 20.51% 보다 우수
#   - E2-D (Logit KD + Disent) = 25.50% → phys_loss 폭주로 실패
#   - Fix: Layer KD를 base로 + Disentanglement 적용 + phys_loss_lambda 수정
# 비교 기준:
#   - E2-C (Layer KD only): disentanglement 추가 효과 검증
#   - E2-D (Logit KD + Disent, 이전 실패): 구조 재설계 후 성능 변화 확인
OUT=outputs/wav2vec/e2_layer_dag_kd
mkdir -p "$OUT"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec \
  --wandb_run wav2vec_e2_layer_dag_kd \
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
  --use_disent True \
  --tch_spk_layers "8" \
  --tch_txt_layers "24" \
  --stu_spk_layers "4" \
  --stu_txt_layers "12" \
  --use_txt_spk_probe True \
  --phys_loss_lambda 1e-3 \
  --batch_size 4 \
  --epochs 100 \
  --gpus 4 \
  --learning_rate 1e-4 \
  --warmup_epochs 5 \
  --kd_warmup_epochs 10 \
  2>&1 | tee "$OUT/train.log"
