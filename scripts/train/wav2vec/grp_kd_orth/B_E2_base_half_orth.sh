#!/bin/bash
export WANDB_API_KEY=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
# Track B / E2 — wav2vec2 GRP-KD + Orthogonal disentanglement (disen_mode=1)
# B_E1 과 동일, grp_disen_mode=1 + orth + spk_cls 추가
# per-layer teacher 병렬 인코더(z_t_text / z_t_spk), text subspace에만 FM+Diff KD
# Conformer 대응: experiments/grp_kd_based/E2_disen_orth.sh
OUT=outputs/wav2vec/grp_kd_orth/B_E2_base_half_orth
mkdir -p "$OUT"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2,3 python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec \
  --wandb_run wav2vec_grp_kd_B_E2 \
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
  --grp_disen_mode 1 \
  --grp_orth_weight 1.0 \
  --grp_spk_cls_weight 1.0 \
  --batch_size 8 \
  --epochs 100 \
  --gpus 2 \
  --learning_rate 1.5e-4 \
  --warmup_epochs 5 \
  --kd_warmup_epochs 10 \
  2>&1 | tee "$OUT/train.log"
