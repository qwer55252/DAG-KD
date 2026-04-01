#!/bin/bash
# E0: Minimal — 모든 disentanglement 컴포넌트 비활성 (공통 베이스라인)
# MI 없음 / Phys 없음 / Rec(txt) 없음 (k=5 proj) / Pros 없음
# CTC + LogitKD + Flow + DiffKD + SpkCE만 동작
python train.py \
  --wandb_run ablation-E0_minimal \
  --out outputs/exp_ablation_component/E0_minimal \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name stt_en_conformer_ctc_small \
  --use_ctc True \
  --use_logit_kd True \
  --use_layer_kd False \
  --use_flow True \
  --use_diffkd True \
  --use_disent True \
  --disent_spk_layers "4" \
  --disent_txt_layers "16" \
  --flow_steps 8 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1 \
  --use_txt_spk_probe False \
  --txt_probe_lambda 1.0 \
  --txt_probe_lr 0.001 \
  --disen_mi_pairs "ts,tp,ps" \
  --disen_lll_weight 1.0 \
  --disen_mi_weight 1.0 \
  --use_pros False \
  --use_mi False \
  --use_rec_loss False \
  --use_txt_rec_loss False \
  --use_phys_loss False \
  --use_mse_kd False
