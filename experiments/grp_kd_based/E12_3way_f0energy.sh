#!/bin/bash
# E12: GRP-KD ver4 + Two-Stage (stage1=25) + 3-Way Orthogonal Disentanglement
# E11의 문제: pros_ref(GlobalProsodyReferenceEncoder)가 랜덤 초기화 후 frozen
#             → enc_pros_t가 random noise supervision에 맞추게 됨 → 성능 저하
#
# E12 개선:
#   pros_ref 완전 제거, 대신 F0 + RMS energy를 prosody anchor로 사용
#   - torchaudio.functional.detect_pitch_frequency → 발화 단위 평균 F0 (B, 1)
#   - signal.pow(2).mean.sqrt → RMS energy (B, 1)
#   - pros_proj = Linear(2, 96): 학습 가능한 투영 (F0+energy → latent)
#   - pros_sup_loss = MSE(enc_pros_t.mean(-1), pros_proj([f0, energy]))
#   - enc_pros_t와 pros_proj 모두 학습됨 (multi-view: teacher hidden ↔ acoustic)
#   orth 3쌍: text⊥spk + text⊥pros + spk⊥pros (변경 없음)
#   재구성: z_t_text + z_t_spk + z_t_pros → lat_dec (변경 없음)
#
# 가설: 의미 있는 acoustic anchor(F0/energy)로 enc_pros_t를 유도하면
#       prosody 분리가 실제로 일어나고 z_t_text가 더 순수한 linguistic 표현이 됨

CUDA_VISIBLE_DEVICES=0 /opt/venv/bin/python3 train_grp_kd.py \
  --wandb_run grp_kd_E12_3way_f0energy \
  --wandb_project GRP-based \
  --out outputs/grp_kd_based/E12_3way_f0energy \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name stt_en_conformer_ctc_small \
  --use_ctc True \
  --use_logit_distillation True \
  --kd_alpha 0.1 \
  --kd_temperature 1.0 \
  --model_version 4 \
  --latent_dim 96 \
  --diffusion_steps 9 \
  --flow_steps 8 \
  --kd_loss_type mse \
  --disen_mode 5 \
  --orth_weight 1.0 \
  --spk_cls_weight 1.0 \
  --grl_weight 1.0 \
  --grl_alpha 0.1 \
  --pros_orth_weight 1.0 \
  --pros_sup_weight 1.0 \
  --stage1_epochs 25 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
