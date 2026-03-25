#!/bin/bash
# E1: No Prosody — GST 제거, MI는 ts쌍만, Flow+DiffKD는 txt_emb 그대로 사용
# 가정: Prosody factor가 없어도 text/speaker 분리와 generative KD만으로 충분한가?
python train.py \
  --wandb_run ablation-E1_no_pros \
  --out outputs/exp_ablation_component/E1_no_pros \
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
  --use_txt_spk_probe True \
  --txt_probe_lambda 1.0 \
  --txt_probe_lr 0.001 \
  --disen_mi_pairs "ts,tp,ps" \
  --disen_lll_weight 1.0 \
  --disen_mi_weight 1.0 \
  --use_pros False \
  --use_mi True \
  --use_rec_loss True \
  --use_phys_loss True \
  --use_mse_kd False
