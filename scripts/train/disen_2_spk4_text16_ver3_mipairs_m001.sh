mkdir -p outputs/disen/disen_2_spk4_text16_ver3_nomi_mi001 \
         outputs/disen/disen_2_spk4_text16_ver3_tp_mi001 \
         outputs/disen/disen_2_spk4_text16_ver3_tstp_mi001 \
         outputs/disen/disen_2_spk4_text16_ver3_nomi_mi0001

CUDA_VISIBLE_DEVICES=0 python train.py \
  --wandb_run disen_2_spk4_text16_ver3_nomi_mi001 \
  --out outputs/disen/disen_2_spk4_text16_ver3_nomi_mi001 \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name stt_en_conformer_ctc_small \
  --use_ctc True \
  --use_logit_kd True \
  --use_layer_kd False \
  --use_flow False \
  --use_diffkd False \
  --use_disent False \
  --disent_spk_layers "4" \
  --disent_txt_layers "16" \
  --flow_steps 8 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1 \
  --use_txt_spk_probe True \
  --txt_probe_lambda 1.0 \
  --txt_probe_lr 0.001 \
  --disen_lll_weight 1.0 \
  --disen_mi_weight 0.01 \
  2>&1 | tee outputs/disen/disen_2_spk4_text16_ver3_nomi_mi001/log.log

CUDA_VISIBLE_DEVICES=0 python train.py \
  --wandb_run disen_2_spk4_text16_ver3_tp_mi001 \
  --out outputs/disen/disen_2_spk4_text16_ver3_tp_mi001 \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name stt_en_conformer_ctc_small \
  --use_ctc True \
  --use_logit_kd True \
  --use_layer_kd False \
  --use_flow False \
  --use_diffkd False \
  --use_disent False \
  --disent_spk_layers "4" \
  --disent_txt_layers "16" \
  --flow_steps 8 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1 \
  --use_txt_spk_probe True \
  --txt_probe_lambda 1.0 \
  --txt_probe_lr 0.001 \
  --disen_mi_pairs "tp" \
  --disen_lll_weight 1.0 \
  --disen_mi_weight 0.01 \
  2>&1 | tee outputs/disen/disen_2_spk4_text16_ver3_tp_mi001/log.log

CUDA_VISIBLE_DEVICES=0 python train.py \
  --wandb_run disen_2_spk4_text16_ver3_tstp_mi001 \
  --out outputs/disen/disen_2_spk4_text16_ver3_tstp_mi001 \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name stt_en_conformer_ctc_small \
  --use_ctc True \
  --use_logit_kd True \
  --use_layer_kd False \
  --use_flow False \
  --use_diffkd False \
  --use_disent False \
  --disent_spk_layers "4" \
  --disent_txt_layers "16" \
  --flow_steps 8 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1 \
  --use_txt_spk_probe True \
  --txt_probe_lambda 1.0 \
  --txt_probe_lr 0.001 \
  --disen_lll_weight 1.0 \
  --disen_mi_weight 0.01 \
  2>&1 | tee outputs/disen/disen_2_spk4_text16_ver3_tstp_mi001/log.log

CUDA_VISIBLE_DEVICES=0 python train.py \
  --wandb_run disen_2_spk4_text16_ver3_nomi_mi0001 \
  --out outputs/disen/disen_2_spk4_text16_ver3_nomi_mi0001 \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name stt_en_conformer_ctc_small \
  --use_ctc True \
  --use_logit_kd True \
  --use_layer_kd False \
  --use_flow False \
  --use_diffkd False \
  --use_disent False \
  --disent_spk_layers "4" \
  --disent_txt_layers "16" \
  --flow_steps 8 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1 \
  --use_txt_spk_probe True \
  --txt_probe_lambda 1.0 \
  --txt_probe_lr 0.001 \
  --disen_lll_weight 1.0 \
  --disen_mi_weight 0.001 \
  2>&1 | tee outputs/disen/disen_2_spk4_text16_ver3_nomi_mi0001/log.log
