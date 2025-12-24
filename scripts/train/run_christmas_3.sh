mkdir -p outputs/disen/disen_4_spk4_text16_ver3_tsps_mi001_fmdf \
         outputs/disen/disen_4_spk4_text16_ver3_tsps_mi001_fm \
         outputs/disen/disen_4_spk4_text16_ver3_tsps_mi001_df \
         outputs/disen/disen_4_spk4_text16_ver3_tsps_mi001_nogen

CUDA_VISIBLE_DEVICES=3 python train.py \
  --wandb_run disen_4_spk4_text16_ver3_tsps_mi001_fmdf \
  --out outputs/disen/disen_4_spk4_text16_ver3_tsps_mi001_fmdf \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name stt_en_conformer_ctc_small \
  --use_ctc True \
  --use_logit_kd True \
  --use_layer_kd True \
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
  --disen_mi_pairs "ts,ps" \
  --disen_lll_weight 1.0 \
  --disen_mi_weight 0.01 \


CUDA_VISIBLE_DEVICES=3 python train.py \
  --wandb_run disen_4_spk4_text16_ver3_tsps_mi001_fm \
  --out outputs/disen/disen_4_spk4_text16_ver3_tsps_mi001_fm \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name stt_en_conformer_ctc_small \
  --use_ctc True \
  --use_logit_kd True \
  --use_layer_kd True \
  --use_flow True \
  --use_diffkd False \
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
  --disen_mi_pairs "ts,ps" \
  --disen_lll_weight 1.0 \
  --disen_mi_weight 0.01 \


CUDA_VISIBLE_DEVICES=3 python train.py \
  --wandb_run disen_4_spk4_text16_ver3_tsps_mi001_df \
  --out outputs/disen/disen_4_spk4_text16_ver3_tsps_mi001_df \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name stt_en_conformer_ctc_small \
  --use_ctc True \
  --use_logit_kd True \
  --use_layer_kd True \
  --use_flow False \
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
  --disen_mi_pairs "ts,ps" \
  --disen_lll_weight 1.0 \
  --disen_mi_weight 0.01 \


CUDA_VISIBLE_DEVICES=3 python train.py \
  --wandb_run disen_4_spk4_text16_ver3_tsps_mi001_nogen \
  --out outputs/disen/disen_4_spk4_text16_ver3_tsps_mi001_nogen \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name stt_en_conformer_ctc_small \
  --use_ctc True \
  --use_logit_kd True \
  --use_layer_kd True \
  --use_flow False \
  --use_diffkd False \
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
  --disen_mi_pairs "ts,ps" \
  --disen_lll_weight 1.0 \
  --disen_mi_weight 0.01 \
