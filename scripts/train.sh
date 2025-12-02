python train.py \
  --wandb_run test \
  --out output/test \
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
  --flow_steps 8 \
  --batch_size 16 \
  --epochs 1 \
  --gpus 1 \
  --disent_spk_layers "1,2" \
  --disent_txt_layers "15,16" \
  # --test_mode True \


