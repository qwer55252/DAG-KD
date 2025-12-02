python train.py \
  --wandb_run baseline_ctc+logitkd \
  --out outputs/baseline/ctc+logitkd \
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
  --batch_size 32 \
  --epochs 100 \
  --gpus 1 

