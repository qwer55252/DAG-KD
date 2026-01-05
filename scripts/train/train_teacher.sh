EXP_NAME="train_ver2_teacher"

python3 train_teacher.py \
  --wandb_run $EXP_NAME \
  --out outputs/train_ver2/$EXP_NAME \
  --data_script ./librispeech_asr.py \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --val_split dev.clean \
  --test_split test.clean \
  --teacher_name stt_en_conformer_ctc_small \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1 \
  --use_txt_spk_probe True \
  --disen_mi_pairs ts,tp \
  --disen_lll_weight 0.01 \
  --disen_mi_weight 0.01 \
  --extra_eval_data gigaspeech \