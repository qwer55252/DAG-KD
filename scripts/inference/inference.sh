python inference.py \
  --ckpt_path /workspace/DAG-KD/outputs/disen/disen_4_spk12_text16_ver2/checkpoints/last.ckpt \
  --gpus 1 \
  --batch_size 8 \
  --data_dir data \
  --data_cfg train_100 \
  --eval_data librispeech \