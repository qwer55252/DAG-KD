python inference.py \
  --ckpt_path /workspace/DAG-KD/outputs/disen_spk48_txt16/ctc+logitkd+layerkd/checkpoints/last.ckpt \
  --teacher_name stt_en_conformer_ctc_small \
  --gpus 1 \
  --device cuda \
  --manifest data/train_100/manifests/dev_clean.json \
  --batch_size 8 \
#   --save_jsonl outputs/inference/disen_spk48_txt16_ctc+logitkd+layerkd/dev_clean_hyp.jsonl
