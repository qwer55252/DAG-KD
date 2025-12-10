#!/usr/bin/env bash
set -e  # 하나라도 실패하면 바로 종료

# 이 스크립트는 /workspace/DAG-KD 에서 실행된다고 가정
for script in \
    "scripts/train/disen_4_spk1_text16_ver2.sh" \
    "scripts/train/disen_4_spk4_text16_ver2.sh" \
    "scripts/train/disen_4_spk8_text16_ver2.sh" \
    "scripts/train/disen_4_spk12_text1516_ver2.sh"
do
    echo "==============================="
    echo "▶ Running ${script} ..."
    echo "==============================="
    bash "${script}"
    echo "▶ Finished ${script}"
    echo
done

echo "✅ All disen scripts finished."
