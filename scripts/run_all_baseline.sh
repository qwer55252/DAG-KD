#!/usr/bin/env bash
set -e  # 중간에 하나라도 실패하면 바로 스크립트 종료

for i in {1,5}; do
    script="baseline_${i}.sh"
    echo "==============================="
    echo "▶ Running ${script} ..."
    echo "==============================="
    bash "scripts/${script}"
    echo "▶ Finished ${script}"
    echo
done

echo "✅ All baseline scripts finished."
