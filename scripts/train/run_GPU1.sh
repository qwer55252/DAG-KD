#!/usr/bin/env bash

# 실패해도 다음 스크립트로 계속 진행
# (set -e 사용 안 함)

# 이 스크립트는 /workspace/DAG-KD 에서 실행된다고 가정

scripts=(
  "scripts/train/train_teacher_lll1e-3_mi1e-2.sh"
)

fail=0

for script in "${scripts[@]}"; do
  echo "==============================="
  echo "▶ Running ${script} ..."
  echo "==============================="

  bash "${script}"
  rc=$?

  if [ $rc -ne 0 ]; then
    echo "❌ FAILED ${script} (exit code: ${rc})"
    fail=$((fail + 1))
  else
    echo "✅ Finished ${script}"
  fi
  echo
done

echo "==============================="
echo "✅ All disen scripts finished. (failed: ${fail}/${#scripts[@]})"
echo "==============================="