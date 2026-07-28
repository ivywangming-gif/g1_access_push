#!/usr/bin/env bash
set -eo pipefail
set +u
export PYTHONPATH="${PYTHONPATH:-}"
source /root/autodl-tmp/robotics/autodl_env.sh
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate /root/autodl-tmp/conda/envs/agile_env
hash -r

: "${RUN_ROOT:?RUN_ROOT must be provided}"
EXTERNAL_LOG="${RUN_ROOT}.console.log"
set +e
python /root/autodl-tmp/robotics/projects/g1_access_push/scripts/stage2_isaac/train_s2_03t.py \
  --run-root "${RUN_ROOT}" --headless "$@" 2>&1 | tee "${EXTERNAL_LOG}"
RC=${PIPESTATUS[0]}
set -e
if [[ -d "${RUN_ROOT}" ]]; then
  mkdir -p "${RUN_ROOT}/process_rc"
  cp "${EXTERNAL_LOG}" "${RUN_ROOT}/console.log"
  printf '%s\n' "${RC}" > "${RUN_ROOT}/process_rc/training.txt"
fi
exit "${RC}"
