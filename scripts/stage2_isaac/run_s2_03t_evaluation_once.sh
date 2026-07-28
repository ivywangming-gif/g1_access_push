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
python /root/autodl-tmp/robotics/projects/g1_access_push/scripts/stage2_isaac/evaluate_s2_03t_checkpoint.py \
  --run-root "${RUN_ROOT}" --headless --enable_cameras "$@" 2>&1 | tee "${EXTERNAL_LOG}"
RUNNER_RC=${PIPESTATUS[0]}
set -e
EVALUATOR_RC=NOT_RUN
if [[ "${RUNNER_RC}" -eq 0 ]]; then
  set +e
  python /root/autodl-tmp/robotics/projects/g1_access_push/scripts/stage2/evaluate_s2_03_attach_only.py \
    --run-root "${RUN_ROOT}" \
    --config /root/autodl-tmp/robotics/projects/g1_access_push/configs/stage2/s2_03_attach_only.yaml \
    2>&1 | tee -a "${EXTERNAL_LOG}"
  EVALUATOR_RC=${PIPESTATUS[0]}
  set -e
fi
if [[ -d "${RUN_ROOT}" ]]; then
  mkdir -p "${RUN_ROOT}/process_rc"
  cp "${EXTERNAL_LOG}" "${RUN_ROOT}/console.log"
  printf '%s\n' "${RUNNER_RC}" > "${RUN_ROOT}/process_rc/runner.txt"
  printf '%s\n' "${EVALUATOR_RC}" > "${RUN_ROOT}/process_rc/evaluator.txt"
fi
if [[ "${RUNNER_RC}" -ne 0 ]]; then
  exit "${RUNNER_RC}"
fi
exit "${EVALUATOR_RC}"
