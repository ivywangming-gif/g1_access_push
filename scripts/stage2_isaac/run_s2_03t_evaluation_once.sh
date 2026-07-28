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
python -c "import json,sys; payload=json.loads(open(sys.argv[1], encoding=\"utf-8\").read()); status=payload[\"status\"]; print(f\"AUTHORITATIVE_STATUS={status} PATH={sys.argv[1]}\", flush=True); raise SystemExit(0 if status in sys.argv[2:] else 2)" "${RUN_ROOT}/runner_status.json" COMPLETE 2>&1 | tee -a "${EXTERNAL_LOG}"
STATUS_RC=${PIPESTATUS[0]}
if [[ "${STATUS_RC}" -ne 0 ]]; then
  RUNNER_RC=${STATUS_RC}
fi
set -e
EVALUATOR_RC=NOT_RUN
if [[ "${RUNNER_RC}" -eq 0 ]]; then
  set +e
  python /root/autodl-tmp/robotics/projects/g1_access_push/scripts/stage2/evaluate_s2_03_attach_only.py \
    --run-root "${RUN_ROOT}" \
    --config /root/autodl-tmp/robotics/projects/g1_access_push/configs/stage2/s2_03_attach_only.yaml \
    2>&1 | tee -a "${EXTERNAL_LOG}"
  EVALUATOR_RC=${PIPESTATUS[0]}
  python -c "import json,sys; payload=json.loads(open(sys.argv[1], encoding=\"utf-8\").read()); status=payload[\"status\"]; print(f\"AUTHORITATIVE_STATUS={status} PATH={sys.argv[1]}\", flush=True); raise SystemExit(0 if status in sys.argv[2:] else 2)" "${RUN_ROOT}/result.json" PASS FAIL 2>&1 | tee -a "${EXTERNAL_LOG}"
  STATUS_RC=${PIPESTATUS[0]}
  if [[ "${STATUS_RC}" -ne 0 ]]; then
    EVALUATOR_RC=${STATUS_RC}
  fi
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
