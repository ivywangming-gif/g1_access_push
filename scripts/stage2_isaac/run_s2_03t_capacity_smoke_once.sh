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
python /root/autodl-tmp/robotics/projects/g1_access_push/scripts/stage2_isaac/run_s2_03t_capacity_smoke.py \
  --run-root "${RUN_ROOT}" --capacity-smoke --headless "$@" 2>&1 | tee "${EXTERNAL_LOG}"
RC=${PIPESTATUS[0]}
python -c "import json,sys; payload=json.loads(open(sys.argv[1], encoding=\"utf-8\").read()); status=payload[\"status\"]; print(f\"AUTHORITATIVE_STATUS={status} PATH={sys.argv[1]}\", flush=True); raise SystemExit(0 if status in sys.argv[2:] else 2)" "${RUN_ROOT}/capacity_smoke_result.json" PASS 2>&1 | tee -a "${EXTERNAL_LOG}"
STATUS_RC=${PIPESTATUS[0]}
if [[ "${STATUS_RC}" -ne 0 ]]; then
  RC=${STATUS_RC}
fi
set -e
if [[ -d "${RUN_ROOT}" ]]; then
  mkdir -p "${RUN_ROOT}/process_rc"
  cp "${EXTERNAL_LOG}" "${RUN_ROOT}/console.log"
  printf '%s\n' "${RC}" > "${RUN_ROOT}/process_rc/capacity_smoke.txt"
fi
exit "${RC}"
