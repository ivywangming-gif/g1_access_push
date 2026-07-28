#!/usr/bin/env bash
set -eo pipefail
set +u

: "${RUN_ROOT:?RUN_ROOT must be provided}"
: "${SOURCE_ROOT:?SOURCE_ROOT must be provided}"
: "${S2_03_RUN_MODE:?S2_03_RUN_MODE must be preflight or formal}"

case "${S2_03_RUN_MODE}" in
  preflight)
    runner_mode_flag="--preflight-only"
    ;;
  formal)
    runner_mode_flag="--formal"
    : "${S2_03_PREFLIGHT_RUN_ROOT:?S2_03_PREFLIGHT_RUN_ROOT must be provided for formal}"
    ;;
  *)
    printf 'unsupported S2_03_RUN_MODE=%s\n' "${S2_03_RUN_MODE}" >&2
    exit 64
    ;;
esac

source /root/autodl-tmp/robotics/autodl_env.sh
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate /root/autodl-tmp/conda/envs/agile_env
hash -r
export PYTHONPATH="${SOURCE_ROOT}/src:${PYTHONPATH:-}"

mkdir -p "${RUN_ROOT}/process_rc"
cp "${SOURCE_ROOT}/configs/stage2/s2_03_attach_only.yaml" "${RUN_ROOT}/config.yaml"
cp "${SOURCE_ROOT}/reports/stage2/s2_03_resolved_config.json" "${RUN_ROOT}/resolved_config.json"
cp "${SOURCE_ROOT}/reports/stage2/s2_03_parameter_provenance.json" "${RUN_ROOT}/parameter_provenance.json"
git -C "${SOURCE_ROOT}" rev-parse HEAD > "${RUN_ROOT}/pre_run_commit_sha.txt"
date +%s > "${RUN_ROOT}/start_epoch_seconds.txt"
printf '%s\n' "${S2_03_RUN_MODE}" > "${RUN_ROOT}/run_mode.txt"
sha256sum \
  "${SOURCE_ROOT}/scripts/stage2_isaac/run_s2_03_attach_only.py" \
  "${SOURCE_ROOT}/src/g1_access_push/sim/stage2/s2_03_attach_env.py" \
  "${SOURCE_ROOT}/src/g1_access_push/stage2/s2_03_contract.py" \
  "${SOURCE_ROOT}/configs/stage2/s2_03_attach_only.yaml" \
  > "${RUN_ROOT}/source_sha256_manifest.txt"

if [[ "${S2_03_RUN_MODE}" == "formal" ]]; then
  python - "${S2_03_PREFLIGHT_RUN_ROOT}" "${RUN_ROOT}/config.yaml" "${RUN_ROOT}/resolved_config.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
preflight = Path(sys.argv[1])
result = json.loads((preflight / "preflight_result.json").read_text())
if result.get("status") != "PASS":
    raise SystemExit("S2_03_PREFLIGHT_NOT_PASSED")
for path, key in ((Path(sys.argv[2]), "config_sha256"), (Path(sys.argv[3]), "resolved_config_sha256")):
    if hashlib.sha256(path.read_bytes()).hexdigest() != result.get(key):
        raise SystemExit(f"S2_03_PREFLIGHT_CONFIG_SHA_MISMATCH:{key}")
PY
fi

set +e
python "${SOURCE_ROOT}/scripts/stage2_isaac/run_s2_03_attach_only.py" \
  --run-root "${RUN_ROOT}" --config "${RUN_ROOT}/config.yaml" --resolved-config "${RUN_ROOT}/resolved_config.json" \
  "${runner_mode_flag}" --preflight-steps 3 --headless --enable_cameras --device cuda:0 \
  >"${RUN_ROOT}/stdout.log" 2>"${RUN_ROOT}/stderr.log"
runner_raw_rc=$?
python "${SOURCE_ROOT}/scripts/stage2/derive_s2_03_runner_status.py" \
  --run-root "${RUN_ROOT}" --raw-rc "${runner_raw_rc}" --mode "${S2_03_RUN_MODE}" \
  >"${RUN_ROOT}/runner_supervisor.log" 2>&1
supervisor_rc=$?
if (( supervisor_rc != 0 )) && [[ ! -f "${RUN_ROOT}/process_rc/runner_effective.txt" ]]; then
  printf '%s\n' "${runner_raw_rc}" > "${RUN_ROOT}/process_rc/runner_raw.txt"
  printf '1\n' > "${RUN_ROOT}/process_rc/runner_effective.txt"
  printf '1\n' > "${RUN_ROOT}/process_rc/runner.txt"
fi
runner_effective_rc=$(<"${RUN_ROOT}/process_rc/runner_effective.txt")
if [[ "${S2_03_RUN_MODE}" == "formal" ]]; then
  python "${SOURCE_ROOT}/scripts/stage2/evaluate_s2_03_attach_only.py" --run-root "${RUN_ROOT}" --config "${RUN_ROOT}/config.yaml" \
    >"${RUN_ROOT}/evaluator_stdout.log" 2>"${RUN_ROOT}/evaluator_stderr.log"
  evaluator_rc=$?
else
  preflight_status=$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status","INVALID"))' "${RUN_ROOT}/preflight_result.json" 2>/dev/null)
  [[ "${preflight_status}" == "PASS" ]]
  evaluator_rc=$?
fi
printf '%s\n' "${evaluator_rc}" > "${RUN_ROOT}/process_rc/evaluator.txt"
set -e
end_epoch=$(date +%s)
printf '%s\n' "${end_epoch}" > "${RUN_ROOT}/end_epoch_seconds.txt"
start_epoch=$(<"${RUN_ROOT}/start_epoch_seconds.txt")
printf '%s\n' "$((end_epoch - start_epoch))" > "${RUN_ROOT}/wall_time_seconds.txt"
if (( runner_effective_rc != 0 )); then exit "${runner_effective_rc}"; fi
exit "${evaluator_rc}"
