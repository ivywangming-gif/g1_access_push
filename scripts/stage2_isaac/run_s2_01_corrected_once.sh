#!/usr/bin/env bash
set -eo pipefail
set +u

: "${RUN_ROOT:?RUN_ROOT must be provided}"
: "${SOURCE_ROOT:?SOURCE_ROOT must be provided}"

source /root/autodl-tmp/robotics/autodl_env.sh
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate /root/autodl-tmp/conda/envs/agile_env
hash -r
export PYTHONPATH="${SOURCE_ROOT}/src:${PYTHONPATH:-}"

mkdir -p "${RUN_ROOT}/process_rc"
cp "${SOURCE_ROOT}/configs/stage2/s2_01_box_stand_sanity.yaml" "${RUN_ROOT}/config.yaml"
cp "${SOURCE_ROOT}/reports/stage2/s2_01_resolved_config.json" "${RUN_ROOT}/resolved_config.json"
cp "${SOURCE_ROOT}/reports/stage2/s2_01_parameter_provenance.json" "${RUN_ROOT}/parameter_provenance.json"
git -C "${SOURCE_ROOT}" rev-parse HEAD > "${RUN_ROOT}/pre_run_commit_sha.txt"
date +%s > "${RUN_ROOT}/start_epoch_seconds.txt"

set +e
python "${SOURCE_ROOT}/scripts/stage2_isaac/run_s2_01_box_stand_sanity.py" \
  --run-root "${RUN_ROOT}" \
  --config "${RUN_ROOT}/config.yaml" \
  --resolved-config "${RUN_ROOT}/resolved_config.json" \
  --headless --enable_cameras --device cuda:0 \
  >"${RUN_ROOT}/stdout.log" 2>"${RUN_ROOT}/stderr.log"
runner_raw_rc=$?

python "${SOURCE_ROOT}/scripts/stage2/derive_s2_01_runner_status.py" \
  --run-root "${RUN_ROOT}" \
  --raw-rc "${runner_raw_rc}" \
  --expected-frames 3000 \
  >"${RUN_ROOT}/runner_supervisor.log" 2>&1
supervisor_rc=$?
if (( supervisor_rc != 0 )); then
  printf '%s\n' "${runner_raw_rc}" > "${RUN_ROOT}/process_rc/runner_raw.txt"
  printf '1\n' > "${RUN_ROOT}/process_rc/runner_effective.txt"
  printf '1\n' > "${RUN_ROOT}/process_rc/runner.txt"
fi
runner_effective_rc=$(<"${RUN_ROOT}/process_rc/runner_effective.txt")

python "${SOURCE_ROOT}/scripts/stage2/evaluate_s2_01_box_stand_sanity.py" \
  --run-root "${RUN_ROOT}" \
  --config "${RUN_ROOT}/config.yaml" \
  >"${RUN_ROOT}/evaluator_stdout.log" 2>"${RUN_ROOT}/evaluator_stderr.log"
evaluator_rc=$?
printf '%s\n' "${evaluator_rc}" > "${RUN_ROOT}/process_rc/evaluator.txt"
set -e

end_epoch=$(date +%s)
printf '%s\n' "${end_epoch}" > "${RUN_ROOT}/end_epoch_seconds.txt"
start_epoch=$(<"${RUN_ROOT}/start_epoch_seconds.txt")
printf '%s\n' "$((end_epoch - start_epoch))" > "${RUN_ROOT}/wall_time_seconds.txt"

if (( runner_effective_rc != 0 )); then
  exit "${runner_effective_rc}"
fi
exit "${evaluator_rc}"
