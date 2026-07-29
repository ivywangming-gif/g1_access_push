#!/usr/bin/env bash
set -eo pipefail
set +u
export PYTHONPATH="${PYTHONPATH:-}"

: "${RUN_ROOT:?RUN_ROOT must be provided}"
: "${SOURCE_ROOT:?SOURCE_ROOT must be provided}"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "RUN_ROOT_ALREADY_EXISTS=${RUN_ROOT}" >&2
  exit 2
fi

source /root/autodl-tmp/robotics/autodl_env.sh
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate /root/autodl-tmp/conda/envs/agile_env
hash -r
export PYTHONPATH="${SOURCE_ROOT}/src:/root/autodl-tmp/robotics/third_party/WBC-AGILE:/root/autodl-tmp/robotics/third_party/WBC-AGILE/scripts:${PYTHONPATH:-}"

mkdir -p "${RUN_ROOT}/process_rc"
date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_ROOT}/started_at_utc.txt"
git -C "${SOURCE_ROOT}" rev-parse HEAD > "${RUN_ROOT}/pre_run_commit_sha.txt"
sha256sum \
  "${SOURCE_ROOT}/scripts/stage2_isaac/run_s2_03t_safe_chest_joint_reference.py" \
  "${SOURCE_ROOT}/src/g1_access_push/sim/stage2/s2_03t_safe_chest_joint_reference_env_cfg.py" \
  "${SOURCE_ROOT}/src/g1_access_push/stage2/s2_03t_safe_chest_joint_reference_contract.py" \
  "${SOURCE_ROOT}/configs/stage2/s2_03t_safe_chest_joint_reference.yaml" \
  > "${RUN_ROOT}/source_sha256_manifest.txt"

start_epoch=$(date +%s)
printf '%s\n' "${start_epoch}" > "${RUN_ROOT}/start_epoch_seconds.txt"
set +e
python "${SOURCE_ROOT}/scripts/stage2_isaac/run_s2_03t_safe_chest_joint_reference.py" \
  --run-root "${RUN_ROOT}" \
  --reference "/root/autodl-tmp/robotics/runs/g1_access_push/stage2/s2_03t_contract_smoke_20260728_104740/precontact_reference.pt" \
  --seed 42 --headless --enable_cameras --device cuda:0 \
  > "${RUN_ROOT}/console.log" 2>&1
runner_rc=$?
set -e
printf '%s\n' "${runner_rc}" > "${RUN_ROOT}/process_rc/runner.txt"
end_epoch=$(date +%s)
printf '%s\n' "${end_epoch}" > "${RUN_ROOT}/end_epoch_seconds.txt"
printf '%s\n' "$((end_epoch - start_epoch))" > "${RUN_ROOT}/wall_time_seconds.txt"
exit "${runner_rc}"
