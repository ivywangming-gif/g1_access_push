#!/usr/bin/env bash
set -Eeuo pipefail

REPOSITORY_ROOT="/root/autodl-tmp/robotics/projects/g1_access_push"
CAMPAIGN_DRIVER="${REPOSITORY_ROOT}/scripts/stage2/s2_03t_redesign_campaign.py"

: "${RUN_ROOT:?RUN_ROOT must be provided}"
: "${TMUX_SESSION:?TMUX_SESSION must be provided}"
: "${IMPLEMENTATION_COMMIT:?IMPLEMENTATION_COMMIT must be provided}"

if [[ "${RUN_ROOT}" != /root/autodl-tmp/robotics/runs/g1_access_push/stage2/s2_03t_redesign_campaign_* ]]; then
  echo "RUN_ROOT_OUTSIDE_FROZEN_CAMPAIGN_NAMESPACE=${RUN_ROOT}" >&2
  exit 64
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "RUN_ROOT_ALREADY_EXISTS=${RUN_ROOT}" >&2
  exit 65
fi

mkdir -p "${RUN_ROOT}"
CAMPAIGN_LOG="${RUN_ROOT}/campaign.console.log"
LOCK_PATH="/tmp/g1_access_push_s2_03t_redesign_campaign.global.flock"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "DUPLICATE_CAMPAIGN_LOCKED=${LOCK_PATH}" >&2
  exit 73
fi

exec > >(tee -a "${CAMPAIGN_LOG}") 2>&1

CURRENT_COMMAND="ENVIRONMENT_ACTIVATION"
shell_exit_trap() {
  local rc=$?
  trap - EXIT ERR INT TERM
  set +e
  if [[ -f "${CAMPAIGN_DRIVER}" ]]; then
    /root/miniconda3/bin/python "${CAMPAIGN_DRIVER}" \
      --record-shell-exit \
      --run-root "${RUN_ROOT}" \
      --exit-code "${rc}" \
      --command "${CURRENT_COMMAND}" >/dev/null 2>&1
  fi
  exit "${rc}"
}
trap shell_exit_trap EXIT ERR
trap 'exit 130' INT
trap 'exit 143' TERM

# The shared environment setup is not nounset-clean.  The launcher itself remains
# strict, while only the sourced setup is given the compatibility window it needs.
set +u
export PYTHONPATH="${PYTHONPATH:-}"
source /root/autodl-tmp/robotics/autodl_env.sh
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate /root/autodl-tmp/conda/envs/agile_env
hash -r
set -u

cd "${REPOSITORY_ROOT}"
CURRENT_COMMAND="S2_03T_REDESIGN_CAMPAIGN_ORCHESTRATOR"
python "${CAMPAIGN_DRIVER}" \
  --run-root "${RUN_ROOT}" \
  --tmux-session "${TMUX_SESSION}" \
  --implementation-commit "${IMPLEMENTATION_COMMIT}" \
  --repository-root "${REPOSITORY_ROOT}" \
  --preservation-baseline /tmp/g1_s2_01_20260727_124751/worktree_before.json \
  --reference /root/autodl-tmp/robotics/runs/g1_access_push/stage2/s2_03t_contract_smoke_20260728_104740/precontact_reference.pt \
  --reference-sha256 1e537e075af7f888fb95a63501d3976d0d7740eff51c6bd9000f574b00cd0e6c
