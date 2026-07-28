#!/usr/bin/env bash
set -eo pipefail
set +u
export PYTHONPATH="${PYTHONPATH:-}"
source /root/autodl-tmp/robotics/autodl_env.sh
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate /root/autodl-tmp/conda/envs/agile_env
hash -r

: "${PACKAGE_ROOT:?PACKAGE_ROOT must be provided and must not exist}"
: "${REFERENCE:?REFERENCE must be provided}"
: "${REFERENCE_SHA256:?REFERENCE_SHA256 must be provided}"
: "${BEST_CHECKPOINT:?BEST_CHECKPOINT must be provided}"
: "${BEST_CHECKPOINT_SHA256:?BEST_CHECKPOINT_SHA256 must be provided}"
: "${BEST_CHECKPOINT_ITERATION:?BEST_CHECKPOINT_ITERATION must be provided}"
: "${DEVELOPMENT_SEED:?DEVELOPMENT_SEED must be provided}"

if [[ -e "${PACKAGE_ROOT}" ]]; then
  echo "PACKAGE_ROOT_ALREADY_EXISTS=${PACKAGE_ROOT}" >&2
  exit 2
fi
mkdir -p "${PACKAGE_ROOT}"
if [[ "$(sha256sum "${REFERENCE}" | awk '{print $1}')" != "${REFERENCE_SHA256}" ]]; then
  echo "REFERENCE_SHA_MISMATCH" >&2
  exit 2
fi
if [[ "$(sha256sum "${BEST_CHECKPOINT}" | awk '{print $1}')" != "${BEST_CHECKPOINT_SHA256}" ]]; then
  echo "BEST_CHECKPOINT_SHA_MISMATCH" >&2
  exit 2
fi

run_episode() {
  local label="$1"
  shift
  local episode_root="${PACKAGE_ROOT}/${label}"
  if [[ -e "${episode_root}" ]]; then
    echo "EPISODE_ROOT_ALREADY_EXISTS=${episode_root}" >&2
    return 2
  fi
  echo "VISUAL_EPISODE=${label} RUN_ROOT=${episode_root} CAMERAS_ENABLED=true"
  set +e
  RUN_ROOT="${episode_root}" bash \
    /root/autodl-tmp/robotics/projects/g1_access_push/scripts/stage2_isaac/run_s2_03t_evaluation_once.sh \
    --reference "${REFERENCE}" \
    --reference-sha256 "${REFERENCE_SHA256}" \
    --mode formal \
    --development-seed "${DEVELOPMENT_SEED}" \
    --visual-evidence \
    --visual-frame-stride 2 \
    "$@" 2>&1 | tee "${PACKAGE_ROOT}/${label}.console.log"
  local rc=${PIPESTATUS[0]}
  printf '%s\n' "${rc}" > "${PACKAGE_ROOT}/${label}.launcher_rc.txt"
  return "${rc}"
}

set +e
run_episode CLEAN_UNTRAINED_ACTOR \
  --checkpoint NONE --checkpoint-sha256 NONE \
  --controller-id actor --visual-evidence-id CLEAN_UNTRAINED_ACTOR
RC_CLEAN=$?
run_episode BEST_GAP_CHECKPOINT \
  --checkpoint "${BEST_CHECKPOINT}" \
  --checkpoint-sha256 "${BEST_CHECKPOINT_SHA256}" \
  --checkpoint-iteration "${BEST_CHECKPOINT_ITERATION}" \
  --controller-id actor --visual-evidence-id BEST_GAP_CHECKPOINT
RC_BEST=$?
run_episode ORIGINAL_S2_03_CONTROLLER \
  --checkpoint NONE --checkpoint-sha256 NONE \
  --controller-id original_s2_03 --visual-evidence-id ORIGINAL_S2_03_CONTROLLER
RC_ORIGINAL=$?
set -e

/root/miniconda3/bin/python \
  /root/autodl-tmp/robotics/projects/g1_access_push/scripts/stage2/derive_s2_03t_visual_summary.py \
  --package-root "${PACKAGE_ROOT}" \
  --screening-summary /root/autodl-tmp/robotics/projects/g1_access_push/reports/stage2/s2_03t_screening_summary.json \
  --best-checkpoint "${BEST_CHECKPOINT}" \
  --best-checkpoint-sha256 "${BEST_CHECKPOINT_SHA256}" \
  --best-checkpoint-iteration "${BEST_CHECKPOINT_ITERATION}" \
  --development-seed "${DEVELOPMENT_SEED}" \
  --launcher-rc-clean "${RC_CLEAN}" \
  --launcher-rc-best "${RC_BEST}" \
  --launcher-rc-original "${RC_ORIGINAL}"

if [[ "${RC_CLEAN}" -ne 0 || "${RC_BEST}" -ne 0 || "${RC_ORIGINAL}" -ne 0 ]]; then
  exit 1
fi
exit 0
