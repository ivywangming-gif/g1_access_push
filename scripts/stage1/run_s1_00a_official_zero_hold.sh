#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

AGILE_ROOT="${AGILE_PATH:-/root/autodl-tmp/robotics/third_party/WBC-AGILE}"
CONFIG="${PROJECT_ROOT}/configs/stage1/s1_00a_official_zero_hold.yaml"
MANIFEST="${PROJECT_ROOT}/configs/stage1/wbc_baseline.yaml"

RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/robotics/runs/g1_access_push/stage1}"
RUN_DIR="${RUN_ROOT}/s1_00a_official_zero_hold_$(date +%Y%m%d_%H%M%S)"

CHECKPOINT_RELATIVE="$(
python - "${MANIFEST}" <<'PY'
from pathlib import Path
import sys

import yaml

manifest = yaml.safe_load(Path(sys.argv[1]).read_text())
print(manifest["checkpoint"]["path_relative_to_agile"])
PY
)"

EXPECTED_SHA="$(
python - "${MANIFEST}" <<'PY'
from pathlib import Path
import sys

import yaml

manifest = yaml.safe_load(Path(sys.argv[1]).read_text())
print(manifest["checkpoint"]["sha256"])
PY
)"

CHECKPOINT="${AGILE_ROOT}/${CHECKPOINT_RELATIVE}"

echo "========== S1-00a PRECHECK =========="
echo "project_root=${PROJECT_ROOT}"
echo "agile_root=${AGILE_ROOT}"
echo "config=${CONFIG}"
echo "checkpoint=${CHECKPOINT}"
echo "python=$(command -v python)"
python --version

if [[ ! -d "${AGILE_ROOT}" ]]; then
    echo "S1-00a: FAIL_AGILE_ROOT_MISSING"
    exit 2
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "S1-00a: FAIL_CONFIG_MISSING"
    exit 2
fi

if [[ ! -s "${CHECKPOINT}" ]]; then
    echo "S1-00a: FAIL_CHECKPOINT_MISSING"
    exit 2
fi

ACTUAL_SHA="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"

echo "expected_checkpoint_sha=${EXPECTED_SHA}"
echo "actual_checkpoint_sha=${ACTUAL_SHA}"

if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
    echo "S1-00a: FAIL_CHECKPOINT_SHA"
    exit 2
fi

mkdir -p "${RUN_DIR}"

echo "run_dir=${RUN_DIR}"

export ISAACLAB_HEADLESS=1

cd "${AGILE_ROOT}"

timeout --signal=INT --kill-after=30s 600s \
python scripts/eval.py \
    --task Velocity-Height-G1-v0 \
    --checkpoint "${CHECKPOINT}" \
    --eval_config "${CONFIG}" \
    --num_envs 1 \
    --num_steps 3200 \
    --run_evaluation \
    --save_trajectories \
    --metrics_file "${RUN_DIR}/metrics.json" \
    --headless \
    --device cuda:0 \
    2>&1 | tee "${RUN_DIR}/console.log"

EVAL_RC=${PIPESTATUS[0]}

METRICS="${RUN_DIR}/metrics.json"
METADATA="${RUN_DIR}/trajectories/metadata.json"

PARQUET_COUNT="$(
find "${RUN_DIR}/trajectories" \
    -maxdepth 1 \
    -type f \
    -name 'episode_*.parquet' \
    2>/dev/null | wc -l
)"

echo
echo "========== S1-00a RESULT =========="
echo "eval_rc=${EVAL_RC}"
echo "metrics=${METRICS}"
echo "metadata=${METADATA}"
echo "parquet_count=${PARQUET_COUNT}"
echo "run_dir=${RUN_DIR}"

ARTIFACTS_OK=false

if [[ -s "${METRICS}" ]] \
    && [[ -s "${METADATA}" ]] \
    && [[ "${PARQUET_COUNT}" -eq 1 ]]; then
    ARTIFACTS_OK=true
fi

if [[ "${EVAL_RC}" -eq 0 ]] && [[ "${ARTIFACTS_OK}" == true ]]; then
    echo "S1-00a: PASS"
    exit 0
fi

if [[ "${EVAL_RC}" -eq 124 ]] && [[ "${ARTIFACTS_OK}" == true ]]; then
    echo "S1-00a: PASS_WITH_KNOWN_SHUTDOWN_TIMEOUT"
    exit 0
fi

echo "S1-00a: FAIL"
exit 1
