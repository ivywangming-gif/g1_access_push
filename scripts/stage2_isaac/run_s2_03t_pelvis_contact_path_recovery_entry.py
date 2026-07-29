#!/usr/bin/env python3
"""Thin entry point: start Isaac once, then execute the isolated campaign."""

from __future__ import annotations

import runpy
from pathlib import Path


RUNNER = Path(__file__).with_name("run_s2_03t_pelvis_contact_path_recovery.py")
namespace = runpy.run_path(str(RUNNER), run_name="s2_03t_pelvis_contact_path_recovery_runner")
namespace["OLD_RUN"] = Path(
    "/root/autodl-tmp/robotics/runs/g1_access_push/stage2/"
    "s2_03t_safe_chest_joint_reference_20260729_033538"
)
from s2_03t_pelvis_contact_campaign import main  # noqa: E402

return_code = main(namespace)
try:
    namespace["SIMULATION_APP"].close()
except Exception:
    pass
raise SystemExit(return_code)
