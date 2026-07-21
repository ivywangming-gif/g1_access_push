"""Run WBC-AGILE eval.py with deterministic Isaac Lab compatibility fixes."""

from __future__ import annotations

import sys
from pathlib import Path

WBC_ROOT = Path("/root/autodl-tmp/robotics/third_party/WBC-AGILE")
OFFICIAL_EVAL = WBC_ROOT / "scripts/eval.py"

if not OFFICIAL_EVAL.is_file():
    raise FileNotFoundError(OFFICIAL_EVAL)

# eval.py imports cli_args from its own scripts directory.
sys.path.insert(0, str(OFFICIAL_EVAL.parent))

source = OFFICIAL_EVAL.read_text(encoding="utf-8")

needle = """    if hasattr(env_cfg, "eval"):
        env_cfg.eval()
"""

injection = needle + """
    # Stage-1 deterministic evaluation compatibility.
    #
    # Some WBC-AGILE eval() implementations set events=None, but our fixed
    # Isaac Lab version still constructs EventManager unconditionally.
    # Use an empty dict: no events, but still a valid manager configuration.
    if getattr(env_cfg, "events", None) is None:
        env_cfg.events = {}

    # Force deterministic flat-ground evaluation.
    env_cfg.seed = 42

    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "terrain"):
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None

    # Disable all observation corruption before ObservationManager is created.
    observations = getattr(env_cfg, "observations", None)
    if observations is not None:
        for group_name in ("policy", "teacher", "critic"):
            group = getattr(observations, group_name, None)
            if group is not None and hasattr(group, "enable_corruption"):
                group.enable_corruption = False

    print("[S1-COMPAT] seed=42")
    print("[S1-COMPAT] terrain=plane")
    print("[S1-COMPAT] observation_corruption=disabled")
    print("[S1-COMPAT] empty_events=" + str(env_cfg.events == {}))
"""

count = source.count(needle)

if count != 1:
    raise RuntimeError(
        f"Expected one eval() insertion point in {OFFICIAL_EVAL}, found {count}"
    )

source = source.replace(needle, injection)

namespace = {
    "__name__": "__main__",
    "__file__": str(OFFICIAL_EVAL),
    "__package__": None,
}

exec(
    compile(source, str(OFFICIAL_EVAL), "exec"),
    namespace,
)
