# g1_access_push

Stage-0 research repository for access-aware planning and object-feedback execution for
Unitree G1 bimanual non-prehensile pushing.

## Current scope

- Coordinate-frame contract and SE(2) transforms.
- Rectangular-box and doorway geometry.
- Frozen success/failure taxonomy.
- Simulator-agnostic WBC adapter interface.
- JSONL episode logging.
- Deterministic unit tests.

The project intentionally does not yet contain a planner, primitive library, recovery
policy, or custom Isaac Lab environment.

## Development

```bash
conda activate agile_env
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```
