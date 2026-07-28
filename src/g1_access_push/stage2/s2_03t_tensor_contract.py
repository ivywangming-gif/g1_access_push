"""Dependency-light tensor contracts shared by S2-03T runtime and tests."""

from __future__ import annotations

import torch


def filtered_contact_activity(force_matrix: torch.Tensor | None, num_envs: int) -> torch.Tensor:
    """Fail closed unless a finite one-body filtered contact matrix is available."""

    if force_matrix is None:
        raise RuntimeError("FORBIDDEN_CONTACT_FORCE_MATRIX_MISSING")
    expected_prefix = (num_envs, 1)
    if (
        force_matrix.ndim != 4
        or tuple(force_matrix.shape[:2]) != expected_prefix
        or force_matrix.shape[2] < 1
        or force_matrix.shape[3] != 3
    ):
        raise RuntimeError(f"FORBIDDEN_CONTACT_FORCE_MATRIX_SHAPE:{tuple(force_matrix.shape)}")
    if not bool(torch.isfinite(force_matrix).all()):
        raise RuntimeError("FORBIDDEN_CONTACT_FORCE_MATRIX_NONFINITE")
    force_norms = torch.linalg.vector_norm(force_matrix, dim=-1).flatten(start_dim=1)
    return force_norms.max(dim=-1).values > 1.0e-6
