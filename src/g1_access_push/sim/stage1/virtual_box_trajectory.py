"""Pure trajectory helpers for the Stage-1 virtual-box tests."""

from __future__ import annotations

import math

import torch


def straight_out_hold_back_displacement(
    step_index: int,
    *,
    amplitude_m: float,
    move_out_steps: int,
    hold_steps: int,
    move_back_steps: int,
) -> float:
    """Return a smooth 0 -> amplitude -> hold -> 0 displacement."""

    if not math.isfinite(amplitude_m) or amplitude_m <= 0.0:
        raise ValueError("amplitude_m must be finite and positive.")

    step_counts = {
        "move_out_steps": move_out_steps,
        "hold_steps": hold_steps,
        "move_back_steps": move_back_steps,
    }

    for name, count in step_counts.items():
        if count < 2:
            raise ValueError(f"{name} must be at least 2.")

    total_steps = move_out_steps + hold_steps + move_back_steps

    if not 0 <= step_index < total_steps:
        raise IndexError(f"step_index must be in [0, {total_steps}), got {step_index}.")

    if step_index < move_out_steps:
        phase = step_index / (move_out_steps - 1)
        return amplitude_m * 0.5 * (1.0 - math.cos(math.pi * phase))

    hold_end = move_out_steps + hold_steps

    if step_index < hold_end:
        return amplitude_m

    local_step = step_index - hold_end
    phase = local_step / (move_back_steps - 1)

    return amplitude_m * 0.5 * (1.0 + math.cos(math.pi * phase))


def virtual_surface_contact_targets(
    baseline_contact_positions: torch.Tensor,
    translation_in_pelvis: torch.Tensor,
) -> torch.Tensor:
    """Translate two fixed virtual-surface contacts as one rigid pair."""

    if tuple(baseline_contact_positions.shape) != (2, 3):
        raise ValueError(
            "baseline_contact_positions must have shape (2, 3), "
            f"got {tuple(baseline_contact_positions.shape)}."
        )

    if tuple(translation_in_pelvis.shape) != (3,):
        raise ValueError(
            f"translation_in_pelvis must have shape (3,), got {tuple(translation_in_pelvis.shape)}."
        )

    if not bool(torch.isfinite(baseline_contact_positions).all()):
        raise ValueError("Baseline contact positions must be finite.")

    if not bool(torch.isfinite(translation_in_pelvis).all()):
        raise ValueError("Translation must be finite.")

    baseline_center = baseline_contact_positions.mean(dim=0)
    contact_offsets = baseline_contact_positions - baseline_center.unsqueeze(0)
    target_center = baseline_center + translation_in_pelvis

    return target_center.unsqueeze(0) + contact_offsets
