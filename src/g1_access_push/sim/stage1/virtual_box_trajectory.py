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


def derive_equal_budget_small_arc(
    *,
    contact_separation_m: float,
    local_motion_budget_m: float,
) -> dict[str, float]:
    """Derive a conservative arc from an already validated motion budget.

    Half the budget is assigned to midpoint arc length. The other
    half is assigned to the rotation-induced chord displacement of
    either contact about the pair midpoint.
    """

    values = (
        contact_separation_m,
        local_motion_budget_m,
    )

    if not all(math.isfinite(value) for value in values):
        raise ValueError("Arc geometry values must be finite.")

    if contact_separation_m <= 0.0:
        raise ValueError("contact_separation_m must be positive.")

    if local_motion_budget_m <= 0.0:
        raise ValueError("local_motion_budget_m must be positive.")

    midpoint_arc_length_m = 0.5 * local_motion_budget_m
    contact_rotation_chord_m = 0.5 * local_motion_budget_m

    ratio = contact_rotation_chord_m / contact_separation_m

    if not 0.0 < ratio < 1.0:
        raise ValueError("Motion budget is too large for the resolved contact separation.")

    peak_yaw_rad = 2.0 * math.asin(ratio)

    center_radius_m = midpoint_arc_length_m / peak_yaw_rad

    contact_radius_m = 0.5 * contact_separation_m

    midpoint_chord_m = 2.0 * center_radius_m * math.sin(0.5 * peak_yaw_rad)

    conservative_contact_displacement_bound_m = midpoint_chord_m + contact_rotation_chord_m

    return {
        "contact_separation_m": (contact_separation_m),
        "contact_radius_m": contact_radius_m,
        "local_motion_budget_m": (local_motion_budget_m),
        "midpoint_arc_length_m": (midpoint_arc_length_m),
        "midpoint_chord_m": midpoint_chord_m,
        "contact_rotation_chord_m": (contact_rotation_chord_m),
        "peak_yaw_rad": peak_yaw_rad,
        "center_radius_m": center_radius_m,
        "conservative_contact_displacement_bound_m": (conservative_contact_displacement_bound_m),
    }


def quaternion_multiply_wxyz(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Multiply broadcast-compatible wxyz quaternions."""

    if left.shape[-1] != 4 or right.shape[-1] != 4:
        raise ValueError("Quaternion tensors must end with dimension 4.")

    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)

    result = torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )

    return result / torch.linalg.vector_norm(
        result,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-9)


def virtual_surface_arc_targets(
    baseline_contact_positions: torch.Tensor,
    baseline_contact_quaternions: torch.Tensor,
    *,
    fraction: float,
    direction: int,
    center_radius_m: float,
    peak_yaw_rad: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
]:
    """Move two virtual contacts as one rigid SE(2) body arc."""

    if tuple(baseline_contact_positions.shape) != (2, 3):
        raise ValueError("baseline_contact_positions must have shape (2, 3).")

    if tuple(baseline_contact_quaternions.shape) != (2, 4):
        raise ValueError("baseline_contact_quaternions must have shape (2, 4).")

    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1.")

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1].")

    if center_radius_m <= 0.0 or peak_yaw_rad <= 0.0:
        raise ValueError("center_radius_m and peak_yaw_rad must be positive.")

    if not bool(torch.isfinite(baseline_contact_positions).all()):
        raise ValueError("Baseline positions must be finite.")

    if not bool(torch.isfinite(baseline_contact_quaternions).all()):
        raise ValueError("Baseline quaternions must be finite.")

    unsigned_yaw = peak_yaw_rad * fraction
    signed_yaw = float(direction) * unsigned_yaw

    midpoint_translation = torch.tensor(
        [
            center_radius_m * math.sin(unsigned_yaw),
            float(direction) * center_radius_m * (1.0 - math.cos(unsigned_yaw)),
            0.0,
        ],
        dtype=baseline_contact_positions.dtype,
        device=baseline_contact_positions.device,
    )

    cosine = math.cos(signed_yaw)
    sine = math.sin(signed_yaw)

    rotation = torch.tensor(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=baseline_contact_positions.dtype,
        device=baseline_contact_positions.device,
    )

    baseline_midpoint = baseline_contact_positions.mean(dim=0)
    offsets = baseline_contact_positions - baseline_midpoint.unsqueeze(0)

    target_positions = (
        baseline_midpoint.unsqueeze(0) + midpoint_translation.unsqueeze(0) + offsets @ rotation.T
    )

    yaw_quaternion = torch.tensor(
        [
            math.cos(0.5 * signed_yaw),
            0.0,
            0.0,
            math.sin(0.5 * signed_yaw),
        ],
        dtype=baseline_contact_quaternions.dtype,
        device=baseline_contact_quaternions.device,
    ).expand_as(baseline_contact_quaternions)

    target_quaternions = quaternion_multiply_wxyz(
        yaw_quaternion,
        baseline_contact_quaternions,
    )

    return (
        target_positions,
        target_quaternions,
        midpoint_translation,
        signed_yaw,
    )
