"""Pure helpers and constants for the Sharpa HB1 Sim2Real V1 setup.

This module deliberately has no Isaac Sim imports.  The training environment,
cache builder, and configuration auditor all use the same implementations so
that the documented setup cannot silently drift between tools.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Mapping, Sequence

import torch
from poise.paths import resolve_path


HB1_SIM2REAL_V1_ASSET_ENV = "SHARPA_HB1_USD_PATH"
HB1_SIM2REAL_V1_DEFAULT_USD_PATH = resolve_path("data/robots/sharpa_wave/right_sharpa_wave.usda")
HB1_SIM2REAL_V1_EXPECTED_USDA_SHA256 = (
    "043bce67fd6b059bd2290b40667ad3b161b4b7f2a76eebf86b5fd757838552b7"
)
SHARPA_RL_LAB_USDA09_REFERENCE_PATH = resolve_path("data/robots/sharpa_limits/right_sharpa_wave.usda")
SHARPA_RL_LAB_USDA09_REFERENCE_SHA256 = (
    "cb2ee455690cab3a7fb6912b9886d9205e57ca977d6ea2f132ac413442c9a09d"
)


def lookup_mesh_surface_field(
    points_object_m: torch.Tensor,
    object_indices: torch.Tensor,
    surface_delta_grid_m: torch.Tensor,
    outward_normal_grid: torch.Tensor,
    grid_half_extent_m: float,
    object_scale_xyz: torch.Tensor | None = None,
    *,
    validate_values: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Look up nearest mesh surface in the nominal object frame.

    ``surface_delta_grid_m`` and ``outward_normal_grid`` are generated for the
    nominal mesh.  Runtime object scale DR is handled by mapping query points
    back to that nominal frame and transforming projected points and normals
    back to the scaled object frame.  The inverse-transpose normal transform
    also keeps this helper correct for anisotropic scale.
    """

    points = torch.as_tensor(points_object_m)
    object_ids = torch.as_tensor(object_indices, device=points.device, dtype=torch.long)
    delta_grid = torch.as_tensor(surface_delta_grid_m, device=points.device, dtype=points.dtype)
    normal_grid = torch.as_tensor(outward_normal_grid, device=points.device, dtype=points.dtype)
    resolution = int(delta_grid.shape[1])
    if object_scale_xyz is None:
        scale_xyz = torch.ones((points.shape[0], 1, 3), device=points.device, dtype=points.dtype)
    else:
        scale_xyz = torch.as_tensor(object_scale_xyz, device=points.device, dtype=points.dtype)
        if scale_xyz.ndim == 1:
            if tuple(scale_xyz.shape) != (points.shape[0],):
                raise ValueError(
                    "1-D object_scale_xyz must have one uniform scale per batch row; "
                    f"got {tuple(scale_xyz.shape)} for batch size {points.shape[0]}."
                )
            scale_xyz = scale_xyz.view(-1, 1, 1).expand(-1, 1, 3)
        elif scale_xyz.ndim == 2 and tuple(scale_xyz.shape) == (points.shape[0], 3):
            scale_xyz = scale_xyz.unsqueeze(1)
        else:
            raise ValueError(
                "object_scale_xyz must have shape (N,) or (N, 3), got "
                f"{tuple(scale_xyz.shape)}."
            )
        if validate_values and (
            not bool(torch.isfinite(scale_xyz).all()) or bool((scale_xyz <= 0.0).any())
        ):
            raise ValueError("object_scale_xyz must contain finite positive scales.")
    nominal_points = points / scale_xyz
    sample_points = torch.clamp(
        nominal_points,
        -float(grid_half_extent_m),
        float(grid_half_extent_m),
    )
    coords = (sample_points / float(grid_half_extent_m) + 1.0) * (0.5 * (resolution - 1))
    lower = torch.floor(coords).to(dtype=torch.long)
    upper = torch.clamp(lower + 1, max=resolution - 1)
    weight = coords - lower.to(dtype=coords.dtype)

    delta = torch.zeros_like(points)
    normal = torch.zeros_like(points)
    object_ids = object_ids.view(-1, 1).expand(-1, points.shape[1])
    for bx in (0, 1):
        ix = upper[..., 0] if bx else lower[..., 0]
        wx = weight[..., 0] if bx else 1.0 - weight[..., 0]
        for by in (0, 1):
            iy = upper[..., 1] if by else lower[..., 1]
            wy = weight[..., 1] if by else 1.0 - weight[..., 1]
            for bz in (0, 1):
                iz = upper[..., 2] if bz else lower[..., 2]
                wz = weight[..., 2] if bz else 1.0 - weight[..., 2]
                corner_weight = (wx * wy * wz).unsqueeze(-1)
                delta = delta + corner_weight * delta_grid[object_ids, ix, iy, iz]
                normal = normal + corner_weight * normal_grid[object_ids, ix, iy, iz]
    projected_nominal = sample_points + delta
    projected_scaled = projected_nominal * scale_xyz
    normal_scaled = normal / scale_xyz
    normal_scaled = normal_scaled / torch.clamp(
        torch.linalg.norm(normal_scaled, dim=-1, keepdim=True), min=1.0e-12
    )
    return projected_scaled, normal_scaled


def conservative_saturated_grasp_wrench_coverage(
    contact_body_pos_object_m: torch.Tensor,
    contact_force_norm_n: torch.Tensor,
    object_dims_m: torch.Tensor,
    contact_friction: torch.Tensor,
    *,
    projected_contact_points_object_m: torch.Tensor | None = None,
    outward_contact_normals_object: torch.Tensor | None = None,
    contact_confidence_cap: torch.Tensor | None = None,
    contact_primary_mask: torch.Tensor | None = None,
    contact_group_ids: torch.Tensor | None = None,
    secondary_group_confidence_cap: float | None = None,
    spatial_dedup_distance_m: float = 0.0,
    contact_on_n: float = 0.05,
    contact_sat_n: float = 0.15,
    force_margin_erosion: float = 0.0,
    torque_margin_erosion: float = 0.0,
    softmin_power: float = 8.0,
    validate_values: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate conservative saturated six-axis grasp-wrench coverage.

    Body centers are projected to the nearest cuboid face and used only as a
    deliberately coarse contact-location proxy.  Each active contact emits
    four rays from a fixed, conservative friction cone.  Measured force only
    controls a smooth confidence gate which saturates at ``contact_sat_n``;
    therefore squeezing harder above that threshold cannot increase quality.

    The returned tuple contains ``quality``, minimum force-axis margin,
    minimum torque-axis margin, all six bidirectional margins, per-contact
    effective confidence, and projected contact points in the object frame.
    Optional all-body controls are applied in this order: per-contact confidence
    cap, primary-over-secondary spatial suppression, secondary-secondary spatial
    deduplication, and proportional per-group secondary cap. Primary contacts
    are never reduced by secondary contacts. Quality is a zero-preserving
    generalized soft minimum in ``[0, 1]``: any completely uncovered wrench
    axis makes it exactly zero.
    """

    positions = torch.as_tensor(contact_body_pos_object_m)
    forces = torch.as_tensor(contact_force_norm_n, device=positions.device, dtype=positions.dtype)
    dims = torch.as_tensor(object_dims_m, device=positions.device, dtype=positions.dtype)
    friction = torch.as_tensor(contact_friction, device=positions.device, dtype=positions.dtype)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(
            "contact_body_pos_object_m must have shape (N, C, 3), got "
            f"{tuple(positions.shape)}."
        )
    env_count, contact_count, _ = positions.shape
    if forces.shape != (env_count, contact_count):
        raise ValueError(
            "contact_force_norm_n must have shape (N, C), got "
            f"{tuple(forces.shape)} for positions {tuple(positions.shape)}."
        )
    if dims.shape != (env_count, 3):
        raise ValueError(f"object_dims_m must have shape (N, 3), got {tuple(dims.shape)}.")
    if friction.ndim == 1:
        if friction.shape != (contact_count,):
            raise ValueError(
                f"1-D contact_friction must have shape ({contact_count},), got {tuple(friction.shape)}."
            )
        friction = friction.unsqueeze(0).expand(env_count, -1)
    elif friction.shape != (env_count, contact_count):
        raise ValueError(
            "contact_friction must have shape (C,) or (N, C), got "
            f"{tuple(friction.shape)}."
        )

    confidence_cap = None
    if contact_confidence_cap is not None:
        confidence_cap = torch.as_tensor(
            contact_confidence_cap, device=positions.device, dtype=positions.dtype
        )
        if confidence_cap.ndim == 1:
            if confidence_cap.shape != (contact_count,):
                raise ValueError(
                    "1-D contact_confidence_cap must have shape "
                    f"({contact_count},), got {tuple(confidence_cap.shape)}."
                )
            confidence_cap = confidence_cap.unsqueeze(0).expand(env_count, -1)
        elif confidence_cap.shape != (env_count, contact_count):
            raise ValueError(
                "contact_confidence_cap must have shape (C,) or (N, C), got "
                f"{tuple(confidence_cap.shape)}."
            )

    primary_mask = None
    if contact_primary_mask is not None:
        primary_mask = torch.as_tensor(contact_primary_mask, device=positions.device, dtype=torch.bool)
        if primary_mask.ndim == 1:
            if primary_mask.shape != (contact_count,):
                raise ValueError(
                    f"contact_primary_mask must have shape ({contact_count},) or "
                    f"({env_count}, {contact_count}), got {tuple(primary_mask.shape)}."
                )
            primary_mask = primary_mask.unsqueeze(0).expand(env_count, -1)
        elif primary_mask.shape != (env_count, contact_count):
            raise ValueError(
                f"contact_primary_mask must have shape ({contact_count},) or "
                f"({env_count}, {contact_count}), got {tuple(primary_mask.shape)}."
            )

    group_ids = None
    if contact_group_ids is not None:
        group_ids = torch.as_tensor(contact_group_ids, device=positions.device, dtype=torch.long)
        if group_ids.shape != (contact_count,):
            raise ValueError(
                f"contact_group_ids must have shape ({contact_count},), got {tuple(group_ids.shape)}."
            )
    if secondary_group_confidence_cap is not None and (primary_mask is None or group_ids is None):
        raise ValueError(
            "secondary_group_confidence_cap requires contact_primary_mask and contact_group_ids."
        )
    if spatial_dedup_distance_m > 0.0 and (primary_mask is None or group_ids is None):
        raise ValueError(
            "spatial_dedup_distance_m > 0 requires contact_primary_mask and contact_group_ids."
        )
    scalar_values = (
        contact_on_n,
        contact_sat_n,
        force_margin_erosion,
        torque_margin_erosion,
        softmin_power,
        spatial_dedup_distance_m,
        *( () if secondary_group_confidence_cap is None else (secondary_group_confidence_cap,) ),
    )
    if not all(math.isfinite(float(value)) for value in scalar_values):
        raise ValueError("CSGWCD scalar parameters must be finite.")
    # Value validation is useful for standalone diagnostics and unit tests, but
    # each bool() below synchronizes a CUDA stream.  The Isaac environment
    # validates its configuration once and disables these checks in the hot
    # reward path; rollout tools separately fail fast on non-finite tensors.
    if validate_values:
        if not bool(torch.isfinite(positions).all()):
            raise ValueError("contact_body_pos_object_m must be finite.")
        if not bool(torch.isfinite(forces).all()) or bool((forces < 0.0).any()):
            raise ValueError("contact_force_norm_n must be finite and non-negative.")
        if not bool(torch.isfinite(dims).all()) or bool((dims <= 0.0).any()):
            raise ValueError("object_dims_m must contain finite positive dimensions.")
        if not bool(torch.isfinite(friction).all()) or bool((friction < 0.0).any()):
            raise ValueError("contact_friction must be finite and non-negative.")
        if confidence_cap is not None and (
            not bool(torch.isfinite(confidence_cap).all())
            or bool((confidence_cap < 0.0).any())
            or bool((confidence_cap > 1.0).any())
        ):
            raise ValueError("contact_confidence_cap must be finite and within [0, 1].")
        if group_ids is not None and bool((group_ids < -1).any()):
            raise ValueError("contact_group_ids values must be >= -1.")
    if contact_on_n < 0.0 or contact_sat_n <= contact_on_n:
        raise ValueError("contact thresholds must satisfy 0 <= contact_on_n < contact_sat_n.")
    if force_margin_erosion < 0.0 or torque_margin_erosion < 0.0:
        raise ValueError("margin erosion values must be non-negative.")
    if softmin_power <= 0.0:
        raise ValueError("softmin_power must be positive.")
    if spatial_dedup_distance_m < 0.0:
        raise ValueError("spatial_dedup_distance_m must be non-negative.")
    if secondary_group_confidence_cap is not None and not (
        0.0 <= float(secondary_group_confidence_cap) <= 1.0
    ):
        raise ValueError("secondary_group_confidence_cap must be within [0, 1].")

    half_dims = 0.5 * dims
    mesh_projection_enabled = projected_contact_points_object_m is not None
    if mesh_projection_enabled:
        if outward_contact_normals_object is None:
            raise ValueError("outward_contact_normals_object is required with projected mesh contacts.")
        projected = torch.as_tensor(
            projected_contact_points_object_m, device=positions.device, dtype=positions.dtype
        )
        outward_normal = torch.as_tensor(
            outward_contact_normals_object, device=positions.device, dtype=positions.dtype
        )
        if projected.shape != positions.shape or outward_normal.shape != positions.shape:
            raise ValueError("mesh projected points and normals must match contact position shape.")
        outward_normal = outward_normal / torch.clamp(
            torch.linalg.norm(outward_normal, dim=-1, keepdim=True), min=1.0e-12
        )
        inward_normal = -outward_normal
        reference_axis = torch.argmin(torch.abs(outward_normal), dim=-1)
        reference = torch.nn.functional.one_hot(reference_axis, num_classes=3).to(dtype=positions.dtype)
        tangent1 = torch.linalg.cross(outward_normal, reference, dim=-1)
        tangent1 = tangent1 / torch.clamp(torch.linalg.norm(tangent1, dim=-1, keepdim=True), min=1.0e-12)
        tangent2 = torch.linalg.cross(outward_normal, tangent1, dim=-1)
    else:
        normalized_abs = torch.abs(positions) / half_dims.unsqueeze(1)
        face_axis = torch.argmax(normalized_abs, dim=-1)
        projected = torch.maximum(torch.minimum(positions, half_dims.unsqueeze(1)), -half_dims.unsqueeze(1))
        selected_coordinate = torch.gather(positions, -1, face_axis.unsqueeze(-1)).squeeze(-1)
        face_sign = torch.where(selected_coordinate >= 0.0, 1.0, -1.0).to(dtype=positions.dtype)
        face_coordinate = torch.gather(half_dims.unsqueeze(1).expand(-1, contact_count, -1), -1, face_axis.unsqueeze(-1))
        projected = projected.scatter(-1, face_axis.unsqueeze(-1), face_sign.unsqueeze(-1) * face_coordinate)
        axis_one_hot = torch.nn.functional.one_hot(face_axis, num_classes=3).to(dtype=positions.dtype)
        outward_normal = axis_one_hot * face_sign.unsqueeze(-1)
        inward_normal = -outward_normal
        # Preserve the historical cuboid tangent basis exactly.
        tangent1_table = positions.new_tensor(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))
        tangent2_table = positions.new_tensor(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        tangent1 = tangent1_table[face_axis]
        tangent2 = tangent2_table[face_axis]
    tangents = torch.stack((tangent1, -tangent1, tangent2, -tangent2), dim=2)
    rays = inward_normal.unsqueeze(2) + friction.unsqueeze(-1).unsqueeze(-1) * tangents
    rays = rays / torch.clamp(torch.linalg.norm(rays, dim=-1, keepdim=True), min=1.0e-12)

    gate = torch.clamp((forces - float(contact_on_n)) / float(contact_sat_n - contact_on_n), 0.0, 1.0)
    confidence = gate.square() * (3.0 - 2.0 * gate)
    if confidence_cap is not None:
        confidence = torch.minimum(confidence, confidence_cap)

    if spatial_dedup_distance_m > 0.0:
        assert primary_mask is not None and group_ids is not None
        distance = torch.linalg.norm(projected.unsqueeze(2) - projected.unsqueeze(1), dim=-1)
        if mesh_projection_enabled:
            same_normal = torch.sum(
                inward_normal.unsqueeze(2) * inward_normal.unsqueeze(1), dim=-1
            ) >= 0.99
        else:
            same_normal = torch.all(
                inward_normal.unsqueeze(2) == inward_normal.unsqueeze(1), dim=-1
            )
        same_group = group_ids.view(1, contact_count, 1) == group_ids.view(1, 1, contact_count)
        valid_group = group_ids.view(1, contact_count, 1) >= 0
        near = (
            same_normal
            & same_group
            & valid_group
            & (distance <= float(spatial_dedup_distance_m))
        )
        active = confidence > 0.0
        # Query axis is the contact to suppress; reference axis is the contact
        # that dominates it. Primary contacts dominate only secondary contacts.
        primary_reference = primary_mask.unsqueeze(1) & active.unsqueeze(1)
        suppress_secondary = (~primary_mask) & torch.any(near & primary_reference, dim=-1)
        confidence = torch.where(suppress_secondary, torch.zeros_like(confidence), confidence)

        # Among remaining secondary contacts in one spatial/group neighborhood,
        # retain the highest confidence. Exact ties retain the lower body index.
        active = confidence > 0.0
        query_confidence = confidence.unsqueeze(2)
        reference_confidence = confidence.unsqueeze(1)
        index = torch.arange(contact_count, device=positions.device)
        lower_reference_index = index.view(1, 1, contact_count) < index.view(1, contact_count, 1)
        dominates = (reference_confidence > query_confidence) | (
            (reference_confidence == query_confidence) & lower_reference_index
        )
        secondary_pair = (~primary_mask).unsqueeze(2) & (~primary_mask).unsqueeze(1)
        suppress_duplicate = torch.any(
            near & secondary_pair & active.unsqueeze(1) & dominates,
            dim=-1,
        )
        confidence = torch.where(suppress_duplicate, torch.zeros_like(confidence), confidence)

    if secondary_group_confidence_cap is not None:
        assert primary_mask is not None and group_ids is not None
        secondary_mask = ~primary_mask
        valid_group = group_ids >= 0
        # Compute each contact's secondary group total through pairwise ID
        # equality. This remains fully vectorized on CUDA without assuming that
        # group IDs are dense or bounded by the contact count.
        same_group = group_ids.view(1, contact_count, 1) == group_ids.view(
            1, 1, contact_count
        )
        secondary_reference = secondary_mask.unsqueeze(1)
        valid_reference = valid_group.view(1, 1, contact_count)
        group_total_for_contact = torch.sum(
            confidence.unsqueeze(1)
            * (same_group & secondary_reference & valid_reference).to(dtype=positions.dtype),
            dim=-1,
        )
        contact_scale = torch.clamp(
            float(secondary_group_confidence_cap)
            / torch.clamp(group_total_for_contact, min=1.0e-12),
            max=1.0,
        )
        capped_secondary = secondary_mask & valid_group.unsqueeze(0)
        confidence = torch.where(capped_secondary, confidence * contact_scale, confidence)

    force_rays = confidence.unsqueeze(-1).unsqueeze(-1) * rays
    radius = torch.linalg.norm(half_dims, dim=-1, keepdim=True)
    lever = projected / torch.clamp(radius.unsqueeze(1), min=1.0e-12)
    torque_rays = torch.linalg.cross(lever.unsqueeze(2).expand_as(force_rays), force_rays, dim=-1)
    wrench_rays = torch.cat((force_rays, torque_rays), dim=-1).reshape(env_count, contact_count * 4, 6)
    positive = torch.relu(wrench_rays).amax(dim=1)
    negative = torch.relu(-wrench_rays).amax(dim=1)
    directional_margins = torch.minimum(positive, negative)
    erosion = positions.new_tensor(
        (force_margin_erosion,) * 3 + (torque_margin_erosion,) * 3
    )
    robust_margins = torch.clamp(directional_margins - erosion.unsqueeze(0), min=0.0, max=1.0)
    force_margin = robust_margins[:, :3].amin(dim=-1)
    torque_margin = robust_margins[:, 3:].amin(dim=-1)
    minimum_margin = robust_margins.amin(dim=-1)
    epsilon = torch.finfo(positions.dtype).eps
    soft_min = (
        torch.mean(torch.pow(robust_margins + epsilon, -float(softmin_power)), dim=-1)
        .pow(-1.0 / float(softmin_power))
        - epsilon
    )
    quality = torch.where(minimum_margin <= epsilon, torch.zeros_like(soft_min), soft_min)
    quality = torch.clamp(quality, min=0.0, max=1.0)
    return quality, force_margin, torque_margin, robust_margins, confidence, projected


def grasp_wrench_coverage_deficit_penalty(
    quality: torch.Tensor,
    target_quality: torch.Tensor,
    *,
    validate_values: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return non-negative coverage deficit and its conservative penalty."""

    quality_tensor = torch.as_tensor(quality)
    target = torch.as_tensor(target_quality, device=quality_tensor.device, dtype=quality_tensor.dtype)
    if target.shape != quality_tensor.shape:
        raise ValueError(
            f"target_quality shape {tuple(target.shape)} must match quality shape {tuple(quality_tensor.shape)}."
        )
    if validate_values and (
        not bool(torch.isfinite(quality_tensor).all())
        or not bool(torch.isfinite(target).all())
    ):
        raise ValueError("quality and target_quality must be finite.")
    deficit = torch.relu(target - quality_tensor)
    return deficit, -deficit.square()


def normalized_grasp_wrench_maintenance_reward(
    quality: torch.Tensor,
    target_quality: torch.Tensor,
    *,
    validate_values: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return coverage deficit and a zero-to-one positive maintenance reward.

    The normalized reward is ``1 - clip(deficit / target, 0, 1) ** 2``.
    A non-positive target yields zero reward so a disabled/invalid target cannot
    grant a free unit bonus.
    """

    quality_tensor = torch.as_tensor(quality)
    if not quality_tensor.is_floating_point():
        raise ValueError("quality must use a floating-point dtype.")
    target = torch.as_tensor(target_quality, device=quality_tensor.device, dtype=quality_tensor.dtype)
    if target.shape != quality_tensor.shape:
        raise ValueError(
            f"target_quality shape {tuple(target.shape)} must match quality shape {tuple(quality_tensor.shape)}."
        )
    if validate_values:
        if (
            not bool(torch.isfinite(quality_tensor).all())
            or not bool(torch.isfinite(target).all())
        ):
            raise ValueError("quality and target_quality must be finite.")
        if bool((quality_tensor < 0.0).any()) or bool((target < 0.0).any()):
            raise ValueError("quality and target_quality must be non-negative.")
    deficit = torch.relu(target - quality_tensor)
    epsilon = torch.finfo(quality_tensor.dtype).eps
    normalized_deficit = torch.clamp(
        deficit / torch.clamp(target, min=epsilon),
        min=0.0,
        max=1.0,
    )
    reward = 1.0 - normalized_deficit.square()
    reward = torch.where(target > epsilon, reward, torch.zeros_like(reward))
    return deficit, reward


def contact_gated_excess_load_terms(
    q_action: torch.Tensor,
    q_command: torch.Tensor,
    actual_pgain: torch.Tensor,
    nominal_pgain: torch.Tensor,
    finger_contact_force_norm: torch.Tensor,
    joint_finger_ids: torch.Tensor,
    *,
    reference_lead_rad: float,
    contact_force_threshold_n: float,
    normalized_excess_clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a contact-gated, deadbanded elastic-load penalty.

    The numerator uses the episode-randomized P gain so it represents the
    elastic part of the torque actually commanded by the explicit PD
    controller.  The deadband uses the nominal P gain and a reference lead,
    yielding a fixed per-joint torque budget across P-gain DR.

    Returns the per-environment mean penalty, normalized excess per joint,
    contact gate per joint, and elastic-load ratio per joint.
    """

    spring_torque = actual_pgain * (q_command - q_action)
    torque_deadband = nominal_pgain * float(reference_lead_rad)
    load_ratio = torch.abs(spring_torque) / torch.clamp(torque_deadband, min=1.0e-12)
    normalized_excess = torch.clamp(
        torch.relu(load_ratio - 1.0),
        max=float(normalized_excess_clip),
    )
    finger_contact = finger_contact_force_norm > float(contact_force_threshold_n)
    joint_contact = torch.index_select(finger_contact, 1, joint_finger_ids)
    penalty = torch.mean(joint_contact.to(dtype=normalized_excess.dtype) * normalized_excess.square(), dim=-1)
    return penalty, normalized_excess, joint_contact, load_ratio


def world_torque_for_target_angular_acceleration(
    inertia_tensor_b_kg_m2: torch.Tensor,
    object_quat_wxyz: torch.Tensor,
    target_angular_acceleration_w_rad_s2: torch.Tensor,
) -> torch.Tensor:
    """Map a world-frame target angular acceleration to world-frame torque.

    ``inertia_tensor_b_kg_m2`` is the full inertia tensor about the center of
    mass in object/body axes. ``object_quat_wxyz`` rotates body vectors into
    world axes. The returned torque is ``R I_body R^T alpha_world``.
    """

    inertia_b = torch.as_tensor(inertia_tensor_b_kg_m2)
    quat = torch.as_tensor(
        object_quat_wxyz,
        device=inertia_b.device,
        dtype=inertia_b.dtype,
    )
    alpha_w = torch.as_tensor(
        target_angular_acceleration_w_rad_s2,
        device=inertia_b.device,
        dtype=inertia_b.dtype,
    )
    if inertia_b.ndim < 2 or inertia_b.shape[-2:] != (3, 3):
        raise ValueError(
            "inertia_tensor_b_kg_m2 must end in (3, 3), got "
            f"{tuple(inertia_b.shape)}."
        )
    if quat.shape != inertia_b.shape[:-2] + (4,):
        raise ValueError(
            "object_quat_wxyz shape must match the inertia leading shape, got "
            f"{tuple(quat.shape)} and {tuple(inertia_b.shape)}."
        )
    if alpha_w.shape != inertia_b.shape[:-2] + (3,):
        raise ValueError(
            "target angular-acceleration shape must match the inertia leading "
            f"shape, got {tuple(alpha_w.shape)} and {tuple(inertia_b.shape)}."
        )

    quat = quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1.0e-12)

    def rotate(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        scalar = q[..., :1]
        axis = q[..., 1:]
        twice_cross = 2.0 * torch.linalg.cross(axis, vector, dim=-1)
        return vector + scalar * twice_cross + torch.linalg.cross(axis, twice_cross, dim=-1)

    quat_conjugate = torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)
    alpha_b = rotate(quat_conjugate, alpha_w)
    torque_b = torch.matmul(inertia_b, alpha_b.unsqueeze(-1)).squeeze(-1)
    return rotate(quat, torque_b)


def resolve_hb1_usd_path() -> Path:
    """Resolve the exact HB1 wrapper USDA selected for Sim2Real V1."""

    return Path(os.environ.get(HB1_SIM2REAL_V1_ASSET_ENV, HB1_SIM2REAL_V1_DEFAULT_USD_PATH)).resolve()


def parse_usda_action_joint_limits(
    usd_path: Path,
    joint_names: Sequence[str],
) -> torch.Tensor:
    """Parse action-ordered revolute limits from an ASCII USDA layer.

    The returned tensor is in radians with shape ``(len(joint_names), 2)``.
    This parser intentionally reads the explicit override blocks in the pinned
    USDA reference, so the joint-limit contract is independent of the separate
    hand asset loaded into Isaac/PhysX.
    """

    path = Path(usd_path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    rows: list[tuple[float, float]] = []
    for name in joint_names:
        match = re.search(
            rf'over "{re.escape(name)}"\s*\{{(.*?)\n\s*\}}',
            text,
            flags=re.DOTALL,
        )
        if match is None:
            raise ValueError(f"USDA joint block missing for {name!r}: {path}")
        block = match.group(1)
        lower = re.search(r"physics:lowerLimit\s*=\s*([-+0-9.eE]+)", block)
        upper = re.search(r"physics:upperLimit\s*=\s*([-+0-9.eE]+)", block)
        if lower is None or upper is None:
            raise ValueError(f"USDA lower/upper limit missing for {name!r}: {path}")
        rows.append((float(lower.group(1)), float(upper.group(1))))
    limits = torch.deg2rad(torch.tensor(rows, dtype=torch.float32))
    if not bool(torch.isfinite(limits).all()) or bool((limits[:, 0] >= limits[:, 1]).any()):
        raise ValueError(f"USDA contains invalid joint limits: {path}")
    return limits


def scaled_signed_usd_limits(
    usd_limits_rad: torch.Tensor,
    *,
    endpoint_scale: float = 0.9,
) -> torch.Tensor:
    """Scale both signed USD joint-limit endpoints around zero.

    For example, ``[-10, 110] deg`` with ``endpoint_scale=0.9`` becomes
    ``[-9, 99] deg``.  This is the pure USDA×scale profile and does not
    intersect with the separate deployment joint-limit table.
    """

    if usd_limits_rad.ndim < 2 or usd_limits_rad.shape[-1] != 2:
        raise ValueError(
            "usd_limits_rad must end in a lower/upper endpoint pair; "
            f"got shape={tuple(usd_limits_rad.shape)}."
        )
    if not 0.0 < float(endpoint_scale) <= 1.0:
        raise ValueError(f"endpoint_scale must be in (0, 1], got {endpoint_scale}.")
    scaled = usd_limits_rad * float(endpoint_scale)
    if bool((scaled[..., 0] >= scaled[..., 1]).any()):
        raise ValueError("Scaled USD joint limits contain an empty interval.")
    return scaled


def composite_deploy_usd_limits(
    usd_limits_rad: torch.Tensor,
    joint_names: Sequence[str],
    deployment_limits_deg: Mapping[str, Sequence[float]],
    *,
    usd_endpoint_scale: float = 0.9,
) -> torch.Tensor:
    """Return ``deployment profile intersect (USD endpoints * scale)``.

    ``usd_limits_rad`` must be action ordered and have shape ``(..., J, 2)``.
    The multiplication is applied to both signed USD endpoints before the
    intersection; for example ``[-10, 110] deg * 0.9 -> [-9, 99] deg``.
    """

    if usd_limits_rad.ndim < 2 or usd_limits_rad.shape[-2:] != (len(joint_names), 2):
        raise ValueError(
            "usd_limits_rad must end in (len(joint_names), 2); "
            f"got shape={tuple(usd_limits_rad.shape)}, joints={len(joint_names)}."
        )
    if not 0.0 < float(usd_endpoint_scale) <= 1.0:
        raise ValueError(f"usd_endpoint_scale must be in (0, 1], got {usd_endpoint_scale}.")
    missing = tuple(name for name in joint_names if name not in deployment_limits_deg)
    if missing:
        raise ValueError(f"Missing deployment limits for joints: {missing}.")

    deploy = torch.tensor(
        [deployment_limits_deg[name] for name in joint_names],
        device=usd_limits_rad.device,
        dtype=usd_limits_rad.dtype,
    )
    deploy = torch.deg2rad(deploy)
    while deploy.ndim < usd_limits_rad.ndim:
        deploy = deploy.unsqueeze(0)

    usd_scaled = usd_limits_rad * float(usd_endpoint_scale)
    lower = torch.maximum(deploy[..., 0], usd_scaled[..., 0])
    upper = torch.minimum(deploy[..., 1], usd_scaled[..., 1])
    invalid = lower >= upper
    if bool(invalid.any()):
        ids = torch.nonzero(invalid.reshape(-1), as_tuple=False).squeeze(-1).tolist()
        raise ValueError(f"Empty deployment/USD limit intersections at flattened joint ids {ids}.")
    return torch.stack((lower, upper), dim=-1)


def project_joint_positions_with_margin(
    positions_rad: torch.Tensor,
    limits_rad: torch.Tensor,
    *,
    margin_rad: float,
) -> torch.Tensor:
    """Project action-ordered joint positions into limits with a soft margin."""

    if positions_rad.shape[-1] != limits_rad.shape[-2] or limits_rad.shape[-1] != 2:
        raise ValueError(
            f"Incompatible position/limit shapes: {tuple(positions_rad.shape)} and {tuple(limits_rad.shape)}."
        )
    margin = float(margin_rad)
    if margin < 0.0:
        raise ValueError(f"margin_rad must be non-negative, got {margin}.")
    lower = limits_rad[..., 0] + margin
    upper = limits_rad[..., 1] - margin
    if bool((lower >= upper).any()):
        raise ValueError("Soft margin leaves at least one joint with an empty feasible interval.")
    return torch.minimum(torch.maximum(positions_rad, lower), upper)


def sample_nominal_log_uniform_masses(
    shape: int | Sequence[int],
    *,
    wide_range_kg: Sequence[float],
    nominal_range_kg: Sequence[float],
    nominal_weight: float,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a nominal-uniform + wide-log-uniform mass mixture.

    Branch id 0 is the nominal uniform component and branch id 1 is the
    wide log-uniform component. Sampling is independent per requested entry.
    """

    sample_shape = (shape,) if isinstance(shape, int) else tuple(int(v) for v in shape)
    if any(v < 0 for v in sample_shape):
        raise ValueError(f"shape must be non-negative, got {sample_shape!r}.")
    wide_lo, wide_hi = (float(v) for v in wide_range_kg)
    nominal_lo, nominal_hi = (float(v) for v in nominal_range_kg)
    weight = float(nominal_weight)
    if not (0.0 < wide_lo <= wide_hi):
        raise ValueError(f"Invalid wide_range_kg={tuple(wide_range_kg)!r}.")
    if not (0.0 < nominal_lo <= nominal_hi):
        raise ValueError(f"Invalid nominal_range_kg={tuple(nominal_range_kg)!r}.")
    if nominal_lo < wide_lo or nominal_hi > wide_hi:
        raise ValueError(
            "nominal_range_kg must lie within wide_range_kg, got "
            f"nominal={tuple(nominal_range_kg)!r}, wide={tuple(wide_range_kg)!r}."
        )
    if not (0.0 <= weight <= 1.0):
        raise ValueError(f"nominal_weight must be in [0, 1], got {weight}.")

    branch_uniform = torch.rand(sample_shape, device=device, generator=generator, dtype=dtype)
    branch_id = (branch_uniform >= weight).to(dtype=torch.long)
    value_uniform = torch.rand(sample_shape, device=device, generator=generator, dtype=dtype)
    nominal = nominal_lo + (nominal_hi - nominal_lo) * value_uniform
    wide = torch.exp(
        math.log(wide_lo) + (math.log(wide_hi) - math.log(wide_lo)) * value_uniform
    )
    return torch.where(branch_id == 0, nominal, wide), branch_id


def mass_bucket_ids(
    masses_kg: torch.Tensor,
    edges_kg: Sequence[float],
) -> torch.Tensor:
    """Map masses to left-closed buckets, with the final upper edge included.

    The returned IDs are in ``[0, len(edges_kg) - 2]`` for covered values and
    ``-1`` outside the configured interval.  Interior boundaries belong to
    the bucket on their right: for edges ``(0.01, 0.02, 0.04)``, exactly
    ``0.02`` kg maps to bucket 1.
    """

    masses = torch.as_tensor(masses_kg)
    if masses.ndim != 1:
        raise ValueError(f"masses_kg must be one-dimensional, got {tuple(masses.shape)}.")
    if not bool(torch.isfinite(masses).all()):
        raise ValueError("masses_kg must contain only finite values.")
    edges = torch.as_tensor(tuple(float(value) for value in edges_kg), device=masses.device, dtype=masses.dtype)
    if edges.ndim != 1 or edges.numel() < 2:
        raise ValueError("edges_kg must contain at least two values.")
    if not bool(torch.isfinite(edges).all()) or bool((edges <= 0.0).any()):
        raise ValueError("edges_kg must contain finite positive values.")
    if bool((edges[1:] <= edges[:-1]).any()):
        raise ValueError("edges_kg must be strictly increasing.")

    bucket_ids = torch.bucketize(masses, edges[1:-1], right=True)
    covered = (masses >= edges[0]) & (masses <= edges[-1])
    return torch.where(covered, bucket_ids, torch.full_like(bucket_ids, -1))


def sample_independent_two_sided_scales(
    shape: Sequence[int],
    *,
    low: float,
    high: float,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample independent scales with equal probability below/above one.

    This reproduces the Sharpa-RL-Lab reset-time gain randomization: a 50/50
    mixture of ``Uniform(low, 1)`` and ``Uniform(1, high)``.  Every tensor
    element is sampled independently, so P and D can vary independently for
    every environment and every joint.
    """

    low = float(low)
    high = float(high)
    if not (0.0 < low <= 1.0 <= high):
        raise ValueError(f"Expected 0 < low <= 1 <= high, got {(low, high)}.")
    uniform = torch.rand(tuple(shape), device=device, dtype=dtype, generator=generator)
    upper_half = torch.rand(tuple(shape), device=device, dtype=dtype, generator=generator) >= 0.5
    below = low + (1.0 - low) * uniform
    above = 1.0 + (high - 1.0) * uniform
    return torch.where(upper_half, above, below)


def cuboid_inertia_diagonal(mass_kg: torch.Tensor, dims_m: torch.Tensor) -> torch.Tensor:
    """Compute principal inertia ``[Ixx, Iyy, Izz]`` for solid cuboids."""

    if dims_m.shape[-1] != 3:
        raise ValueError(f"dims_m must end in 3, got {tuple(dims_m.shape)}.")
    if mass_kg.shape != dims_m.shape[:-1]:
        raise ValueError(f"mass/dims shapes do not align: {tuple(mass_kg.shape)} vs {tuple(dims_m.shape)}.")
    if bool((mass_kg <= 0.0).any()) or bool((dims_m <= 0.0).any()):
        raise ValueError("Cuboid masses and dimensions must be positive.")
    x, y, z = dims_m.unbind(dim=-1)
    factor = mass_kg / 12.0
    return torch.stack(
        (
            factor * (y.square() + z.square()),
            factor * (x.square() + z.square()),
            factor * (x.square() + y.square()),
        ),
        dim=-1,
    )


def sample_inertia_multipliers(
    count: int,
    *,
    low: float,
    high: float,
    distribution: str = "log_uniform",
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample shared scalar multipliers for full inertia tensors."""

    low = float(low)
    high = float(high)
    if not (0.0 < low <= high):
        raise ValueError(f"Expected 0 < low <= high, got {(low, high)}.")
    unit = torch.rand((int(count),), device=device, dtype=dtype, generator=generator)
    distribution = str(distribution).strip().lower()
    if distribution == "uniform":
        return low + (high - low) * unit
    if distribution == "log_uniform":
        return torch.exp(math.log(low) + (math.log(high) - math.log(low)) * unit)
    raise ValueError(f"Inertia distribution must be 'uniform' or 'log_uniform', got {distribution!r}.")


def sample_real_to_legacy_cuboid_inertia_diagonal(
    mass_kg: torch.Tensor,
    dims_m: torch.Tensor,
    *,
    upper_diag_kg_m2: float = 1.0e-4,
    distribution: str = "log_uniform",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample cuboid inertia between its physical value and a legacy scale.

    The physical solid-cuboid diagonal is computed from ``mass_kg`` and
    ``dims_m``.  One scalar multiplier is then sampled per cuboid and applied
    to all three principal moments, preserving their physically meaningful
    axis ratios.  The maximum principal moment reaches
    ``upper_diag_kg_m2`` at the top of the sampling interval.

    Returns ``(sampled_diagonal, multiplier)``.  The configured upper value
    must cover every input cuboid's physical maximum principal moment; this
    makes a bad DR contract fail loudly instead of silently exceeding its
    advertised upper endpoint.
    """

    physical_diag = cuboid_inertia_diagonal(mass_kg, dims_m)
    upper = float(upper_diag_kg_m2)
    if not math.isfinite(upper) or upper <= 0.0:
        raise ValueError(f"upper_diag_kg_m2 must be finite and positive, got {upper!r}.")

    physical_max = physical_diag.amax(dim=-1)
    tolerance = max(1.0e-12, upper * 1.0e-6)
    if bool((physical_max > upper + tolerance).any()):
        actual_max = float(physical_max.max().item())
        raise ValueError(
            "upper_diag_kg_m2 must cover the physical cuboid inertia; "
            f"configured upper={upper:.9g}, physical maximum={actual_max:.9g}."
        )

    maximum_multiplier = torch.clamp(
        torch.as_tensor(upper, device=physical_diag.device, dtype=physical_diag.dtype)
        / physical_max,
        min=1.0,
    )
    unit = torch.rand(
        physical_max.shape,
        device=physical_diag.device,
        dtype=physical_diag.dtype,
        generator=generator,
    )
    normalized_distribution = str(distribution).strip().lower()
    if normalized_distribution == "uniform":
        multiplier = 1.0 + (maximum_multiplier - 1.0) * unit
    elif normalized_distribution == "log_uniform":
        multiplier = torch.exp(torch.log(maximum_multiplier) * unit)
    else:
        raise ValueError(
            "Inertia distribution must be 'uniform' or 'log_uniform', "
            f"got {distribution!r}."
        )
    return physical_diag * multiplier.unsqueeze(-1), multiplier


def compound_cuboid_mass_properties(
    total_mass_kg: torch.Tensor,
    part_dims_m: torch.Tensor,
    part_centers_m: torch.Tensor,
    *,
    relative_density: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute center of mass and inertia for a rigid compound of cuboids.

    ``part_dims_m`` and ``part_centers_m`` use shape ``(..., P, 3)`` while
    ``total_mass_kg`` uses the matching leading shape ``(...)``. Part masses
    are proportional to ``volume * relative_density``. Returned inertia
    tensors are expressed about the compound center of mass in object axes.
    """

    total_mass = torch.as_tensor(total_mass_kg)
    dims = torch.as_tensor(part_dims_m, device=total_mass.device, dtype=total_mass.dtype)
    centers = torch.as_tensor(part_centers_m, device=total_mass.device, dtype=total_mass.dtype)
    if dims.ndim < 2 or dims.shape[-1] != 3:
        raise ValueError(f"part_dims_m must have shape (..., P, 3), got {tuple(dims.shape)}.")
    if centers.shape != dims.shape:
        raise ValueError(
            "part_dims_m and part_centers_m must have identical shapes, got "
            f"{tuple(dims.shape)} and {tuple(centers.shape)}."
        )
    if total_mass.shape != dims.shape[:-2]:
        raise ValueError(
            "total_mass_kg leading shape must match compound parts, got "
            f"{tuple(total_mass.shape)} and {tuple(dims.shape)}."
        )
    if bool((total_mass <= 0.0).any()) or bool((dims <= 0.0).any()):
        raise ValueError("Compound masses and cuboid dimensions must be positive.")

    if relative_density is None:
        density = torch.ones(dims.shape[:-1], device=dims.device, dtype=dims.dtype)
    else:
        density = torch.as_tensor(relative_density, device=dims.device, dtype=dims.dtype)
        if density.shape != dims.shape[:-1]:
            raise ValueError(
                "relative_density must have shape (..., P), got "
                f"{tuple(density.shape)} for parts {tuple(dims.shape)}."
            )
        if bool((density <= 0.0).any()):
            raise ValueError("Compound relative densities must be positive.")

    weighted_volume = dims.prod(dim=-1) * density
    part_mass_fraction = weighted_volume / weighted_volume.sum(dim=-1, keepdim=True)
    part_mass = total_mass.unsqueeze(-1) * part_mass_fraction
    center_of_mass = (part_mass.unsqueeze(-1) * centers).sum(dim=-2) / total_mass.unsqueeze(-1)

    x, y, z = dims.unbind(dim=-1)
    intrinsic_diagonal = torch.stack(
        (
            y.square() + z.square(),
            x.square() + z.square(),
            x.square() + y.square(),
        ),
        dim=-1,
    ) * (part_mass / 12.0).unsqueeze(-1)
    inertia = torch.diag_embed(intrinsic_diagonal).sum(dim=-3)

    offset = centers - center_of_mass.unsqueeze(-2)
    offset_sq = offset.square().sum(dim=-1)
    identity = torch.eye(3, device=dims.device, dtype=dims.dtype)
    parallel_axis = part_mass.unsqueeze(-1).unsqueeze(-1) * (
        offset_sq.unsqueeze(-1).unsqueeze(-1) * identity
        - offset.unsqueeze(-1) * offset.unsqueeze(-2)
    )
    inertia = inertia + parallel_axis.sum(dim=-3)
    return center_of_mass, inertia


def mass_size_scaled_inertia_diagonal(
    nominal_inertia_diag_kg_m2: torch.Tensor,
    sampled_mass_kg: torch.Tensor,
    reference_mass_kg: torch.Tensor,
    side_length_m: torch.Tensor,
    *,
    nominal_side_length_m: float,
) -> torch.Tensor:
    """Scale legacy nominal inertia by mass ratio and squared size ratio."""

    nominal = torch.as_tensor(nominal_inertia_diag_kg_m2)
    sampled_mass = torch.as_tensor(
        sampled_mass_kg, device=nominal.device, dtype=nominal.dtype
    )
    reference_mass = torch.as_tensor(
        reference_mass_kg, device=nominal.device, dtype=nominal.dtype
    )
    side_length = torch.as_tensor(
        side_length_m, device=nominal.device, dtype=nominal.dtype
    )
    if nominal.ndim != 2 or nominal.shape[-1] != 3:
        raise ValueError(
            "nominal_inertia_diag_kg_m2 must have shape (N, 3), "
            f"got {tuple(nominal.shape)}."
        )
    expected = nominal.shape[:-1]
    if sampled_mass.shape != expected or reference_mass.shape != expected or side_length.shape != expected:
        raise ValueError(
            "Mass and side tensors must match the nominal inertia batch: "
            f"nominal={tuple(nominal.shape)}, sampled_mass={tuple(sampled_mass.shape)}, "
            f"reference_mass={tuple(reference_mass.shape)}, side={tuple(side_length.shape)}."
        )
    nominal_side = float(nominal_side_length_m)
    if nominal_side <= 0.0:
        raise ValueError(f"nominal_side_length_m must be positive, got {nominal_side}.")
    if bool((sampled_mass <= 0.0).any()) or bool((reference_mass <= 0.0).any()) or bool(
        (side_length <= 0.0).any()
    ):
        raise ValueError("Masses and side lengths must be positive.")
    factor = (sampled_mass / reference_mass) * (side_length / nominal_side).square()
    return nominal * factor.unsqueeze(-1)


def density_range_for_global_mass_and_size_bounds(
    *,
    minimum_mass_kg: float,
    maximum_mass_kg: float,
    minimum_side_length_m: float,
    maximum_side_length_m: float,
) -> tuple[float, float]:
    """Return density bounds that realize global mass bounds at size endpoints.

    The lower density produces ``minimum_mass_kg`` for the smallest cube and
    the upper density produces ``maximum_mass_kg`` for the largest cube.  A
    density sampled inside this interval therefore keeps mass correlated with
    cube volume while preserving the requested global mass envelope.
    """

    minimum_mass_kg = float(minimum_mass_kg)
    maximum_mass_kg = float(maximum_mass_kg)
    minimum_side_length_m = float(minimum_side_length_m)
    maximum_side_length_m = float(maximum_side_length_m)
    if not (
        0.0 < minimum_mass_kg <= maximum_mass_kg
        and 0.0 < minimum_side_length_m <= maximum_side_length_m
    ):
        raise ValueError(
            "Mass and side-length bounds must be positive and ordered; "
            f"got mass={(minimum_mass_kg, maximum_mass_kg)}, "
            f"side={(minimum_side_length_m, maximum_side_length_m)}."
        )
    minimum_density = minimum_mass_kg / minimum_side_length_m**3
    maximum_density = maximum_mass_kg / maximum_side_length_m**3
    if minimum_density > maximum_density:
        raise ValueError(
            "Requested global mass/size bounds imply an inverted density range: "
            f"{(minimum_density, maximum_density)} kg/m^3."
        )
    return minimum_density, maximum_density


def sample_density_correlated_cube_masses(
    side_lengths_m: torch.Tensor,
    *,
    minimum_mass_kg: float,
    maximum_mass_kg: float,
    minimum_side_length_m: float,
    maximum_side_length_m: float,
    distribution: str = "log_uniform",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample cube mass through density so mass remains proportional to scale cubed.

    Returns ``(mass_kg, density_kg_m3)``.  ``log_uniform`` is the default so
    the broad 5--60 g envelope does not over-weight its high-density end.
    """

    side_lengths = torch.as_tensor(side_lengths_m)
    if side_lengths.ndim != 1 or not bool(torch.isfinite(side_lengths).all()):
        raise ValueError("side_lengths_m must be a finite one-dimensional tensor.")
    tolerance = 1.0e-7
    if bool((side_lengths < float(minimum_side_length_m) - tolerance).any()) or bool(
        (side_lengths > float(maximum_side_length_m) + tolerance).any()
    ):
        raise ValueError(
            f"Side lengths must stay inside [{minimum_side_length_m}, {maximum_side_length_m}] m."
        )
    density_min, density_max = density_range_for_global_mass_and_size_bounds(
        minimum_mass_kg=minimum_mass_kg,
        maximum_mass_kg=maximum_mass_kg,
        minimum_side_length_m=minimum_side_length_m,
        maximum_side_length_m=maximum_side_length_m,
    )
    uniform = torch.rand(
        side_lengths.shape,
        device=side_lengths.device,
        dtype=side_lengths.dtype,
        generator=generator,
    )
    normalized_distribution = str(distribution).strip().lower()
    if normalized_distribution == "uniform":
        density = density_min + (density_max - density_min) * uniform
    elif normalized_distribution == "log_uniform":
        log_min = math.log(density_min)
        density = torch.exp(log_min + (math.log(density_max) - log_min) * uniform)
    else:
        raise ValueError(
            "Density distribution must be 'uniform' or 'log_uniform', "
            f"got {distribution!r}."
        )
    mass = density * side_lengths.pow(3)
    # Numerical roundoff at the two endpoints must not escape the declared
    # global DR envelope.
    mass = torch.clamp(mass, min=float(minimum_mass_kg), max=float(maximum_mass_kg))
    return mass, density


def sample_density_correlated_cuboid_masses(
    dims_m: torch.Tensor,
    *,
    density_range_kg_m3: tuple[float, float],
    distribution: str = "uniform",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample density and compute physically correlated cuboid masses.

    Unlike :func:`sample_density_correlated_cube_masses`, this helper does not
    force a global mass envelope.  The density contract remains fixed while
    mass follows the actual per-environment volume ``dx * dy * dz``.
    """

    dims = torch.as_tensor(dims_m)
    if dims.ndim != 2 or dims.shape[-1] != 3:
        raise ValueError(f"dims_m must have shape (N, 3), got {tuple(dims.shape)}.")
    if not bool(torch.isfinite(dims).all()) or bool((dims <= 0.0).any()):
        raise ValueError("dims_m must contain finite positive dimensions.")
    density_lo, density_hi = (float(v) for v in density_range_kg_m3)
    if density_lo <= 0.0 or density_hi < density_lo:
        raise ValueError(f"Invalid density_range_kg_m3={density_range_kg_m3!r}.")
    uniform = torch.rand(
        (dims.shape[0],),
        device=dims.device,
        dtype=dims.dtype,
        generator=generator,
    )
    normalized_distribution = str(distribution).strip().lower()
    if normalized_distribution == "uniform":
        density = density_lo + (density_hi - density_lo) * uniform
    elif normalized_distribution == "log_uniform":
        density = torch.exp(math.log(density_lo) + (math.log(density_hi) - math.log(density_lo)) * uniform)
    else:
        raise ValueError(
            "Density distribution must be 'uniform' or 'log_uniform', "
            f"got {distribution!r}."
        )
    return density * dims.prod(dim=-1), density


def sample_nearcube_geometry_cpu(
    num_envs: int,
    *,
    mixture_weights: tuple[float, float, float] = (0.60, 0.20, 0.20),
    nominal_side_range_m: tuple[float, float] = (0.039, 0.041),
    nominal_side_mode_m: float = 0.040,
    wide_side_range_m: tuple[float, float] = (0.038, 0.042),
    cuboid_dim_range_m: tuple[float, float] = (0.038, 0.042),
    maximum_aspect_ratio: float = 1.08,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample an exact-count, shuffled near-cube geometry population on CPU.

    Branch 0 is a triangular nominal isotropic cube distribution, branch 1 is
    a wider uniform isotropic cube distribution, and branch 2 is a continuous
    anisotropic cuboid distribution rejected outside the aspect-ratio bound.
    """

    count = int(num_envs)
    if count <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}.")
    weights = torch.tensor(mixture_weights, dtype=torch.float64)
    if weights.shape != (3,) or not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
        raise ValueError(f"mixture_weights must contain three finite non-negative values, got {mixture_weights!r}.")
    if float(weights.sum()) <= 0.0:
        raise ValueError("mixture_weights must have a positive sum.")
    weights /= weights.sum()
    raw_counts = weights * count
    branch_counts = torch.floor(raw_counts).to(torch.long)
    remainder = count - int(branch_counts.sum())
    if remainder:
        fractional = raw_counts - branch_counts.to(raw_counts.dtype)
        order = torch.argsort(fractional, descending=True, stable=True)
        branch_counts[order[:remainder]] += 1

    nominal_lo, nominal_hi = (float(v) for v in nominal_side_range_m)
    wide_lo, wide_hi = (float(v) for v in wide_side_range_m)
    dim_lo, dim_hi = (float(v) for v in cuboid_dim_range_m)
    nominal_mode = float(nominal_side_mode_m)
    aspect_limit = float(maximum_aspect_ratio)
    scalar_params = torch.tensor(
        [nominal_lo, nominal_hi, nominal_mode, wide_lo, wide_hi, dim_lo, dim_hi, aspect_limit],
        dtype=torch.float64,
    )
    if not bool(torch.isfinite(scalar_params).all()):
        raise ValueError("Near-cube geometry ranges, mode, and aspect ratio must be finite.")
    if not (0.0 < nominal_lo <= nominal_mode <= nominal_hi):
        raise ValueError("nominal side range/mode must be positive and ordered.")
    if not (0.0 < wide_lo <= wide_hi and 0.0 < dim_lo <= dim_hi):
        raise ValueError("wide side and cuboid dimension ranges must be positive and ordered.")
    if aspect_limit < 1.0 or aspect_limit > dim_hi / dim_lo + 1.0e-9:
        raise ValueError(f"Invalid maximum_aspect_ratio={aspect_limit}.")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    dims = torch.empty((count, 3), dtype=torch.float32)
    branch_ids = torch.repeat_interleave(torch.arange(3, dtype=torch.long), branch_counts)
    cursor = 0
    nominal_count = int(branch_counts[0])
    if nominal_count:
        if nominal_hi == nominal_lo:
            side = torch.full((nominal_count,), nominal_lo, dtype=dims.dtype)
        else:
            u = torch.rand((nominal_count,), generator=generator)
            split = (nominal_mode - nominal_lo) / (nominal_hi - nominal_lo)
            side = torch.where(
                u < split,
                nominal_lo + torch.sqrt(u * (nominal_hi - nominal_lo) * (nominal_mode - nominal_lo)),
                nominal_hi - torch.sqrt((1.0 - u) * (nominal_hi - nominal_lo) * (nominal_hi - nominal_mode)),
            )
        dims[cursor : cursor + nominal_count] = side.unsqueeze(-1)
        cursor += nominal_count
    wide_count = int(branch_counts[1])
    if wide_count:
        side = wide_lo + (wide_hi - wide_lo) * torch.rand((wide_count,), generator=generator)
        dims[cursor : cursor + wide_count] = side.unsqueeze(-1)
        cursor += wide_count
    cuboid_count = int(branch_counts[2])
    accepted: list[torch.Tensor] = []
    accepted_count = 0
    while accepted_count < cuboid_count:
        proposal_count = max(64, 2 * (cuboid_count - accepted_count))
        proposal = dim_lo + (dim_hi - dim_lo) * torch.rand((proposal_count, 3), generator=generator)
        aspect = proposal.max(dim=-1).values / proposal.min(dim=-1).values
        valid = proposal[aspect <= aspect_limit]
        if valid.numel():
            accepted.append(valid)
            accepted_count += int(valid.shape[0])
    if cuboid_count:
        dims[cursor:] = torch.cat(accepted, dim=0)[:cuboid_count]

    permutation = torch.randperm(count, generator=generator)
    return dims[permutation], branch_ids[permutation]


def geometry_cache_anchor_ids(
    dims_m: torch.Tensor,
    anchor_dims_m: torch.Tensor,
    *,
    metric_scale_m: torch.Tensor | tuple[float, float, float],
) -> torch.Tensor:
    """Assign each geometry to its nearest normalized-L-infinity anchor."""

    dims = torch.as_tensor(dims_m)
    anchors = torch.as_tensor(anchor_dims_m, device=dims.device, dtype=dims.dtype)
    scale = torch.as_tensor(metric_scale_m, device=dims.device, dtype=dims.dtype)
    if dims.ndim != 2 or dims.shape[-1] != 3:
        raise ValueError(f"dims_m must have shape (N, 3), got {tuple(dims.shape)}.")
    if anchors.ndim != 2 or anchors.shape[-1] != 3 or anchors.shape[0] == 0:
        raise ValueError(f"anchor_dims_m must have shape (K, 3), got {tuple(anchors.shape)}.")
    if not bool(torch.isfinite(dims).all()) or bool((dims <= 0.0).any()):
        raise ValueError("dims_m must contain finite positive dimensions.")
    if not bool(torch.isfinite(anchors).all()) or bool((anchors <= 0.0).any()):
        raise ValueError("anchor_dims_m must contain finite positive dimensions.")
    if scale.shape != (3,) or not bool(torch.isfinite(scale).all()) or bool((scale <= 0.0).any()):
        raise ValueError(f"metric_scale_m must contain three finite positive values, got {metric_scale_m!r}.")
    distance = torch.abs(dims[:, None, :] - anchors[None, :, :]) / scale
    return torch.argmin(distance.amax(dim=-1), dim=-1)


def timebase_matched_wrench_scale(reference_scale: float, reference_hz: float, control_hz: float) -> float:
    """Scale a held wrench so one event has the reference-policy impulse."""

    reference_hz = float(reference_hz)
    control_hz = float(control_hz)
    if reference_hz <= 0.0 or control_hz <= 0.0:
        raise ValueError("reference_hz and control_hz must be positive.")
    return float(reference_scale) * control_hz / reference_hz


def timebase_matched_trigger_probability(
    probability: torch.Tensor | float,
    *,
    reference_hz: float,
    control_hz: float,
) -> torch.Tensor | float:
    """Convert per-reference-step event probability to a control-step value."""

    reference_hz = float(reference_hz)
    control_hz = float(control_hz)
    if reference_hz <= 0.0 or control_hz <= 0.0:
        raise ValueError("reference_hz and control_hz must be positive.")
    exponent = reference_hz / control_hz
    if isinstance(probability, torch.Tensor):
        if bool(((probability < 0.0) | (probability > 1.0)).any()):
            raise ValueError("probability must be in [0, 1].")
        return 1.0 - torch.pow(1.0 - probability, exponent)
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}.")
    return 1.0 - math.pow(1.0 - probability, exponent)


def scale_cache_bucket_ids(
    cube_sizes_m: torch.Tensor,
    interval_upper_bounds_m: torch.Tensor,
    *,
    minimum_size_m: float,
    maximum_size_m: float,
    tolerance_m: float = 1.0e-7,
) -> torch.Tensor:
    """Map continuous cube sizes to contiguous cache intervals.

    ``interval_upper_bounds_m`` contains the inclusive upper endpoint for
    every bucket, including the final bucket.  Exact midpoint ties therefore
    remain in the smaller-size bucket, matching intervals such as
    ``[39.00,39.25]`` and ``(39.25,39.75]`` mm.
    """

    sizes = torch.as_tensor(cube_sizes_m)
    upper = torch.as_tensor(interval_upper_bounds_m, device=sizes.device, dtype=sizes.dtype)
    if sizes.ndim != 1:
        raise ValueError(f"cube_sizes_m must be one-dimensional, got {tuple(sizes.shape)}.")
    if upper.ndim != 1 or upper.numel() == 0:
        raise ValueError("interval_upper_bounds_m must be a non-empty one-dimensional tensor.")
    if not bool(torch.isfinite(sizes).all()) or not bool(torch.isfinite(upper).all()):
        raise ValueError("Cube sizes and interval bounds must be finite.")
    if upper.numel() > 1 and not bool((upper[1:] > upper[:-1]).all()):
        raise ValueError("Scale-cache interval upper bounds must be strictly increasing.")
    minimum = float(minimum_size_m)
    maximum = float(maximum_size_m)
    tolerance = float(tolerance_m)
    if minimum <= 0.0 or maximum < minimum or tolerance < 0.0:
        raise ValueError(
            f"Invalid size range/tolerance: minimum={minimum}, maximum={maximum}, tolerance={tolerance}."
        )
    if bool((sizes < minimum - tolerance).any()) or bool((sizes > maximum + tolerance).any()):
        actual_min = float(sizes.min()) if sizes.numel() else math.nan
        actual_max = float(sizes.max()) if sizes.numel() else math.nan
        raise ValueError(
            f"Cube sizes [{actual_min}, {actual_max}] fall outside scale-cache range [{minimum}, {maximum}]."
        )
    # Exclude the last upper endpoint from bucketize: everything above the
    # penultimate endpoint belongs to the final bucket. ``right=False`` keeps
    # exact boundary values in the bucket on their left.
    return torch.bucketize(sizes, upper[:-1], right=False).to(dtype=torch.long)
