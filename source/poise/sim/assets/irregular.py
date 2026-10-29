"""Lightweight irregular object assets for G1 geometry-latent experiments."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch


IRREGULAR_DEFAULT_MASS_KG = 0.05
"""Default irregular-object mass in kilograms."""

IRREGULAR_DEFAULT_FRICTION = 1.0
"""Default static/dynamic friction coefficient."""

IRREGULAR_SHAPE_NAMES = (
    "cone",
    "frustum",
    "ellipsoid",
    "icosahedron20",
    "triangular_prism",
    "rounded_box",
    "asym_capsule",
    "convex_mesh",
    "l_shape",
    "t_shape",
    "u_shape",
    "c_shape",
    "cross_shape",
    "notched_box",
    "step_block",
)
"""Shape families encoded by the G1 irregular object ids."""

G1_IRREGULAR_OBJECT_SPECS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("cone_4x5cm", (0.04, 0.04, 0.05)),
    ("frustum_4x5cm", (0.04, 0.04, 0.05)),
    ("ellipsoid_3x4x5cm", (0.03, 0.04, 0.05)),
    ("triangular_prism_4x4x5cm", (0.04, 0.04, 0.05)),
    ("rounded_box_4x5x3cm", (0.04, 0.05, 0.03)),
    ("asym_capsule_4x5cm", (0.04, 0.04, 0.05)),
    ("convex_mesh_wedge_4x5x4cm", (0.04, 0.05, 0.04)),
    ("l_shape_5x5x4cm", (0.05, 0.05, 0.04)),
    ("t_shape_5x5x4cm", (0.05, 0.05, 0.04)),
)
"""Default G1 irregular object set.

The dimensions are total bbox dimensions in meters, matching the convention
used by primitive multi-shape configs and support descriptors.
"""

CONSERVATIVE_NONCONVEX_OBJECT_SPECS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("u_shape_5x5x4cm", (0.05, 0.05, 0.04)),
    ("c_shape_5x5x4cm", (0.05, 0.05, 0.04)),
    ("cross_shape_5x5x4cm", (0.05, 0.05, 0.04)),
    ("notched_box_5x4x4cm", (0.05, 0.04, 0.04)),
    ("step_block_5x4x4cm", (0.05, 0.04, 0.04)),
)
"""Conservative multi-convex nonconvex objects for G2 held-out evaluation."""

SHARPA_HAMMER_HANDLE2X20X2_HEAD5X3X3_OBJECT_ID = "t_shape_hammer_handle2x20x2cm_y_head5x3x3cm_x"
"""Sharpa hammer-like object id: cuboid handle on local Y with cuboid head on +Y."""

SHARPA_HAMMER_HANDLE2X20X2_HEAD5X3X3_DIMS_M = (0.05, 0.215, 0.03)
"""Total bbox dimensions for the Sharpa hammer-like object."""

SHARPA_HEX_PRISM_HAMMER_CHAMFERED_OBJECT_ID = "t_shape_hex_prism_hammer_chamfered"
"""CAD-authored hammer made from joined chamfered regular-hex-prism solids."""

SHARPA_HEX_PRISM_HAMMER_CHAMFERED_DIMS_M = (0.05, 0.215, 0.03364102)
"""Canonical XYZ bbox dimensions of the CAD-authored hex-prism hammer."""


@dataclass(frozen=True)
class MeshPart:
    """A convex collision/visual mesh part in object-local coordinates."""

    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, ...], ...]


def workspace_root() -> Path:
    """Resolve the workspace root without importing package-global config."""

    from poise.paths import workspace_root as resolve_root
    return resolve_root()


def irregular_shape_name_from_object_id(object_id: str) -> str:
    """Return the irregular shape family encoded in an object id."""

    for shape_name in IRREGULAR_SHAPE_NAMES:
        if object_id.startswith(f"{shape_name}_"):
            return shape_name
    raise ValueError(
        f"Unsupported irregular object_id={object_id!r}; expected one of "
        f"{', '.join(f'{name}_*' for name in IRREGULAR_SHAPE_NAMES)}."
    )


def is_irregular_object_id(object_id: str) -> bool:
    """Return whether an object id belongs to the G1 irregular family."""

    return any(object_id.startswith(f"{shape_name}_") for shape_name in IRREGULAR_SHAPE_NAMES)


def validate_irregular_dims_m(dims_m: tuple[float, float, float]) -> tuple[float, float, float]:
    """Validate bbox dimensions and return them as floats."""

    dims = tuple(float(v) for v in dims_m)
    if len(dims) != 3 or any(v <= 0.0 for v in dims):
        raise ValueError(f"dims_m must contain three positive values, got {dims_m!r}.")
    return dims


def irregular_asset_path(object_id: str) -> Path:
    """Return the generated USD path for an irregular object."""

    if object_id == SHARPA_HEX_PRISM_HAMMER_CHAMFERED_OBJECT_ID:
        return (
            workspace_root()
            / "data/objects/sharpa_hex_prism_hammer"
            / object_id
            / f"{object_id}.usda"
        )
    return workspace_root() / "data/objects/g1_irregular_usd" / f"{object_id}.usda"


def _mesh_part_from_obj(path: Path) -> MeshPart:
    """Load a triangulated OBJ as a dependency-free ``MeshPart``."""

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("v "):
            fields = line.split()
            vertices.append(tuple(float(value) for value in fields[1:4]))
        elif line.startswith("f "):
            indices = tuple(int(field.split("/", 1)[0]) - 1 for field in line.split()[1:])
            if len(indices) < 3:
                raise ValueError(f"Invalid OBJ face in {path}: {line!r}")
            faces.append(indices)
    if not vertices or not faces:
        raise ValueError(f"OBJ mesh is empty or invalid: {path}")
    return MeshPart(vertices=tuple(vertices), faces=tuple(faces))


def _hex_prism_hammer_collision_parts() -> tuple[MeshPart, ...]:
    asset_dir = (
        workspace_root()
        / "data/objects/sharpa_hex_prism_hammer"
        / SHARPA_HEX_PRISM_HAMMER_CHAMFERED_OBJECT_ID
    )
    stem = SHARPA_HEX_PRISM_HAMMER_CHAMFERED_OBJECT_ID
    return (
        _mesh_part_from_obj(asset_dir / f"{stem}.collision_handle.obj"),
        _mesh_part_from_obj(asset_dir / f"{stem}.collision_head.obj"),
    )


def _bbox_scale_vertices(vertices: torch.Tensor, dims_m: tuple[float, float, float]) -> torch.Tensor:
    """Center and scale vertices so their bbox exactly matches ``dims_m``."""

    dims = torch.tensor(dims_m, dtype=torch.float32).view(1, 3)
    lower = vertices.amin(dim=0, keepdim=True)
    upper = vertices.amax(dim=0, keepdim=True)
    center = 0.5 * (lower + upper)
    span = torch.clamp(upper - lower, min=1.0e-8)
    return (vertices - center) / span * dims


def _cuboid_part(
    dims_m: tuple[float, float, float],
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> MeshPart:
    hx, hy, hz = (0.5 * float(v) for v in dims_m)
    cx, cy, cz = (float(v) for v in center_m)
    vertices = (
        (cx - hx, cy - hy, cz - hz),
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
    )
    faces = (
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    )
    return MeshPart(vertices=vertices, faces=faces)


def _ring_mesh(
    radius_profile: torch.Tensor,
    z_values: torch.Tensor,
    *,
    theta_resolution: int,
    x_offsets: torch.Tensor | None = None,
) -> MeshPart:
    theta = torch.linspace(0.0, 2.0 * math.pi, theta_resolution + 1, dtype=torch.float32)[:-1]
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    if x_offsets is None:
        x_offsets = torch.zeros_like(z_values)
    rings = []
    for radius, z_value, x_offset in zip(radius_profile, z_values, x_offsets, strict=True):
        rings.append(
            torch.stack(
                (
                    x_offset + radius * cos_t,
                    radius * sin_t,
                    torch.full_like(theta, z_value),
                ),
                dim=-1,
            )
        )
    vertices = torch.cat(rings, dim=0)
    faces: list[tuple[int, ...]] = []
    ring_count = int(z_values.numel())
    for ring_idx in range(ring_count - 1):
        base = ring_idx * theta_resolution
        next_base = (ring_idx + 1) * theta_resolution
        for theta_idx in range(theta_resolution):
            j = (theta_idx + 1) % theta_resolution
            faces.append((base + theta_idx, base + j, next_base + j, next_base + theta_idx))
    faces.append(tuple(range(theta_resolution - 1, -1, -1)))
    top_start = (ring_count - 1) * theta_resolution
    faces.append(tuple(top_start + i for i in range(theta_resolution)))
    return MeshPart(
        vertices=tuple(tuple(float(v) for v in row) for row in vertices.tolist()),
        faces=tuple(faces),
    )


def _cone_or_frustum_part(
    dims_m: tuple[float, float, float],
    *,
    top_radius_ratio: float,
    theta_resolution: int = 48,
) -> MeshPart:
    radius = 0.5 * float(dims_m[0])
    half_height = 0.5 * float(dims_m[2])
    bottom_radius = radius
    top_radius = max(0.001 * radius, float(top_radius_ratio) * radius)
    z_values = torch.tensor((-half_height, half_height), dtype=torch.float32)
    radii = torch.tensor((bottom_radius, top_radius), dtype=torch.float32)
    return _ring_mesh(radii, z_values, theta_resolution=theta_resolution)


def _ellipsoid_like_part(
    dims_m: tuple[float, float, float],
    *,
    theta_resolution: int = 48,
    axial_resolution: int = 25,
    boxiness: float = 1.0,
    skew_x_m: float = 0.0,
) -> MeshPart:
    # boxiness=1 is an ellipsoid. Larger values make a rounded-box/superellipsoid.
    theta = torch.linspace(0.0, 2.0 * math.pi, theta_resolution + 1, dtype=torch.float32)[:-1]
    phi = torch.linspace(0.0, math.pi, axial_resolution, dtype=torch.float32)
    pp, tt = torch.meshgrid(phi, theta, indexing="ij")

    def signed_pow(value: torch.Tensor, power: float) -> torch.Tensor:
        return torch.sign(value) * torch.abs(value).pow(power)

    power = 1.0 / max(float(boxiness), 1.0e-6)
    sx, sy, sz = (0.5 * float(v) for v in dims_m)
    sin_p = torch.sin(pp)
    cos_p = torch.cos(pp)
    cos_t = torch.cos(tt)
    sin_t = torch.sin(tt)
    x = sx * signed_pow(sin_p, power) * signed_pow(cos_t, power)
    y = sy * signed_pow(sin_p, power) * signed_pow(sin_t, power)
    z = sz * signed_pow(cos_p, power)
    if skew_x_m != 0.0:
        x = x + float(skew_x_m) * (z / max(sz, 1.0e-8))
    vertices = torch.stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)), dim=-1)

    faces: list[tuple[int, ...]] = []
    for p_idx in range(axial_resolution - 1):
        base = p_idx * theta_resolution
        next_base = (p_idx + 1) * theta_resolution
        for t_idx in range(theta_resolution):
            j = (t_idx + 1) % theta_resolution
            faces.append((base + t_idx, base + j, next_base + j, next_base + t_idx))
    vertices = _bbox_scale_vertices(vertices, dims_m)
    return MeshPart(
        vertices=tuple(tuple(float(v) for v in row) for row in vertices.tolist()),
        faces=tuple(faces),
    )


def _triangular_prism_part(
    dims_m: tuple[float, float, float],
    *,
    apex_x_ratio: float = 0.0,
) -> MeshPart:
    x, y, z = (float(v) for v in dims_m)
    hx, hy, hz = 0.5 * x, 0.5 * y, 0.5 * z
    apex_x = float(apex_x_ratio) * hx
    vertices = (
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (apex_x, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (apex_x, hy, hz),
    )
    faces = ((0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5))
    return MeshPart(vertices=vertices, faces=faces)


def _convex_wedge_part(dims_m: tuple[float, float, float]) -> MeshPart:
    x, y, z = (float(v) for v in dims_m)
    hx, hy, hz = 0.5 * x, 0.5 * y, 0.5 * z
    vertices = torch.tensor(
        (
            (-hx, -hy, -hz),
            (hx, -hy, -hz),
            (hx, hy, -hz),
            (-hx, hy, -hz),
            (-0.55 * hx, -0.80 * hy, hz),
            (0.75 * hx, -0.80 * hy, hz),
            (0.75 * hx, 0.64 * hy, hz),
            (-0.55 * hx, 0.64 * hy, hz),
        ),
        dtype=torch.float32,
    )
    vertices = _bbox_scale_vertices(vertices, dims_m)
    faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    return MeshPart(
        vertices=tuple(tuple(float(v) for v in row) for row in vertices.tolist()),
        faces=faces,
    )


def _icosahedron20_part(dims_m: tuple[float, float, float]) -> MeshPart:
    """Regular icosahedron: sphere-like, with 20 equal triangular faces."""

    phi = 0.5 * (1.0 + math.sqrt(5.0))
    vertices = torch.tensor(
        (
            (0.0, -1.0, -phi), (0.0, -1.0, phi),
            (0.0, 1.0, -phi), (0.0, 1.0, phi),
            (-1.0, -phi, 0.0), (-1.0, phi, 0.0),
            (1.0, -phi, 0.0), (1.0, phi, 0.0),
            (-phi, 0.0, -1.0), (phi, 0.0, -1.0),
            (-phi, 0.0, 1.0), (phi, 0.0, 1.0),
        ),
        dtype=torch.float32,
    )
    vertices = _bbox_scale_vertices(vertices, dims_m)
    faces = (
        (0, 4, 8), (2, 0, 8), (2, 8, 5), (2, 9, 0), (6, 0, 9),
        (6, 1, 4), (6, 4, 0), (7, 2, 5), (7, 5, 3), (7, 9, 2),
        (10, 1, 3), (10, 3, 5), (10, 4, 1), (10, 5, 8), (10, 8, 4),
        (11, 1, 6), (11, 3, 1), (11, 6, 9), (11, 7, 3), (11, 9, 7),
    )
    return MeshPart(
        vertices=tuple(tuple(float(value) for value in row) for row in vertices.tolist()),
        faces=faces,
    )


def _asym_capsule_part(
    dims_m: tuple[float, float, float],
    *,
    theta_resolution: int = 48,
    axial_resolution: int = 33,
) -> MeshPart:
    z = torch.linspace(-0.5, 0.5, axial_resolution, dtype=torch.float32)
    # A pear-like capsule profile: rounded ends, larger upper shoulder, slight
    # centerline skew. It is scaled back to the requested bbox after sampling.
    rounded = torch.clamp(1.0 - (2.0 * z).abs().pow(3.2), min=0.0).sqrt()
    asym = 0.82 + 0.30 * (z + 0.5)
    radii = 0.5 * float(dims_m[0]) * rounded * asym
    radii = torch.clamp(radii, min=0.001 * float(dims_m[0]))
    z_values = z * float(dims_m[2])
    x_offsets = 0.10 * float(dims_m[0]) * torch.sin(math.pi * z)
    part = _ring_mesh(radii, z_values, theta_resolution=theta_resolution, x_offsets=x_offsets)
    vertices = _bbox_scale_vertices(torch.tensor(part.vertices, dtype=torch.float32), dims_m)
    return MeshPart(
        vertices=tuple(tuple(float(v) for v in row) for row in vertices.tolist()),
        faces=part.faces,
    )


def mesh_parts_for_irregular_object(
    object_id: str,
    dims_m: tuple[float, float, float],
) -> tuple[MeshPart, ...]:
    """Return one or more convex mesh parts for an irregular object."""

    dims = validate_irregular_dims_m(dims_m)
    if object_id == SHARPA_HEX_PRISM_HAMMER_CHAMFERED_OBJECT_ID:
        expected_dims = SHARPA_HEX_PRISM_HAMMER_CHAMFERED_DIMS_M
        if any(abs(float(got) - float(expected)) > 1.0e-6 for got, expected in zip(dims, expected_dims, strict=True)):
            raise ValueError(
                f"{object_id!r} expects bbox dims {expected_dims!r}; got {dims!r}."
            )
        return _hex_prism_hammer_collision_parts()
    if object_id == SHARPA_HAMMER_HANDLE2X20X2_HEAD5X3X3_OBJECT_ID:
        expected_dims = SHARPA_HAMMER_HANDLE2X20X2_HEAD5X3X3_DIMS_M
        if any(abs(float(got) - float(expected)) > 1.0e-6 for got, expected in zip(dims, expected_dims, strict=True)):
            raise ValueError(
                f"{object_id!r} expects bbox dims {expected_dims!r}; got {dims!r}."
            )
        return (
            _cuboid_part((0.02, 0.20, 0.02), center_m=(0.0, 0.0, 0.0)),
            _cuboid_part((0.05, 0.03, 0.03), center_m=(0.0, 0.10, 0.0)),
        )
    shape_name = irregular_shape_name_from_object_id(object_id)
    if shape_name == "cone":
        return (_cone_or_frustum_part(dims, top_radius_ratio=0.0),)
    if shape_name == "frustum":
        return (_cone_or_frustum_part(dims, top_radius_ratio=0.55),)
    if shape_name == "ellipsoid":
        return (_ellipsoid_like_part(dims, boxiness=1.0),)
    if shape_name == "icosahedron20":
        return (_icosahedron20_part(dims),)
    if shape_name == "triangular_prism":
        apex_x_ratio = -0.28 if object_id.startswith("triangular_prism_scalene_") else 0.0
        return (_triangular_prism_part(dims, apex_x_ratio=apex_x_ratio),)
    if shape_name == "rounded_box":
        return (_ellipsoid_like_part(dims, boxiness=3.0),)
    if shape_name == "asym_capsule":
        return (_asym_capsule_part(dims),)
    if shape_name == "convex_mesh":
        return (_convex_wedge_part(dims),)
    if shape_name == "l_shape":
        x, y, z = dims
        width = 0.45 * min(x, y)
        return (
            _cuboid_part((width, y, z), center_m=(-0.5 * x + 0.5 * width, 0.0, 0.0)),
            _cuboid_part((x, width, z), center_m=(0.0, -0.5 * y + 0.5 * width, 0.0)),
        )
    if shape_name == "t_shape":
        x, y, z = dims
        width = 0.36 * min(x, y)
        return (
            _cuboid_part((width, y, z), center_m=(0.0, 0.0, 0.0)),
            _cuboid_part((x, width, z), center_m=(0.0, 0.5 * y - 0.5 * width, 0.0)),
        )
    if shape_name == "u_shape":
        x, y, z = dims
        width = 0.30 * min(x, y)
        return (
            _cuboid_part((width, y, z), center_m=(-0.5 * x + 0.5 * width, 0.0, 0.0)),
            _cuboid_part((width, y, z), center_m=(0.5 * x - 0.5 * width, 0.0, 0.0)),
            _cuboid_part((x, width, z), center_m=(0.0, -0.5 * y + 0.5 * width, 0.0)),
        )
    if shape_name == "c_shape":
        x, y, z = dims
        width = 0.30 * min(x, y)
        return (
            _cuboid_part((width, y, z), center_m=(-0.5 * x + 0.5 * width, 0.0, 0.0)),
            _cuboid_part((x, width, z), center_m=(0.0, -0.5 * y + 0.5 * width, 0.0)),
            _cuboid_part((x, width, z), center_m=(0.0, 0.5 * y - 0.5 * width, 0.0)),
        )
    if shape_name == "cross_shape":
        x, y, z = dims
        width = 0.34 * min(x, y)
        return (
            _cuboid_part((width, y, z), center_m=(0.0, 0.0, 0.0)),
            _cuboid_part((x, width, z), center_m=(0.0, 0.0, 0.0)),
        )
    if shape_name == "notched_box":
        x, y, z = dims
        notch_x = 0.38 * x
        notch_y = 0.38 * y
        return (
            _cuboid_part((x, y - notch_y, z), center_m=(0.0, -0.5 * notch_y, 0.0)),
            _cuboid_part((x - notch_x, notch_y, z), center_m=(-0.5 * notch_x, 0.5 * y - 0.5 * notch_y, 0.0)),
        )
    if shape_name == "step_block":
        x, y, z = dims
        step_w = x / 3.0
        parts = []
        for idx, height_scale in enumerate((1.0 / 3.0, 2.0 / 3.0, 1.0)):
            height = height_scale * z
            center_x = -0.5 * x + (idx + 0.5) * step_w
            center_z = -0.5 * z + 0.5 * height
            parts.append(_cuboid_part((step_w, y, height), center_m=(center_x, 0.0, center_z)))
        return tuple(parts)
    raise ValueError(f"Unsupported irregular object_id={object_id!r}.")


def _format_tuple(values: tuple[float, ...]) -> str:
    return "(" + ", ".join(f"{float(v):.8g}" for v in values) + ")"


def _format_face_indices(faces: tuple[tuple[int, ...], ...]) -> tuple[str, str]:
    counts = ", ".join(str(len(face)) for face in faces)
    indices = ", ".join(str(index) for face in faces for index in face)
    return counts, indices


def write_irregular_object_usd(
    object_id: str,
    dims_m: tuple[float, float, float],
    output_path: Path | None = None,
) -> Path:
    """Write a compact USD mesh asset for an irregular object."""

    output = irregular_asset_path(object_id) if output_path is None else Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    parts = mesh_parts_for_irregular_object(object_id, dims_m)
    color_cycle = (
        (0.30, 0.74, 0.94),
        (0.95, 0.68, 0.32),
        (0.42, 0.78, 0.67),
    )

    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Object"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "Object" (',
        '    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]',
        ")",
        "{",
        "    bool physics:rigidBodyEnabled = 1",
        f"    float physics:mass = {IRREGULAR_DEFAULT_MASS_KG:.8g}",
    ]
    for part_idx, part in enumerate(parts):
        counts, indices = _format_face_indices(part.faces)
        color = color_cycle[part_idx % len(color_cycle)]
        lines.extend(
            [
                f'    def Mesh "Part{part_idx}" (',
                '        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI"]',
                "    )",
                "    {",
                f"        int[] faceVertexCounts = [{counts}]",
                f"        int[] faceVertexIndices = [{indices}]",
                "        point3f[] points = [",
            ]
        )
        for vertex in part.vertices:
            lines.append(f"            {_format_tuple(vertex)},")
        lines.extend(
            [
                "        ]",
                '        uniform token subdivisionScheme = "none"',
                "        bool physics:collisionEnabled = 1",
                '        uniform token physics:approximation = "convexHull"',
                '        token physxCollision:approximation = "convexHull"',
                f"        color3f[] primvars:displayColor = [{_format_tuple(color)}] (",
                '            interpolation = "constant"',
                "        )",
                "    }",
            ]
        )
    lines.append("}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def ensure_irregular_object_usd(object_id: str, dims_m: tuple[float, float, float]) -> Path:
    """Create the irregular USD asset if it does not already exist."""

    path = irregular_asset_path(object_id)
    if not path.is_file():
        write_irregular_object_usd(object_id, dims_m, path)
    return path


def get_irregular_object_cfg(
    *,
    object_id: str,
    dims_m: tuple[float, float, float],
    prim_path: str = "/World/envs/env_.*/Object",
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    disable_gravity: bool = False,
    mass_kg: float = IRREGULAR_DEFAULT_MASS_KG,
    friction: float = IRREGULAR_DEFAULT_FRICTION,
):
    """Create an Isaac Lab rigid object config backed by a generated USD mesh."""

    usd_path = ensure_irregular_object_usd(object_id, dims_m)

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=disable_gravity,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(mass_kg)),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=init_rot),
    )


def sample_mesh_parts_surface_points(parts: tuple[MeshPart, ...], samples_per_edge: int = 9) -> torch.Tensor:
    """Deterministically sample mesh faces in object-local coordinates."""

    resolution = max(2, int(samples_per_edge))
    sampled: list[torch.Tensor] = []
    for part in parts:
        vertices = torch.tensor(part.vertices, dtype=torch.float32)
        for face in part.faces:
            if len(face) < 3:
                continue
            # Fan triangulation works for the convex polygon faces used here.
            for tri_idx in range(1, len(face) - 1):
                a = vertices[face[0]]
                b = vertices[face[tri_idx]]
                c = vertices[face[tri_idx + 1]]
                tri_points = []
                for i in range(resolution):
                    for j in range(resolution - i):
                        u = float(i) / float(resolution - 1)
                        v = float(j) / float(resolution - 1)
                        if u + v > 1.0:
                            continue
                        tri_points.append(a + u * (b - a) + v * (c - a))
                sampled.append(torch.stack(tri_points, dim=0))
    if not sampled:
        raise ValueError("No mesh surface points were sampled.")
    return torch.cat(sampled, dim=0)


def sample_irregular_surface_points(
    object_id: str,
    dims_m: tuple[float, float, float],
    samples_per_edge: int = 9,
) -> torch.Tensor:
    """Sample local surface points for support descriptors and contact proxies."""

    return sample_mesh_parts_surface_points(mesh_parts_for_irregular_object(object_id, dims_m), samples_per_edge)
