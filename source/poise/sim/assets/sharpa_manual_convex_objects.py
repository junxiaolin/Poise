"""User-authored convex rigid assets converted from external CAD meshes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


SHARPA_MANUAL_CONVEX_DEFAULT_MASS_KG = 0.05
SHARPA_MANUAL_CONVEX_DEFAULT_FRICTION = 1.0
SHARPA_TRIANGULAR_PYRAMID_FAMILY = "triangular_pyramid"
SHARPA_TRUNCATED_SQUARE_BIPYRAMID_FAMILY = "truncated_square_bipyramid"
SHARPA_MANUAL_CONVEX_HEX_PRISM_FAMILY = "hex_prism"
SHARPA_CHAMFERED_CUBE_FAMILY = "cuboid"
SHARPA_TRUNCATED_SQUARE_BIPYRAMID_OBJECT_ID = "truncated_square_bipyramid_40mm_chamfered"
SHARPA_TRUNCATED_SQUARE_BIPYRAMID_DIMS_M = (
    0.039105571746826176,
    0.039105571746826176,
    0.040,
)
SHARPA_TRIANGULAR_PYRAMID70_OBJECT_ID = "triangular_pyramid_70mm_chamfered"
SHARPA_TRIANGULAR_PYRAMID70_DIMS_M = (
    0.05462178134918213,
    0.06307179641723633,
    0.05149790573120117,
)


@dataclass(frozen=True)
class ManualConvexObjectSpec:
    object_id: str
    display_name: str
    shape_family: str
    extents_xyz_m: tuple[float, float, float]
    source_sha256: str
    obj_sha256: str
    stl_sha256: str
    rigid_usd_sha256: str
    mesh_data_path: str
    mesh_data_sha256: str
    stable_cache_path: str | None = None
    stable_cache_sha256: str | None = None
    stable_cache_row_count: int = 0


SHARPA_TRUNCATED_SQUARE_BIPYRAMID_SPEC = ManualConvexObjectSpec(
    object_id=SHARPA_TRUNCATED_SQUARE_BIPYRAMID_OBJECT_ID,
    display_name="Chamfered Truncated Square Bipyramid 40mm",
    shape_family=SHARPA_TRUNCATED_SQUARE_BIPYRAMID_FAMILY,
    extents_xyz_m=SHARPA_TRUNCATED_SQUARE_BIPYRAMID_DIMS_M,
    source_sha256="1ec0089bd567027821780d7afe9705a0dfb7b0a4bb6812ed6536f9ac1d292ec9",
    obj_sha256="7c6b3c7be1dcc03129f896617897ab4a11921ae11b080be790441fc769b19020",
    stl_sha256="d4ba1509993dad3b52bafcfdc1763901b5b7b1460f986730231426e4adab7b53",
    rigid_usd_sha256="d1f9debba3168817911c4c3ef44952351a5ef405a348a85183a0ff686682f754",
    stable_cache_path=(
        "data/stable_grasps/sharpa_manual_convex_objects/"
        "truncated_square_bipyramid_40mm_chamfered_griddiverse430_"
        "variants10to20_palmup_omniwrist8_hold2s_training.pt"
    ),
    stable_cache_sha256="329688e6757a0cf2f66670e3eca944ec4b16b3196b6812960c653e54bfc59f17",
    stable_cache_row_count=4343,
    mesh_data_path=(
        "data/geometry_latents/"
        "sharpa_truncated_square_bipyramid40_bps64_mesh_data_64_v1.pt"
    ),
    mesh_data_sha256="540f4c7f597443cdfbba29b9b16888595c10815dcfa776e68fc9aac044ed08e3",
)

SHARPA_TRIANGULAR_PYRAMID70_SPEC = ManualConvexObjectSpec(
    object_id=SHARPA_TRIANGULAR_PYRAMID70_OBJECT_ID,
    display_name="Chamfered Triangular Pyramid 70mm",
    shape_family=SHARPA_TRIANGULAR_PYRAMID_FAMILY,
    extents_xyz_m=SHARPA_TRIANGULAR_PYRAMID70_DIMS_M,
    source_sha256="f6f22378fce2c795baed215248844ab96910c4d482f4cab604e438638e577a70",
    obj_sha256="4812068edf5d8d49f0f622c2361f7c0ea2372bb6d3109e5db5b9f3429bb8b35a",
    stl_sha256="b9a8a15a7a0b775d0f3534a61c9a14bbc5a160f5b7ad0188ee8d8f546c836617",
    rigid_usd_sha256="20b3f8f1ff526e387c8b479113f5d9f0271345776256e3abddc02f7444daa9f8",
    stable_cache_path=(
        "data/stable_grasps/sharpa_manual_convex_objects/"
        "triangular_pyramid_70mm_chamfered_griddiverse380_"
        "variants10to20_palmup_omniwrist8_hold2s_training.pt"
    ),
    stable_cache_sha256="3e0e54e47e21954347d907ba64ea5bd2b2cd726704fff54562be9ec51cca3934",
    stable_cache_row_count=3075,
    mesh_data_path=(
        "data/geometry_latents/"
        "sharpa_triangularpyramid70_bps64_mesh_data_64_v1.pt"
    ),
    mesh_data_sha256="1552b811f380a79b6cf8fe72ff7471e041fc48f74f64e0b9d690f1eef1caf11a",
)


SHARPA_NEW_MANUAL_TRAINING_OBJECT_SPECS = (
    ManualConvexObjectSpec(
        object_id="triangular_pyramid_65mm_chamfered",
        display_name="Chamfered Triangular Pyramid 65mm",
        shape_family=SHARPA_TRIANGULAR_PYRAMID_FAMILY,
        extents_xyz_m=(0.05029164981842041, 0.05807179641723633, 0.04741542434692383),
        source_sha256="29b99fac076eeda3b1ff7ce11275ca75522cf85ad4f854d13768acb3b8ccd2d6",
        obj_sha256="90d4970342a910d6081019de27acc649f05989aa2af7dc0918c83d7d9b35de5a",
        stl_sha256="7059e06a919dc31a35229a399e110a0109649846a4adc845102226134624658f",
        rigid_usd_sha256="75c3563990f73e80bf43cfed5007e501a5bece7a104b3a24daaafdcbd3a9afd6",
        mesh_data_path="data/geometry_latents/sharpa_triangularpyramid65_bps64_mesh_data_64_v1.pt",
        mesh_data_sha256="132683f9381b7deb86afa3b4811dc95ad8eff31aca9ee2e88beed426290b58b3",
    ),
    SHARPA_TRIANGULAR_PYRAMID70_SPEC,
    ManualConvexObjectSpec(
        object_id="triangular_pyramid_75mm_chamfered",
        display_name="Chamfered Triangular Pyramid 75mm",
        shape_family=SHARPA_TRIANGULAR_PYRAMID_FAMILY,
        extents_xyz_m=(0.058951905250549316, 0.0680718002319336, 0.05558038711547852),
        source_sha256="d3e15623044d1a174384ad1483b128522f841ec0a8a47ac0d5c4fba9b0203721",
        obj_sha256="d20bb00332493277454091bc7443d700fdfa71d0281feed2bafe0b1d7554b699",
        stl_sha256="be82c2d7997eb0da13e3b72c9ec94eed6377ec517e8760aef52bdf138131d815",
        rigid_usd_sha256="8dc08759aa5eb8ad412d8c36813eb6af4354ee817aa2ec6c19f887b45c4eea49",
        mesh_data_path="data/geometry_latents/sharpa_triangularpyramid75_bps64_mesh_data_64_v1.pt",
        mesh_data_sha256="a600c489a3d48d2e46168c17dd0e046e5af9a56d2e74a906510234e97856e080",
    ),
    ManualConvexObjectSpec(
        object_id="truncated_square_bipyramid_xy45mm_h40mm_chamfered",
        display_name="Chamfered Truncated Square Bipyramid XY45mm H40mm",
        shape_family=SHARPA_TRUNCATED_SQUARE_BIPYRAMID_FAMILY,
        extents_xyz_m=(0.044019477844238285, 0.044019477844238285, 0.040),
        source_sha256="1220de2527edfe2f150e274bba6f75e579e93a43bad57eb1b778e7490797e785",
        obj_sha256="cd2adcbcd1e8cf6d8c5eba20d24b6bd4fb92a3cfe5e6f5dbaa4309afb8df857a",
        stl_sha256="7fd3c30d2b3db2fec1debb4e6515caab439a1aa7a9fa6c661b90feaf66bc4ca6",
        rigid_usd_sha256="9a54346ca7b0357fe1a8688035cdddc7321e3bf8117951ba90930bd6d026b23c",
        mesh_data_path="data/geometry_latents/sharpa_truncated_square_bipyramid45_bps64_mesh_data_64_v1.pt",
        mesh_data_sha256="6340193389b32012b9f738799906112141abc4420def3a13c4a530d3578e02c5",
    ),
    ManualConvexObjectSpec(
        object_id="truncated_square_bipyramid_xy50mm_h50mm_chamfered",
        display_name="Chamfered Truncated Square Bipyramid XY50mm H50mm",
        shape_family=SHARPA_TRUNCATED_SQUARE_BIPYRAMID_FAMILY,
        extents_xyz_m=(0.04910557174682617, 0.04910557174682617, 0.050),
        source_sha256="c9802d0677d0d9b5e0bf16b830935fba56efbd13e8307028da48ea82eabaf303",
        obj_sha256="05d9ce8ffe03ad7f45f591d452f6752743cacfeed186d7f7560a459713e79146",
        stl_sha256="5c3270a992532aefb898204f8c7b5374746b4c535888a75874f9e7b4c211d242",
        rigid_usd_sha256="fd297d4ce104b74a47e920a9f875bbe54130e24533ae9ac31e8137bf16e90df1",
        mesh_data_path="data/geometry_latents/sharpa_truncated_square_bipyramid50_bps64_mesh_data_64_v1.pt",
        mesh_data_sha256="9ca53bb511420add00621127cfc32d3f88b916414d7821f1a2da7cf7de07064d",
    ),
    ManualConvexObjectSpec(
        object_id="regular_hex_prism_af45mm_h45mm_chamfered",
        display_name="Chamfered Regular Hex Prism AF45mm H45mm",
        shape_family=SHARPA_MANUAL_CONVEX_HEX_PRISM_FAMILY,
        extents_xyz_m=(0.05138417434692383, 0.045, 0.045),
        source_sha256="3d7a78b1cb84d3d611814401f1fc860db2c0c1c67258c6a0be1632730d4f977a",
        obj_sha256="e2a2917d4e83b54030490c9df1e37ff4cb84287600542132913c053cb99fea11",
        stl_sha256="372fa7f10cf08bb929400bbb932000c20db25f2a8034498574326ef3c15e901a",
        rigid_usd_sha256="fab056d658ff3181d2be39d71a2705b59c02dc68d9d9e02bea20a48ba079f2bf",
        mesh_data_path="data/geometry_latents/sharpa_hexprism45_bps64_mesh_data_64_v1.pt",
        mesh_data_sha256="0f039a0af1b170072dc8248c28513d7d9223f9e5b189e9ae26276f77ebd68226",
    ),
)

SHARPA_CHAMFERED_CUBE_SPECS = (
    ManualConvexObjectSpec(
        object_id="chamfered_cube_35mm",
        display_name="Chamfered Cube 35mm",
        shape_family=SHARPA_CHAMFERED_CUBE_FAMILY,
        extents_xyz_m=(0.035, 0.035, 0.035),
        source_sha256="b86c8f4134b08bcc59e3347a2cd90e6d1419d87c97b07907738bc3cea159f3b2",
        obj_sha256="c1a30111bfeaaf4f2ec8a3bda035342c18f516c9c88676aaf34b2c064bb195ac",
        stl_sha256="d55d9264872cf0012037f4595a92800897894c7524a292bafc508d10daabb278",
        rigid_usd_sha256="06d25c560e221c51d808add107bf22b7dabc12eb3e1f68c99a3e1f78ea64e3fa",
        mesh_data_path="data/geometry_latents/sharpa_chamfered_cube35_bps64_mesh_data_64_v1.pt",
        mesh_data_sha256="361278d22de9ac0e084663b4f95c75dc6a9386fa749c9ec95c186ef7308aedc5",
    ),
    ManualConvexObjectSpec(
        object_id="chamfered_cube_40mm",
        display_name="Chamfered Cube 40mm",
        shape_family=SHARPA_CHAMFERED_CUBE_FAMILY,
        extents_xyz_m=(0.040, 0.040, 0.040),
        source_sha256="5e3ec5e01731c2517e4267e461d3ce8d40003c65910eea3dede412372e8fda6c",
        obj_sha256="21d6558f07f006153fa1187f32a7e22af978214ecc2a86e0aa59231b223a5a98",
        stl_sha256="4af5f22877b1fd942bd046505b5ec688d2f9200c247065de0c814eeb14b0e43e",
        rigid_usd_sha256="40568a0381bbb5c0edba2d684a8b97e9118cf4aa8f9b8c4a824844648d55a3cf",
        mesh_data_path="data/geometry_latents/sharpa_chamfered_cube40_bps64_mesh_data_64_v1.pt",
        mesh_data_sha256="c80dee4d786b500878468bdf1be94dff626311b1368b8de628c965a487cf241c",
    ),
    ManualConvexObjectSpec(
        object_id="chamfered_cube_45mm",
        display_name="Chamfered Cube 45mm",
        shape_family=SHARPA_CHAMFERED_CUBE_FAMILY,
        extents_xyz_m=(0.045, 0.045, 0.045),
        source_sha256="4d7813fa9d3d2eef63c385a83f12f065f80d7ef98f022664ba6a23d58a039a00",
        obj_sha256="874d0bf3ef5ccff3ef75d1eab3c979e9a3e5213c9bf275cd00998502af05cd2d",
        stl_sha256="c0dd250dce7b7ca96dc0e37a943f5e823b96a848f1a57691e66999e8c76831e5",
        rigid_usd_sha256="b018317c8f83934e9dbf8af4975559cee1a5bffe764a8338fe9f7d4bf2e00815",
        mesh_data_path="data/geometry_latents/sharpa_chamfered_cube45_bps64_mesh_data_64_v1.pt",
        mesh_data_sha256="8b872c6a11f62d1957e6ff48517c3ef061f60eb985514d4ea02584654127bdaf",
    ),
)
SHARPA_CHAMFERED_CUBE_OBJECT_IDS = tuple(spec.object_id for spec in SHARPA_CHAMFERED_CUBE_SPECS)

SHARPA_MANUAL_CONVEX_OBJECT_SPECS = (
    SHARPA_TRUNCATED_SQUARE_BIPYRAMID_SPEC,
    *SHARPA_NEW_MANUAL_TRAINING_OBJECT_SPECS,
    *SHARPA_CHAMFERED_CUBE_SPECS,
)
SHARPA_MANUAL_CONVEX_OBJECT_IDS = tuple(spec.object_id for spec in SHARPA_MANUAL_CONVEX_OBJECT_SPECS)


def workspace_root() -> Path:
    from poise.paths import workspace_root as resolve_root
    return resolve_root()


def get_manual_convex_object_spec(object_id: str) -> ManualConvexObjectSpec:
    for spec in SHARPA_MANUAL_CONVEX_OBJECT_SPECS:
        if spec.object_id == object_id:
            return spec
    raise ValueError(f"Unknown manual convex object_id={object_id!r}; expected {SHARPA_MANUAL_CONVEX_OBJECT_IDS}")


def get_manual_convex_object_asset_dir(object_id: str) -> Path:
    get_manual_convex_object_spec(object_id)
    return workspace_root() / "data/objects/sharpa_manual_convex_objects" / object_id


def get_manual_convex_object_obj_path(object_id: str) -> Path:
    path = get_manual_convex_object_asset_dir(object_id) / f"{object_id}.obj"
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical OBJ asset: {path}")
    return path


def get_manual_convex_object_usd_path(object_id: str) -> Path:
    path = get_manual_convex_object_asset_dir(object_id) / f"{object_id}.usda"
    if not path.is_file():
        raise FileNotFoundError(f"Missing converted rigid asset: {path}")
    return path


def get_sharpa_manual_convex_object_cfg(
    *,
    object_id: str,
    prim_path: str = "/World/envs/env_.*/Object",
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.58),
    init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    disable_gravity: bool = False,
    mass_kg: float = SHARPA_MANUAL_CONVEX_DEFAULT_MASS_KG,
    friction: float = SHARPA_MANUAL_CONVEX_DEFAULT_FRICTION,
):
    """Create an Isaac Lab rigid-object config for a converted manual convex mesh."""

    if float(friction) <= 0.0:
        raise ValueError(f"friction must be positive, got {friction}")

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(get_manual_convex_object_usd_path(object_id)),
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
