"""USD composition regression; run in an environment providing pxr."""

from importlib.resources import files

import pytest

Usd = pytest.importorskip("pxr.Usd")


def test_collision_overlay_restores_body_pairs_and_blocks_mesh_pairs(tmp_path):
    for name in ("right_sharpa_wave.usda", "poise_collision.usda"):
        (tmp_path / name).write_text(files("poise.assets.sharpa_wave").joinpath(name).read_text())
    # Minimal weaker public layer, including one mesh-level exclusion.
    (tmp_path / "upstream.usda").write_text('''#usda 1.0
def Xform "right_sharpa_wave" {
    def Xform "root_joint" {
        bool physxArticulation:enabledSelfCollisions = false
    }
    def Xform "right_hand_C_MC" {
        def Xform "collisions" {
            def Xform "right_hand_C_MC" {
                def Mesh "mesh" (prepend apiSchemas = ["PhysicsFilteredPairsAPI"]) {
                    rel physics:filteredPairs = </right_sharpa_wave/wrong_mesh>
                }
            }
        }
    }
}
''')
    stage = Usd.Stage.Open(str(tmp_path / "right_sharpa_wave.usda"))
    root = "/right_sharpa_wave"
    assert stage.GetPrimAtPath(root + "/root_joint").GetAttribute("physxArticulation:enabledSelfCollisions").Get() is True
    mesh = stage.GetPrimAtPath(root + "/right_hand_C_MC/collisions/right_hand_C_MC/mesh")
    assert "PhysicsFilteredPairsAPI" not in mesh.GetAppliedSchemas()
    assert not mesh.GetRelationship("physics:filteredPairs").GetTargets()
    body = stage.GetPrimAtPath(root + "/right_hand_C_MC")
    assert len(body.GetRelationship("physics:filteredPairs").GetTargets()) == 9
    assert "PhysicsFilteredPairsAPI" in body.GetAppliedSchemas()
    # Isaac Lab references the default prim under each environment's Robot.
    host = Usd.Stage.CreateInMemory()
    robot = host.DefinePrim("/World/env_0/Robot", "Xform")
    robot.GetReferences().AddReference(str(tmp_path / "right_sharpa_wave.usda"))
    targets = host.GetPrimAtPath("/World/env_0/Robot/right_hand_C_MC").GetRelationship("physics:filteredPairs").GetTargets()
    assert len(targets) == 9
    assert all(str(t).startswith("/World/env_0/Robot/") for t in targets)
