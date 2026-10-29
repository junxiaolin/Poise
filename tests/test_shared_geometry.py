"""Shared USD defaults preserve physics and per-environment authoring."""

import pytest

Usd = pytest.importorskip("pxr.Usd")
from pxr import Sdf, UsdGeom, UsdPhysics
from poise.sim.shared_geometry import reference_hand_definition


def composed_values(stage):
    return {
        str(p.GetPath()): ({a.GetName(): a.Get() for a in p.GetAttributes()},
                          {r.GetName(): r.GetTargets() for r in p.GetRelationships()})
        for p in stage.Traverse()
    }


def test_independent_body_shape_and_joint_overrides():
    stage = Usd.Stage.CreateInMemory()
    root = "/World/env_0/Robot"
    UsdGeom.Xform.Define(stage, root)
    for name in ("palm", "finger"):
        path = f"{root}/{name}"
        body = UsdGeom.Xform.Define(stage, path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.MassAPI.Apply(body).CreateMassAttr(0.2)
        shape = UsdGeom.Cube.Define(stage, path + "/shape").GetPrim()
        UsdPhysics.CollisionAPI.Apply(shape)
        shape.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.002)
        shape.CreateRelationship("material:binding").SetTargets([root + "/Looks/mat"])
    stage.DefinePrim(root + "/Looks/mat", "Material")
    UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(root + "/palm")).CreateFilteredPairsRel().SetTargets([root + "/finger"])
    joint = UsdPhysics.RevoluteJoint.Define(stage, root + "/joint")
    joint.CreateBody0Rel().SetTargets([root + "/palm"])
    joint.CreateBody1Rel().SetTargets([root + "/finger"])
    joint.CreateLowerLimitAttr(-10)
    before = composed_values(stage)
    layers = reference_hand_definition(stage, root)
    assert len(layers) == 1
    assert composed_values(stage) == before
    assert not stage.GetRootLayer().GetPrimAtPath(root).nameChildren
    other = "/World/env_1/Robot"
    stage.DefinePrim("/World/env_1", "Xform")
    Sdf.CopySpec(stage.GetRootLayer(), root, stage.GetRootLayer(), other)
    UsdPhysics.MassAPI(stage.GetPrimAtPath(root + "/palm")).GetMassAttr().Set(0.8)
    UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(root + "/joint")).GetLowerLimitAttr().Set(-20)
    stage.GetPrimAtPath(other + "/palm/shape").GetAttribute("physxCollision:contactOffset").Set(0.007)
    assert UsdPhysics.MassAPI(stage.GetPrimAtPath(other + "/palm")).GetMassAttr().Get() == pytest.approx(0.2)
    assert UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(other + "/joint")).GetLowerLimitAttr().Get() == -10
    assert stage.GetPrimAtPath(root + "/palm/shape").GetAttribute("physxCollision:contactOffset").Get() == pytest.approx(0.002)
    assert stage.GetPrimAtPath(other + "/palm").GetRelationship("physics:filteredPairs").GetTargets() == [Sdf.Path(other + "/finger")]
    assert stage.GetPrimAtPath(other + "/joint").GetRelationship("physics:body0").GetTargets() == [Sdf.Path(other + "/palm")]
    assert stage.GetPrimAtPath(other + "/palm/shape").GetRelationship("material:binding").GetTargets() == [Sdf.Path(other + "/Looks/mat")]
    assert not any(p.IsInstance() for p in stage.Traverse())


def test_public_hand_all_attributes_and_relationships_are_preserved():
    from poise.paths import resolve_path
    asset = resolve_path("data/robots/sharpa_wave/right_sharpa_wave.usda")
    if not asset.exists():
        pytest.skip("Public Sharpa assets are not downloaded")
    stage = Usd.Stage.CreateInMemory()
    root = "/World/env_0/Robot"
    stage.DefinePrim(root, "Xform").GetReferences().AddReference(str(asset))
    before = composed_values(stage)
    layers = reference_hand_definition(stage, root)
    assert layers
    assert composed_values(stage) == before
