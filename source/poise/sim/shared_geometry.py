"""Share immutable USD hand definitions without sharing live physics state."""

from pxr import Sdf, Usd


def reference_hand_definition(stage, source_path):
    """Share authored defaults, preserving independently editable composed prims.

    All spawner overrides are captured before cloning. Each environment gets a
    reference to an immutable layer, not inheritance from a live environment.
    The layer is kept alive by the environment and never written to disk.
    """
    path = Sdf.Path(source_path)
    temporary = Usd.Stage.CreateInMemory()
    temporary.DefinePrim(path.GetParentPath(), "Xform")
    Sdf.CopySpec(stage.GetRootLayer(), path, temporary.GetRootLayer(), path)
    flat = temporary.Flatten()
    layer = Sdf.Layer.CreateAnonymous("poise_hand_definition.usda")
    Sdf.CopySpec(flat, path, layer, "/Robot")
    with Sdf.ChangeBlock():
        spec = stage.GetRootLayer().GetPrimAtPath(path)
        del spec.nameParent.nameChildren[spec.name]
        spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), path)
        spec.referenceList.explicitItems = [Sdf.Reference(layer.identifier, "/Robot")]
    return [layer]
