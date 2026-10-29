"""Compare composed hand physics and geometry; import USD only when requested."""


def compare_hand_physics(reference, candidate):
    from pxr import Usd

    stages = [Usd.Stage.Open(str(path)) for path in (reference, candidate)]
    if not all(stages):
        raise ValueError("Could not open both hand USD stages")
    def snapshot(stage):
        records = {}
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/right_sharpa_wave"), Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            # The model uses embedded mesh data. Shader/viewport settings do not
            # affect the task's collision shapes, inertias, actuators or joints.
            if "/Looks" in path or "/visuals" in path:
                continue
            records[path + ":type"] = prim.GetTypeName()
            records[path + ":active"] = prim.IsActive()
            records[path + ":instanceable"] = prim.IsInstanceable()
            records[path + ":schemas"] = sorted(prim.GetAppliedSchemas())
            for attr in prim.GetAttributes():
                if attr.GetName().startswith(("isaac:", "primvars:", "userProperties:")):
                    continue
                value = attr.Get()
                if value is not None:
                    records[str(attr.GetPath())] = (str(attr.GetTypeName()), value)
            for rel in prim.GetRelationships():
                if rel.GetName().startswith("material:"):
                    continue
                targets = rel.GetTargets()
                # An explicit empty target list blocks a weaker exclusion and
                # is equivalent to an absent relation for collision filtering.
                if targets:
                    records[str(rel.GetPath())] = sorted(str(t) for t in targets)
        return records
    old, new = [snapshot(stage) for stage in stages]
    differences = [key for key in sorted(old.keys() | new.keys()) if old.get(key) != new.get(key)]
    pairs = {k: v for k, v in new.items() if k.endswith(".physics:filteredPairs")}
    self_collision = {k: bool(v[1]) for k, v in new.items() if k.endswith(".physxArticulation:enabledSelfCollisions")}
    return {"passed": not differences, "compared_records": len(old.keys() | new.keys()),
            "reference": str(reference), "candidate": str(candidate),
            "differences": differences, "filtered_pairs": pairs, "self_collision": self_collision}
