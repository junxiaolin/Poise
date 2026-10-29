"""Import the user-provided, metre-scale D40 x L160 STL without changing its frame."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


OBJECT_ID = "hexprism_d40_l160_original_m"
SOURCE_SHA256 = "62b01d0c15a0fec410647e55a7e0298f5921f5ff989665d58ce7e4c61b301e12"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("This importer is specific to the supplied D40 x L160 STL")
    mesh = trimesh.load_mesh(args.source, process=True)
    if not mesh.is_watertight or not mesh.is_convex:
        raise ValueError("Expected a watertight convex prism")
    if not np.allclose(mesh.bounds, -mesh.bounds[::-1], atol=1e-8):
        raise ValueError("Expected an origin-centred mesh")
    root = Path(__file__).resolve().parents[1]
    out = root / "data/objects/sharpa_manual_hex_prisms" / OBJECT_ID
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{OBJECT_ID}.stl").write_bytes(raw)
    obj = "# Original STL frame and metre units preserved\n"
    obj += "".join("v " + " ".join(f"{x:.12g}" for x in v) + "\n" for v in mesh.vertices)
    obj += "".join("f " + " ".join(str(int(x) + 1) for x in f) + "\n" for f in mesh.faces)
    (out / f"{OBJECT_ID}.obj").write_text(obj)
    points = ",\n".join("            (" + ", ".join(f"{x:.12g}" for x in v) + ")" for v in mesh.vertices)
    usd = '''#usda 1.0
(
    defaultPrim = "Object"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Object" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
    bool physics:rigidBodyEnabled = 1
    float physics:mass = 0.05
    def Mesh "Part0" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI"]) {
        int[] faceVertexCounts = COUNTS
        int[] faceVertexIndices = INDICES
        point3f[] points = [
POINTS
        ]
        uniform token subdivisionScheme = "none"
        bool physics:collisionEnabled = 1
        uniform token physics:approximation = "convexHull"
        token physxCollision:approximation = "convexHull"
        color3f[] primvars:displayColor = [(0.95, 0.66, 0.12)] (interpolation = "constant")
    }
}
'''.replace("COUNTS", str([3] * len(mesh.faces))).replace("INDICES", str(mesh.faces.reshape(-1).tolist())).replace("POINTS", points)
    (out / f"{OBJECT_ID}.usda").write_text(usd)
    report = {"object_id": OBJECT_ID, "source_sha256": SOURCE_SHA256,
              "units": "metres", "transform": "identity", "long_axis": "Z",
              "extents_xyz_m": mesh.extents.tolist(), "volume_m3": mesh.volume,
              "diameter_definition": "40 mm vertex-to-vertex; flat-to-flat 34.641 mm",
              "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(out.iterdir()) if p.suffix in (".stl", ".obj", ".usda")}}
    (out / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
