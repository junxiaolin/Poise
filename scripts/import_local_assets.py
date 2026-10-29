"""Stage the HexPrism40 reference inputs into an independent local data tree.

This developer utility requires explicitly supplied source directories. Data is
ignored by Git; redistribution is subject to each upstream asset's license.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", required=True, type=Path)
    parser.add_argument("--limits-usda", required=True, type=Path)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    relative_inputs = [
        "data/stable_grasps/sharpa_manual_hex_prisms/usda09/regular_hex_prism_af40mm_h40mm_chamfered_hb1wave14_usda09_lightning711_variants10to20_palmup_omniwrist8_hold2s_training.pt",
        "data/geometry_latents/sharpa_hexprism40_bps64_mesh_data_64_v1.pt",
        "data/hand_fields/sharpa_wave14_hb1/right_hand_C_MC_collision_sdf_2mm.npz",
        "data/objects/sharpa_manual_hex_prisms/regular_hex_prism_af40mm_h40mm_chamfered",
    ]
    pairs = [(args.research_root / rel, args.destination / rel) for rel in relative_inputs]
    pairs += [
        (args.limits_usda, args.destination / "data/robots/sharpa_limits/right_sharpa_wave.usda"),
    ]
    records = []
    for source, target in pairs:
        if not source.exists():
            raise FileNotFoundError(source)
        entries = sorted(source.rglob("*")) if source.is_dir() else [source]
        for entry in entries:
            if not entry.is_file():
                continue
            output = target / entry.relative_to(source) if source.is_dir() else target
            digest = hashlib.sha256(entry.read_bytes()).hexdigest()
            if output.exists() and hashlib.sha256(output.read_bytes()).hexdigest() != digest:
                raise FileExistsError(f"Refusing to replace different local asset: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists():
                shutil.copy2(entry, output)
            records.append({"path": str(output.relative_to(args.destination)), "sha256": digest})
    manifest = args.destination / "data/local_assets.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Staged and hashed {len(records)} files: {manifest}")
    print("Install the public hand separately with: poise fetch-hand")


if __name__ == "__main__":
    main()
