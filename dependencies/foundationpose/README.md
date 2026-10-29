# FoundationPose source and weight pins

Upstream: https://github.com/NVlabs/FoundationPose

The source revision, PyTorch3D/NVDiffRast revisions and locally inspected weight
hashes are in `manifest.json`. `scripts/setup_foundationpose.py` stages sources
in a new directory, checks revisions and refuses to replace an existing tree.
No upstream source, native binary or weight is copied into POISE Git.

The upstream license and separate weight terms apply. No POISE-wide license
overrides them. Follow [environment setup](../../environments/foundationpose-poise/README.md).
