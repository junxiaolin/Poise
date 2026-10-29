# Environments and assets

Start with [setup reproduction](reproduction.md). Existing-setup training and
reference rollout require only a prepared Isaac Lab environment and the matching
asset bundle. Install optional optimizer or deployment dependencies only for
those tasks; they have incompatible Torch/NumPy/native stacks.

| Environment | Inspected / tested stack | Installation |
|---|---|---|
| `train` | Python 3.11, Isaac Sim 5.1, Isaac Lab 0.48, Torch 2.7 cu128, NumPy 1.26, RSL-RL 3.0.1 | [Train](../environments/train/README.md) |
| `lygra-poise` | Python 3.9, Torch 2.8 cu126, NumPy 1.23.5, urdfpy, Open3D | [LyGra](../environments/lygra-poise/README.md) |
| `foundationpose-poise` | Python 3.11, Torch 2.6 cu124, NumPy 2.4.4, OpenCV 5 with ArUco, PyTorch3D, NVDiffRast | [FoundationPose](../environments/foundationpose-poise/README.md) |

The local simulation host uses Ubuntu 22.04 and an RTX 4090 reporting about
48 GiB. Exact versions and stage checks live in `environments/<name>/manifest.json`.
FoundationPose's existing environment passed dependency/weight/import checks and
POISE CPU interface tests; a fresh native build and robot session remain unverified.
LyGra passed an independent optimizer-venv install plus fresh search, perturbation
augmentation and physics filtering on this host; see
[that record](release/validation.md).

All three environments install the same **source checkout**, editable:

```bash
python -m pip install --no-deps -e .
```

Use each environment's requirements rather than installing all extras together.
In particular, the older generic `.[deploy]` OpenCV range is not the inspected
FoundationPose stack. Python 3.9 support applies to the optimizer/core;
simulation and deployment use Python 3.11. The hand SDK uses its own configured
container/Python ABI. No SDK or native upstream binary is bundled with POISE.

## Assets for provided setups

```bash
poise fetch-assets --setup hexprism40_palmup_v1 \
  --source /path/to/release/assets --with-reference
poise check-assets --setup hexprism40_palmup_v1 --deep --with-reference
```

Bundles include the simulation hand and collision composition, object meshes/USD,
BPS/contact geometry, palm SDF, reset cache and optional checkpoint. They can be
imported into the checkout or an independent `POISE_ROOT` workspace. All selected
files are checked against setup hashes. Public URLs remain unset; no clean-machine
public download is claimed. See [reproduction](reproduction.md) for the implemented
local-bundle path and [validation](release/validation.md) for results.

## Additional generation assets

For the supplied Hex40 generation recipe, install the shared inputs and the
optional numerical joint-limit reference:

```bash
poise fetch-grasp-assets --source /path/to/release/assets
```

The public hand can also be installed independently:

```bash
poise fetch-hand
poise check-hand
```

It is pinned to Sharpa's upstream commit and composed with POISE's collision
overlay; see [hand assets](hand_assets.md). The separate numerical joint-limit /
drive-stiffness USDA reference is a grasp-generation input, not the simulated
hand. It is included in the optional generation package, with provenance and
terms recorded in `manifests/rights.json`.

`scripts/import_local_assets.py` is retained for internal migration from the old
research workspace. It is not part of the reproduction route or a public
downloader. Training with a complete setup bundle does not need that workspace.

## Offline development

In a separate environment without Isaac Lab:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

Some simulator/RSL-RL tests skip when their dependencies are unavailable. Native
physics and GPU validation must run in the matching environment. These package
extras are conveniences for development, not a combined lockfile for all stages.
