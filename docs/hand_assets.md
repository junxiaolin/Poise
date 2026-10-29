# Public Sharpa Wave assets

POISE uses [sharpa-robotics/sharpa-urdf-usd-xml](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml),
revision `0d447b6889e6d993758169dfc0aa75ee9f6ad8d7`, directory
`wave_01/right_sharpa_wave`. The vendor assets carry Sharpa Group's Apache-2.0
notice. Their URDF, base geometry, robot and sensor layers match the previous
Wave1.4 training assets byte for byte. The public top/physics layers differ in
self-collision and collision exclusions, so POISE supplies a small composition
overlay to restore the training behavior.

## Install and verify

```bash
poise fetch-hand
poise check-hand
```

`fetch-hand` downloads the pinned public HTTPS archive without Git credentials.
Only the standard right-hand URDF, its 22 referenced mesh files, four USD
configuration layers, upstream entrypoint and license notice are installed.
Wrist/flange, floating-base, dual-hand, high-fidelity and MIT-mode variants are
not required by POISE. All 29 selected upstream files and both POISE layers are
checked against packaged SHA256 values. No simulator or device SDK is needed
for download/verification. Installed assets remain under ignored `data/`.

An existing local public checkout can avoid downloading the archive:

```bash
poise fetch-hand --source /path/to/sharpa-urdf-usd-xml
```

The checkout must be at the pinned commit and selected file hashes must match.
To migrate an existing hand installation, add `--replace`; all new files are
verified before the old directory is moved to a sibling `sharpa_wave.backup-*`.
A failed download or checksum check leaves the existing installation intact.
Repeated installation of matching assets is a no-op.

`--destination` selects a custom hand directory. Defaults use
`$POISE_ROOT/data/robots/sharpa_wave`, or the repository data directory for a
source installation. With wheels, set `POISE_ROOT` to your data workspace.

## Model composition

The training/grasp-filtering entrypoint remains:

```text
data/robots/sharpa_wave/right_sharpa_wave.usda
```

It composes these layers, strongest first:

1. `poise_collision.usda`: self-collision enabled at `root_joint`; seven body
   exclusion relationships with 16 target pairs; three public mesh-level
   exclusion relationships cleared and their `PhysicsFilteredPairsAPI` removed.
2. `upstream.usda`: unmodified public `wave_01/right_sharpa_wave/right_sharpa_wave.usda`,
   referencing the unmodified public base/physics/sensor/robot layers.

The collision overlay does not alter limits, joint frames, masses, inertias,
drive gains, armature, friction or mesh geometry. USD remaps the exclusion paths
when Isaac Lab references the hand beneath each environment's `Robot` prim.
Upstream viewport/render settings are retained; they differ from the old lab
wrapper and do not form part of the physics comparison.

The new entrypoint SHA256 is
`043bce67fd6b059bd2290b40667ad3b161b4b7f2a76eebf86b5fd757838552b7`.
Its hash alone is insufficient to identify the composed asset; `check-hand`
checks all dependencies against `source/poise/assets/sharpa_wave/manifest.json`.
Default simulation startup also performs this check. Each installation records
the revision, hashes and modification description in `provenance.json`; new
deployment bundles record the hand source under `training.hand_asset`.

Do not launch the unchanged `upstream.usda` directly for POISE training: it
disables self-collision and has a different exclusion set.

## Scope of this migration

The previous hand directory is no longer needed for asset installation.
`right_sharpa_wave.urdf` remains byte-identical, so grasp kinematics use the same
model. The palm collision STL is also identical; existing palm SDF data remains
applicable. Separate USDA09 joint-limit/drive reference data still lives at
`data/robots/sharpa_limits/right_sharpa_wave.usda` and has not been changed.
Object data, reference reset caches and optimizer dependencies have separate
acquisition steps; this hand migration does not complete their public release.

The previous hand's wrapper hash `a9404964...` belongs to historical artifacts.
It must not be assigned to the new composition. Local physics comparison and
simulator validation are recorded in [validation](validation.md).
