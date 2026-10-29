# Reproduce a provided setup

Use a **source checkout**, installed editable, in a prepared Isaac Lab environment.
The default route needs no LyGra, FoundationPose, camera, SDK or research checkout.
Code and assets are versioned together in the
[Poise release](https://github.com/junxiaolin/Poise/releases/tag/v0.1.0-rc1).
Commands below run from the POISE root.

## 1. Prepare the simulation environment

Follow [environments/train](../environments/train/README.md) for the pinned
Isaac Sim 5.1 / Isaac Lab 0.48 / Python 3.11 / Torch 2.7 stack, then:

```bash
python -m pip install -r environments/train/requirements.txt
python -m pip install --no-deps -e .
poise check-env train --native
poise list-setups
poise show-setup hexprism40_palmup_v1
```

`check-env` reports missing packages and differences from the tested versions;
its success is an import/version check, not a physics test. Setup commands need
the checkout's configs and scripts; installing a standalone wheel is insufficient.

| Setup | Objects | Global frontier gate | Reference candidate |
|---|---:|---:|---|
| `hexprism40_palmup_v1` | 1 | 20,000 | baseline `model_31200.pt` |
| `manualconvex9_palmup_v1` | 9 | 5,000 per object | evaluated `model_23300.pt` |
| `hexprism40_omniwrist_fullpose_v1` | 1 | 160,000 | baseline `model_35800.pt` |
| `hexhammer_palmup_fullpose_192k_v1` | 1 | 20,000 | baseline `model_89800.pt` |

The palm-up setups use a fixed wrist. The OmniWrist FullPose setup randomizes
wrist RPY at reset, keeps it fixed within each episode, and uses strict full-pose
success with no tactile input. All use Sharpa Wave. HexHammer uses its historical
11,630-row cache; the other setups use 6,842 rows per object. These are **not** the newly generated public-hand caches. Object order,
dimensions, BPS64, cache hashes, hand composition, recipes and PPO settings are
pinned together. The older `hexprism40` recipe has a different curriculum gate;
it is preserved for old runs and should not be substituted for this setup.

## 2. Import and check assets

By default data lives under `data/` in the checkout. To keep it on another disk,
set `POISE_ROOT` to a workspace containing a `data/` directory. The repository and
its configuration continue to resolve from the editable installation.

```bash
# Optional; omit this to use the checkout's data/ directory.
export POISE_ROOT=/path/to/poise-workspace
poise fetch-assets --setup hexprism40_palmup_v1 --with-reference
poise check-assets --setup hexprism40_palmup_v1 --with-reference --deep
```

The downloader resolves the pinned release index, verifies hashes and installs
only the selected setup's missing dependencies. Correct shared files are reused.
For offline use, download the Release assets into one directory and add
`--source /path/to/release/assets`. That directory must contain
`release-index.json` and the required package files.

For a repository with restricted access, download through an authenticated GitHub
CLI first; the automatic HTTPS downloader does not use GitHub login credentials:

```bash
gh release download v0.1.0-rc1 --repo junxiaolin/Poise --dir /path/to/release/assets
poise fetch-assets --setup hexprism40_palmup_v1 \
  --source /path/to/release/assets --with-reference
```

`--source` also accepts a legacy complete tar bundle or an existing workspace
containing `data/`. The manifests
include the public hand, its USD dependencies and notices, object meshes/USD,
reset caches, geometry fields, palm SDF and the optional reference checkpoint.
The grasp-only numeric limit reference and optimizer dependencies are not needed
for this path.

Omit `--with-reference` when only training from scratch. Import is idempotent for
identical files and refuses to replace changed data. All selected assets are
staged and verified before installation; arbitrary archive entries are not
extracted. `--deep` additionally checks tensor contents, object order, sizes,
counts, finiteness and the reference policy interface. Tensor files must come
from a trusted source.

For the maintainer, build the shared release directory with:

```bash
PYTHONPATH=source python scripts/release/package_assets.py \
  --output outputs/releases/candidate/assets --bind
```

For a legacy standalone offline bundle containing only one setup:

```bash
poise package-assets --setup hexprism40_palmup_v1 --with-reference \
  --output outputs/private_bundles/hexprism40_palmup_v1.tar.gz
```

This writes an archive, metadata, notices and a checksum locally. It never uploads
or changes repository visibility. [Asset manifests](../manifests/setups/README.md)
pin the release index and the identities of all required files.

## 3. Roll out the reference checkpoint

```bash
poise rollout --setup hexprism40_palmup_v1 --output outputs/rollout/hex_reference
poise rollout --setup manualconvex9_palmup_v1 --headless --steps 400 \
  --output outputs/rollout/nine_reference
```

The default checkpoint is selected by the setup, with SHA256 verification.
GUI mode includes pose frames, live metrics and real-time pacing. The camera
starts near environment 0; `--view-env 3` selects another environment for the
camera and metrics. With nine environments, indices 0–8 follow the setup's object
order. Headless mode
omits those graphics. Default rollout length is 400 control steps, seed 42, with
4 environments for HexPrism40 or 9 for Convex9. `--steps 0` runs until interrupted.
An overridden environment count must cover every setup object.

For Hex40 and Convex9, the separately pinned rollout config samples rotation 0–180° and translation
0–30 mm, disables curricula and most randomization/noise, and refreshes targets
after success. The historical inertia log-uniform multiplier remains enabled.
The two historical palm-up setup entry points passed local reference rollout,
per-object interface checks and 400-step GUI acceptance; see
[historical validation](validation.md). The current four-setup isolated
acceptance is recorded in [release validation](release/validation.md).
This is a qualitative continuous-target rollout, **not** a reproduction of the
archived first-target success benchmark. It does not mutate training settings.
For the new OmniWrist setup, random wrists remain enabled in qualitative rollout;
see its [baseline record](baselines/hexprism40_omniwrist_fullpose_v1/README.md) for
validation and checkpoint provenance.

The [HexHammer 192k baseline](baselines/hexhammer_palmup_fullpose_192k_v1/README.md)
starts reference rollout at its saved 180° / 100 mm curriculum and retains training
randomization and sensing noise. Its reference training profile uses four physical
GPUs, eight ranks, 24,000 environments per rank and GF20k. Rollout remains a local
four-environment qualitative check, not an independent success-rate benchmark.

Each run saves `env.json`, `setup.json`, `rollout.json`, `result.json` and deployment
exports. The first object is under `bundle/`; further objects are under
`bundles/<object-id>/`. Add `--verify-interface` with finite steps for simulator
versus independent observation/action replay and TorchScript checks.

## 4. Train, continue and play your own checkpoint

```bash
poise train --setup hexprism40_palmup_v1 --profile smoke --output outputs/train/hex_smoke
poise train --setup manualconvex9_palmup_v1 --profile smoke --output outputs/train/nine_smoke

poise train --setup hexprism40_palmup_v1 --profile custom \
  --num-envs 128 --iterations 1000 --output outputs/train/hex_custom
poise train --setup hexprism40_palmup_v1 --profile custom \
  --num-envs 128 --iterations 100 --resume outputs/train/hex_custom/model_999.pt \
  --output outputs/train/hex_continued
poise rollout --setup hexprism40_palmup_v1 --checkpoint outputs/train/hex_continued/model_1099.pt \
  --output outputs/rollout/hex_custom
```

`smoke` means 8 HexPrism40 or 18 Convex9 environments and two updates. It verifies
integration, not convergence. `custom` defaults to 128 environments and 100
updates; explicit overrides require this profile. `--iterations` always means
additional PPO updates, each with eight rollout steps. Output directories must
be new. Training saves configs, topology, logs, setup-bound checkpoints and
per-object policy bundles. Inspect logs with `scripts/summarize_training.py`.

Resume requires the same setup, seed, environment count and distributed topology.
It restores optimizer, normalizers, curriculum and random streams; PhysX episode
trajectories restart. Newly trained checkpoints carry the setup contract. Older
unbound custom checkpoints must use their original `--recipe` entry; pinned
reference candidates are recognized by hash for rollout, not automatically
converted into new-format training runs.

## 5. Run at the reference training scale

Inspect the command before starting a large job:

```bash
poise train --setup hexprism40_palmup_v1 --profile reference \
  --output outputs/train/hex_reference_scale --dry-run
poise train --setup manualconvex9_palmup_v1 --profile reference \
  --output outputs/train/nine_reference_scale --dry-run
```

Remove `--dry-run` to launch. The Hex40 palm-up and Convex9 profiles use two ranks, 32,000 environments
per rank, 150,000 updates and base seed 42. Hex uses two physical GPUs; Convex9
uses one large GPU with two workers (the historical A800 layout). This is not a
small workstation default. See [distributed training](training.md) and the
[hardware notes](release/README.md#hardware-and-profiles); that memory measurement
does not predict Convex9 usage. A reference-scale run is a specified training
configuration, not a promise of bitwise-identical learning across hardware.

`hexprism40_omniwrist_fullpose_v1 --profile reference` uses **4 physical GPUs,
8 ranks ×24,000 =192,000 environments**, Gloo, GF160,000, 150,000 updates and
base seed 42. It reproduces the A800 launch topology. Its smoke profile uses
8 environments and two updates on one GPU. Choose `--profile reference`
explicitly on a prepared four-GPU machine; the default remains a small smoke run.

For a different object set or wrist distribution, first use a new recipe and
validate it, then register a new versioned setup. See [new objects](new_objects.md).
Do not edit the identity of an existing reference setup in place.

## Layered release assets

The four setups share dependencies; correctly installed files are reused, and
`--with-reference` adds only the selected checkpoint. Use `--source` for offline
imports. Full tar bundles remain supported. See [release contents](release/README.md).
