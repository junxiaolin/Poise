# Repository boundaries

POISE uses one source repository and three independent Python environments.
Users obtain the POISE adapters with the main checkout. External optimizer and
tracker implementations are optional, pinned dependencies fetched on demand;
they are not Git submodules and a normal clone does not download them.

```text
source/poise/
  core/                 observation, actions, transforms, policy bundle
  sim/                  Isaac Lab environment
    recipes/            canonical packaged training/PPO JSON
  training/             PPO, checkpoint/resume, distributed launch, export
  grasp/                cache contract, geometry, filtering support
    lightning/          POISE's LyGra hand/search/q_cmd adapters
  deploy/               FoundationPose adapter, camera, calibration, control
  assets/               hand acquisition and setup bundle import/export
  setups.py             setup identity and input validation
  launch.py             train / rollout command resolution
configs/
  setups/               versioned experiment descriptors
  rollout/              explicit, separate rollout settings
  grasp_hexprism40.json  grasp-generation pipeline settings
manifests/setups/        exact asset paths, sizes, hashes, distribution status
manifests/migration.csv  historical source provenance
Environments below are separate, not installed together:
environments/train/
environments/lygra-poise/
environments/foundationpose-poise/
dependencies/           upstream revision, patch, binary/weight metadata
scripts/                task entry points and optional dependency fetchers
tests/                  hardware-free regression checks
docs/                   guides plus frozen historical evidence
third_party/            ignored optional upstream source checkouts
data/                   ignored assets, caches and reference checkpoints
outputs/                ignored training, rollout and private bundle artifacts
```

Training configuration stays in `source/poise/sim/recipes/`, so existing packaged
recipe imports and legacy runs remain compatible. `configs/setups/` points to
these recipes by name/hash; there is no second editable copy in `configs/train/`.
A setup binds an ordered object set, the hand, recipe/PPO, asset manifest,
reference checkpoint, rollout config and launch profiles.

The `core` import path does not import Isaac Lab, LyGra, FoundationPose or the
SDK. Isaac Lab consumes prepared caches. Grasp generation runs optimizer stages
using the explicitly selected LyGra Python and physics stages using Isaac Lab
Python. Deployment consumes a per-object exported bundle and uses FoundationPose
in the control process; the vendor SDK runs in its configured container.

`POISE_ROOT` changes the data workspace only. Source/config/script locations come
from the editable checkout. The setup workflow requires that checkout; standalone
wheel distribution of the full task launcher is not currently supported.

Historical baseline/evaluation/training documents remain evidence, not active
configuration. New experiments should get new recipes and versioned setups,
without changing archived configs or silently rebinding a reference checkpoint.
Large data is distributed through GitHub Releases and ignored by Git. Distribution
URLs and hashes are pinned; local bundle generation does not publish anything.
