# Release-candidate acceptance

The acceptance suite imports the candidate's layered packages into an empty data
workspace, uses a fresh source checkout, and runs each of the four reference
policies, small from-scratch training and checkpoint continuation. The original
workspace and research checkouts are hidden from simulator/optimizer processes
with a filesystem namespace. The prepared Isaac Lab runtime is reused: this
validates the documented prerequisite, not installation of Isaac on a new OS.

The optional LyGra check fetches pinned upstream sources, applies the recorded
patch, verifies CUDA binaries and executes a small fresh generation/augmentation/
physics-filter chain. It uses a separate optimizer Python environment. A small
chain checks integration, not the formal generation quota or coverage quality.

Machine-readable results are written during the acceptance run. No full 192k
training run, fresh operating-system installation or real-robot test is implied.
The historical wrist success-rate benchmark used model_35600; it is not a score
for the fixed model_35800 or model_89800 reference candidates.

## Completed on 2026-10-09

The final private candidate passed all four setups using its pinned 12-package
release index, starting with a fresh local Git clone and an empty data directory.
The original development/research workspace was hidden from every simulator
process. Packages were shared as subsequent setups were installed.

| Setup | Reference model | Headless rollout | Fresh training and resume |
|---|---|---|---|
| Hex40 palm-up | 31200 | 4 environments, 64 steps | 8 environments, 2 + 2 PPO updates |
| Convex9 palm-up | 23300 | 9 environments, 64 steps | 18 environments, 2 + 2 PPO updates |
| Hex40 OmniWrist FullPose | 35800 | 4 environments, 64 steps | 8 environments, 2 + 2 PPO updates |
| HexHammer palm-up FullPose | 89800 | 4 environments, 64 steps | 8 environments, 2 + 2 PPO updates |

Each continuation restored the matching setup checkpoint, advanced the next
iteration from 2 to 4, increased total timesteps and the environment step counter,
and produced finite policy tensors. All four frozen reference hashes matched.
The development suite passed 189 tests and 6 subtests, with 2 skips. The exported
candidate passed 185 tests and 6 subtests, with 2 skips; internal cloud-launcher
tests are excluded. GitHub CI was added but has not run remotely.

The separately installed, pinned LyGra checkout passed its CUDA extension checks.
Fresh search produced 16 candidates, of which 14 passed stability filtering.
Two parents were selected, 32 perturbed states were tested, 27 passed, and the
final cache contains 12 states, six per parent. Both filters used eight wrist
trials and a two-second hold. The generation chain took 113.4 seconds on this
host. This small profile is not a timing guarantee for formal generation.

The checked runtime uses Python 3.11 / Torch 2.7 cu128 / Isaac Sim 5.1 / Isaac Lab
0.48 for simulation and Python 3.9 / Torch 2.8 cu126 for optimization. Separate
venvs reused installed dependencies; upstream LyGra sources, Poise source and
candidate assets were installed afresh. Hardware was an RTX 4090 reporting
49,140 MiB VRAM, driver 580.178.04. This was not a GUI or full-scale training test.

Source and history pattern scans found no credential candidates. The 413
reachable historical blobs contained 192 internal path/node findings, so old
history remains private. The exported tree has no findings under those rules.
Asset metadata exceptions are described in [release notes](README.md#historical-cache-provenance).

The [machine-readable receipt](validation-2026-10-09.json) pins the package index,
tested runtime files, recipes, model identities, generation outputs and limits.
Documentation may be updated after testing; executable/configuration hashes
in that receipt must continue to match the candidate.

## Repeat the check

Create a new source clone and put the candidate packages outside the original
workspace. On Linux with bubblewrap and a prepared simulator environment:

```bash
python scripts/release/acceptance.py \
  --checkout /tmp/poise-clean \
  --assets /tmp/poise-assets \
  --python /path/to/isaac-environment/bin/python \
  --hide-workspace /path/to/original-workspace \
  --output /tmp/poise-acceptance
```

The checkout, package directory, Python environment and output must be outside
the directory being hidden. The output directory must be new. Run the independent
optimizer check using [the generation smoke profile](../new_objects.md#release-generation-assets).
