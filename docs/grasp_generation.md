# Generate diverse reset states

Provided [setups](reproduction.md) already contain grasp caches; this optional
workflow is needed for fresh grasps. For new object/hand scope and environment
boundaries, see [new objects](new_objects.md).

Run from the repository root after [installation](installation.md). The complete
pipeline searches fresh grasps, derives preload commands, validates parent grasps
in physics, expands them and validates the resulting reset states. Existing
training caches are not generation inputs.

## Fast HexPrism40 build

For a small training-ready cache, use the same environments and assets with:

```bash
python scripts/build_grasp_cache.py \
  --config configs/grasp_hexprism40_fast.json \
  --optimizer-python /path/to/lightning-env/bin/python \
  --output outputs/grasp_generation/hexprism40_fast \
  --device cuda:0
```

This profile targets completion within 30 minutes on an RTX 4090 with dependencies
and assets already prepared. It sets a **1,740-second generation budget**, with
up to 15 seconds for child-process termination on timeout. Installation and
training are separate. A timeout is a failed build, not a successful partial
cache; inspect `.running.json` and stage logs. Hardware speed and grasp yield
vary, so this is a bounded quick-generation option, not a promise of success
for arbitrary objects or environments.

On the validated RTX 4090 run, fresh generation finished in **13 min 12 s**:
38 search candidates → 28 stable parents → 840 perturbations → 609 states
passing all wrist trials → 280 selected unique reset states. See the
[measured run](release/validation.md).

For the low search yield, see the [collision audit](release/validation.md).
Five-contact candidates leave no free joints for assignment retries. The audit
confirmed substantial inter-finger intersections and an implicit 2.5 mm tolerance
in the pinned object-collision kernel; increasing retries or globally relaxing
collision checks is not a validated way to improve yield.

| Setting | Formal | Fast |
|---|---|---|
| Search | Three full grids, 10/20/20 passes, plus corner searches | One full grid, 4 passes |
| Position/orientation grid | 4,050 poses per full-grid pass | Same grid |
| Parents | Exactly 486 | Up to 32; require at least 8 stable parents |
| Final reset states | Exactly 6,842 | 10 × selected parents, normally up to 320 |
| Final variants per parent | 10–20 | 5–15, balanced toward 10 |
| Initial expansion | 20 per parent | 30 per parent |
| Supplement budget | 12 rounds × 80 per deficient parent | 2 rounds × 20 per deficient parent |
| Parent and final physics checks | Eight wrist poses, 2 seconds each | Unchanged |
| Perturbation, settling, collision/limit/stability criteria | Formal settings | Unchanged |

Fast generation still runs **fresh LyGra search → q_cmd → physical parent
validation → diverse parent selection → perturbation/settling → eight-wrist
validation → final selection**. It sacrifices search coverage and cache size;
it does not substitute copied rows for perturbations or omit physics checks.
If fewer than 32 but at least 8 stable parents are available, the selected count
and total row target shrink together. `targets.json` records requested and actual
counts. Too few parents, insufficient stable variants, or timeout cause failure.
The formal profile keeps its exact-count policy.

`result.json` includes actual counts, the cache hash, wall-time budget and elapsed
time for the successful invocation. `--resume` gets a new invocation budget and
retains the usual hash checks; total time across retries can exceed 30 minutes.
Only consume `training.pt` when `result.json` reports `status: complete` and its
SHA256 matches. An interrupted stage may leave unreceipted files for inspection.

For palm-up training, copy `source/poise/sim/recipes/hexprism40_palmup_v1.json`
to a new recipe and set both `stable_cache_path` and
`primitive_stable_cache_paths` to the fast cache. Existing HexPrism40 geometry
and hand fields can be reused. Keep the versioned reference setup unchanged:
this smaller cache is a different initialization distribution and is not a
replacement for the historical reference cache or an equivalent-quality policy.

## Formal HexPrism40 build

Install the public hand with `poise fetch-hand` and verify it with
`poise check-hand`. Search uses `data/robots/sharpa_wave/right_sharpa_wave.urdf`;
physics uses `data/robots/sharpa_wave/right_sharpa_wave.usda`. Both come from the
pinned public Sharpa repository. The builder clears the optional
`SHARPA_WAVE_URDF_PATH` override and records the public manifest, entrypoint and
collision overlay hashes. No old private hand directory is used.

The separate `data/robots/sharpa_limits/right_sharpa_wave.usda` is a numeric
joint-limit/drive-stiffness reference, not the simulated hand. USDA09 scales both
signed endpoints by 0.9. Keep this reference unchanged when comparing against the
current training distribution.

Use Isaac Lab Python for the builder and supply the Lightning Grasp Python
executable explicitly. Install the [pinned LyGra sources, patch and CUDA binaries](lightning_grasp.md)
in that environment first. The builder checks required interfaces and loaded
extensions before searching. Object and limit
reference assets must also be installed before this command.

```bash
python scripts/build_grasp_cache.py \
  --config configs/grasp_hexprism40.json \
  --optimizer-python /path/to/lightning-env/bin/python \
  --output outputs/grasp_generation/hexprism40_public \
  --device cuda:0
```

The builder runs one stage at a time and takes a workspace-wide device lock.
The supplied configuration uses 32 grid poses × 32 inner search candidates,
128 rows per q_cmd batch, and 64 simulation environments for filtering and
expansion. These defaults leave substantially more desktop VRAM available than
concurrent searches. Small batches do not cap GPU compute utilization.

The formal configuration targets **486 stable parents and 6,842 unique reset
states**, with **10–20 variants per parent**. Counts are output requirements, not
promises that arbitrary seeds or hardware will produce sufficient stable grasps.
The builder stops rather than copying old rows or weakening stability criteria.
On the tested workstation, allow substantial time for search and physics; stage
logs show progress. Existing cloud training and baseline cache paths are not
modified.

| Stage | Behavior | Main artifacts |
|---|---|---|
| Fresh search | Three full XYZ/RPY grids with independent seeds, plus three lower-corner contact searches | `search_*.npy`, metadata |
| Preload and conversion | Compute q_cmd from contact wrenches/Jacobians/stiffness, convert to schema v2 | `search_*.qcmd.npz`, `search_*.pt` |
| Candidate merge | Retain distinct q/q_cmd/object-pose tuples and source lineage | `candidates.pt` |
| Parent validation | Palm up plus seven random wrist orientations; hold each for 240 steps at 120 Hz | `stable_candidates.pt`, replay report |
| Parent selection | Highest stability first, then deterministic farthest-point selection over standardized joints and object pose | `parents.pt`, `parents.json` |
| Local expansion | Gaussian perturbations, 80-step settling and 240-step hold; initially 20 states per parent | `initial_variants.pt` |
| Expanded-state validation | Repeat all eight wrist trials; every trial must pass | `initial_filtered.pt` |
| Replenishment | Target deficient parents, generate additional candidates, remap parent IDs and recheck physics | `supplement_*.pt` |
| Final selection | Capacity-aware exact quotas and within-parent pose maximin selection | `training.pt`, `training.json`, `result.json` |

The full grid is X=0.03–0.11 m, Y=−0.05–0.05 m, Z=0.05–0.14 m
with steps 0.02/0.025/0.02 m (inclusive endpoint), and roll/pitch/yaw each in
{−45°, 0°, 45°}. It contains 4,050 poses. The full searches use 10/20/20
passes starting at seeds 0/10/20 and five candidate contacts. The corner
searches use four contacts, X=0.05 m and Z=0.05/0.07 m, retaining up to three
solutions per grid pose. Z=0.07 m is searched again starting at seed 100 to
provide additional stable-parent capacity. All search arguments are in the JSON configuration.

Expansion standard deviations are q=0.025 rad, q_cmd=0.012 rad,
position=0.004 m and rotation=0.08 rad. As in the historical expansion,
joint perturbations and saved settled q are clamped to USDA09 endpoints.
The subsequent filter validates the saved values without projection.
Object mass is 0.05 kg and friction is 1.0.

The hold filter requires at least two fingertip contact proxies (distance below
25 mm), object drift at most 40 mm, final linear speed at most 0.25 m/s and
angular speed at most 4 rad/s. The palm-frame workspace is X=[0.025, 0.18] m,
|Y|≤0.14 m, Z=[0.02, 0.22] m. Passing eight sampled wrist orientations is a
finite stability check, not a guarantee for every possible orientation.

This build produces a fresh distribution at the reference scale. It is not a
bitwise reproduction of the historical Lightning711 cache: merging preserves
all distinct configurations, parent selection uses farthest-point sampling, and
an exact 6,842-row target uses balanced quotas with redistribution for parents
with limited stable capacity. The historical 10–20 count histogram is not forced.

## Resume and audit

Repeat the command with `--resume` to reuse completed stages only when commands,
inputs, outputs, source files, configuration and asset hashes still match. Each
stage records a `.stage.json` completion receipt, a `.log`, and a running/failure
record while incomplete. A file lock prevents two builders from writing into
one output directory, and a separate device lock prevents simultaneous builds
in different directories. SIGINT/SIGTERM stops the active child process group.
A failed or interrupted stage never receives a completion receipt.

Inspect failed logs before retrying. If a failed stage left partial outputs,
move those outputs aside before resuming; they are deliberately not accepted as
completed work. Use a new output directory after changing code or configuration.
`contract.json` records the full generation contract and the installed optimizer
source hashes/revision and native binary hashes. `result.json` is written only after final count, lineage
and duplicate checks pass. Generated data remains under ignored `outputs/`.

For an insufficient parent pool, increase fresh search coverage/passes in a new
configuration and output directory. For missing per-parent variants, increase
expansion/replenishment budgets. Do not compensate with repeated copies.

Individual stages remain available through `poise.grasp.lightning.generate`,
`poise.grasp.lightning.qcmd`, `poise.grasp.convert`, `poise.grasp.merge`,
`scripts/filter_grasps.py`, `poise.grasp.select_parents`,
`scripts/expand_grasps.py` and `poise.grasp.finalize`. Use `--help` in the
corresponding environment. The builder's optional `--search-results` accepts a
hash-verified manifest of already completed fresh searches; missing searches
are executed sequentially. Imported searches retain their original batch sizes
and provenance. This option is unnecessary for a normal build.

## Compare with an existing cache

```bash
python scripts/compare_grasp_caches.py \
  --reference /path/to/reference_training.pt \
  --new outputs/grasp_generation/hexprism40_public/training.pt \
  --output outputs/grasp_generation/hexprism40_public/comparison
```

The report includes row/parent counts, exact duplicates and overlap, USDA09
violations, position/orientation/joint spread, spatial-bin coverage and a plot.
Position and orientation nearest-neighbor distances are separate marginal
coverage measures, not joint SE(3) matches. `--reference-replay` and
`--new-replay` add JSON reports from a fresh identical wrist-hold protocol under
the public hand. Use a held-out wrist seed for both, keeping filter settings and
batch size the same. Cache coverage and hold stability do not establish equal
reaching-policy performance.

## Build geometry fields and train

```bash
python -m poise.grasp.mesh_data \
  --object-id regular_hex_prism_af40mm_h40mm_chamfered \
  --combined-output outputs/grasp_generation/hexprism40_mesh_data.pt
```

Copy `source/poise/sim/recipes/hexprism40.json` to a new JSON recipe. Set
`stable_cache_path` and `primitive_stable_cache_paths` to the generated `training.pt`;
set `object_mesh_data_path` and `primitive_geometry_latent_path` to the new
geometry file. Run `scripts/train.py --recipe /path/to/recipe.json`.

For multiple objects, keep `primitive_object_ids`, `primitive_dims_m`,
`primitive_object_shape_names`, `primitive_stable_cache_paths` and
`pose6d_anchor_per_object_ids` in the same order. The geometry payload must cover
all object IDs. Set `dexndm_omni_wrist_ornt=true` for random wrist roots.
Per-object anchor and swing success-threshold arrays must also match this order.
Arbitrary new meshes still require adding an asset specification; the geometry
CLI currently handles the included hex-prism family.

## Regenerate the palm collision field

```bash
python -m poise.grasp.hand_sdf \
  --mesh data/robots/sharpa_wave/meshes/right_hand_C_MC.STL \
  --output outputs/grasp_generation/palm_sdf.npz
```

The mesh must be watertight, in meters and in the hand-root frame. The tool
prints the SHA256; update both `pose6d_feasible_target_forbidden_sdf_path` and
`pose6d_feasible_target_forbidden_sdf_sha256` in your recipe. Generated archive
hashes can differ from the formal reference because metadata and grid padding
are included; do not relabel generated data with the reference hash.
