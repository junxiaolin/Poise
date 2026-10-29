# New objects and grasp generation

Skip this path when using a provided setup. Its asset bundle already contains
reset states, meshes and geometry descriptors. LyGra is needed only to generate
new grasps; Isaac Lab then filters them physically before training.

## Install the optional optimizer

Follow [environments/lygra-poise](../environments/lygra-poise/README.md) and the
[pinned dependency recipe](lightning_grasp.md). POISE's Sharpa adapter ships with
the main repository; upstream LyGra is fetched only when requested. The tested
optimizer uses Python 3.9 / Torch 2.8 cu126 in a separate environment. Its CUDA
extensions currently come from the upstream precompiled archive because kernel
sources are not available at the pinned revision.

```bash
# In the optimizer environment:
python -m pip install --no-deps -e .
poise check-env lygra-poise --native
python -m poise.grasp.lightning.runtime --cuda-smoke
```

The complete generation builder runs in Isaac Lab and launches the optimizer with
an explicit Python path:

```bash
python scripts/build_grasp_cache.py --config configs/grasp_hexprism40.json \
  --optimizer-python /path/to/lygra-poise/bin/python \
  --output outputs/grasp_generation/new_hex --device cuda:0
```

This is a formal generation job, not a quick check. Read the stage sizes,
expected runtime, filtering criteria and resume rules in
[grasp generation](grasp_generation.md) before launching it. The separate numeric
joint-limit/drive-stiffness reference is an additional generation input; it is
not included in the training-only setup closure. Its path/hash is recorded in
the generation config. Do not substitute unrelated hand limits.

For a smaller HexPrism40 cache with a 30-minute runtime target, replace the
config with `configs/grasp_hexprism40_fast.json` and use a fresh output directory.
The [fast profile](grasp_generation.md#fast-hexprism40-build) retains physical
perturbation, settling and eight-wrist filtering while reducing search work and
parent/state counts. New geometries may need different search settings; their
success rate and runtime have not been established by the HexPrism40 check.

## Prepare a new object

1. Provide a metric, centered tracking/collision mesh and a simulator USD with
   consistent object-frame convention, dimensions, mass and inertia. Record its
   provenance and hashes. Register the object in the asset/specification code
   used by generation and simulation (`source/poise/sim/assets/`, especially
   `sharpa_manual_hex_prisms.py` and `sharpa_manual_convex_objects.py`). Arbitrary meshes are not yet a one-command
   import; current geometry tooling focuses on the included hex-prism family.
2. Copy a generation config and set the object identity, dimensions, mesh/hash,
   search coverage and output targets. Generate q/q_cmd and schema-v2 grasp states,
   then run the Isaac Lab stability filtering/expansion pipeline. Check counts,
   finite tensors, joint bounds and lineage. A new mesh does not inherit stability
   from the old mesh just because the grasp coordinates are similar.
3. Build BPS64 and contact/geometry fields. Create a custom training recipe with
   the new mesh geometry and cache. Keep IDs, dimensions, shapes, cache paths,
   geometry rows, anchor arrays and per-object success thresholds in the same
   order for multi-object training. Changing wrist mode is also a new setup.
4. Run `scripts/train.py --recipe /path/to/custom.json` at small scale and inspect
   reset stability and exported policy interfaces. After validating the assets
   and training configuration, register a new versioned setup with a reference
   checkpoint; see [setup schema](../configs/setups/README.md). Keep existing
   reference setups unchanged.

A regenerated grasp cache is a new training distribution. Reaching performance
needs a new training/evaluation comparison, even if cache size and hold-stability
rate are similar. The provided setups intentionally retain their historical
caches and checkpoint pairing.

## A different hand

The supported implementation is Sharpa Wave right hand only. Installing LyGra
for another hand does not make POISE hand-agnostic. A new hand requires adapters
for kinematics/collisions, joint ordering and limits, grasp preload, simulator
articulation, observation/action dimensions, rewards/contact proxies, export,
and hardware control. The current policy/checkpoint contract is 244D input and
22D action; it cannot be reused unchanged with an arbitrary new hand.

## Release generation assets

Before the cache builder, install the optional numeric joint-limit reference and
Hex40 search inputs. This step does not download a reference policy:

```bash
poise fetch-grasp-assets --source /path/to/release/assets
python scripts/build_grasp_cache.py --config configs/grasp_hexprism40_smoke.json \
  --optimizer-python /path/to/lygra-poise/bin/python --output outputs/grasp_smoke
```

The smoke profile requests two parents and twelve final states (one parent/six
states is allowed), with fresh search, perturbation and eight-wrist physics
filtering. It is an integration test, not a coverage or formal-cache substitute.
Use the full/fast configurations for production data.
