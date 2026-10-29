# Shared hand USD definitions during training

The requested full-scale follow-up is now complete: on **2×4090D, 32,000
environments per card**, sustained summed worker PSS fell from **62.724 GiB to
48.250 GiB**, saving **14.473 GiB (23.07%)**. See
[the 64k benchmark](training_memory_64k_20260920.md) for protocol and limits.

## Sustained-memory result

On one RTX 4090D, with 4,096 environments and 120 PPO updates, the mean
training-process PSS over updates 60–120 decreased from **7.870 GiB to 6.820 GiB**:
**1.049 GiB / 13.33% less system RAM during training**.

| Measurement, updates 60–120 | Original hand definition | Shared hand definition |
|---|---:|---:|
| Mean PSS | 7.8698 GiB | 6.8204 GiB |
| PSS range | 7.8670–7.8716 GiB | 6.8201–6.8206 GiB |
| Median iteration time | 1.51 s | 1.46 s |
| PyTorch CUDA reserved | 464 MiB | 464 MiB |

The gain persists throughout the measured second half of training; it is not
the transient startup-trim effect. Timing is one serial A/B pair, so it does not
establish a speedup. Both completed all 120 updates, saved finite checkpoints and
exported policies successfully. The paired resolved configs are identical except
for output directory; checkpoint configuration hash is identical:
`a745402d447bd3a633f17cf33655a22fbcfe16e822d6ebf733d005e7fecca345`.

This is a host-RAM optimization. CUDA storage, network sizes and PPO batch sizes
are unchanged. The separate full-scale benchmark above validates 64k / two-worker
savings; no active formal training has been migrated to this implementation.

## Implementation

`PoiseEnv._setup_scene()` first spawns one Sharpa hand through the original Isaac
Lab USD spawner. After all rigid-body, articulation, collision and sensor
overrides have been applied, `reference_hand_definition()` captures the resolved
hand into one anonymous, read-only USD layer. Each environment's hand references
this layer instead of containing a copied hierarchy of identical authored
attribute overrides. The existing Isaac Lab cloner still creates every robot.

All composed prims are ordinary, independently editable prims. This is not whole
articulation instancing, physics replication, or inheritance from environment 0.
Edits to one live environment do not propagate to another. Bodies, joints,
collisions, materials, scale randomization and PhysX tensor views remain separate.
The anonymous layer stays alive with the environment and is never written into
the public asset directory. Existing asset identity checks and files are intact.

This reduces the USD data retained throughout training. No recipe/PPO parameter,
environment count, world size, device mapping, contact-buffer capacity, collision
filter, reward, curriculum, randomization or observation contract was edited.
The change applies to the common hand setup for both HexPrism40 and Convex9.

`POISE_SHARED_HAND_GEOMETRY=0` selects the original spawner for A/B measurement.
Normal launches enable sharing. Checkpoint configuration identity stays unchanged.
`POISE_MEMORY_REPORT=1` reports per-worker PSS/RSS and PyTorch CUDA memory every
ten updates, as well as learning boundaries. Diagnostics are disabled by default.

The two initialization trims from the earlier patch remain enabled on both sides
of the steady-state comparison, isolating the benefit of shared USD definitions.
There is no periodic GC, heap trimming, or CUDA-cache flush during the comparison.

## Correctness verification

- CPU USD regression compares **every original attribute and relationship** of
  the public Sharpa hand before/after factoring out its definition.
- A two-environment regression edits mass and joint limits in environment 0,
  and contact offset in environment 1. These remain independent; collision-pair,
  joint-body and material targets resolve under the correct environment.
- A serial 256-environment Isaac/PhysX A/B run compares actual tensor views with
  zero tolerance: body/joint ordering, 26 collision shapes, masses, inertias,
  centers of mass, material properties, joint limits/drives/armature/friction/
  velocity/force limits, contact/rest offsets, object physical properties, object
  friction samples and scale samples. All match exactly.
- The same run compares composed physics attributes and relationships of robots
  0 and 1, including self-collision and excluded pairs: exact match.
- Changing environment 0's PhysX material coefficients leaves every other
  environment unchanged. The original values are restored before learning.
- Existing CPU PPO/Gloo checkpoint continuation tests also pass. These tests
  cover algorithms/state preservation, not bitwise deterministic GPU physics.
- On Isaac/PhysX, a checkpoint created with the original USD representation
  (`baseline_audit256/model_2.pt`) resumes with the shared definition: all 82
  curriculum fields are restored, learning continues from iteration 3 through
  iteration 5, and the final checkpoint and exported policy are produced.
- Convex9 palm-up GF5k completes a 288-environment / three-update smoke run,
  including nine policy bundles. This establishes compatibility, not the
  nine-object task's steady-state memory savings.

The targeted CPU/USD suite completed with **15 passed**:
`test_shared_geometry.py`, `test_hand_usd.py`, `test_training_memory.py`,
`test_distributed_training.py`, `test_resume.py`, and `test_action_std.py`.
USD tests need the Isaac Sim `pxr` Python/library paths available. All test GPU
processes have exited; each of the four server GPUs returned to 1 MiB usage.

The initial experiment with 66 per-hand geometry instances passed physics checks
but **increased** 4,096-environment PSS to roughly 8.35 GiB. That implementation
was rejected and removed. Final sharing uses one hand-definition reference and
does not create mesh prototypes.

## Reproduction and scope

Experiments run on GPU 0 of the idle four-card 4090D server in the isolated folder
`<TRAINING_PATH>`. The existing A800
source/job is untouched; formal 4090D training is not resumed by these tests.
Test environment counts are reduced via CLI, with equal counts on both sides.
Original recipe files remain unchanged. The steady pair uses 4,096 environments,
120 PPO updates, seed 42, and the existing random-wrist HexPrism40 GF20k recipe.
Both runs produce 3,932,160 transitions. PSS is sampled every ten updates; use
updates 60–120 to compare sustained memory rather than the startup peak.

Logs, config snapshots, checkpoints, physics tensors, launchers and comparison
scripts are retained locally in `outputs/verification/steady_memory_20260920/`
(git-ignored). This is a minutes-long regression/capacity test, not validation of
long-horizon learning performance. The separate 32k-per-worker result is measured
directly rather than inferred by multiplying the small-run result.
