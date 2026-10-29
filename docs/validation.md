# Historical validation

These dated records preserve earlier evidence. Current release acceptance and
its environment boundaries are in [release validation](release/validation.md).

## Diverse reset generation (2026-09-18)

The public-hand formal pipeline completed a fresh 486-parent / 6,842-state build,
a small end-to-end build, strict resume, and 32-environment training/export smoke
tests with both the small cache and final cache. Matched independent eight-wrist
replay retained 85.31% of regenerated states versus 87.52% of reference states.
This measures reset stability, not reaching-policy performance equivalence.
See [the validation record](release/validation.md)
for measured counts, hashes and scope.


## A800 distributed training — 2026-09-13

Training commit `7fb22e6` was tested on one A800-SXM4-80GB with two Gloo
workers sharing GPU0. The simulator stack is Python 3.11.15, Torch 2.7.0+cu128,
Isaac Sim 5.1.0, Isaac Lab 0.48.0 and RSL-RL 3.0.1.

- CPU Gloo test: synchronized real PPO updates, independent worker seeds and
  observation statistics, checkpoint save/load and continued updates passed.
- GPU smoke: 2 × 128 environments, three updates, 6,144 transitions; checkpoint
  and policy bundle export succeeded. Continuation added two updates to reach
  five completed updates and 10,240 transitions.
- Capacity test: 2 × 32,000 environments, three updates, 1,536,000 transitions;
  checkpoint and bundle export succeeded. Observed GPU memory was approximately
  42 GiB, with 34,855–38,761 transitions/s across these first three updates.
- The 64k checkpoint contained identical values for all 82 curriculum fields
  across workers. Model tensors and all 279 logged scalar series were finite.
- Formal from-scratch training with 64,000 environments, seeds 42/43 and
  150,000 planned updates started after these checks. Its first checkpoint
  `model_0.pt` was written following 512,000 transitions.

Scene creation took approximately three minutes, followed by ten minutes of
PhysX initialization at 32,000 environments per worker. This is startup time,
not the per-update training cost. Public hand files and the reference object,
reset states, BPS, SDF and joint-limit asset hashes were verified on the server.

The 192k-environment reference recipe is preserved except for the requested
64k environment topology and public hand source with the collision overlay.
Per-update batch size is one third of the reference run; iteration-wise learning
curves are not expected to match. These startup checks establish operability,
not converged reaching performance. See [training](training.md) for launch,
logging and continuation instructions.

## Public hand migration — 2026-09-13

Public source: `https://github.com/sharpa-robotics/sharpa-urdf-usd-xml`, commit
`0d447b6889e6d993758169dfc0aa75ee9f6ad8d7`. Both HTTPS archive installation and
installation from the local public checkout passed verification of 29 upstream
files plus 2 POISE composition layers. No credentials/private hand directory
are needed by the downloader. The previous installed tree was preserved at
`data/robots/sharpa_wave.backup-a662a3b1c25d`.

- Composed geometry/physics: **2,696 records compared, zero differences** from
  the previous Wave1.4 asset (joint/rigid-body properties, collision geometry,
  transforms, enabled schemas and exclusion targets). Rendering/visual settings
  and empty blocked relationships are excluded from this comparison.
  `outputs/validation/public_hand_physics.json`.
- Tests: **91 passed, 6 subtests**, including archive/file-integrity faults,
  backup preservation and USD relationship remapping under an environment's
  `Robot` reference. With pxr unavailable, the USD-only test is skipped; it was
  run here using the installed Isaac Sim USD libraries.
- Existing checkpoint continuation: 1,024 environments × 20 updates, 163,840
  additional transitions; `outputs/training/public_hand_resume1024/model_1229.pt`.
  The run and bundle export completed successfully.
- Same-checkpoint/same-seed replay: 16 steps × 4 environments, no resets.
  Joint positions, joint commands, object poses and policy observations match
  the prior original-asset replay **exactly in this short run**. This does not
  establish bitwise equality for arbitrary long GPU physics trajectories.
  `outputs/validation/public_hand_rollout_comparison.json`.
- Independent deployment replay: max observation/action error
  2.443790436e-6 / 4.768371582e-6, below 2e-5.
  `outputs/verification/public_hand/result.json`.
- Grasp environment: the public URDF and selected meshes load with urdfpy;
  22 actuated joints / 33 links. `outputs/validation/public_hand_grasp_urdf.json`.
- Physical grasp filtering: the existing 34 candidates were rerun through
  4 wrist orientations × 240 physics steps; **29/34 pass**, as before.
  `outputs/grasp_generation/public_hand.filter.json`.
- Wheel rebuilt and installed separately; from `/tmp` with `python -I`, the
  packaged manifest/overlays install and verify the hand without importing USD
  or Isaac Lab. No vendor mesh/URDF/USD binaries are included in the wheel.
  `outputs/validation/public_hand_wheel_result.json`.

The separate USDA09 joint-limit/drive reference, object/reset data and palm SDF
were not changed. These checks establish asset compatibility, not converged
reaching performance. Original research and public source checkouts were not
modified; no push or hardware actuation occurred.

## Deployment integration — 2026-09-13

### Longer local continuation

From `final_resume1024/model_209.pt`, completed another 1,000 PPO updates with
1,024 environments and 8 steps/update: **8,192,000 new / 9,912,320 total
transitions**. Final checkpoint: `outputs/training/continued_1024/model_1209.pt`.
The process exited successfully and exported the per-object bundle.

Final checkpoint checks: next iteration 1,210, 82 curriculum fields, global
control step 9,680, Adam step 33,280, LR 0.00013168724; model tensors are finite.
See `outputs/validation/continued_checkpoint.json` and the run's `result.json`.

The last 100 logged entries average reward 3.7823 and episode length 355.16
control steps (about 17.76 s), but both episode/first-target success signals
average **0**. Curriculum remains 5° rotation and 10 mm translation, with no
promotion. This is **not a converged reaching policy**. Tail averages summarize
logged scalars, not pooled episode statistics or held-out evaluation. Reproduce
the report with `scripts/summarize_training.py`; saved output is
`outputs/validation/continued_training_metrics.json`.

The final model's fresh Isaac Lab replay passed 16 steps × 4 environments with
no resets. Independent deployment observation/action errors are respectively
2.443790436e-6 / 4.768371582e-6 (tolerance 2e-5). Artifacts:
`outputs/verification/continued_1024/{bundle,golden.pt,result.json}`. This short
noise-disabled replay checks the interface, not task success.

### Offline deployment checks

85 tests and 6 subtests pass. Added configuration/artifact fault checks, container
SDK path tests, camera disconnect/interrupted capture tests, and shutdown order
tests. Exceptions stop the robot helper before closing capture and camera;
secondary cleanup failures preserve the original failure.

Testing the real exported actor through `PolicyController` uncovered a
single-frame shape error: fused normalization returned `(1,22)` instead of
`(22,)`. `PolicyBundle.infer` now preserves single versus batched input shape,
checks finite output, and has an export-to-controller regression test. Offline
preflight exercises this inference path as well.

- `outputs/validation/deployment_preflight/result.json`: actual HexPrism40
  bundle/mesh/target pass preflight. Mask, tag mounting and hardware-limit values
  are an **offline fixture**, not a robot installation or valid live calibration.
- `outputs/validation/controller_real_bundle.json`: all 16 recorded frames of
  environment 3 produced acknowledged 22D commands. Environments 0/1/2 stopped
  at frames 2/2/8 because their recorded motion exceeded the existing jump limits;
  limits were not relaxed. Acknowledgments were local, with no SDK process.
- `outputs/validation/golden_after_deploy_fixes.json`: 17 frames × 4 environments;
  maximum observation error 1.847743988e-6, action error 7.152557373e-7, pass.
- `outputs/validation/golden_python39.json`: the same independent replay also
  passes in the Python 3.9/Torch 2.8 grasp environment after the inference change.
- `outputs/validation/wheel_result.json`: rebuilt wheel installed to a separate
  target directory and used from `/tmp` with `python -I`; template, preflight,
  single-frame inference and golden replay passed. Isaac Lab, RSL-RL, AP2AP,
  RealSense, Sharpa SDK and FoundationPose were not imported. Existing host Torch,
  OpenCV and trimesh were used; this is not a fresh-machine dependency install.

`scripts/deploy.py --help` and syntax checks passed. No camera acquisition,
Docker SDK launch, robot actuation, Git push or publication was performed.

## Earlier full-chain validation — 2026-09-12

Ubuntu 22.04, RTX 4090 (~48 GiB), Isaac Sim 5.1, Isaac Lab 0.48.0,
Python 3.11.15, Torch 2.7.0+cu128, RSL-RL 3.0.1. See installation.md for
optimizer versions. All simulator/optimizer jobs below finished; none was left
running unattended. Original research source was not modified.

| Check | Actual result / local artifact |
|---|---|
| Unit/regression tests | 49 passed, 6 subtests; core, cache, calibration, command guards, timing faults, curriculum snapshots, SDF |
| Formal-cache training, final extracted environment | 1,024 environments × 200 PPO updates × 8 steps = 1,638,400 transitions; `outputs/training/final1024/model_199.pt` |
| Checkpoint continuation | Additional 10 updates / 81,920 transitions; next iteration 200→210; `outputs/training/final_resume1024/model_209.pt` |
| Continuation state | 82 curriculum fields; global control steps 1,600→1,680; Adam steps 5,592→5,832; optimizer LR and normalizer continued |
| New grasp generation | 1,620 grid poses × 2 passes → 34 candidates; `outputs/grasp_generation/grid.npy` |
| New q_cmd and schema conversion | 34 rows; q_cmd derived from wrench/Jacobian/stiffness and retained |
| Physical grasp filtering | 29/34 survived all 4 wrist orientations × 240 physics steps (2 s each); `outputs/grasp_generation/grid.filter.json` |
| Training on newly generated reset states and geometry | 128 environments × 20 updates = 20,480 transitions; `outputs/training/generated128/model_19.pt` |
| Two objects + random wrist | HexPrism35/40, 32 environments each, 5 updates; `outputs/training/multiobject64_v2/model_4.pt`; two object bundles exported |
| Geometry regeneration | 64D BPS, volume/COM/inertia and 64³ surface fields regenerated from OBJ |
| Palm SDF generation | 2 mm grid, 57×74×76, generated from the local palm collision STL |
| Final policy export and play | 16 control steps, 4 environments, no resets; `outputs/verification/final/` |
| Independent observation replay | 17 frames × 4 envs; max observation error 1.847743988e-6; max action error 7.152557373e-7; tolerance 2e-5 |
| Wheel | Built, installed into a temporary directory, packaged JSON recipe read outside checkout, final golden replay passed without Isaac/AP2AP/RSL imports |
| Hardware | No camera acquisition or robot actuation performed |

Checkpoint resume validates a configuration fingerprint, including object order,
seed and environment count. It restores PPO/normalizer/optimizer, curriculum,
learning rate and random streams, then begins fresh simulation episodes. PhysX
contact state is not serialized; this is not a bitwise trajectory restart.

Two older reference bundle golden recordings also passed the initial extraction
checks: observation/action errors 1.431e-6/3.338e-6 (ManualConvex9 model21800) and
1.550e-6/1.526e-5 (HexPrism40 OmniWrist model29000). Their recordings remain in the
private reference checkout.

The runtime uses a local static floor instead of a network-hosted visual ground
asset. All results are integration checks in existing environments, not a fresh
machine reproduction or converged POISE performance evaluation. GPU physics has
small numerical variability: one same-seed 5-update comparison before/after dead
code removal differed by at most 3.3844e-6 across checkpoint tensors. The final
independent observation/action replay is the numerical deployment contract check.

Public asset acquisition, external dependency packaging, release license review,
longer performance training and physical closed-loop validation remain unfinished.
