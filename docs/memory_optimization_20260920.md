# POISE training memory audit — 2026-09-20

For the subsequent **sustained training-memory optimization**, see
[training_memory_shared_usd_20260920.md](training_memory_shared_usd_20260920.md).
The allocator-trim measurements in this document only establish earlier release
of initialization memory; they do not establish a substantial steady-state gain.

The initial read-only audit is followed below by an implemented host-memory
reclamation change and isolated short-run measurements. Training recipes and
physics/PPO parameters are unchanged. No allocator was injected into a running
process; the existing A800 job and formal 4090D job were not restarted.

## Measurements and scope

The upgraded server at port 35424 exposes four RTX 4090 D GPUs, each reporting
24,564 MiB VRAM, and 336 GiB system RAM. No training was running on this server.
Before stopping the two-worker, 64,000-environment random-wrist run, summed
training-process PSS was 84.8 GiB and device memory was about 21.5 GiB per GPU.
These are historical measurements, not measurements of a four-worker job.

A read-only smaps sample of the resumed A800 two-worker nine-object training
shows the following proportional set sizes. This is a related workload, not a
substitute for profiling the stopped HexPrism40 job.

| Mapping category | Worker PID 308 | Worker PID 309 |
|---|---:|---:|
| Main heap | 16.669 GiB | 38.131 GiB |
| Anonymous mappings | 10.549 GiB | 10.658 GiB |
| Other files/mappings | 3.684 GiB | 3.683 GiB |
| NVIDIA device mappings | 0.134 GiB | 0.134 GiB |

The roughly 21.5 GiB difference is concentrated in the main heap. smaps cannot
identify ownership or prove fragmentation/leakage; native allocation profiling
or controlled allocator experiments are needed. Both ranks' resolved configs
are equivalent apart from seed/rank-related metadata.

## Evidence from code

- `robot_cfg.spawn.copy_from_source=true`, `scene.replicate_physics=false`, and
  `scene.clone_in_fabric=false`. `PoiseEnv.__init__` also forces the latter false.
  The installed Isaac Lab spawner passes copy_from_source to the USD cloner.
  Independent USD/PhysX scene construction is a plausible major native-memory
  contributor, but this audit does not attribute a precise byte count to it.
- Per-environment object scale is sampled at startup in [0.975, 1.025] and authored
  into USD before physics starts. `_sample_startup_object_scales_cpu` explicitly
  rejects replicate_physics=true. A blanket flag flip is not a valid optimization
  of the same task; preserve size randomization and collision behavior.
- Nine-object training already uses one fixed object asset per environment and
  aliases that asset in its lookup dictionary. It does not simulate all nine
  objects inside every environment.
- Each PhysX scene reserves 8,388,608 rigid contact entries and 4,194,304 patch
  entries. The installed Isaac Lab default patch count is 163,840: the configured
  patch capacity is 25.6 times that default. The documented contact and patch
  streams use pinned HOST memory, despite the gpu_ prefix. Some other capacities
  use GPU memory. Capacity is entries, not bytes, and 25.6x does not apply to total
  process memory. A lower safe capacity must be established with contact stress.
- The HexPrism40 reset cache is 3.56 MiB on disk, geometry data 6.01 MiB, and palm
  SDF 0.53 MiB. The saved model_38300 checkpoint is 9.22 MiB. Disk size is not a
  complete in-memory measure, but these are not evidence of a tens-of-GiB dataset.
- For one 32,000-env worker, eight steps, 244D actor/492D critic and 22D actions,
  the basic float32 PPO storage is approximately 0.77 GiB and is allocated on the
  runner device (CUDA). This excludes activations, temporary minibatches, KL
  reference tensors, observation histories and PhysX buffers. Shrinking the policy
  is therefore not the first intervention for the system-RAM problem.

## Recommended experiments, in order

1. Add per-worker PSS/private memory and CUDA allocated/reserved measurements at
   pre-launch, scene creation, first simulation step, post-reset, rollout/update,
   checkpoint and steady-state boundaries. Record startup peak as well as steady
   usage, with equal environment counts and the same recipe.
2. Test bounded allocator arenas and safe post-initialization reclamation in an
   isolated short run (for example MALLOC_ARENA_MAX=2/4 and an allocator-compatible
   trim after construction). The main-heap asymmetry makes this worth testing;
   it is not proof that 21.5 GiB is reclaimable. Do not trim every update or call
   into the running A800 process with a debugger.
3. Test inherited/shared hand USD structure using copy_from_source=false for the
   unchanged hand first, retaining independent object scales. Verify composed
   collision filters, joint drives, material randomization and per-environment
   independence. Further mesh instancing/visual-only asset pruning can follow;
   hiding rendering alone does not remove scene data.
4. Sweep PhysX contact/patch capacities at small scale, then at the target scale
   with difficult contact/reset states. Require no capacity overflow, dropped
   contacts or changed task behavior before adopting a reduction. Do not blindly
   use upstream defaults or halve all capacities.
5. Evaluate physical/Fabric cloning only after resolving heterogeneous object and
   continuous scale authoring. Grouping by a finite scale set would change the
   original continuous randomization distribution and needs explicit treatment.

Run experiments serially at 1,024/4,096 environments first. Measure the winning
candidate at the intended per-worker environment count before claiming savings.
Retain self-collision, collision geometry, reward, curriculum and observation
contracts while testing memory-only changes.

## Four-GPU topology

Four workers with 16,000 environments each retain 64,000 total environments and
512,000 transitions per eight-step global rollout. Per-device variable storage
will decrease, but host memory is not expected to halve: there are four Kit/PhysX
processes and four sets of fixed buffers instead of two. Normalization/random
streams differ, so this is not bitwise-equivalent training.

Four workers with 32,000 environments each create 128,000 environments and
1,024,000 transitions per rollout. It increases total memory and batch size;
it is capacity expansion, not a memory optimization.

The existing train_4090d.sh hardcodes two workers/two GPUs. It must be generalized
before using four GPUs. Checkpoint loading also rejects a changed world size or
environment count: model_38300 from two workers cannot be directly resumed as
four workers with the current implementation. Either keep two workers for exact
supported continuation, start a new four-worker run, or implement and validate
an explicit optimizer/curriculum/normalizer/RNG migration policy.

## Implemented: return unused host heap pages

`source/poise/training/memory.py` provides a bounded CPU reclamation operation:
collect unreachable Python objects and call glibc `malloc_trim(0)` to return
already-free native heap pages to the OS. It does not release live tensors or
flush PyTorch's CUDA allocator. Before calling the native function it verifies
that the process's `malloc` and `free` resolve to `libc.so.6`; other allocators
and non-Linux platforms skip the native trim.

`PoiseRunner.learn()` calls this on each worker at two boundaries:

1. After environment, runner and any checkpoint restoration are initialized,
   immediately before learning.
2. After the first PPO update and any distributed checkpoint-state gathering,
   to cover lazy initialization during the first rollout.

There is no per-step or recurring per-update reclamation. The normal training
launch commands enable the change automatically. Recipes, environment counts,
GPU topology, PhysX capacities, self-collision/filtering, object randomization,
PPO updates, reward, observation and curriculum settings were not edited.
No existing process is modified by editing this code: A800 continues using its
original snapshot, and the formal 4090D training remains stopped.

For diagnostics, prefix the usual launch command with
`POISE_MEMORY_REPORT=1`. This emits per-worker `POISE_MEMORY` JSON records at the
two boundaries, including before/after PSS, RSS, private pages, peak RSS, PyTorch
CUDA allocated/reserved bytes and reclamation duration. It also emits a snapshot
after learning finishes. `POISE_MEMORY_RECLAIM=0` disables both explicit GC and
native trimming for a baseline; reporting still works. Neither variable enters
the environment or PPO configuration/checkpoint identity.

### Isolated validation

Tests ran serially on GPU 0 of the idle 4090D server, in
`<TRAINING_PATH>`, with existing assets
referenced read-only. Each run used the original random-wrist HexPrism40 recipe,
GF20k, seed 42 and ten PPO updates. Only the test environment count was reduced
to 1,024 or 4,096; both sides of each A/B pair used the same count. No training
recipe file was edited. Runs had a 12-minute timeout, and each finished before
the next started.

Local raw logs, resolved configs, final checkpoints, `comparison.json`, and the
exact launcher/comparison scripts are retained under
`outputs/verification/memory_20260920/` (git-ignored).

The reproducible runtime comparison is the same training command, output to
different directories, with these environment prefixes:

```bash
POISE_MEMORY_REPORT=1 POISE_MEMORY_RECLAIM=0  # baseline
POISE_MEMORY_REPORT=1 POISE_MEMORY_RECLAIM=1  # reclamation
```

The paired resolved environment configs (excluding output log directory), PPO
configs and topology must match, as must checkpoint configuration hashes.
Checkpoint tensors and reported PPO losses are checked for finite values.
Short PhysX GPU runs are not asserted to be bitwise identical: this configuration
does not enable deterministic physics. The CPU PPO regression instead checks
exact equality of model/optimizer states, normalizer buffers, curricula and all
saved RNG states with reclamation off/on, excluding elapsed wall time.

Targeted CPU checks (including actual two-worker Gloo save/resume):

```bash
python -m pytest -q tests/test_training_memory.py tests/test_distributed_training.py \
  tests/test_resume.py tests/test_action_std.py
```

Result: **12 passed**. This also tests disabled reclamation and safe fallback
when a supported allocator is unavailable.

### Measured results and limits

Host numbers below are per-process **PSS**, in GiB (2^30 bytes).

| Environments | Measurement | Baseline | Reclamation |
|---|---|---:|---:|
| 1,024 | Before learning, after optional reclamation | 5.298 | 4.913 |
| 1,024 | After first update, after optional reclamation | 5.510 | 5.182 |
| 4,096 | Before learning, after optional reclamation | 10.005 | 7.470 |
| 4,096 | After first update, after optional reclamation | 10.340 | 7.840 |
| 4,096 | After ten updates and checkpoint save | 7.931 | 7.851 |

Within the optimized process itself, the first trim returned **392 MiB** at
1,024 environments (0.210 seconds), and **2.521 GiB** at 4,096 environments
(0.543 seconds). The second trim returned another 3.9 MiB / 23.5 MiB and cost
0.249 / 0.298 seconds respectively. PyTorch allocated/reserved CUDA bytes were
identical immediately before/after every reclamation operation.

The 4,096-env process peak RSS reported by `/proc/self/status` was 10.339 GiB
for baseline and 9.983 GiB for reclamation. Startup allocations before the
first trim remain necessary: this is not a solution to an OOM during scene
construction itself.

By the end of ten updates, baseline had also released much of its transient
heap without explicit reclamation. The final difference was only **82 MiB**
(about 1%). The demonstrated benefit is earlier release of initialization
memory; **multi-GiB long-term savings have not been established**. End-of-run
telemetry was added after the 1,024-env pair, so no final PSS claim is made for
that pair. PyTorch CUDA accounting excludes native PhysX/Kit allocations.

Median iteration times excluding the first update were 1.15 vs 1.21 seconds
at 1,024 environments, and 1.77 vs 1.82 seconds at 4,096 environments
(baseline vs reclamation). These ten-update, single-pair measurements cannot
establish a throughput change; no speedup is claimed. All four runs completed
with exit code zero, finite model/loss values, matching paired resolved configs
and checkpoint identities, and successful policy export. The test host had no
remaining training processes and each GPU reported 1 MiB afterward.

This allocator-only experiment did not benchmark 32,000 environments per worker
or long-running heap growth. The subsequent
[64k comparison](training_memory_64k_20260920.md) measures shared USD definitions
with reclamation enabled in both arms; it does not isolate allocator-only savings
at that scale. Neither result establishes the running nine-object A800 task's
memory savings. PhysX capacities and physics cloning settings remain unchanged.
