# Two-4090D / 64k training-memory benchmark

## Results

Both jobs completed 120 updates / 61,440,000 transitions, saved their final
checkpoint and exported the policy. Both exit codes were zero. All four GPUs
returned to 1 MiB usage after the test; no formal training was restarted.

| Measurement | Original USD representation | Shared USD definition |
|---|---:|---:|
| Mean summed worker PSS, updates 60–120 | **62.724 GiB** | **48.250 GiB** |
| Summed PSS range, updates 60–120 | 62.716–62.737 GiB | 48.239–48.264 GiB |
| Rank 0 mean PSS | 31.293 GiB | 24.221 GiB |
| Rank 1 mean PSS | 31.430 GiB | 24.029 GiB |
| Sampled peak summed PSS, entire run | 102.902 GiB | 48.264 GiB |
| GPU 0 sampled peak VRAM | 21.406 GiB | 21.410 GiB |
| GPU 1 sampled peak VRAM | 21.451 GiB | 21.451 GiB |
| Median iteration time, iterations 60–120 | 6.16 s | 5.96 s |

**Sustained host memory saved: 14.473 GiB (23.07%) across two workers.** This
confirms the optimization at the requested 2×32,000 scale. VRAM is effectively
unchanged. Median iteration time did not increase in this pair; no statistical
speedup claim is made.

The baseline's approximately 103 GiB peak occurs during initialization. It must
not be compared with optimized steady-state usage to claim a 53% *training*
memory reduction. The correct steady-to-steady comparison is 62.724→48.250 GiB.
Reported peaks are 10-second samples, not guarantees of the largest transient.

Scene construction took 144.1/147.9 seconds for baseline versus 44.9/44.5 seconds
with sharing. Simulation startup took 565.0/558.1 seconds versus 434.0/433.8
seconds (ranks 0/1). A single pair includes file/cache/run-order effects.

The comparison script verifies equal paired environment/PPO/topology configs
(excluding log directory), finite final model tensors, 120 completed updates,
64,000 total environments, 61,440,000 transitions, and identical checkpoint
configuration hashes for each rank:

- Rank 0: `0e3a3d491f89e4db00f290827056f3f91d837f472d9e410783526d02699ace86`
- Rank 1: `9aa8d9831705d34f7329f7433211d2b4ab65b283f14fb1b13513e0d81f89f7c1`

These measurements cover the existing random-wrist HexPrism40 task. They do not
quantify Convex9 memory or establish leak-free behavior over days of training.
The implementation is enabled by default in this revision; see
[training defaults](training.md#default-training-memory-management). This test
did not inject changes into the running A800 job. New launches must use the
updated checkout to receive the optimization.

## Protocol

Run two serial from-scratch jobs on physical GPUs 0 and 1 of the four-card
RTX 4090D host (SSH port 35424). Each worker uses 32,000 environments: 64,000
total, not 65,536. Use the existing HexPrism40 random-wrist GF20k recipe with
the original seeds 42/43, Gloo, physics, PPO, observations, rewards, curriculum,
asset source and randomization. Each job runs 120 updates / 61,440,000 total
transitions. The existing A800 job is not touched.

The original formal run's resolved `env.json`, `agent.json` and `topology.json`
were compared per rank. They match after excluding output directory, the asset
path's isolated checkout prefix, and the intentionally shortened iteration limit.
Within the A/B pair even the asset path is identical.

Both jobs enable the earlier initialization reclamation. The sole implementation
difference is `POISE_SHARED_HAND_GEOMETRY=0` (baseline) versus `=1` (shared USD
definition). This isolates the sustained-memory effect from allocator trimming.
Recipes, launch scripts and assets used by the formal jobs are not edited.

Every ten PPO updates, each worker logs PSS/RSS, peak RSS and PyTorch CUDA
allocated/reserved bytes. An external monitor independently records both worker
PSS values and full-device `nvidia-smi` memory every ten seconds throughout
initialization and training. GPU 2/3 stay idle. Host memory is the sum of the two
workers' **PSS**; it excludes the small torchrun launcher and monitor. Summing PSS
avoids counting their shared library pages twice, unlike summed RSS.

Use updates 60–120 for the sustained comparison. Report startup/whole-run peak
separately. GPU memory includes native PhysX/Kit, not just PyTorch tensors.
Ten-second sampling may miss brief peaks. A single serial pair does not establish
long-horizon memory growth or policy performance equivalence.

## Reproducibility

Remote isolated root:
`<TRAINING_PATH>`.
Runs: `outputs/verification/baseline_2x32000` and `shared_2x32000`.
Each job has a 35-minute hard timeout; the second starts only if the first exits
successfully. These are bounded tests, not a formal training restart.

Local evidence is retained in `outputs/verification/memory64k_20260920/`:
launcher/monitor/comparison scripts, source checksums, per-rank logs, resolved
configs, final checkpoints and resource samples. This folder is git-ignored.
