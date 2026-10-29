# Provisional baseline: hexprism40_palmup_v0

Designated by the user on 2026-09-17 as the reference for subsequent POISE
development and training comparisons. This is the paper-aligned dual-4090D run
started on September 15, not the earlier dual-4090D run with component success
tolerances. Training is stopped; designation does not launch or resume it.

## Frozen identity

- Training implementation: Git commit `248111365d538cc686a08ce2b3e27cfc76f0df71`.
- Run: `hexprism40_fixedpalmup_2x4090d_32000pergpu_paper_gf20k_std030_s42_20260915`.
- Reference checkpoint: `model_31200.pt`, 31,201 completed updates and
  15,974,912,000 transitions. This is the last complete checkpoint when training
  was stopped, not a checkpoint selected as best by evaluation.
- Checkpoint SHA256:
  `f801420829d6d92a31c7cf0a634a7d893e8211b89dcaac744d46be00591da87c`.
- Local artifact directory relative to the repository:
  `outputs/training/hexprism40_fixedpalmup_2x4090d_32000pergpu_paper_gf20k_std030_s42_20260915/`.

[env.json](env.json), [agent.json](agent.json), and [topology.json](topology.json)
freeze the actual rank-zero training configuration. The environment snapshot
changes only the robot USD path to a repository-relative path and clears the
run-specific log directory. Original downloaded configs remain in the artifact
directory. [manifest.json](manifest.json) records source/rank configuration
hashes, code identity, launch settings, checkpoint identity and evaluation.

## Training contract

| Setting | Baseline |
|---|---|
| Hand / object / wrist | Sharpa Wave right / HexPrism40 / fixed palm-up |
| Initialization | Diverse 6,842-row stable-grasp cache; from scratch |
| Parallelism | 2 physical RTX 4090 D GPUs, 32,000 envs each, Gloo |
| Seeds | Base 42, ranks 42/43 |
| Budget | 150,000 planned PPO updates; rollout 8; save every 100 updates |
| Success | Object-origin distance <10 mm and full SO(3) angle <10 degrees, 20 consecutive steps without drop |
| Target continuation | Refresh after success without resetting the hand/object |
| Curriculum limits | Rotation 5–180 degrees, translation 10–30 mm |
| Curriculum gate | 20,000 completed frontier events globally per level |
| Hard promotion | Frontier success 0.40; +10 degrees / +10 mm |
| Soft promotion | Enabled: success 0.30; +5 degrees / +5 mm; plateau 2,000 updates, cooldown 1,000 updates, minimum history 8,000 batches, completion gate satisfied |
| Target feasibility | HB1 root SDF at 2 mm, compound27 proxy, 2 mm allowed penetration |
| Action std | Effective sampling bounds [0.30, 1.50] |
| Critic / actor noise | Critic added noise disabled; actor joint/pose noise, delay and dropout retained |
| CoM / grasp reward | Per-axis offset ±5 mm / reward weight 0.05 |
| Policy | Actor 244D, critic 492D; both [512,256,128] ELU; BPS64 |
| Simulation / control | 120 Hz / 20 Hz; incremental joint commands, action scale 1/24 rad |

All remaining PPO, reward, sensing and dynamics parameters are in the snapshots.
Only q/q_cmd have three-frame observation histories. The episode is configured
for 400 steps; the current timeout implementation ends on the 399th env.step.
This baseline covers one object and one wrist orientation. It does not cover
the paper's joint multi-object training or randomized wrist orientation. The
paper text does not yet fully describe the soft-promotion rule.

## Recorded performance

Same 6,842 reset rows and explicit generation-0 target bank for each policy;
deterministic actor mean, no dynamics DR or observation corruption. This is a
training-cache first-target evaluation, not held-out or real-world performance.

| First-target success criterion | Baseline model31200 | Historical latest model33200 |
|---|---:|---:|
| Historical component tolerances | 68.6641% | 68.1818% |
| Full object-origin / SO(3), 10 mm / 10 degrees | 63.8556% | 63.5048% |

For the full-pose criterion, the difference is +0.35 percentage points, with a
parent-grasp-cluster bootstrap 95% interval of [-1.00, +1.71]. Overall reaching
is close; performance in the low-Z region still lags the historical model.
Results and pairing audits are under `outputs/evaluation/poise_comparison_20260917/`.

## Future comparisons

Use this configuration and checkpoint as the default reference. Record changed
parameters, topology and sample budget explicitly. Use the same success
criterion, reset rows, target bank and DR settings when claiming a performance
change. Retain v0 and assign a new baseline version if a later run replaces it.

To rerun from scratch, use the recorded training commit in an isolated checkout,
the required assets, the frozen environment snapshot and the recorded PPO config.
The following is a launch example for that implementation; the output must be new:

```bash
POISE_PYTHON=/path/to/isaaclab/bin/python bash scripts/train_4090d.sh \
  --recipe /path/to/hexprism40_palmup_v0/env.json \
  --num-envs 32000 --global-frontier-min 20000 --seed 42 --iterations 150000 \
  --output outputs/training/hexprism40_palmup_v0_rerun
```

`train.py` loads PPO settings from `source/poise/sim/recipes/ppo.json`; its hash
at the recorded commit is in the manifest. The frozen `agent.json` is an audit
snapshot, not an additional CLI argument. A different GPU count or 24k envs per
GPU is a configuration variant, not the exact v0 topology.
