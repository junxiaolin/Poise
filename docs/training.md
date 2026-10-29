# Training and continuation

Use [versioned setups](reproduction.md) for the four supported baselines.
Their [reference topology and hardware](release/README.md#hardware-and-profiles)
are explicit; the default profile is a small single-GPU smoke run.

```bash
poise train --setup hexprism40_palmup_v1 --profile smoke --output outputs/smoke
poise train --setup hexprism40_palmup_v1 --profile custom --num-envs 128 \
  --iterations 100 --output outputs/train
poise train --setup hexprism40_palmup_v1 --profile custom --num-envs 128 \
  --iterations 100 --resume outputs/train/model_99.pt --output outputs/continued
poise train --setup hexhammer_palmup_fullpose_192k_v1 --profile reference \
  --output outputs/reference --dry-run
```

Outputs must be new directories. Iterations means additional PPO updates.
Continuation requires matching setup, environment count, seed and topology. It
restores optimizer, policy normalizers, curriculum and per-rank random streams;
PhysX episodes restart, so this is not bitwise trajectory continuation.

Reference checkpoints without a setup identity are recognized by their pinned
hash for rollout. For continuing an older custom training run, use its original
recipe/topology with scripts/train.py; do not rewrite its setup identity.

For a deliberate curriculum-gate experiment on a custom recipe, pass both
`--global-frontier-min 160000 --resume-global-frontier-from 20000` together with
`--resume original-run/model_11000.pt`. The old value must reproduce the saved
configuration fingerprint; every other environment setting remains checked.
This keeps the original checkpoint, optimizer, curriculum history and counters.
The gate applies to future promotion decisions; it does not reset a level or
undo previous promotions. Subsequent checkpoints use the new fingerprint and
resume without `--resume-global-frontier-from`.

Legacy/custom recipes remain available via scripts/train.py --recipe and
scripts/play.py --recipe. Tactile recipes use 304D actor / 552D critic with three
frames of fingertip force/contact-position observations. They are experimental
and are not among the four frozen reference setups.

Shared-GPU distributed jobs use Gloo and map local ranks onto physical GPUs.
Use the setup reference profile to preserve world size and curriculum gate.
The CLI's --dry-run prints the exact command without starting a job.
