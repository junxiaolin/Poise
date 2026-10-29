# Training and simulation

Install Isaac Sim 5.1.0.0 and Isaac Lab 0.48.0 using their upstream instructions.
The validated Isaac Lab commit is `5c2ec81cb17532d32f7922dd7fcaae40d123b71a`
from https://github.com/isaac-sim/IsaacLab. Use its Python 3.11 environment with
Torch 2.7.0+cu128; exact inspected versions are in [manifest.json](manifest.json).

From the POISE root, using that Python:

```bash
python -m pip install -r environments/train/requirements.txt
python -m pip install --no-deps -e .
poise check-env train --native
poise list-setups
```

Continue with [setup reproduction](../../docs/reproduction.md). No LyGra,
FoundationPose, camera or robot SDK installation is needed. The smoke profile uses 8 environments (18 for Convex9) and two PPO updates.
Reference profiles differ: fixed Hex40 is 2×32k on two GPUs; Convex9 is 2×32k
on one large GPU; OmniWrist Hex40 and HexHammer are 8×24k on four A800 80GB GPUs.
See [hardware and profiles](../../docs/release/README.md#hardware-and-profiles).
Inspect --dry-run before launching reference scale.
