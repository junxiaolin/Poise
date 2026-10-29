# Optional grasp optimizer

Use Python 3.9.23, Torch 2.8.0+cu126 and the separate environment described in
[the tested LyGra installation recipe](../../docs/lightning_grasp.md).
The complete POISE adapter ships with the main repository; upstream sources and
CUDA binaries are fetched only when `scripts/setup_lightning_grasp.py` is run.

```bash
# In the separately configured optimizer environment:
python -m pip install --no-deps -e .
poise check-env lygra-poise --native
python -m poise.grasp.lightning.runtime --cuda-smoke
```

Existing setup training and rollout do not need this environment. For new object
geometry follow [new objects](../../docs/new_objects.md), then use Isaac Lab to
filter/expand the generated caches. New hands require additional implementation;
this release supports only Sharpa Wave right hand.
