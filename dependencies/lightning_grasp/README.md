# POISE's Lightning Grasp compatibility patch

`poise.patch` contains changes derived from Lightning Grasp by Zhao-Heng Yin and
the Sharpa adaptation recorded in `manifest.json`. It is distributed under
CC BY-NC 4.0; see [the full license](../../licenses/Lightning-Grasp.txt).
Existing copyright notices are preserved. The patch adds external-wrench and
force-coefficient support, metadata propagation, explicit collision resting poses
and URDF limit/resource utilities; these are modifications to upstream.

Source: https://github.com/zhaohengyin/lightning-grasp

Sharpa adaptation: https://github.com/ZJU-jxlin/lightning-grasp-sharpa

POISE distributes the patch and fetches pinned public upstream sources and
upstream release binaries separately. No private fork checkout, hand assets or
CUDA binaries are included here. See [installation and audit](../../docs/lightning_grasp.md).
