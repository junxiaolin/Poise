# A800 Convex9: model23300 versus model19100

Latest complete checkpoint at evaluation start (2026-09-21T02:25:53.479024+00:00):
`model_23300.pt`, SHA256 `9619a7c55a4b874c3b92b952ffdf83d589d216227c541b0d86cec58be55e5fdd`.
The training configuration is identical to the previous checkpoint; 4,200 more
iterations. Frozen deterministic actor and frozen normalizers. The remote
training job continued unchanged.

| Suite | Episodes | model19100 | model23300 | Difference (pp) | Paired 95% interval (pp) |
|---|---:|---:|---:|---:|---:|
| seen | 9216 | 68.00% | **69.61%** | +1.61 | +0.70 to +2.58 |
| unseen | 6144 | 57.45% | **57.19%** | -0.26 | -1.50 to +0.91 |

Interpretation under this fixed evaluation protocol:

- seen: a small positive change supported by the paired interval.
- unseen: no clear improvement or regression; the paired interval includes zero.

Unseen subgroups:

| Group | Episodes | model19100 | model23300 | Difference (pp) | Paired 95% interval (pp) |
|---|---:|---:|---:|---:|---:|
| new_dimensions | 3072 | 58.85% | 55.86% | -2.99 | -4.72 to -1.32 |
| new_shapes | 3072 | 56.05% | 58.53% | +2.47 | +0.72 to +4.17 |

Whole-episode drop rates (including any drop after a first-target success):

- seen: 1.76% → 2.07%.
- unseen: 3.87% → 4.62%.

## Seen objects

All objects: 1,024 episodes each. Dimensions in mm.

| Object | model19100 | model23300 | Difference (pp) | Paired parent-grasp 95% interval (pp) |
|---|---:|---:|---:|---:|
| Truncated bipyramid 40x40 | 65.72% | **67.09%** | +1.37 | -1.31 to +4.13 |
| Truncated bipyramid 45x40 | 74.32% | **75.59%** | +1.27 | -1.32 to +3.93 |
| Truncated bipyramid 50x50 | 67.09% | **67.58%** | +0.49 | -2.40 to +3.30 |
| Hex prism AF35 H35 | 73.44% | **74.41%** | +0.98 | -2.00 to +3.94 |
| Hex prism AF40 H40 | 60.06% | **60.74%** | +0.68 | -1.75 to +3.13 |
| Hex prism AF45 H45 | 65.62% | **67.97%** | +2.34 | -0.40 to +5.13 |
| Chamfered cube 35 | 73.44% | **74.71%** | +1.27 | -1.65 to +4.08 |
| Chamfered cube 40 | 68.46% | **72.75%** | +4.30 | +1.36 to +7.25 |
| Chamfered cube 45 | 63.87% | **65.62%** | +1.76 | -1.39 to +4.97 |

## Unseen objects

All objects: 1,024 episodes each. Dimensions in mm.

| Object | model19100 | model23300 | Difference (pp) | Paired parent-grasp 95% interval (pp) |
|---|---:|---:|---:|---:|
| Hex prism AF40 H30 mm | 67.68% | **66.80%** | -0.88 | -3.30 to +1.62 |
| Hex prism AF40 H50 mm | 56.15% | **53.52%** | -2.64 | -5.71 to +0.29 |
| Chamfered cuboid 50 x 35 x 30 mm | 52.73% | **47.27%** | -5.47 | -8.85 to -2.17 |
| Regular pentagonal prism D44 H40 mm | 65.23% | **66.41%** | +1.17 | -1.85 to +4.01 |
| Wedge 40 x 36 x 36 mm | 45.12% | **50.59%** | +5.47 | +2.32 to +8.69 |
| Cylinder D40 H40 mm (32-sided collision mesh) | 57.81% | **58.59%** | +0.78 | -2.04 to +3.47 |

## Matched protocol and limits

Each suite reuses its original cache rows, source poses, target poses and target
axis commands. Every initial target matches the archived float32 arrays bit for
bit before inference. Zero-translation fallback directions, which otherwise
inherit policy-dependent episode history, are restored from the archived command.
Exact original batch topology is preserved (seen: 8 x
1,152 environments; unseen: 32 x 192 environments), with the same reset/target
seed schedule. Only policy/checkpoint and output paths change. The original
mesh inertia multiplier log-uniform [1,4] remains enabled; all other configured
DR and observation noise are disabled. Unseen inertia samples also match saved
arrays exactly. Seen inertia was not recorded previously; it is reproduced by
the unchanged reset code/topology/seed, whose saved target outputs match exactly.

Fixed palm up, first target, configured 20 s horizon (existing timeout step399).
Success: object-origin distance <10 mm and full SO(3) error <10 degrees for
20 consecutive control steps before drop. Targets do not refresh in this
evaluation. Symmetries are not discounted. A later drop does not undo a recorded
first-target success. This is separate from the continuous-target videos.

Seen evaluation uses the original 1,024 unique training-cache rows per object.
Unseen evaluation reuses the same accepted reset population (78,41,99,65,94,78
states), with repeated targets and the same final geometric fallback check.
No assets, grasps or target samples are reselected based on the new policy.
The unseen rate is conditional on these stable resets, not arbitrary grasps.

Per-object paired intervals use 5,000 donor-parent cluster bootstrap replicates,
retaining success/failure pairs and repeated/related grasp variants. Suite-level
intervals keep the object set fixed and stratify these draws by object. They do
not represent uncertainty across independent training runs or new object sets.
Per-object intervals are descriptive and not adjusted for multiple comparisons.
Small changes should also be assessed against the GPU simulator replay control.

## Old-model replay control

- seen: same model19100 on 2304 archived episodes: historical 67.62%, rerun 67.62%, difference +0.00 pp; 0 episode outcomes differ.
- unseen: same model19100 on 384 archived episodes: historical 54.17%, rerun 54.17%, difference +0.00 pp; 0 episode outcomes differ.

## Reproducibility

`manifest.json`: downloaded checkpoint/policy hashes, iteration and export audit.
`training_config_diff.json`: training env configuration comparison (empty).
`source_audit.json`: unchanged environment, success and config source.
`unseen_asset_hash_audit.json`: unchanged unseen BPS, meshes and collision assets.
`seen_latest/`, `unseen_latest/`: per-episode results, exact-input reset audits,
resolved configs and first-target success statistics.
`seen_reference/`, `unseen_reference/`: old-model repeatability checks.
`comparison.json`: paired differences, clustered intervals, flips, drops and latency.
`comparison.png`: old/new per-object success rates.

Run `scripts/evaluation/unseen_objects/compare_checkpoint.py --comparison-dir
<this-directory> --suite seen|unseen --policy latest|reference --headless` in the
configured Isaac Lab Python environment. Output directories must not already
exist. Add `--batches 2` for the reference controls. Then run
`summarize_checkpoint_comparison.py --comparison-dir <this-directory>`.
