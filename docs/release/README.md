# Private release candidate

The first release scope is training and simulation reproduction of four fixed
Sharpa Wave baselines, with optional new-grasp generation. Deployment is
experimental. General new-hand support and independently benchmarked robot
control are not promised.

## Rights and distribution

The owner confirmed authority to distribute Poise-owned and AP2AP-migrated code,
authored object assets, grasp caches and checkpoints on 2026-10-09. This is the
basis of the rights record, not an independent ownership investigation.

- Poise-owned code/documentation: MIT, with exceptions in NOTICE.
- Authored meshes, caches, geometry descriptors and policy weights: CC BY 4.0.
- Sharpa upstream files: Apache 2.0; Poise collision layers: MIT.
- LyGra-derived adapter and compatibility patch: CC BY-NC 4.0, with attribution.
- FoundationPose source/weights: external, non-commercial research/evaluation
  terms at the pinned revision; not redistributed by Poise.

See [rights manifest](../../manifests/rights.json) and [NOTICE](../../NOTICE).
The MIT core does not change optional components' non-commercial restrictions.

## Shared asset transport

The four setups use a single pinned release index and 11 training packages: common hand
assets, shared Hex40 assets, extra Convex9 assets, HexHammer assets, three
geometry files and four independent checkpoints. A twelfth, optional generation package contains the
numeric joint-limit reference used by grasp search. Files appear in one package;
setups reference shared dependencies. Geometry tensors retain their exact object
order. SHA256 identifies archives and every extracted file.

```bash
# Maintainer: builds locally and binds the index; never uploads
PYTHONPATH=source python scripts/release/package_assets.py \
  --output outputs/releases/candidate/assets --bind

# User: private/offline candidate directory downloaded or provided separately
poise fetch-assets --setup hexhammer_palmup_fullpose_192k_v1 \
  --source /path/to/release/assets --with-reference
```

The importer reuses correct existing files, skips checkpoints unless requested,
verifies all missing dependencies before installation, and rejects conflicting
files. Legacy complete tar bundles and data workspaces are still supported.
Automatic HTTPS download works when the pinned index is explicitly published
with archive URLs. URLs/status remain unset/private in this candidate; no fake
public endpoint is advertised. A future GitHub Release uses the same repository
as the source; large assets never enter Git history. Raw geometry/checkpoint
files are accompanied by release NOTICE, licenses and the rights manifest.

## Source boundary

The working development repository and its history remain private. Internal
operations/handoff records are preserved there; they are not needed by users.
Generate the public-facing candidate with:

```bash
PYTHONPATH=source python scripts/release/export_source.py --output /tmp/poise-source
python scripts/release/audit.py --root /tmp/poise-source --output /tmp/source-audit.json
python scripts/release/audit.py --history --output outputs/history-audit.json
```

The exporter includes code, tests, configurations and public guides, retains
needed baseline/evaluation evidence, redacts machine-specific paths in evidence,
and excludes internal cloud launchers and operations records. Exported evidence
hashes are recomputed; model/asset bytes and training recipes are not changed.
The candidate has no private Git history. Do not make the old development
repository/history public by simply flipping its visibility. Publishing a new
clean history or replacing remote history is a separate user-approved action.

## Hardware and profiles

| Setup | Reference hardware / topology | Gate | Target range | Reference |
|---|---|---|---|---|
| Hex40 palm-up | 2 GPUs, 2×32k = 64k | GF20k | 180° / 30 mm | 31200 |
| Convex9 palm-up | 1 large GPU, 2×32k = 64k | GF5k per object | 180° / 30 mm | 23300 |
| Hex40 OmniWrist | 4×A800 80GB, 8×24k = 192k | GF160k | 180° / 30 mm | 35800 |
| HexHammer palm-up | 4×A800 80GB, 8×24k = 192k | GF20k | 180° / 100 mm | 89800 |

All start from seed 42, use eight rollout steps, strict FullPose success and no
tactile input. The 192k cloud runs used 80GB A800s; use that configuration for
reference-scale reproduction. The historical 2-GPU Hex40 measurement was about
21.45 GiB VRAM per GPU; this is not a peak guarantee or a Convex9 estimate.
For 192k jobs, the tested launch guidance is at least 64 vCPUs / 256 GiB host RAM.
Small smoke (8 environments; Convex9 18) and reference rollout (4; Convex9 9)
are separate from formal-scale requirements. Local acceptance uses an RTX 4090
with approximately 48 GiB reported memory, not a standard 24GB memory guarantee.
No minimum VRAM claim is made without a measured result for that setup.

Hex40/Convex9 qualitative rollout disables most noise/randomization and samples
0–180° / 0–30 mm uniformly. HexHammer reference rollout preserves training noise,
randomization and target sampling, initialized at the saved 180° / 100 mm range.
Full curriculum is not a 100% success-rate claim. See each baseline for provenance.

## Historical cache provenance

The 92 unique training data files were scanned before packaging. Ten grasp-cache files
retain five historical local source paths each in metadata (seed/intermediate
cache, original hand and limit-reference locations). These are provenance strings,
not runtime dependencies or download locations. They are retained deliberately
to preserve the frozen cache hashes; no node address or credential marker was
found in these asset files by the scan. Tensor loading and simulator use are
validated with those original paths inaccessible. Runtime hand assets instead
come from the pinned public Sharpa model plus Poise's collision overlay.
