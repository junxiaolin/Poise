# Setup asset manifests

These manifests list the actual file closure needed for simulation/training and
optional reference rollout. Paths start with `data/` and are relative to
`POISE_ROOT` (the checkout by default). Each record has `path`, `bytes`, `sha256`
and `role`; only `reference_checkpoint` records are optional for training.

The hand includes vendor files, composition layers, its provenance and LICENSE.
Object entries include meshes, USD and available per-object provenance. Geometry
and grasp caches retain the historical training identities. Checkpoint candidates
are copied under ignored `data/checkpoints/<setup>/` and pinned by SHA256.
Their historical source records live in the setup descriptors.

Each distribution now pins a shared release index by path/SHA256. It maps setup
IDs to deduplicated packages, raw geometry files and optional checkpoints.
The release index pins HTTPS URLs, archive sizes and hashes. `--source` accepts
a downloaded release directory, a data workspace or a legacy complete tar bundle.
See [release contents](../../docs/release/README.md).

Local import verifies every selected asset before publishing files and refuses
to overwrite a different existing file. Unknown archive entries are ignored;
selected entries must be unique regular files. Hashes verify identity, not rights
to redistribute or suitability of untrusted Python tensor serialization.

Future releases must retain origin/license records, choose a versioned reference
set and update URLs/hashes explicitly. Preparing these manifests does not change
repository visibility or upload files.
