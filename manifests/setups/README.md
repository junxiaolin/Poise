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
The index remains private_staging with null URLs; --source accepts the candidate
release directory, a data workspace or a legacy complete tar bundle. Publishing
requires an explicit index/version change after approval. See docs/release/README.md.

Local import verifies every selected asset before publishing files and refuses
to overwrite a different existing file. Unknown archive entries are ignored;
selected entries must be unique regular files. Hashes verify identity, not rights
to redistribute or suitability of untrusted Python tensor serialization.

Before any future public release, complete origin/license review for every data
class and weight, choose a release version/reference set, stage approved archives
and add their verified URLs/hashes as an explicit release change. Preparing these
manifests grants no new permission and performs no publication.
