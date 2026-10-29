# Versioned setups

Each `<id>.json` binds one reproducible experiment. `poise list-setups` lists
registered IDs; `poise show-setup <id>` prints the descriptor.

Schema version 1 includes:

- `id`, `status`, `description`, `hand`.
- `recipe` and `ppo`: packaged resource name and file SHA256; recipe origin is
  recorded separately as historical evidence.
- `objects`: ordered IDs, dimensions in meters, shape names, mesh/USD/cache paths
  and expected cache row counts. Order matches recipe and BPS geometry tensors.
- `assets_manifest` and its SHA256: a closed set of required input files.
- `reference_checkpoint`: path, SHA256, iteration and explicit candidate status.
- `rollout`: separate pinned config; it does not change training defaults.
- `training`: base seed, updates, ranks, physical GPUs, environment count,
  curriculum gate and small smoke profile.

Recipes remain canonical under `source/poise/sim/recipes/`. Configuration hashes
are computed from file bytes; formatting changes therefore require an intentional
hash update. `poise check-assets --deep` checks cached tensor content as well as
file identity. A setup-bound checkpoint stores recipe/PPO/manifest hashes and
ordered object IDs; a mismatched setup cannot silently consume it.

For a new registered setup, first validate the custom recipe, stage its assets,
create the manifest, choose a tested reference checkpoint and explicit rollout
settings, then compute hashes and add a new descriptor. Validate with
`show-setup`, `check-assets --deep --with-reference`, a smoke run and rollout.
Do not copy the existing setup ID onto changed inputs. The four reference setups
and checkpoint identities are pinned for release v0.1.0-rc1.
