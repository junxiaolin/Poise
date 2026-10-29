# Lightning Grasp dependency

This is the optional [lygra-poise environment](../environments/lygra-poise/README.md).
Existing [setup reproduction](reproduction.md) does not require it. The POISE
adapters and dependency metadata ship with the main repository; upstream sources
and binaries are fetched separately on demand.

POISE uses public Lightning Grasp Python sources plus a small compatibility
patch extracted from the previously tested Sharpa fork. Fetching this dependency
does not require access to the research repository or the fork. The optimizer
environment is separate from Isaac Lab.

Upstream currently distributes **precompiled CUDA extensions only**; the pinned
repository does not contain their CUDA C++ sources. There is no reproducible
CUDA source-build procedure to document. The recipe below downloads the public
Linux x86-64 Python 3.9/CUDA 12.6 binaries, checks their SHA256 and keeps the
upstream license. Other Python ABIs/platforms are not covered by this recipe.

## Install from public sources

Run from the POISE repository root. Use an NVIDIA driver compatible with CUDA
12.6; the native extensions also require Linux system libraries. On Ubuntu 22.04,
Open3D needs `libGL.so.1` and `libgomp.so.1` (packages `libgl1`, `libgomp1`).

```bash
conda create -n lygra-poise python=3.9.23 pip -y
conda activate lygra-poise
python -m pip install pip==25.2 setuptools==80.9.0 wheel==0.45.1
python -m pip install torch==2.8.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
python -m pip install -r dependencies/lightning_grasp/requirements.txt
python -m pip install --no-deps urdfpy==0.0.22

python scripts/setup_lightning_grasp.py --destination "$PWD/third_party/lightning-grasp"
python -m pip install --no-deps --no-build-isolation -e third_party/lightning-grasp
python -m pip install --no-deps --no-build-isolation -e .
python -m poise.grasp.lightning.runtime --cuda-smoke
```

The script refuses an existing destination. It stages downloads and patches in
a temporary directory and publishes the checkout only after every hash matches.
Use a new destination for an upgrade. `--kernel-archive /path/to/download.zip`
reuses an existing release archive with the same hash check. Python installation
is a separate explicit step; the script does not change active environments.

`urdfpy==0.0.22` declares an obsolete `networkx==2.2` requirement. As recommended
by LyGra upstream, this recipe uses `networkx==3.2.1`; its other dependencies are
listed explicitly before installing urdfpy with `--no-deps`. `pip check` will
report that known metadata conflict. Do not let a subsequent urdfpy installation
downgrade NetworkX. NumPy stays at 1.23.5 for urdfpy's old NumPy aliases.

The runtime check imports both native libraries and all POISE search/q_cmd
interfaces. `--cuda-smoke` also executes a tiny geometry gather, LBVH query and
wrench solve. It prints package paths, versions, Python source hashes and native
binary hashes. These checks do not require hand/object assets and do not prove
the full grasp-generation or training workflow.

Use the optimizer environment's Python path with `--optimizer-python` in
[the cache builder](grasp_generation.md). Install LyGra editable as above: the
builder replaces `PYTHONPATH` for child stages, so a shell-only LyGra path is not
sufficient. The builder now runs the same import/interface preflight and records
native hashes in its generation contract. This changes the contract; previous
build outputs remain valid data, but cannot be resumed under the changed code.

## Source audit and pins

The machine-readable [manifest](../dependencies/lightning_grasp/manifest.json)
pins upstream revision `af43818e864b0389c97b73429e5e60de2a2de593`, the patch hash,
each resulting modified source hash and the release archive hash. The patch was
extracted from Sharpa fork revision `802d1f428fda0164127c62206cae7cad18b1fecd`.
It preserves that implementation rather than changing the grasp algorithm.

| Included file(s), under `lygra/` | Required behavior |
|---|---|
| `batch_grasp_optimizer.py` | External force/torque, winning NNLS pivot, per-contact force coefficients used by q_cmd |
| `pipeline/module/contact_optimization.py` | Transform external loads into object coordinates and return wrench metadata |
| `pipeline/module/kinematics.py` | Carry wrench metadata through IK/contact-adjustment filtering |
| `pipeline/module/collision.py` | Check initial collision with Sharpa's explicit open resting pose |
| `urdf_utils.py`, `kinematics.py`, `mesh.py` | Resolve URDF resources and override joint endpoints for USDA09 |
| `pipeline/module/contact_collection.py` | Missing `sys` import on the failure path |

The fork's robot registration, `robot/sharpa_wave.py`, generic object-placement
changes and `RobotInterface.get_resting_q` addition are unnecessary for POISE's
grid entry point: POISE constructs its own Sharpa adapter and overrides the
resting-pose method. Fork demos, batch scripts and private hand/object assets are
excluded. The patch is CC BY-NC 4.0, with attribution in
[the dependency notice](../dependencies/lightning_grasp/README.md) and
[the existing license](../licenses/Lightning-Grasp.txt). No binary is vendored.

The public release archive is `v1.0.0/lygra_kernel_py39_cu126.zip`, SHA256
`ec72b919d8ff74742e8371dc2a02b26a2c3fc326c4d91b08fd2d0b55f45cdac1`.
Its extracted files match the historical locally tested binaries:

| File | SHA256 |
|---|---|
| `geometry.so` | `3bb317eac2a1c5c79cf82efb8d595611734c4e8fa98e5a432447cd8395436cbd` |
| `lbvh.so` | `4126a65ae3877c061facddb45b06152fc6e7d883c289d915dd583796f2c54a12` |

## Remaining limits

The public setup and fresh search → q_cmd → schema conversion passed in an
independent Python 3.9 venv on the existing host. The resulting cache also passed
small physics-filtering and training/export checks in the installed Isaac Lab; see
[the validation record](release/validation.md).

Public optimizer acquisition does not provide the separate USDA limit/stiffness
reference or the reference object meshes. Those asset acquisition paths and a
complete clean-machine search → filtering → training validation remain release
work. See [installation](installation.md). A local successful run using existing
assets is not evidence that a fresh clone contains all required inputs.
