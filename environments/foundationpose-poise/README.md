# Optional perception and deployment

POISE's tracker/control adapters ship in the main repository. FoundationPose
sources, native extensions and weights are installed only for deployment.
This environment uses Python 3.11 and its own CUDA Torch; it is independent of
the simulator. The inspected stack is Torch 2.6.0+cu124, NumPy 2.4.4,
OpenCV 5.0.0.93 (with ArUco), PyTorch3D 0.7.9 and NVDiffRast 0.4.0.

```bash
# Fetch source only; does not install Python packages or download weights.
python scripts/setup_foundationpose.py --destination third_party/FoundationPose
conda env create -n foundationpose-poise -f third_party/FoundationPose/environment.yml
conda activate foundationpose-poise
python -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --extra-index-url https://download.pytorch.org/whl/cu124
# Configure CUDA_HOME/PATH for your matching nvcc toolkit before native builds.
python -m pip install --no-build-isolation ./third_party/FoundationPose/poise-deps/pytorch3d
python -m pip install --no-build-isolation ./third_party/FoundationPose/poise-deps/nvdiffrast
python -m pip install -r third_party/FoundationPose/requirements.txt \
  -c environments/foundationpose-poise/requirements.txt
python -m pip install -r environments/foundationpose-poise/requirements.txt
bash third_party/FoundationPose/build_all_conda.sh
python -m pip install --no-deps -e .
```

Download the two pretrained weight directories from the upstream link recorded
in [the dependency manifest](../../dependencies/foundationpose/manifest.json)
into `third_party/FoundationPose/weights/`, preserving its directory structure.
Their config/model hashes are pinned. No weights are bundled or uploaded here.

```bash
poise check-env foundationpose-poise --native \
  --foundationpose-root third_party/FoundationPose
poise init-deploy outputs/deploy/robot.json
```

Fill measured transforms, hardware limits, mesh and device settings, then follow
[deployment](../../docs/deployment.md). The tracker is currently in the same
process as POISE control; the vendor hand SDK remains a separate container.
Do not install `.[deploy]` into this environment: that generic older OpenCV extra
targets another stack. Use the explicit environment requirements and `--no-deps`
so POISE does not replace FoundationPose's dependencies. The shared package now
accepts Torch 2.6; training and grasp still pin their respective newer versions.

The manifest records the inspected local stack. A full independent native build,
tracking throughput and physical robot validation remain separate work; source
fetch and dependency import checks do not establish hardware readiness.
