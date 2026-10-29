# Sharpa Wave closed-loop deployment

`scripts/deploy.py` connects RealSense RGB-D, AprilTag hand-camera calibration,
FoundationPose object tracking, the 244D policy and an independently guarded SDK
subprocess. This new integration has only offline tests; robot execution and
camera throughput have not yet been validated in this repository.

Install the independent [foundationpose-poise environment](../environments/foundationpose-poise/README.md)
with POISE, Torch, OpenCV with ArUco, RealSense and pinned FoundationPose
sources/weights/native dependencies. Its adapters are already in the main POISE
checkout; optional upstream sources are fetched only for this path. Training
and rollout do not require this environment. Export a per-object bundle with
`poise rollout --setup <id> --output <new-directory>` in Isaac Lab first. The SDK helper currently runs
through Docker in an explicitly named container. Its SDK path and Python
executable are configurable; their defaults are `/opt/sharpa-wave-sdk` and
`python3`. The container's Python must match the vendor extension's ABI (the
locally used SDK requires Python 3.10). Container provisioning remains release work.

## Installation-specific configuration

Generate a template without connecting to any device:

```bash
poise init-deploy outputs/deploy/robot.json
```

Fill the `null` installation fields using your measured setup. The template
intentionally does not invent a tag mounting transform or hardware joint limits.
The command refuses to overwrite an existing configuration.

| Field | Meaning |
|---|---|
| `foundationpose_root`, `mesh` | Paths to FoundationPose and a metric tracking mesh |
| `registration_mask` | Grayscale object mask aligned to the current 1280×720 registration view |
| `sdk_container` | Running container providing `python3` and the Sharpa SDK |
| `sdk_root`, `sdk_python` | SDK directory **inside the container**, containing `python/` and `lib/`; Python executable inside that container |
| `tag_id`, `tag_size_m`, `t_hand_tag` | Tag identity, printed size and measured 4×4 hand-from-tag transform |
| `t_mesh_object` | 4×4 mesh-from-simulator-object transform |
| `t_hand_target` | Desired 4×4 object pose in the hand frame |
| `gravity_direction_palm` | Unit gravity vector in the fixed hand frame; palm-up uses `[-1,0,0]` |
| `hard_joint_lower_rad`, `hard_joint_upper_rad` | 22 SDK-order hardware limits from your installation |
| `camera_serial`, `hand_serial` | Optional explicit device serials |
| `speed_coeff`, `current_coeff` | Optional arming coefficients, at most 0.30 |

Validate local inputs before starting the preview:

```bash
poise check-deploy --bundle /path/to/bundle --config outputs/deploy/robot.json
```

This checks required fields, rigid transforms, finite 22D hardware limits, unit
gravity, policy/initial-grasp hashes, a single-frame policy/command inference,
the 20 Hz control contract, target workspace
and SDF, mesh dimensions in the simulator object frame, and the 1280×720 mask.
Mesh size tolerance is 1 mm plus 2% per axis. Unknown configuration fields are
rejected to catch typos. The check does not start Docker, discover devices, import
FoundationPose, or load its CUDA models. It cannot establish mask alignment to a
future camera frame, native dependency compatibility, or calibration accuracy.
Live deployment repeats this same check and saves `preflight.json`.

Use the vendor's SDK installation procedure inside your container. Configure
`sdk_root` to that installation instead of moving files to a lab-specific path;
the launcher sets `SHARPA_SDK_ROOT` and `LD_LIBRARY_PATH` and imports `sharpa`
from its `python/` subdirectory. Host paths and container paths are separate.

Relative file paths resolve from the JSON directory. Matrices use `T_A_B` to map
coordinates in B into A. The observation pose is
`T_hand_camera @ T_camera_mesh @ T_mesh_object`. Do not reuse the research lab's
camera/tag transform or widen hardware limits to fit a policy. This controller
assumes a fixed hand root during a session; random-wrist simulation is supported
separately. Tracking symmetric untextured objects requires resolving the visual
pose convention for the physical object; that setup has not been validated here.

The bundle contains the actor normalizer, BPS descriptor, simulator joint limits,
initial grasp and palm SDF. Deployment checks target workspace and compound
sphere/SDF collision before starting the camera. Initial grasp q and q_cmd are
verified against the bundle hash.

## Preview and operation

```bash
python scripts/deploy.py --bundle /path/to/bundle --config /path/to/robot.json \
  --output outputs/deploy/preview
```

Preview reads camera and joint state but does not arm the hand. Keep the object
stationary during calibration and registration so the supplied mask remains
aligned. For an attended
hardware session, add `--execute`. Startup collects 100 calibration frames and
registers the object from the supplied mask. Keyboard controls:

- `a`: arm and move gradually toward pregrasp.
- Place the object at the displayed initial-grasp pose, then `c`: close.
- Once q_cmd reaches the cache command and pose is within 10 mm/15°, `p`: run policy.
- Space: HOLD; `q`/Escape: stop. Restart after HOLD or a completed policy trial.

The default trial is 200 commands at 20 Hz. Pose/joint freshness, pose jumps,
command acknowledgments, limits and SDK heartbeat are checked. A missed control
deadline is checked before issuing the next policy command. The helper
independently enforces policy joint bounds and command count; failures latch
stop. The preview uses a simple status overlay; paper experiment UI and recovery
protocols are excluded. Recordings contain state/command data in `control.jsonl`.
On failure, the host requests robot stop and closes the SDK helper before joining
the capture worker and releasing the camera. A disconnected SDK or a camera
cleanup failure does not replace the original control exception in the log.

Real hardware remains unverified. A passing offline test or simulator replay does
not mean a pilot checkpoint is ready for robot operation.
