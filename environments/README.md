# Independent environments

| Environment | Needed for | External dependencies |
|---|---|---|
| [train](train/README.md) | Existing-setup training, reference rollout, physics filtering | Isaac Sim / Isaac Lab |
| [lygra-poise](lygra-poise/README.md) | Fresh grasp search and preload commands | LyGra and its native CUDA extensions |
| [foundationpose-poise](foundationpose-poise/README.md) | RGB-D tracking and policy deployment | FoundationPose, weights, camera SDK |

Each environment installs this same POISE checkout editable with `--no-deps`.
Training never imports LyGra/FoundationPose. Grasp search never imports Isaac
Lab. Deployment never imports Isaac Lab or LyGra. The hand SDK uses a separate
container/ABI configured by the deployment template.

`manifest.json` records inspected/tested package versions; `poise check-env STAGE`
reports missing packages and version deviations without camera/robot access.
`--native` additionally checks stage imports. Version agreement does not replace
the stage's smoke test. Use `python -m poise.cli` if the console entry is absent.
Source-checkout installation is the supported reproduction path; setup manifests
and simulation launch scripts are repository resources, not a standalone wheel API.
