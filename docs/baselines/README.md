# Training baselines

The private reference baselines cover four setups:

| Setup | Configuration | Reference checkpoint |
| --- | --- | --- |
| `hexprism40_palmup_v1` | [Fixed palm-up Hex40](hexprism40_palmup_v0/README.md), 64k / GF20k | `model_31200.pt` |
| `manualconvex9_palmup_v1` | [Nine diverse objects](../../configs/setups/manualconvex9_palmup_v1.json), fixed palm-up, 64k / GF5k per object | `model_23300.pt` |
| `hexprism40_omniwrist_fullpose_v1` | [Random-wrist Hex40 FullPose](hexprism40_omniwrist_fullpose_v1/README.md), 192k / GF160k | `model_35800.pt` |
| `hexhammer_palmup_fullpose_192k_v1` | [Fixed palm-up HexHammer FullPose, no tactile](hexhammer_palmup_fullpose_192k_v1/README.md), 192k / GF20k | `model_89800.pt` |

Use each setup's recorded configuration and checkpoint for comparisons in its
task. Baseline designation does not restart training or publish assets.

Keep each version's snapshots unchanged. Record any changed settings and use
a new version when replacing the baseline; historical research policies remain
comparison references.
