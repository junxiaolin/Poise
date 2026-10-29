# Hex40 OmniWrist FullPose v1 基准

2026-09-28 按用户要求，将 `poise-hex40-omniwrist-fullpose-4xa800-192k-gf160k-s42`
的训练配置和当时最新完整模型固定为第三个私有基准。
注册 setup：`hexprism40_omniwrist_fullpose_v1`，与固定掌心朝上的 Hex40、
九物体 diverse-object setup 使用相同的资产导入、训练和 rollout 接口。

## 固定配置

| 项目 | 值 |
| --- | --- |
| 对象 / 抓取缓存 | HexPrism40 AF40×H40 mm，6,842 行 / 486 母手势 |
| 腕姿 | 每次 reset 独立随机 RPY，episode 内固定 |
| 成功判据 | 三维位置误差 <10 mm、SO(3) 旋转误差 <10°，连续 20 个控制步 |
| 触觉 / actor / critic / action | 无 / 244 / 492 / 22 |
| GPU / rank / 环境数 | 4×A800 80GB / 8 / 8×24,000＝192,000 |
| GF / rollout | 160,000 / 8 步，1,536,000 transitions/update |
| seed | base 42，rank 42–49 |
| PPO | 150,000 次计划更新，每 100 次保存，Gloo |
| DR 关键参数 | CoM ±2 mm，critic 关节噪声 0.02，grasp reward 0.0512 |
| 参考模型 | `model_35800.pt`，已完成 35,801 次更新 |
| 已学课程上限 | 旋转约 180° / 平移约 30 mm |

模型保存在 `data/checkpoints/hexprism40_omniwrist_fullpose_v1/model_35800.pt`，SHA256：
`015cce3ad9dfdb5c76fa270ac1c328812cac6463bd7965425bc896331f2102cf`。
这是指定时最新完整模型，未按成功率挑选“最佳”。模型字节未修改，也未伪造新的训练身份；
setup 通过固定参考 SHA256 接纳原始历史 checkpoint。

`env.source.json`、`agent.json`、`topology.json` 保存实际训练记录。
`env.json` 仅将手资产路径改为仓库相对路径、log_dir 设为 null，
其余内容与实际训练配置逐字段一致，也与打包 recipe 一致。
来源 commit、源码哈希、训练规模、模型身份见 `manifest.json`；实际启动记录见
`source-launch.json`。此指定不启动新的正式训练。

## 基准入口

在已有 IsaacLab Python 环境中，从仓库根目录运行：

```bash
poise check-assets --setup hexprism40_omniwrist_fullpose_v1 --with-reference --deep

# GUI 连续 rollout，默认使用 model_35800，无需指定模型路径
poise rollout --setup hexprism40_omniwrist_fullpose_v1 \
  --steps 0 --output outputs/rollout/omniwrist_reference

# 小规模训练贯通；新的输出目录
poise train --setup hexprism40_omniwrist_fullpose_v1 \
  --profile smoke --output outputs/train/omniwrist_smoke

# 核对原训练规模的入口；仅输出命令，不启动正式训练
poise train --setup hexprism40_omniwrist_fullpose_v1 \
  --profile reference --output outputs/train/omniwrist_reference --dry-run
```

`reference` profile 为四物理 GPU、八 rank 的正式规模；只有在对应硬件上去掉
`--dry-run` 才会开始从零训练。原火山专用入口
`scripts/train_volcengine_hexprism40_fullpose_4xa800.sh` 仍保持原配方。
原 checkpoint 可用 `--resume` 恢复，但需要匹配原分布式拓扑和训练配方。

## 独立 rollout 配方与评估边界

`configs/rollout/hexprism40_omniwrist_fullpose_v1.json` 与已有基准的定性展示习惯一致：
目标幅度均匀提议 0–180° / 0–30 mm，经现有可行性筛选；关闭课程和大部分 DR/观测噪声，
成功后刷新目标。随机腕姿和严格 FullPose 判据保持开启；历史惯量 log-uniform 随机项保留。
这些覆盖只用于 rollout，不修改基准训练配置。默认 4 环境、400 控制步；`--steps 0` 无限播放。

**model_35800 尚未做定量成功率评估。** 先前 46.5% 随机腕姿成功率属于
[model_35600 的独立首目标评估](../../evaluations/2026-09-28-hex40-fullpose-model35600-wrist/README.md)，
不能作为 model_35800 的成绩，也不能与关闭大部分随机化的定性 rollout 混为一谈。

## 私有数据包

本地数据包：`outputs/private_bundles/hexprism40_omniwrist_fullpose_v1.tar.gz`，
包含固定清单内的 39 个资产文件和 1 个原始参考 checkpoint；所有文件均校验 SHA256。

```bash
poise fetch-assets --setup hexprism40_omniwrist_fullpose_v1 --with-reference \
  --source /path/to/hexprism40_omniwrist_fullpose_v1.tar.gz
```

数据、模型和压缩包不进入 Git；仓库保存可复现配置与文件清单。未上传或公开。
验证记录见 `validation.json`。
