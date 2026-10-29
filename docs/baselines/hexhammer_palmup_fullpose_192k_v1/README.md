# HexHammer PalmUp FullPose 192k v1 基准

2026-10-09 将 `poise-hexhammer-palmup-fullpose-notactile-4xa800-192k-gf20k-s42`
指定为私有参考基准，setup 为 `hexhammer_palmup_fullpose_192k_v1`。
固定当天已回传并验证的 `model_89800.pt`，后续训练产生的 checkpoint 不会自动替换它。

## 固定训练配置

| 项目 | 值 |
| --- | --- |
| 物体 / 抓取缓存 | HexHammer / 11,630 行 |
| 手 / 腕姿 | Sharpa Wave 右手，固定 palm-up |
| 成功判据 | 三维位置误差 <10 mm、SO(3) 旋转误差 <10°，连续 20 控制步 |
| 触觉 / actor / critic / action | 无 / 244 / 492 / 22 |
| GPU / rank / 环境数 | 4×A800 80GB / 8 / 8×24,000＝192,000 |
| GF / rollout | 20,000 / 8 步，1,536,000 transitions/update |
| seed / backend | base 42，rank 42–49 / Gloo |
| 计划更新 / 保存间隔 | 从零训练 150,000 次 / 每 100 次 |
| 关键参数 | CoM ±5 mm，critic 关节噪声 0.02，grasp reward 0.0512 |
| 参考 checkpoint | `model_89800.pt`，已完成 89,801 次更新 |
| 保存时课程上限 | 旋转约 180° / 平移约 100 mm |

checkpoint 路径：`data/checkpoints/hexhammer_palmup_fullpose_192k_v1/model_89800.pt`。
SHA-256：`5c6e3674b055574c8a75e5ac800d3534b29fa39ac3d9dbb7a5640dbb7e7180c2`。
模型字节保持原样，setup 通过固定哈希识别原始训练 checkpoint。

`env.source.json`、`agent.json`、`topology.json` 是实际训练记录；`source-launch.json`
保存真实启动命令、软件版本及源码哈希。`env.json` 仅将手资产改为仓库相对路径、
清空 log_dir，其余字段与训练记录一致。对应的新版本 recipe 保留每 rank 24,000 环境，
原有 64k 和火山任务配方保持原样。来源与模型身份见 `manifest.json`。

## 复现入口

在配置好的 IsaacLab Python 环境内，从 Poise 仓库根目录执行：

```bash
poise check-assets --setup hexhammer_palmup_fullpose_192k_v1 --with-reference --deep

# GUI 参考 rollout；自动选择固定的 model_89800
poise rollout --setup hexhammer_palmup_fullpose_192k_v1 \
  --steps 0 --output outputs/rollout/hexhammer_192k_reference

# 小规模训练贯通
poise train --setup hexhammer_palmup_fullpose_192k_v1 \
  --profile smoke --output outputs/train/hexhammer_192k_smoke

# 查看原规模训练命令；去掉 --dry-run 才会在对应硬件上启动训练
poise train --setup hexhammer_palmup_fullpose_192k_v1 \
  --profile reference --output outputs/train/hexhammer_192k_reference --dry-run
```

输出目录须未存在。reference profile 使用四物理 GPU、八 rank、192k 环境；
smoke 为单卡 8 环境、2 次更新。此次基准登记不启动新的正式训练。

## Rollout 和评估边界

单独固定的 `configs/rollout/hexhammer_palmup_fullpose_192k_v1.json` 从 checkpoint
保存的 180° / 100 mm 课程上限开始，保留训练时的目标采样、可行性筛选、成功刷新、
观测噪声和域随机化，与 10 月 9 日回传模型的 rollout 设置一致。
推理使用均值动作，默认 4 环境、400 步；`--steps 0` 持续播放。
这些覆盖只应用于 rollout，从零训练仍从原始初级课程开始。

满课程表示课程范围达到上限，不代表 100% reaching 成功率。
本 checkpoint 尚无独立成功率评估；短 rollout 验证只用于检查加载与仿真运行。

## 私有资产包

`outputs/private_bundles/hexhammer_palmup_fullpose_192k_v1.tar.gz` 包含 41 项资产和
1 个固定 checkpoint，可导入其他本地数据目录：

```bash
poise fetch-assets --setup hexhammer_palmup_fullpose_192k_v1 --with-reference \
  --source /path/to/hexhammer_palmup_fullpose_192k_v1.tar.gz
```

资产、模型和压缩包保持私有，均不进入 Git。配置、清单与验证结果保存在仓库；
验证详情见 `validation.json`。
