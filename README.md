# Go2 + Piper · RoboDuet PPO / Directional AMP · v11

Unitree Go2 搭载前移 Piper、开口盒和双 D435i 的自由底盘训练项目。包含当前 **v11 静止落脚与承重修复版**的完整训练源码、机器人网格、URDF / MJCF / USD、轨迹库，以及 17 条上游 AMP 动作和 40 条筛选后的补录动作。克隆后无需从原工作目录复制机器人资产。

当前配方：双策略 PPO、方向条件 AMP、无固定步态相位；阶段一保持机械臂折叠，学习底盘行走与停车。所有载荷、自碰撞和非足端接触终止均保留，PhysX 模型总质量约 **19.842947 kg**。

> 这是训练代码与资产快照，当前有界试训属于 stage1。阶段二整机精度、实际 P95、静止峰峰值和模式切换仍未完成验收。阶段一检查点不能作为机械臂双模式策略已训练完成的依据。

## 快速开始

在 Python 3.11、Isaac Sim 5.1 / Isaac Lab 环境中安装依赖。实际使用版本和安装边界见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。

```bash
git clone https://github.com/losehu/go2-arm.git
cd go2-arm
python -m pip install -r requirements-preview.txt
python scripts/verify_bundle.py --load-model
```

小规模启动检查：8 环境、4 更新，检查模型构建、PPO / AMP 更新及检查点重载。它只用于检查运行链路。

```bash
python scripts/train.py --smoke --device cpu --num-envs 8 --updates 4 \
  --run-dir runs/smoke_v11
```

复现当前 **4096 环境、从零、20000 更新**有界试训：

```bash
python scripts/train.py --device cuda:0 --num-envs 4096 --updates 20000 \
  --run-dir runs/v11_scratch_4096
```

启动器在同一台机器上运行训练及独立 CPU 探针，记录退出码、进程身份和日志。训练结束核对 `model_19999.pt` 后，自动执行完整固定指令与两个种子的停车过渡测试。默认保持前台运行，可在自己的终端会话管理器中启动。

```bash
python go2_arm/training_status.py runs/v11_scratch_4096
tensorboard --logdir runs/v11_scratch_4096
```

完整两阶段预算是 **40000 + 40000**；它与当前 20000 更新试训不同。若明确需要完整配方，可在启动器中指定 `--updates 80000`。阶段切换仍由原课程门槛控制，不能保证预算内一定进入或完成阶段二。恢复训练 CLI 当前主动拒绝执行，勿把新训练称为续训。

## 查看权重

将同版本检查点放入本地 `checkpoints/`；权重和运行日志不进入 Git 历史。

```bash
python scripts/preview.py --checkpoint checkpoints/model_3000.pt
```

面板默认开放 `vx ±1.0 m/s`、`vy ±0.6 m/s`、`wz ±0.8 rad/s`，启动后为零指令，可直接查看静止落脚。动力学和策略由 PhysX 执行，MuJoCo 只渲染实际状态与双相机画面。碰撞或跌倒后暂停，需手动复位。机械臂目标需要阶段二检查点。

## 独立测试

```bash
# 全范围固定指令：45 组 × 均值 / 采样策略
python tools/direction_probe.py --package go2_arm --headless --device cpu \
  --checkpoint checkpoints/model_3000.pt --output runs/direction_probe

# 18 项：move 0–6s → stop 6–14s → restart 14–18s → stop 18–24s
python tools/stop_transition_probe.py --package go2_arm --headless --device cpu \
  --checkpoint checkpoints/model_3000.pt --output runs/standing_seed61119 --seed 61119
python tools/stop_transition_probe.py --package go2_arm --headless --device cpu \
  --checkpoint checkpoints/model_3000.pt --output runs/standing_seed61120 --seed 61120

# 按实际时间重算连续满 0.5s 的四脚接触，保留原始结果
python scripts/analyze_settle.py --probe-dir runs/standing_seed61119
```

探针输出目录必须是新目录。已有 `result.json` 的监督任务会跳过；不完整输出需要先查日志，启动器不会覆盖它重新运行。

## v11 配方要点

- 动作幅度 hip / thigh / calf = `0.35 / 0.55 / 0.65 rad`，腿滤波 `alpha=0.35`。
- 速度范围始终 `±1 / ±0.6 / ±0.8`。前 1000 更新以专家帧指令为主，随后 8000 更新连续增加全向比例至 80%，保留 20% 专家；固定扰动 `±0.15 / ±0.10 / ±0.15`，每 5 秒重采样。
- 通过六方向高速门槛、全向比例和成功率条件后，内部姿态范围连续从 3° 扩至 15°。
- 内收、站距和合格完整周期步频成本作用于正向 shaping 折扣；物理失败在 shaping 下限之后单次 `-5`，普通超时不罚。
- 三轴速度指令全零时，静止项经过 0.25s 宽限和 0.50s 淡入，约束足底离地、近地法向力及每脚最低载荷份额。起步立即解除，不要求四脚均分载荷。
- 折叠臂默认 `[0, 40, -3, 0, -32, 0]°`；完整机器人与碰撞保护保留。

奖励阈值是训练设计；不能替代最终用户验收。详细边界与已有结果见 [docs/TRAINING.md](docs/TRAINING.md)。

## 目录

| 路径 | 内容 |
| --- | --- |
| `go2_arm/train_duet.py` | 原版双策略 PPO / AMP 训练入口 |
| `go2_arm/duet_env.py`, `base_env.py` | 观测、动作、奖励、机器人与物理终止 |
| `go2_arm/standing_support.py`, `stance_cadence.py` | 静止承重、内收、站距与完整周期约束 |
| `go2_arm/robot/` | 完整网格、URDF、MJCF、USD 及资产条款 |
| `go2_arm/amp_reference/` | 57 条参考动作与来源 / 筛选清单 |
| `scripts/train.py`, `watch_probes.py` | 可移植的监督启动与计划测试 |
| `scripts/preview.py`, `full_range_panel.py` | 可移植的全速实时预览 |
| `tools/` | 独立探针、录制、参考覆盖审计与资产工具 |
| `asset_manifest.json` | 随仓库提供的数据文件 SHA-256 |
| `provenance/` | 正式训练源码 / 资产哈希与环境快照 |

训练 Python 和物理资产与导出时正式服务器版本逐字节一致；仅整理了路径元数据并新增可移植启动包装。`python scripts/verify_bundle.py` 会核对这些哈希。

## 资产来源与发布

第三方来源和原始条款见 [THIRD_PARTY.md](THIRD_PARTY.md) 与 `go2_arm/robot/licenses/`。相机支架来源中含明确标为 **Proprietary** 的资产；本快照没有为它们授予新的开源许可，也没有为整个混合包指定统一许可证。

上传已有空仓库的命令见 [docs/GITHUB.md](docs/GITHUB.md)。运行数据、权重、机器连接信息和 SSH 凭据不属于此源码仓库。
