# Go2 + Piper · AMP 行走训练与实时仿真

Unitree Go2 搭载前移 Piper 机械臂、开口盒和双 D435i 的自由底盘仿真。代码包含 Isaac Lab 训练、加载检查点后的 PhysX 推理，以及带右侧指令面板的 Tk/MuJoCo 可视化。

当前配方是第一阶段无相位 AMP v4。PPO 优化速度跟踪、姿态稳定、触地防滑、髋部外摆约束和 AMP 风格奖励。底盘策略观测为 69 维。新配方按指令方向筛选参考动作，再按速度加权采样。匹配的归一化速度距离上限为 0.30。80% 的组合目标优先匹配同方向的已验证速度，另保留原随机组合。判别器使用跨 0.20 秒的三个状态，比较关节运动、根链接原点速度、机身高度、相对地面脚高和脚速。只有有参考覆盖且指令稳定的历史窗口参与 AMP。触地时根据实际腾空和抬脚幅度给有上限的奖励。PhysX 中足底球和小腿各有独立的接触信号，小腿碰撞会终止回合。机身低于 28.5 cm 时施加单边软惩罚。阶段一保持折叠臂姿态 `[0, 40, -3, 0, -32, 0]°`。阶段一达到全速度范围、15° 内部姿态指令范围且成功率 EMA ≥ 0.7 后，完整训练才进入底盘与机械臂联合优化。

## 环境

测试环境为 Linux、Python 3.11、Isaac Sim 5.1.0、Isaac Lab 0.54.3（提交 `47c9f95eeab90dc3611981d894d59f163191f5b1`）、PyTorch 2.7.0+cu128 与 RSL-RL 5.0.1。训练使用 NVIDIA GPU，CPU 可用于小规模检查和本地推理。预览面板需要图形会话、Tk 和 EGL。先按 Isaac Lab 文档安装仿真环境，再在同一个 Python 环境中安装项目依赖。 [Isaac Lab 安装文档](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation/index.html)

```bash
git clone https://github.com/losehu/go2-arm.git
cd go2-arm
python -m pip install -r requirements.txt
```

## 准备模型资产

需要已有的 `box_mount_rails_v9` 模型包，目录中包含 `robot/`、`path_bank.npz`、`whole_body_bank.npz`、`head_vertices.npy` 和 `controller_bench_report.json`。脚本核对 SHA-256 后把这些运行文件导入 `go2_arm/`。参考数据使用 17 条固定版本的 `amp_go2` 动作片段，优先从已有包读取，缺少时按固定提交下载。仅推理时可以加 `--skip-reference`。

```bash
python tools/prepare_assets.py --bundle /path/to/go2_piper_roboduet_amp
python tools/split_foot_collisions.py --usd go2_arm/robot/usd/robot.usd --headless --device cpu
```

当前服务器配方使用 40 条自由底盘 PhysX 补录和 17 条上游参考，覆盖全部 27 种方向符号组合。用冻结的旧策略录制，右侧以镜像策略在物理仿真中独立重跑。部分补录用实际离地触发的教师抬脚辅助。筛选要求每只脚至少 3 次完整摆动、摆动峰值中位数至少 1.5 cm，且速度跟踪、对角支撑和防滑通过。最慢通过筛选的补录速度为 ±0.16 m/s，可用于匹配 ±0.12 m/s 指令。补录记录包含教师、参数和文件哈希。数据在本地准备，训练可读取原模型包中的补录目录，或通过 `--recordings` 导入：

```bash
python tools/record_amp_clips.py --package go2_arm --headless --device cpu \
  --checkpoint /path/to/phase_free_teacher.pt --output runs/recorded
python tools/record_amp_clips.py --package go2_arm --headless --device cpu \
  --checkpoint /path/to/phase_free_teacher.pt --output runs/recorded_slow --suite slow \
  --lift-assist-mm 35 --assist-paired --front-lift-scale 1.35
python tools/merge_amp_recordings.py --output runs/recorded_merged runs/recorded runs/recorded_slow
python tools/prepare_assets.py --bundle /path/to/go2_piper_roboduet_amp --recordings runs/recorded_merged
python tools/audit_amp_coverage.py
```

生成的 USD 层将原足底球移到独立刚体，小腿与足底合计的质量、质心和惯量保持一致。URDF 和 MuJoCo 视图资产仍来自原模型包。运行以下命令可分别验证脚底触地、小腿触地和同时触地。

```bash
python tools/verify_foot_contacts.py --package go2_arm --headless --device cpu
```

模型包包含有独立使用条款的相机支架。模型网格、生成的 USD、轨迹库、参考动作和训练权重由使用者在本地准备。具体版本与散列见 [asset_manifest.json](asset_manifest.json)，资产来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 从零训练

先跑小规模短测，确认 AMP 判别器和 PPO 都能更新、保存并重载权重。

```bash
python go2_arm/train_duet.py --headless --device cpu --num_envs 8 \
  --amp --phase_free_amp --directional_amp --hip_splay_penalty --smoke \
  --stop_after_updates 4 --run_dir runs/smoke
```

复现当前第一阶段的 10,000 次无相位 AMP 试验。阶段一从 50% 速度范围起步，每 1,000 次保存，并在试验结束时保存 `model_9999.pt`。

```bash
python go2_arm/train_duet.py --headless --device cuda:0 --num_envs 4096 \
  --amp --phase_free_amp --directional_amp --hip_splay_penalty \
  --stop_after_updates 10000 --run_dir runs/amp_phase_free
python go2_arm/training_status.py runs/amp_phase_free
```

完整双阶段训练省略 `--stop_after_updates`。默认阶段一和阶段二各有 40,000 次更新的预算，第二阶段由课程门槛控制。试验完成与整机验收是两个独立结果。

## 本地交互推理

用同一模型包生成的双策略 `.pt` 文件启动窗口。右侧 `vx / vy / wz` 滑块控制底盘，单位分别为 m/s、m/s、rad/s。三个分量分别在绝对值不超过 0.10 时清零。训练、推理和 AMP 方向分类使用同一死区。AMP 检查 0.20 秒窗口的每帧，跨方向类别或速度变化过大的窗口不参与风格匹配。启动时暂停，点「继续仿真」后调整速度。阶段一检查点用于行走，机械臂控制需要第二阶段检查点。

```bash
python go2_arm/launch_live_preview.py --checkpoint /path/to/model_4000.pt
```

Isaac/PhysX 计算动力学和策略动作。MuJoCo 根据实时状态渲染外部视角与双相机画面。启动器将状态和日志存到 `runs/previews/`。如果 Isaac 环境的 Tk 与系统 Tcl 冲突，可以通过 `--panel-python` 指向带 Tk、NumPy、MuJoCo 和 Pillow 的独立 Python 环境。Ubuntu 22.04 系统 Tcl/Tk 8.6 兼容配置如下。

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcl8.6.so:/usr/lib/x86_64-linux-gnu/libtk8.6.so
export TCL_LIBRARY=/usr/share/tcltk/tcl8.6
export TK_LIBRARY=/usr/share/tcltk/tk8.6
```

固定速度诊断包含前后、左右、原地转向、斜移和带转向组合，并测试确定性动作与采样动作。按每只脚的实际离地与触地统计摆动次数、峰值、持续时间和拖脚比例，另记录支撑脚滑动、对角支撑和低机身时间。用重复的 `--case` 参数选择单独工况。

```bash
python tools/direction_probe.py --package go2_arm --headless --device cpu \
  --checkpoint /path/to/model_4000.pt --output runs/direction_probe
```

## 代码入口

| 文件 | 用途 |
| --- | --- |
| `go2_arm/train_duet.py` | 双策略 PPO、阶段课程、AMP 奖励及检查点 |
| `go2_arm/duet_env.py` | 自由底盘、观测、奖励和碰撞终止 |
| `go2_arm/stage1_amp.py` | 基础 AMP 判别器和参考动作采样 |
| `go2_arm/directional_amp.py` | 方向匹配的参考采样和实测运动历史 |
| `go2_arm/air_time.py` | 基于实际腾空和触地的奖励 |
| `go2_arm/live_controller.py` | 加载策略、执行指令并发布状态 |
| `go2_arm/control_panel.py` | 速度滑块、机械臂控制、实时相机和外部视角 |
