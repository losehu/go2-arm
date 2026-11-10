# Go2 + Piper · AMP 行走训练与实时仿真

Unitree Go2 搭载前移 Piper 机械臂、开口盒和双 D435i 的自由底盘仿真。代码包含 Isaac Lab 训练、加载检查点后的 PhysX 推理，以及带右侧指令面板的 Tk/MuJoCo 可视化。

当前配方是第一阶段无相位 AMP。PPO 优化速度跟踪、姿态稳定、触地防滑、髋部外摆约束和 AMP 风格奖励。底盘策略观测为 69 维，风格判别器比较参考动作和仿真动作的相邻状态。阶段一保持折叠臂姿态 `[0, 40, -3, 0, -32, 0]°`。阶段一达到全速度范围、15° 内部姿态指令范围且成功率 EMA ≥ 0.7 后，完整训练才进入底盘与机械臂联合优化。

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
```

模型包包含有独立使用条款的相机支架。模型网格、生成的 USD、轨迹库、参考动作和训练权重由使用者在本地准备。具体版本与散列见 [asset_manifest.json](asset_manifest.json)，资产来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 从零训练

先跑小规模短测，确认 AMP 判别器和 PPO 都能更新、保存并重载权重。

```bash
python go2_arm/train_duet.py --headless --device cpu --num_envs 8 \
  --amp --phase_free_amp --hip_splay_penalty --smoke \
  --stop_after_updates 4 --run_dir runs/smoke
```

复现当前第一阶段的 10,000 次无相位 AMP 试验。阶段一从 50% 速度范围起步，每 1,000 次保存，并在试验结束时保存 `model_9999.pt`。

```bash
python go2_arm/train_duet.py --headless --device cuda:0 --num_envs 4096 \
  --amp --phase_free_amp --hip_splay_penalty \
  --stop_after_updates 10000 --run_dir runs/amp_phase_free
python go2_arm/training_status.py runs/amp_phase_free
```

完整双阶段训练省略 `--stop_after_updates`。默认阶段一和阶段二各有 40,000 次更新的预算，第二阶段由课程门槛控制。试验完成与整机验收是两个独立结果。

## 本地交互推理

用同一模型包生成的双策略 `.pt` 文件启动窗口。右侧 `vx / vy / wz` 滑块控制底盘，单位分别为 m/s、m/s、rad/s。启动时暂停，点「继续仿真」后调整速度。阶段一检查点用于行走，机械臂控制需要第二阶段检查点。

```bash
python go2_arm/launch_live_preview.py --checkpoint /path/to/model_4000.pt
```

Isaac/PhysX 计算动力学和策略动作。MuJoCo 根据实时状态渲染外部视角与双相机画面。启动器将状态和日志存到 `runs/previews/`。如果 Isaac 环境的 Tk 与系统 Tcl 冲突，可以通过 `--panel-python` 指向带 Tk、NumPy、MuJoCo 和 Pillow 的独立 Python 环境。Ubuntu 22.04 系统 Tcl/Tk 8.6 兼容配置如下。

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcl8.6.so:/usr/lib/x86_64-linux-gnu/libtk8.6.so
export TCL_LIBRARY=/usr/share/tcltk/tcl8.6
export TK_LIBRARY=/usr/share/tcltk/tk8.6
```

固定速度诊断可记录速度误差、支撑脚滑动、对角支撑比例和抬脚高度。默认命令集包含前后、左右、正反转向和站立，测试确定性动作与采样动作。

```bash
python tools/direction_probe.py --package go2_arm --headless --device cpu \
  --checkpoint /path/to/model_4000.pt --output runs/direction_probe
```

## 代码入口

| 文件 | 用途 |
| --- | --- |
| `go2_arm/train_duet.py` | 双策略 PPO、阶段课程、AMP 奖励及检查点 |
| `go2_arm/duet_env.py` | 自由底盘、观测、奖励和碰撞终止 |
| `go2_arm/stage1_amp.py` | 速度条件判别器和参考动作采样 |
| `go2_arm/live_controller.py` | 加载策略、执行指令并发布状态 |
| `go2_arm/control_panel.py` | 速度滑块、机械臂控制、实时相机和外部视角 |
