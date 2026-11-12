# 环境与依赖

导出时正式服务器环境的精确包版本见 `provenance/environment.json`：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.18 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab Python distribution | 0.54.3 |
| PyTorch | 2.7.0+cu128 |
| RSL-RL | 5.0.1 |
| TensorDict | 0.8.3 |
| NumPy / SciPy | 1.26.0 / 1.15.3 |
| TensorBoard | 2.20.0 |
| 本地预览 MuJoCo / Pillow | 3.3.7 / 11.3.0 |

先准备匹配的 Isaac Sim / Isaac Lab 环境，再在它的 Python 中安装本仓库依赖。`requirements.txt` 不负责安装仿真器、显卡驱动或替换 PyTorch。Isaac Lab 和仿真器安装应以所用版本的官方安装说明为准。

```bash
python -m pip install -r requirements.txt
# 需要完整预览 / MuJoCo 几何工具时：
python -m pip install -r requirements-preview.txt
python scripts/verify_bundle.py --load-model
```

本地预览需要 Linux 图形会话、Tk 和 EGL。Tk 应由所用 Python 的环境或系统包提供。若 Isaac Python 的 Tcl/Tk 与 Ubuntu 系统库冲突，可通过 `scripts/preview.py --panel-python /path/to/python` 使用独立的面板 Python；它需要 Tk、MuJoCo、NumPy、SciPy、Pillow。

当前 Ubuntu 22.04 机器曾使用以下可选环境设置。路径存在时才设置；不要在其他系统盲目复制：

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcl8.6.so:/usr/lib/x86_64-linux-gnu/libtk8.6.so
export TCL_LIBRARY=/usr/share/tcltk/tcl8.6
export TK_LIBRARY=/usr/share/tcltk/tk8.6
```

无头训练启动器只在自身子进程设置线程和 Kit 参数，不修改全局环境、系统驱动或桌面。
