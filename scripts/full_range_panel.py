"""Full-range UI overlay; the formal training source remains unchanged."""
import json
from pathlib import Path
import sys
import tkinter as tk

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'go2_arm/control_panel.py'
sys.path.insert(0, str(source.parent))
code = source.read_text()
replacements = {
    "['前后 vx  (m/s)','横移 vy  (m/s)','转向 wz  (rad/s)'],[.5,.3,.4]":
    "['前后 vx  (m/s)','横移 vy  (m/s)','转向 wz  (rad/s)'],[1.,.6,.8]",
    "full=tk.BooleanVar(value=any(abs(v)>lim+1e-6 for v,lim in zip(command['base'],[.5,.3,.4])))":
    "full=tk.BooleanVar(value=True)",
    '使用完整目标范围（超出当前训练范围）':
    '完整速度范围：vx ±1.0 / vy ±0.6 / wz ±0.8',
    'send();root.after(100,update)':
    "(a.ipc/'panel_ranges.json').write_text(json.dumps({'vx':[-1.,1.], 'vy':[-.6,.6], 'wz':[-.8,.8], 'actual_widget_ranges':[[float(c.scale.cget('from')),float(c.scale.cget('to'))] for c in base_controls]}))\nsend();root.after(100,update)",
}
for before, after in replacements.items():
    assert code.count(before) == 1, before
    code = code.replace(before, after)
ipc = Path(sys.argv[sys.argv.index('--ipc') + 1])
iteration = json.loads((ipc / 'state.json').read_text())['checkpoint_iteration']
original_title = tk.Tk.title
title = f'Go2 + Piper · v11 model_{iteration} · 全速控制'
tk.Tk.title = lambda self, *args: original_title(self, title) if args else original_title(self)
exec(compile(code, str(source), 'exec'), {'__name__': '__main__', '__file__': str(source)})
