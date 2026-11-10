"""Single-window live robot view, RGB views and all ten public controls."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
import json
import math
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk
import mujoco
import numpy as np
from PIL import Image,ImageTk

R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--ipc',type=Path,required=True);p.add_argument('--self_test',action='store_true');a=p.parse_args()
a.ipc.mkdir(parents=True,exist_ok=True)
from robot_constants import PREVIEW_DEFAULT_Q as DEFAULT_Q,JOINT_NAMES

root=tk.Tk()
if a.self_test:root.withdraw()
root.title('Go2 + Piper · RoboDuet 双策略 · 实时控制')
root.geometry('1520x950');root.minsize(1320,850);root.configure(bg='#101926')
style=ttk.Style();style.theme_use('clam')
style.configure('.',background='#182436',foreground='#e6edf7',font=('Noto Sans CJK SC',10))
style.configure('TFrame',background='#182436')
style.configure('TLabel',background='#182436',foreground='#dce8f5')
style.configure('TButton',padding=5,background='#29445e',foreground='white')
style.configure('TCheckbutton',background='#182436',foreground='#dce8f5')
style.configure('TRadiobutton',background='#182436',foreground='#dce8f5')
style.configure('TEntry',fieldbackground='#eaf0f7',foreground='#142435')
style.configure('TScale',background='#182436',troughcolor='#31465e')
style.configure('Header.TLabel',font=('Noto Sans CJK SC',15,'bold'),foreground='white')
style.configure('Hint.TLabel',foreground='#f7c77b',font=('Noto Sans CJK SC',9))
root.columnconfigure(0,weight=1);root.columnconfigure(1,weight=0);root.rowconfigure(0,weight=1)
left=ttk.Frame(root,padding=10);left.grid(row=0,column=0,sticky='nsew')
panel=ttk.Frame(root,padding=14,width=450);panel.grid(row=0,column=1,sticky='nsew');panel.grid_propagate(False)
left.columnconfigure(0,weight=1);left.rowconfigure(2,weight=1)
ttk.Label(left,text='实时策略控制 · 自由底盘 · 前置机械臂与开口盒',style='Header.TLabel').grid(row=0,column=0,sticky='w')
live_label=ttk.Label(left,text='正在连接本地控制器…',style='Hint.TLabel');live_label.grid(row=1,column=0,sticky='w',pady=(5,8))
view=tk.Label(left,bg='#101926');view.grid(row=2,column=0,sticky='nsew')
sensors=ttk.Frame(left);sensors.grid(row=3,column=0,sticky='ew',pady=8)
sensors.columnconfigure(0,weight=1);sensors.columnconfigure(1,weight=1)
rgb=[]
for j,title in enumerate(['Go2 头部 D435i · 仿真 RGB','Piper 腕部 D435i · 仿真 RGB']):
 f=ttk.Frame(sensors);f.grid(row=0,column=j,sticky='nsew',padx=3)
 ttk.Label(f,text=title).pack(anchor='w');lab=tk.Label(f,bg='#101926');lab.pack(fill='both',expand=True);rgb.append(lab)
bottom=ttk.Label(left,text='鼠标拖动旋转视角 · 滚轮缩放。右侧面板发送实时指令，非轨迹回放。')
bottom.grid(row=4,column=0,sticky='w')

command={'seq':0,'base':[0.,0.,0.],'grip':.5,'paused':False,'arm_mode':'joint_position',
         'arm_target':DEFAULT_Q[19:25].tolist(),'arm_revision':0,'reset_revision':0,'shutdown':False}
try:
 saved=json.loads((a.ipc/'command.json').read_text())
 if not saved.get('shutdown'):command.update(saved)
except (OSError,ValueError):pass
try:last=json.loads((a.ipc/'state.json').read_text())
except (OSError,ValueError):last={}
pending=None;loading=False;closed=False

def send():
 global pending
 pending=None;command['seq']+=1
 tmp=a.ipc/'command.tmp';tmp.write_text(json.dumps(command));tmp.replace(a.ipc/'command.json')

def debounce():
 global pending
 if loading:return
 if pending:root.after_cancel(pending)
 pending=root.after(120,send)

class Control:
 def __init__(self,parent,label,lo,hi,value,callback,row):
  self.value=tk.DoubleVar(value=value);self.callback=callback
  f=ttk.Frame(parent);f.grid(row=row,column=0,sticky='ew',pady=2);f.columnconfigure(1,weight=1)
  self.label=ttk.Label(f,text=label,width=15);self.label.grid(row=0,column=0,sticky='w')
  self.entry=ttk.Entry(f,textvariable=self.value,width=9,justify='right');self.entry.grid(row=0,column=2)
  self.scale=ttk.Scale(f,from_=lo,to=hi,variable=self.value,command=self.changed);self.scale.grid(row=1,column=0,columnspan=3,sticky='ew')
  self.entry.bind('<Return>',self.enter);self.entry.bind('<FocusOut>',self.enter)
 def changed(self,*args):
  self.value.set(round(self.value.get(),4))
  self.callback()
 def enter(self,*args):
  try:self.value.set(round(float(np.clip(self.value.get(),float(self.scale.cget('from')),float(self.scale.cget('to')))),4));self.callback()
  except (ValueError,tk.TclError):self.value.set(0.)
 def configure(self,label,lo,hi,value):
  self.label.configure(text=label);self.scale.configure(from_=lo,to=hi);self.value.set(round(float(value),4))

panel.columnconfigure(0,weight=1)
ttk.Label(panel,text='控制面板',style='Header.TLabel').grid(row=0,column=0,sticky='w')
checkpoint_label=ttk.Label(panel,text='正在读取双策略权重…',style='Hint.TLabel');checkpoint_label.grid(row=1,column=0,sticky='w',pady=(2,8))
ttk.Label(panel,text='底盘指令  ·  拖动后即时发送',font=('Noto Sans CJK SC',11,'bold')).grid(row=2,column=0,sticky='w')
base_frame=ttk.Frame(panel);base_frame.grid(row=3,column=0,sticky='ew');base_frame.columnconfigure(0,weight=1)
base_controls=[]
def base_changed():
 if loading:return
 try:
  command['base']=[c.value.get() for c in base_controls];debounce()
 except tk.TclError:pass
for j,(name,limit) in enumerate(zip(['前后 vx  (m/s)','横移 vy  (m/s)','转向 wz  (rad/s)'],[.5,.3,.4])):
 value=command['base'][j]
 if abs(value)>limit:limit=[1.,.6,.8][j]
 base_controls.append(Control(base_frame,name,-limit,limit,value,base_changed,j))
full=tk.BooleanVar(value=any(abs(v)>lim+1e-6 for v,lim in zip(command['base'],[.5,.3,.4])))
def range_changed():
 global loading
 loading=True
 scale=1. if full.get() else last.get('trained_range_scale',.5)
 for c,limit in zip(base_controls,[1.,.6,.8]):
  limit*=scale;c.scale.configure(from_=-limit,to=limit);c.value.set(float(np.clip(c.value.get(),-limit,limit)))
 loading=False;base_changed()
ttk.Checkbutton(panel,text='使用完整目标范围（超出当前训练范围）',variable=full,command=range_changed).grid(row=4,column=0,sticky='w',pady=4)
actual_base=ttk.Label(panel,text='速度 / 自动俯仰横滚：等待数据',font=('Noto Sans CJK SC',9));actual_base.grid(row=5,column=0,sticky='w')
ttk.Separator(panel).grid(row=6,column=0,sticky='ew',pady=7)
mode=tk.StringVar(value=command['arm_mode'])
mode_row=ttk.Frame(panel);mode_row.grid(row=7,column=0,sticky='ew')
ttk.Label(mode_row,text='机械臂模式').pack(side='left',padx=(0,8))
arm_controls=[]
def mode_changed():
 global loading
 loading=True;command['arm_mode']=mode.get()
 if mode.get()=='joint_position':
  vals=np.rad2deg(last.get('arm_goal',DEFAULT_Q[19:25]))
  limits=np.rad2deg(last.get('arm_limits',[[-3.14,3.14]]*6))
  for j,c in enumerate(arm_controls):c.configure(f'关节 {j+1}  (°)',*limits[j],vals[j])
 else:
  vals=np.array(last.get('tcp_goal',[.6,0,.4,0,0,0]),dtype=float);vals[3:]=np.rad2deg(vals[3:])
  for c,name,lim,v in zip(arm_controls,['X  (m)','Y  (m)','Z  (m)','Roll  (°)','Pitch  (°)','Yaw  (°)'],[[-.8,1.2],[-1.,1.],[-.5,1.2],[-180,180],[-180,180],[-180,180]],vals):c.configure(name,*lim,v)
 loading=False;send()
for text,value in [('六关节','joint_position'),('末端位姿','end_effector_pose')]:
 ttk.Radiobutton(mode_row,text=text,value=value,variable=mode,command=mode_changed).pack(side='left')
ttk.Label(panel,text='末端 XYZ / RPY 使用航向坐标系；俯仰横滚自动。',font=('Noto Sans CJK SC',9)).grid(row=8,column=0,sticky='w',pady=2)
arm_frame=ttk.Frame(panel);arm_frame.grid(row=9,column=0,sticky='ew');arm_frame.columnconfigure(0,weight=1)
for j in range(6):arm_controls.append(Control(arm_frame,f'关节 {j+1}  (°)',-180,180,np.rad2deg(DEFAULT_Q[19+j]),lambda:None,j))
def send_arm():
 try:
  values=np.array([c.value.get() for c in arm_controls])
  if mode.get()=='joint_position':values=np.deg2rad(values)
  else:values[3:]=np.deg2rad(values[3:])
  command['arm_target']=values.tolist();command['arm_mode']=mode.get();command['arm_revision']+=1;send()
  message.configure(text='已发送目标，正在检查路径…')
 except (tk.TclError,ValueError):message.configure(text='请输入有效数值')
ttk.Button(panel,text='发送机械臂目标  ·  先检查路径',command=send_arm).grid(row=10,column=0,sticky='ew',pady=(5,3))
actual_arm=ttk.Label(panel,text='实际关节：等待数据',font=('Noto Sans CJK SC',9));actual_arm.grid(row=11,column=0,sticky='w')
grip_frame=ttk.Frame(panel);grip_frame.grid(row=12,column=0,sticky='ew',pady=(5,2));grip_frame.columnconfigure(0,weight=1)
def grip_changed():
 if loading:return
 try:command['grip']=grip.value.get()/100.;debounce()
 except tk.TclError:pass
grip=Control(grip_frame,'夹爪开度  (%)',0,100,command['grip']*100,grip_changed,0)
actual_grip=ttk.Label(panel,text='实际夹爪：等待数据',font=('Noto Sans CJK SC',9));actual_grip.grid(row=13,column=0,sticky='w')
buttons=ttk.Frame(panel);buttons.grid(row=14,column=0,sticky='ew',pady=7)
def stop_base():
 global loading
 loading=True
 for c in base_controls:c.value.set(0.)
 loading=False;command['base']=[0.,0.,0.];send()
def pause():
 command['paused']=not command['paused'];pause_button.configure(text='继续仿真' if command['paused'] else '暂停仿真');send()
def reset():
 global loading
 stop_base();command['reset_revision']+=1;command['paused']=last.get('policy_home_matches') is False;command['grip']=.5
 command['arm_target']=DEFAULT_Q[19:25].tolist();mode.set('joint_position');command['arm_mode']='joint_position'
 loading=True
 grip.value.set(50)
 limits=np.rad2deg(last.get('arm_limits',[[-3.14,3.14]]*6))
 for j,(c,v) in enumerate(zip(arm_controls,np.rad2deg(DEFAULT_Q[19:25]))):c.configure(f'关节 {j+1}  (°)',*limits[j],v)
 loading=False;pause_button.configure(text='继续仿真' if command['paused'] else '暂停仿真');send()
ttk.Button(buttons,text='底盘归零',command=stop_base).pack(side='left',expand=True,fill='x',padx=2)
pause_button=ttk.Button(buttons,text='继续仿真' if command['paused'] else '暂停仿真',command=pause);pause_button.pack(side='left',expand=True,fill='x',padx=2)
ttk.Button(buttons,text='复位机器人',command=reset).pack(side='left',expand=True,fill='x',padx=2)
message=ttk.Label(panel,text='准备就绪后，可先小幅调整控制量。',wraplength=410,style='Hint.TLabel');message.grid(row=15,column=0,sticky='ew',pady=3)
ttk.Label(panel,text='机械臂滑块需点“发送”；底盘、夹爪即时生效。\n未通过路径检查的目标会拒绝。碰撞/跌倒后暂停。',wraplength=410,font=('Noto Sans CJK SC',9)).grid(row=16,column=0,sticky='w')

model=mujoco.MjModel.from_xml_path(str(R/'robot/scene.xml'));data=mujoco.MjData(model)
qids=np.array([model.joint(n).qposadr[0] for n in JOINT_NAMES])
data.qpos[:]=DEFAULT_Q;mujoco.mj_forward(model,data)
model.vis.global_.offwidth=1000;model.vis.global_.offheight=600
renderer=mujoco.Renderer(model,562,1000);sensor=mujoco.Renderer(model,225,400)
cam=mujoco.MjvCamera();cam.distance=1.8;cam.azimuth=145;cam.elevation=-22
opt=mujoco.MjvOption();opt.geomgroup[3]=0
sensor_opt=mujoco.MjvOption();sensor_opt.geomgroup[3]=0;sensor_opt.geomgroup[5]=0
drag=[None]
def move(e):
 if drag[0] is not None:
  dx=e.x-drag[0][0];dy=e.y-drag[0][1];cam.azimuth-=dx*.3;cam.elevation=np.clip(cam.elevation-dy*.3,-85,20)
 drag[0]=(e.x,e.y)
view.bind('<ButtonPress-1>',lambda e:drag.__setitem__(0,(e.x,e.y)));view.bind('<B1-Motion>',move)
view.bind('<ButtonRelease-1>',lambda e:drag.__setitem__(0,None))
view.bind('<Button-4>',lambda e:setattr(cam,'distance',max(.6,cam.distance*.9)))
view.bind('<Button-5>',lambda e:setattr(cam,'distance',min(8.,cam.distance/ .9)))
images=[];initialized=False;frames=0;last_failure_time=None
def failure_message(s):
 detail=s.get('failure_details',{});reasons=[]
 if detail.get('nonfoot_contacts'):reasons.append('非足端接触：'+'、'.join(x['body'] for x in detail['nonfoot_contacts']))
 if detail.get('fallen_height') or detail.get('fallen_tilt'):reasons.append('达到跌倒阈值')
 if detail.get('head_too_low'):reasons.append('头部离地不足 2 cm')
 return '；'.join(reasons)+'。已暂停，可复位。' if reasons else s.get('message','几何检查未通过，已暂停。')

def update():
 global last,images,initialized,frames,loading,last_failure_time
 if closed:return
 try:
  s=json.loads((a.ipc/'state.json').read_text())
  if 'qpos' in s:
   last=s;q=np.array(s['qpos']);data.qpos[:7]=q[:7];data.qpos[qids]=q[7:];mujoco.mj_forward(model,data)
   if not initialized:
    initialized=True;mode_changed()
   values=np.array(s['base_actual']);values[3]=np.rad2deg(values[3])
   attitude=np.rad2deg(s.get('automatic_attitude',[0,0]))
   actual_base.configure(text=f'vx {values[0]:+.3f}  vy {values[1]:+.3f}  wz {values[2]:+.3f}\n实际俯仰 {values[3]:+.2f}° 横滚 {np.rad2deg(values[4]):+.2f}°  / 指引 {attitude[0]:+.1f}° {attitude[1]:+.1f}°')
   home_note='折叠默认 · 夹爪朝前' + (' · 旧姿态权重' if s.get('policy_home_matches') is False else '')
   checkpoint_label.configure(text=f"第 {s.get('checkpoint_iteration',0)} 次 · 阶段 {s.get('training_stage',1)} · 速度范围 {s.get('trained_range_scale',.5):.0%}\n{home_note}")
   if mode.get()=='joint_position':
    actual_arm.configure(text='实际关节 °：'+'  '.join(f'{x:.1f}' for x in np.rad2deg(s['arm_actual'])))
   else:
    tp=np.array(s['tcp_actual']);tp[3:]=np.rad2deg(tp[3:])
    actual_arm.configure(text='实际 XYZ m：'+'  '.join(f'{x:+.3f}' for x in tp[:3])+'\n实际 RPY °：'+'  '.join(f'{x:+.1f}' for x in tp[3:]))
   actual_grip.configure(text=f"实际开度 {s['grip_actual']*100:.1f}%  ·  指间距约 {s['grip_actual']*70:.1f} mm")
   paused=s['status'] in ('paused','failed_paused')
   live_label.configure(text=f"{'已暂停' if paused else '实时物理推理'} · 仿真 {s['sim_time']:.1f}s · 速度约 {s['real_time_factor']:.2f}× · 当前模式：{'六关节' if s['arm_mode']=='joint_position' else '末端位姿'}")
  elif s.get('message'):live_label.configure(text=s['message'])
  if s.get('status')=='failed_paused' and 'qpos' in s:
   detail=failure_message(s);message.configure(text=detail)
   pause_button.configure(text='接触/姿态保护暂停',state='disabled')
   if s.get('sim_time')!=last_failure_time:
    last_failure_time=s.get('sim_time')
    (a.ipc/'failure_display_detail.json').write_text(json.dumps({'message':detail,'state':s},ensure_ascii=False,indent=2))
  else:
   pause_button.configure(state='normal',text='继续仿真' if command['paused'] else '暂停仿真')
   if s.get('message'):message.configure(text=s['message'])
  if time.time()-s.get('heartbeat',0)>15:live_label.configure(text='本地控制器暂无新数据，正在等待…')
 except (OSError,ValueError,KeyError):pass
 cam.lookat[:]=data.qpos[:3]+[.08,0,.10]
 renderer.update_scene(data,camera=cam,scene_option=opt)
 im=Image.fromarray(renderer.render());w=max(1,view.winfo_width());h=max(1,view.winfo_height())
 im.thumbnail((w,h));images=[ImageTk.PhotoImage(im)];view.configure(image=images[0])
 for j,name in enumerate(['go2_d435i_rgb','piper_d435i_rgb']):
  sensor.update_scene(data,camera=name,scene_option=sensor_opt)
  image=Image.fromarray(sensor.render());image.thumbnail((max(200,left.winfo_width()//2-16),225))
  images.append(ImageTk.PhotoImage(image));rgb[j].configure(image=images[-1])
 frames+=1
 if initialized and frames==100:
  Image.fromarray(renderer.render()).save(a.ipc/'live_overview.png')
 root.after(65,update)

def close():
 global closed
 closed=True;command['shutdown']=True;send();renderer.close();sensor.close();root.destroy()
root.protocol('WM_DELETE_WINDOW',close)
send();root.after(100,update)
def self_test():
 assert len(base_controls)==3 and len(arm_controls)==6 and initialized
 stop_base();assert len(command['base'])==3
 mode.set('end_effector_pose');mode_changed();assert len(arm_controls)==6
 mode.set('joint_position');mode_changed()
 (a.ipc/'panel_test.json').write_text(json.dumps({'base_controls':3,'arm_controls':6,'gripper_controls':1,'modes':['joint_position','end_effector_pose'],'state_rendered':initialized,'frames':frames}))
 close()
if a.self_test:root.after(2000,self_test)
print('CONTROL_PANEL_READY',flush=True)
root.mainloop()
