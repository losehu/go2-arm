"""Fixed-command free-base diagnostic; no training or automatic policy reset."""
import argparse, hashlib, json, os, sys, time, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent
from isaaclab.app import AppLauncher
from gait_metrics import measured_swing
p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--package', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seed', type=int, default=61119)
p.add_argument('--focus',action='store_true')
p.add_argument('--only_forward',action='store_true')
p.add_argument('--case',action='append',help='Only run a named case (repeatable)')
AppLauncher.add_app_launcher_args(p)
args=p.parse_args();sys.path.insert(0,str(args.package.resolve()));app=AppLauncher(args).app
try:
 import numpy as np, torch
 from duet_env import DuetEnv, DuetCfg
 from cooperative_ppo import make_ppo, observations
 from diagonal_support import DiagonalSupport, support_pairs
 from leg_action import LEG_ACTION_CONTROL,leg_targets
 from isaaclab.utils.math import quat_apply_inverse
 torch.set_num_threads(2);torch.manual_seed(args.seed)
 cases=[('stop',[0.,0.,0.])]
 for axis,lim in enumerate([1.,.6,.8]):
  for speed in ([.12,.23,.40,lim] if axis==1 else ([.12,.26,.40,lim] if axis==0 else [.12,lim])):
   for sign in [1,-1]:
    c=[0.,0.,0.];c[axis]=speed*sign
    cases.append((f'{["vx","vy","wz"][axis]}_{speed*sign:+.2f}',c))
 for sx in (1,-1):
  for sy in (1,-1):
   cases.append((f'diagonal_{sx:+d}_{sy:+d}',[sx*.45,sy*.23,0.]))
   cases.append((f'diagonal_fast_{sx:+d}_{sy:+d}',[sx*.60,sy*.30,0.]))
   cases.append((f'walk_turn_{sx:+d}_{sy:+d}',[sx*.45,0.,sy*.30]))
   cases.append((f'lateral_turn_{sx:+d}_{sy:+d}',[0.,sx*.23,sy*.30]))
   for sw in (1,-1):
    cases.append((f'diagonal_turn_{sx:+d}_{sy:+d}_{sw:+d}',[sx*.45,sy*.23,sw*.30]))
 if args.focus:cases=[x for x in cases if x[0].startswith('vx_')]
 if args.only_forward:cases=[x for x in cases if x[0].startswith('vx_+')]
 if args.case:
  known={name for name,_ in cases}
  if set(args.case)-known:p.error(f'Unknown cases: {sorted(set(args.case)-known)}')
  cases=[x for x in cases if x[0] in args.case]
 N=len(cases);n=2*N;commands=torch.tensor([c for _,c in cases]*2)
 class Probe(DuetEnv):
  def _sample_base_commands(self,ids):
   self.command_targets[ids,:3]=commands[ids];self.command_targets[ids,3]=0
   self.command_age[ids]=0;self.command_period[ids]=1e9
   if hasattr(self,'guidance_target'):self.guidance_target[ids]=0
  def _get_dones(self):
   term,trunc=super()._get_dones()
   if getattr(self,'capture',False):
    point,velocity=self.foot_kinematics()
    root_q=self.robot.data.root_link_quat_w[:,None,:].expand(-1,4,-1)
    foot_base=quat_apply_inverse(root_q,point-self.robot.data.root_link_pos_w[:,None,:])
    hip_outward=torch.rad2deg(self.robot.data.joint_pos[:,self.legs[::3]])*torch.tensor([1.,-1.,1.,-1.])
    clearance=point[:,:,2]-self.scene.env_origins[:,None,2]-.022
    _,_,_,_,head,forces=self._metrics()
    foot_normals=self.foot_contact_normals_w()
    foot_force=foot_normals.norm(dim=-1)
    contact=(foot_force>1)&(clearance<.020)
    translating=self.commands[:,:2].ne(0.).any(dim=-1)&~self.bad_contact
    _,switch,_,_=self.probe_diagonal.step(contact,translating,self.step_dt)
    if getattr(self.cfg,'phase_free_amp',False):
     swing=torch.zeros_like(contact)
    elif getattr(self.cfg,'coordinated_trot',False):
     from foot_gait import trot_phase
     _,swing_progress=trot_phase(self.phase)
     swing=(swing_progress>0)&(self.walk_blend[:,None]>.8)
    else:
     phase=(self.phase[:,None]+torch.tensor([0.,.5,.5,0.]))%1
     swing=(phase>=.5)&(self.walk_blend[:,None]>.8)
    sliding=contact&(velocity[:,:,:2].norm(dim=-1)>.1)
    calf_velocity=self.robot.data.body_lin_vel_w[:,self.calf_bodies]
    legacy=torch.exp(-calf_velocity.square().sum(-1)/.04)
    corrected=torch.exp(-velocity.square().sum(-1)/.04)
    for i in range(n):
     if i in self.finished:continue
     t=float(self.episode_length_buf[i])*self.step_dt
     row={'t':t,'command':self.commands[i,:3].tolist(),'velocity':torch.cat([self.robot.data.root_lin_vel_b[i,:2],self.robot.data.root_ang_vel_b[i,2:3]]).tolist(),
       'height':float(self.robot.data.root_link_pos_w[i,2]-self.scene.env_origins[i,2]),'pitch_deg':float(torch.rad2deg(self.actual_pitch[i])),
       'roll_deg':float(torch.rad2deg(self.actual_roll[i])),'joint_error_deg':torch.rad2deg(self.robot.data.joint_pos[i,self.arm]-self.ref[i]).tolist(),'tcp_error_m':float(self.tcp_pos_error[i]),'tcp_orientation_error_deg':float(torch.rad2deg(self.tcp_rot_error[i])),'head_clearance':float(head[i]),'foot_clearance':clearance[i].tolist(),'foot_speed_xy':velocity[i,:,:2].norm(dim=-1).tolist(),
       'calf_com_speed':calf_velocity[i].norm(dim=-1).tolist(),'foot_center_speed':velocity[i].norm(dim=-1).tolist(),
       'raw_actions':self.raw_actions[i,:12].tolist(),'actions':self.actions[i,:12].tolist(),'filtered':self.filtered[i,:12].tolist(),'leg_joint_q':self.robot.data.joint_pos[i,self.legs].tolist(),'leg_target':leg_targets(self.nominal[i,self.legs],self.filtered[i],self.robot.data.soft_joint_pos_limits[i,self.legs]).tolist(),'leg_torque':self.last_tau[i,:12].tolist(),'foot_force':foot_force[i].tolist(),'foot_position':point[i].tolist(),'root_position':self.robot.data.root_pos_w[i].tolist(),'contact':contact[i].tolist(),'swing':swing[i].tolist(),'sliding':sliding[i].tolist(),
       'legacy_stance_score':legacy[i].tolist(),'correct_stance_score':corrected[i].tolist(),
       'hip_outward_deg':hip_outward[i].tolist(),'calf_contact':self.calf_contact[i].tolist(),
       'foot_normal_load_n':foot_normals[i,:,2].clamp_min(0.).tolist(),'diagonal_switch':bool(switch[i]),
       'foot_pair_width_m':(foot_base[i,[0,2],1]-foot_base[i,[1,3],1]).tolist()}
     self.traces[i].append(row)
     if bool(term[i]) or bool(trunc[i]):
      self.finished[i]={'physical_termination':bool(term[i]),'timeout':bool(trunc[i]),'end_time':t,
       'head_low':bool(head[i]<.02),'root_low':bool(self.robot.data.root_pos_w[i,2]<.16),
       'tilt':bool(self.robot.data.projected_gravity_b[i,2]>-.5),'bad_contact':bool(self.bad_contact[i]),
       'calf_contact':self.calf_contact[i].tolist()}
   return term,trunc
 saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
 assert saved['manifest'].get('leg_action_controller')==LEG_ACTION_CONTROL,'Checkpoint/controller mismatch'
 cfg=DuetCfg();cfg.scene.num_envs=n;cfg.scene.env_spacing=10.;cfg.sim.device='cpu';cfg.seed=args.seed
 if hasattr(cfg,'coordinated_trot'):cfg.coordinated_trot='coordinated_trot' in saved.get('manifest',{})
 if hasattr(cfg,'phase_free_amp'):cfg.phase_free_amp=saved.get('manifest',{}).get('phase_free_amp',False)
 if hasattr(cfg,'directional_amp'):cfg.directional_amp=saved.get('manifest',{}).get('directional_amp',False)
 env=Probe(cfg);env.reset();env.grip_target.fill_(.5)
 env.calf_bodies=[env.robot.body_names.index(f'{x}_calf') for x in ['FL','FR','RL','RR']]
 assert not env.robot.is_fixed_base
 loco=make_ppo(observations(env.get_loco_observation()),12,'cpu');loco.load(saved['loco'],load_cfg=None,strict=True);loco.actor.eval()
 env.probe_diagonal=DiagonalSupport(n,'cpu')
 env.capture=True;env.finished={};env.traces=[[] for _ in range(n)];zero=torch.zeros(n,8)
 with torch.inference_mode():
  for step in range(1050):
   env.prepare_arm(zero);obs=observations(env.get_loco_observation());a=loco.actor(obs);a[N:]=loco.actor(obs,stochastic_output=True)[N:]
   env.raw_actions=a.clone();env.step(torch.cat([a,zero[:,:7]],-1))
   if len(env.finished)==n:break
 result=[]
 for i,trace in enumerate(env.traces):
  clean=[r for r in trace if not any(r['calf_contact'])]
  rows=[r for r in clean if r['t']>1.] or clean or trace
  err=np.array([r['velocity'] for r in rows])-np.array([r['command'] for r in rows])
  swing=np.array([r['swing'] for r in rows]);contact=np.array([r['contact'] for r in rows]);clear=np.array([r['foot_clearance'] for r in rows])
  diagonal_a,diagonal_b,left,right=support_pairs(contact)
  hip=np.array([r['hip_outward_deg'] for r in rows])
  width=np.array([r['foot_pair_width_m'] for r in rows])
  observed=measured_swing(contact,clear,env.step_dt)
  normal_load=np.array([r['foot_normal_load_n'] for r in rows])*contact
  load_sum=normal_load.sum(axis=0)
  load_share=load_sum/max(float(load_sum.sum()),1.)
  window_steps=round(1.2/env.step_dt)
  complete_steps=len(normal_load)//window_steps*window_steps
  windows=normal_load[:complete_steps].reshape(-1,window_steps,4).sum(axis=1)
  window_shares=windows/np.maximum(windows.sum(axis=1,keepdims=True),1.)
  weak_load_fraction=float((window_shares.min(axis=1)<.12).mean()) if len(windows) else None
  swing_samples=sum(observed['swing_sample_count'])
  observed['measurement_duration_s']=len(rows)*env.step_dt
  observed['qualified_stride_hz_by_foot']=[n/(len(rows)*env.step_dt) for n in observed['qualified_swing_count']]
  actual_low=sum(n*(fraction or 0.) for n,fraction in zip(
      observed['swing_sample_count'],observed['swing_samples_below_1cm_fraction']))/swing_samples if swing_samples else None
  heights=np.array([r['height'] for r in rows])
  result.append(dict(env.finished[i],case=cases[i%N][0],policy='mean' if i<N else 'sampled',
    velocity_rmse=np.sqrt(np.mean(err**2,axis=0)).tolist(),pitch_rmse_deg=float(np.sqrt(np.mean(np.array([r['pitch_deg'] for r in rows])**2))),
    exclusive_diagonal_support_fraction=float((diagonal_a|diagonal_b).mean()) if clean else None,
    both_diagonal_pairs_observed=bool(clean and diagonal_a.any() and diagonal_b.any()),
    diagonal_a_fraction=float(diagonal_a.mean()) if clean else None,
    diagonal_b_fraction=float(diagonal_b.mean()) if clean else None,
    diagonal_switch_count=sum(r['diagonal_switch'] for r in rows) if clean else None,
    same_side_support_fraction=float((left|right).mean()) if clean else None,
    calf_free_metric_samples=len(clean),
    all_four_contact_fraction=float(contact.all(axis=1).mean()),
    foot_contact_fraction=contact.mean(axis=0).tolist(),
    foot_touchdowns=((~contact[:-1])&contact[1:]).sum(axis=0).tolist(),
    hip_outward_mean_deg=hip.mean(axis=0).tolist(),
    hip_outward_p95_deg=np.percentile(hip,95,axis=0).tolist(),
    hip_outward_over_18deg_fraction=(hip>18).mean(axis=0).tolist(),
    foot_pair_width_mean_m=width.mean(axis=0).tolist(),
    sliding_contact_fraction=float(np.sum(np.array([r['sliding'] for r in rows]))/max(1,contact.sum())),
    swing_below_1cm_fraction=(actual_low if getattr(cfg,'phase_free_amp',False) else
                               float((clear[swing]<.01).mean()) if swing.any() else None),
    measured_swing=observed,normal_load_share_by_foot=load_share.tolist(),
    foot_load_window_s=1.2,foot_load_window_count=len(windows),
    windows_any_foot_below_12pct_normal_load_fraction=weak_load_fraction,
    root_height_p05_m=float(np.percentile(heights,5)),
    root_below_28p5cm_fraction=float((heights<.285).mean()),
    foot_clearance_p95_m=np.percentile(clear,95,axis=0).tolist(),min_root_height=min(r['height'] for r in trace),
    roll_rmse_deg=float(np.sqrt(np.mean(np.array([r['roll_deg'] for r in rows])**2))),
    joint_rmse_deg=np.sqrt(np.mean(np.array([r['joint_error_deg'] for r in rows])**2,axis=0)).tolist(),
    tcp_rmse_m=float(np.sqrt(np.mean(np.array([r['tcp_error_m'] for r in rows])**2))),
    tcp_orientation_rmse_deg=float(np.sqrt(np.mean(np.array([r['tcp_orientation_error_deg'] for r in rows])**2))),
    legacy_stance_score=float(np.array([r['legacy_stance_score'] for r in rows])[contact].mean()) if contact.any() else None,
    correct_stance_score=float(np.array([r['correct_stance_score'] for r in rows])[contact].mean()) if contact.any() else None))
 args.output.mkdir(exist_ok=True,parents=True)
 (args.output/'traces.json').write_text(json.dumps(env.traces))
 report={'seed':args.seed,'checkpoint':str(args.checkpoint),'sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
  'physics':'PhysX CPU','is_fixed_base':False,'mass_kg':float(env.robot.root_physx_view.get_masses()[0].sum()),
  'protocol':'20s fixed command; deterministic mean and sampled action; zero internal attitude reference; first episode only',
  'measured_swing_available':True,'planned_swing_available':not getattr(cfg,'phase_free_amp',False),'cases':result,'acceptance_passed':False}
 (args.output/'result.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({'cases':len(result),'physical_terminations':sum(row['physical_termination'] for row in result)}),flush=True)
except BaseException:
 traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
sys.stdout.flush();os._exit(0)
