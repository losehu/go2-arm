"""Fixed-command free-base diagnostic; no training or automatic policy reset."""
import argparse, hashlib, json, os, sys, time, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--package', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seed', type=int, default=61119)
p.add_argument('--focus',action='store_true')
p.add_argument('--only_forward',action='store_true')
AppLauncher.add_app_launcher_args(p)
args=p.parse_args();sys.path.insert(0,str(args.package.resolve()));app=AppLauncher(args).app
try:
 import numpy as np, torch
 from duet_env import DuetEnv, DuetCfg
 from cooperative_ppo import make_ppo, observations
 from isaaclab.utils.math import quat_apply, quat_apply_inverse
 torch.set_num_threads(2);torch.manual_seed(args.seed)
 cases=[('stop',[0.,0.,0.])]
 for axis,lim in enumerate([.75,.45,.6]):
  for speed in ([.12,.23,.40,lim] if axis==1 else [.12,lim]):
   for sign in [1,-1]:
    c=[0.,0.,0.];c[axis]=speed*sign
    cases.append((f'{["vx","vy","wz"][axis]}_{speed*sign:+.2f}',c))
 if args.focus:cases=[x for x in cases if x[0].startswith('vx_')]
 if args.only_forward:cases=[x for x in cases if x[0].startswith('vx_+')]
 N=len(cases);n=2*N;commands=torch.tensor([c for _,c in cases]*2)
 class Probe(DuetEnv):
  def _sample_base_commands(self,ids):
   self.command_targets[ids,:3]=commands[ids];self.command_targets[ids,3]=0
   self.command_age[ids]=0;self.command_period[ids]=1e9
   if hasattr(self,'guidance_target'):self.guidance_target[ids]=0
  def _get_dones(self):
   term,trunc=super()._get_dones()
   if getattr(self,'capture',False):
    ids=self.foot_ids
    offset=quat_apply(self.robot.data.body_link_quat_w[:,ids],self.foot_offset)
    point=self.robot.data.body_link_pos_w[:,ids]+offset
    root_q=self.robot.data.root_link_quat_w[:,None,:].expand(-1,4,-1)
    foot_base=quat_apply_inverse(root_q,point-self.robot.data.root_link_pos_w[:,None,:])
    hip_outward=torch.rad2deg(self.robot.data.joint_pos[:,self.legs[::3]])*torch.tensor([1.,-1.,1.,-1.])
    com_offset=point-self.robot.data.body_com_pos_w[:,ids]
    velocity=self.robot.data.body_com_lin_vel_w[:,ids]+torch.linalg.cross(self.robot.data.body_com_ang_vel_w[:,ids],com_offset,dim=-1)
    clearance=point[:,:,2]-self.scene.env_origins[:,None,2]-.022
    _,_,_,_,head,forces=self._metrics()
    contact=(forces[:,self.feet]>1)&(clearance<.020)
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
    calf_velocity=self.robot.data.body_lin_vel_w[:,ids]
    legacy=torch.exp(-calf_velocity.square().sum(-1)/.04)
    corrected=torch.exp(-velocity.square().sum(-1)/.04)
    for i in range(n):
     if i in self.finished:continue
     t=float(self.episode_length_buf[i])*self.step_dt
     row={'t':t,'command':self.commands[i,:3].tolist(),'velocity':torch.cat([self.robot.data.root_lin_vel_b[i,:2],self.robot.data.root_ang_vel_b[i,2:3]]).tolist(),
       'height':float(self.robot.data.root_pos_w[i,2]),'pitch_deg':float(torch.rad2deg(self.actual_pitch[i])),
       'roll_deg':float(torch.rad2deg(self.actual_roll[i])),'joint_error_deg':torch.rad2deg(self.robot.data.joint_pos[i,self.arm]-self.ref[i]).tolist(),'tcp_error_m':float(self.tcp_pos_error[i]),'tcp_orientation_error_deg':float(torch.rad2deg(self.tcp_rot_error[i])),'head_clearance':float(head[i]),'foot_clearance':clearance[i].tolist(),'foot_speed_xy':velocity[i,:,:2].norm(dim=-1).tolist(),
       'calf_com_speed':calf_velocity[i].norm(dim=-1).tolist(),'foot_center_speed':velocity[i].norm(dim=-1).tolist(),
       'raw_actions':self.raw_actions[i,:12].tolist(),'actions':self.actions[i,:12].tolist(),'filtered':self.filtered[i,:12].tolist(),'leg_joint_q':self.robot.data.joint_pos[i,self.legs].tolist(),'leg_target':(self.nominal[i,self.legs]+.35*self.filtered[i,:12]).tolist(),'leg_torque':self.last_tau[i,:12].tolist(),'foot_force':forces[i,self.feet].tolist(),'foot_position':point[i].tolist(),'root_position':self.robot.data.root_pos_w[i].tolist(),'contact':contact[i].tolist(),'swing':swing[i].tolist(),'sliding':sliding[i].tolist(),
       'legacy_stance_score':legacy[i].tolist(),'correct_stance_score':corrected[i].tolist(),
       'hip_outward_deg':hip_outward[i].tolist(),
       'foot_pair_width_m':(foot_base[i,[0,2],1]-foot_base[i,[1,3],1]).tolist()}
     self.traces[i].append(row)
     if bool(term[i]) or bool(trunc[i]):
      self.finished[i]={'physical_termination':bool(term[i]),'timeout':bool(trunc[i]),'end_time':t,
       'head_low':bool(head[i]<.02),'root_low':bool(self.robot.data.root_pos_w[i,2]<.16),
       'tilt':bool(self.robot.data.projected_gravity_b[i,2]>-.5),'bad_contact':bool(self.bad_contact[i])}
   return term,trunc
 saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
 cfg=DuetCfg();cfg.scene.num_envs=n;cfg.scene.env_spacing=10.;cfg.sim.device='cpu';cfg.seed=args.seed
 if hasattr(cfg,'coordinated_trot'):cfg.coordinated_trot='coordinated_trot' in saved.get('manifest',{})
 if hasattr(cfg,'phase_free_amp'):cfg.phase_free_amp=saved.get('manifest',{}).get('phase_free_amp',False)
 if hasattr(cfg,'directional_amp'):cfg.directional_amp=saved.get('manifest',{}).get('directional_amp',False)
 env=Probe(cfg);env.reset();env.grip_target.fill_(.5)
 env.foot_ids=[env.robot.body_names.index(f'{x}_calf') for x in ['FL','FR','RL','RR']]
 env.foot_offset=torch.tensor([-.002,0.,-.213]).expand(n,4,3).clone()
 assert not env.robot.is_fixed_base
 loco=make_ppo(observations(env.get_loco_observation()),12,'cpu');loco.load(saved['loco'],load_cfg=None,strict=True);loco.actor.eval()
 env.capture=True;env.finished={};env.traces=[[] for _ in range(n)];zero=torch.zeros(n,8)
 with torch.inference_mode():
  for step in range(1050):
   env.prepare_arm(zero);obs=observations(env.get_loco_observation());a=loco.actor(obs);a[N:]=loco.actor(obs,stochastic_output=True)[N:]
   env.raw_actions=a.clone();env.step(torch.cat([a,zero[:,:7]],-1))
   if len(env.finished)==n:break
 result=[]
 for i,trace in enumerate(env.traces):
  rows=[r for r in trace if r['t']>1.] or trace
  err=np.array([r['velocity'] for r in rows])-np.array([r['command'] for r in rows])
  swing=np.array([r['swing'] for r in rows]);contact=np.array([r['contact'] for r in rows]);clear=np.array([r['foot_clearance'] for r in rows])
  diagonal_a=contact[:,0]&contact[:,3]&~contact[:,1]&~contact[:,2]
  diagonal_b=contact[:,1]&contact[:,2]&~contact[:,0]&~contact[:,3]
  hip=np.array([r['hip_outward_deg'] for r in rows])
  width=np.array([r['foot_pair_width_m'] for r in rows])
  result.append(dict(env.finished[i],case=cases[i%N][0],policy='mean' if i<N else 'sampled',
    velocity_rmse=np.sqrt(np.mean(err**2,axis=0)).tolist(),pitch_rmse_deg=float(np.sqrt(np.mean(np.array([r['pitch_deg'] for r in rows])**2))),
    exclusive_diagonal_support_fraction=float((diagonal_a|diagonal_b).mean()),
    both_diagonal_pairs_observed=bool(diagonal_a.any() and diagonal_b.any()),
    all_four_contact_fraction=float(contact.all(axis=1).mean()),
    foot_contact_fraction=contact.mean(axis=0).tolist(),
    foot_touchdowns=((~contact[:-1])&contact[1:]).sum(axis=0).tolist(),
    hip_outward_mean_deg=hip.mean(axis=0).tolist(),
    hip_outward_p95_deg=np.percentile(hip,95,axis=0).tolist(),
    hip_outward_over_18deg_fraction=(hip>18).mean(axis=0).tolist(),
    foot_pair_width_mean_m=width.mean(axis=0).tolist(),
    sliding_contact_fraction=float(np.sum(np.array([r['sliding'] for r in rows]))/max(1,contact.sum())),
    swing_below_1cm_fraction=float((clear[swing]<.01).mean()) if swing.any() else None,
    foot_clearance_p95_m=np.percentile(clear,95,axis=0).tolist(),min_root_height=min(r['height'] for r in trace),
    roll_rmse_deg=float(np.sqrt(np.mean(np.array([r['roll_deg'] for r in rows])**2))),
    joint_rmse_deg=np.sqrt(np.mean(np.array([r['joint_error_deg'] for r in rows])**2,axis=0)).tolist(),
    tcp_rmse_m=float(np.sqrt(np.mean(np.array([r['tcp_error_m'] for r in rows])**2))),
    tcp_orientation_rmse_deg=float(np.sqrt(np.mean(np.array([r['tcp_orientation_error_deg'] for r in rows])**2))),
    legacy_stance_score=float(np.array([r['legacy_stance_score'] for r in rows])[contact].mean()) if contact.any() else None,
    correct_stance_score=float(np.array([r['correct_stance_score'] for r in rows])[contact].mean()) if contact.any() else None))
 args.output.mkdir(exist_ok=True,parents=True)
 report={'seed':args.seed,'checkpoint':str(args.checkpoint),'sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
  'physics':'PhysX CPU','is_fixed_base':False,'mass_kg':float(env.robot.root_physx_view.get_masses()[0].sum()),
  'protocol':'20s fixed command; deterministic mean and sampled action; zero internal attitude reference; first episode only',
  'planned_swing_available':not getattr(cfg,'phase_free_amp',False),'cases':result,'acceptance_passed':False}
 (args.output/'result.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({'cases':len(result),'physical_terminations':sum(row['physical_termination'] for row in result)}),flush=True)
except BaseException:
 traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
sys.stdout.flush();os._exit(0)
