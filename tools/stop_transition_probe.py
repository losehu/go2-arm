"""Independent free-base move/stop/restart test with measured per-foot support."""
import argparse,hashlib,json,os,sys,traceback
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--package',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=61119)
p.add_argument('--quick',action='store_true')
AppLauncher.add_app_launcher_args(p);args=p.parse_args();sys.path.insert(0,str(args.package.resolve()));app=AppLauncher(args).app
try:
 import torch,numpy as np
 from duet_env import DuetCfg,DuetEnv
 from cooperative_ppo import make_ppo,observations
 from leg_action import LEG_ACTION_CONTROL
 torch.set_num_threads(2);torch.manual_seed(args.seed)
 cases=[('stop',[0.,0.,0.]),('forward',[.26,0.,0.]),('backward',[-.26,0.,0.]),
        ('left',[0.,.23,0.]),('right',[0.,-.23,0.]),('yaw_left',[0.,0.,.3]),('yaw_right',[0.,0.,-.3]),
        ('forward_fast',[1.,0.,0.]),('backward_fast',[-1.,0.,0.])]
 if args.quick:cases=cases[:3]
 N=len(cases);n=2*N;vel=torch.tensor([c for _,c in cases]*2)
 class Probe(DuetEnv):
  def _sample_base_commands(self,ids):
   self.command_targets[ids,:3]=vel[ids];self.command_age[ids]=0;self.command_period[ids]=1e9
   if hasattr(self,'guidance_target'):self.guidance_target[ids]=0
  def _pre_physics_step(self,actions):
   t=self.episode_length_buf*self.step_dt
   moving=(t<6.)|((t>=14.)&(t<18.))
   self.command_targets[:,:3]=vel.to(self.device)*moving[:,None]
   super()._pre_physics_step(actions)
  def _get_dones(self):
   term,trunc=super()._get_dones()
   if getattr(self,'capture',False):
    point,speed=self.foot_kinematics();force=self.foot_contact_normals_w()[:,:,2].clamp_min(0.)
    clearance=point[:,:,2]-self.scene.env_origins[:,None,2]-.022
    contact=(force>1.)&(clearance<.020)
    for i in range(n):
     if self.failed[i]:continue
     t=float(self.episode_length_buf[i])*self.step_dt
     if bool(term[i]):self.failed[i]=True
     self.traces[i].append(dict(t=t,command=self.commands[i,:3].tolist(),
       target=self.command_targets[i,:3].tolist(),velocity=[*self.robot.data.root_lin_vel_b[i,:2].tolist(),float(self.robot.data.root_ang_vel_b[i,2])],
       clearance=clearance[i].tolist(),contact=contact[i].tolist(),force_n=force[i].tolist(),
       foot_speed_mps=speed[i].norm(dim=-1).tolist(),pitch_deg=float(torch.rad2deg(self.actual_pitch[i])),
       roll_deg=float(torch.rad2deg(self.actual_roll[i])),physical_failure=bool(term[i])))
   return torch.zeros_like(term),torch.zeros_like(trunc)
 saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
 assert saved['manifest']['leg_action_controller']==LEG_ACTION_CONTROL
 assert saved['manifest']['home_consistency_verified'] and not saved['manifest']['is_fixed_base']
 cfg=DuetCfg();cfg.scene.num_envs=n;cfg.scene.env_spacing=10.;cfg.sim.device='cpu';cfg.seed=args.seed;cfg.episode_length_s=26.
 cfg.phase_free_amp=saved['manifest'].get('phase_free_amp',False);cfg.directional_amp=saved['manifest'].get('directional_amp',False)
 env=Probe(cfg);env.reset();env.grip_target.fill_(.5);env.training_stage=1
 assert not env.robot.is_fixed_base
 actor=make_ppo(observations(env.get_loco_observation()),12,'cpu');actor.load(saved['loco'],load_cfg=None,strict=True);actor.actor.eval()
 env.traces=[[] for _ in range(n)];env.failed=[False]*n;env.capture=True;zero=torch.zeros(n,8)
 with torch.inference_mode():
  for step in range(1200):
   env.prepare_arm(zero);obs=observations(env.get_loco_observation());a=actor.actor(obs)
   a[N:]=actor.actor(obs,stochastic_output=True)[N:];env.step(torch.cat((a,zero[:,:7]),-1))
 result=[]
 for i,tr in enumerate(env.traces):
  row=dict(case=cases[i%N][0],policy='mean' if i<N else 'sampled',physical_failure=env.failed[i],segments={})
  for name,a,b,stopped in [('move',2.,5.9,False),('stop',8.,13.9,True),('restart',15.,17.9,False),('restop',20.,23.9,True)]:
   rows=[x for x in tr if a<=x['t']<=b]
   if not rows:row['segments'][name]={'missing':True};continue
   c=np.array([x['contact'] for x in rows]);h=np.array([x['clearance'] for x in rows]);f=np.array([x['force_n'] for x in rows]);v=np.array([x['velocity'] for x in rows]);cmd=np.array([x['command'] for x in rows]);shares=f.sum(0)/max(f.sum(),1.)
   stats=dict(samples=len(rows),all_four_contact_fraction=float(c.all(-1).mean()),per_foot_contact_fraction=c.mean(0).tolist(),
     clearance_mean_m=h.mean(0).tolist(),clearance_p95_m=np.percentile(h,95,axis=0).tolist(),load_share=shares.tolist(),
     velocity_rmse=np.sqrt(((v-cmd)**2).mean(0)).tolist(),actual_mean_velocity=v.mean(0).tolist(),
     pitch_roll_rmse_deg=np.sqrt((np.array([[x['pitch_deg'],x['roll_deg']] for x in rows])**2).mean(0)).tolist())
   if stopped or row['case']=='stop':
    stats['standing_support_passed']=bool(c.all(-1).mean()>=.95 and (c.mean(0)>=.98).all() and
      (np.percentile(h,95,axis=0)<=.01).all() and (shares>=.12).all())
   else:
    active=np.abs(cmd.mean(0))>.1
    gain=np.divide(v.mean(0),cmd.mean(0),out=np.zeros(3),where=active)
    stats['moving_response_passed']=bool((np.sqrt(((v-cmd)**2).mean(0))<=.10).all() and (gain[active]>=.5).all())
   row['segments'][name]=stats
  # Settling time requires continuous all-foot contact, not one lucky frame.
  for name,start,end in [('stop',6.,14.),('restop',18.,24.)]:
   streak=0;settled=None
   for x in tr:
    if not start<=x['t']<end:continue
    streak=streak+1 if all(x['contact']) else 0
    if streak>=25:settled=x['t']-start-.5;break
   row[name+'_continuous_0p5s_contact_settle_s']=settled
  result.append(row)
 args.output.mkdir(parents=True,exist_ok=False)
 report=dict(checkpoint=str(args.checkpoint),sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
   seed=args.seed,physics='PhysX CPU',is_fixed_base=False,mass_kg=float(env.robot.root_physx_view.get_masses()[0].sum()),
   protocol='24s: move0-6s,stop6-14s,restart14-18s,stop18-24s;zero attitude;mean+sampled;no resets',
   foot_order=['FL','FR','RL','RR'],cases=result,whole_policy_acceptance_passed=False)
 (args.output/'result.json').write_text(json.dumps(report,indent=2));(args.output/'traces.json').write_text(json.dumps(env.traces))
 print(json.dumps({'cases':n,'physical_failures':sum(env.failed),'sha256':report['sha256']}),flush=True)
except BaseException:
 traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
sys.stdout.flush();sys.stderr.flush();os._exit(0)
