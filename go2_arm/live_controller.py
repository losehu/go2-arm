"""Local dual-policy inference; same DuetEnv filters/actuators as training."""
import argparse,json,os,time,traceback
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--ipc',type=Path,required=True);p.add_argument('--test_steps',type=int,default=0);AppLauncher.add_app_launcher_args(p);args=p.parse_args()
args.ipc.mkdir(parents=True,exist_ok=True)
def publish(s):
 s['heartbeat']=time.time();tmp=args.ipc/'state.tmp';tmp.write_text(json.dumps(s,ensure_ascii=False));tmp.replace(args.ipc/'state.json')
publish({'status':'starting','message':'正在加载本地双策略…'})
app=AppLauncher(args).app
try:
 import numpy as np,torch
 from scipy.spatial.transform import Rotation
 from isaaclab.utils.math import matrix_from_quat
 from duet_env import DuetEnv,DuetCfg
 from base_commands import apply_velocity_deadzone
 from cooperative_ppo import make_ppo,observations
 from command_session import CommandSession
 from reference_filter import ReferenceFilter
 from geometry import base_to_heading
 from robot_constants import JOINT_NAMES,DEFAULT_Q,PREVIEW_DEFAULT_Q
 torch.set_num_threads(2)
 class LiveEnv(DuetEnv):
  def _sample_base_commands(self,ids):
   self.command_targets[ids]=0;self.command_age[ids]=0;self.command_period[ids]=1e9
  def init_live(self):
   self.external_control=True;self.live_t=0.;self.session=CommandSession(PREVIEW_DEFAULT_Q);self.goal[:]=self.nominal[:,self.arm]
   self.mode.zero_();self.guidance_target.zero_();self.guidance.zero_();self._reference()
  def _trajectory_target(self):
   if not hasattr(self,'session'):return super()._trajectory_target()
   s=self.session.sample(self.live_t);tensor=lambda x:torch.as_tensor(x,device=self.device,dtype=torch.float32).reshape(1,-1)
   self.path_ref=tensor(s['q']);self.path_v=tensor(s['qd']);self.path_a=tensor(s['qdd'])
   self.target_p=tensor(s['target_p']);self.target_R=tensor(s['target_R']).reshape(1,3,3);self.planned_attitude=tensor(s['x'][6:])
   self.mode[:]=float(s['active_mode']=='end_effector_pose')
   final=self.session.sample(1e12);self.goal[:]=tensor(final['q'])
   self.segment_goal=self.goal.clone();self.segment_direction=torch.sign(self.goal-self.nominal[:,self.arm])
   duration=sum(seg.duration for seg in self.session.segments)
   self.segment_duration=torch.tensor([duration],device=self.device);self.segment_moved=(self.goal-self.nominal[:,self.arm]).abs()>np.deg2rad(1)
   self.clock[:]=max(0,self.live_t-self.session.start_time)
   if self.training_stage==1:self.ref=self.path_ref.clone();self.ref_v=self.path_v.clone();self.ref_a=self.path_a.clone()
  def _apply_action(self):
   super()._apply_action()
   if hasattr(self,'session'):self.live_t+=self.physics_dt
  def _get_dones(self):
   done,timeout=super()._get_dones()
   if hasattr(self,'session'):
    self.live_done=bool(done[0]);self.failure_details={}
    if self.live_done:
     _,_,_,_,head,forces=self._metrics()
     self.failure_details={'nonfoot_contacts':[{'body':self.contact.body_names[i],'force_n':float(forces[0,i])} for i in self.forbidden if forces[0,i]>.5],
       'fallen_height':bool(self.robot.data.root_pos_w[0,2]<.16),'fallen_tilt':bool(self.robot.data.projected_gravity_b[0,2]>-.5),'head_too_low':bool(head[0]<.02),'head_clearance_m':float(head[0])}
    # Pause the UI at the actual failing configuration; don't auto-reset it.
    return torch.zeros_like(done),torch.zeros_like(timeout)
   return done,timeout
  def snapshot(self):
   q=self.robot.data.root_quat_w[0].cpu().numpy();j=self.robot.data.joint_pos[0,self.ids].cpu().numpy();pos=self.robot.data.root_pos_w[0].cpu().numpy()
   roll,pitch,_=Rotation.from_quat(q[[1,2,3,0]]).as_euler('xyz')
   p,R=self.fk(self.robot.data.joint_pos[:,self.arm]);p,R=base_to_heading(p,R,matrix_from_quat(self.robot.data.root_quat_w),self.robot.data.root_pos_w[:,2])
   final=self.session.sample(1e12)
   return {'qpos':np.r_[pos,q,j].tolist(),'joint_names':JOINT_NAMES,'base_actual':[*self.robot.data.root_lin_vel_b[0,:2].cpu().tolist(),float(self.robot.data.root_ang_vel_b[0,2]),pitch,roll],
    'automatic_attitude':self.guidance[0].cpu().tolist(),'base_command':self.commands[0,:3].cpu().tolist(),'arm_actual':j[12:18].tolist(),
    'tcp_actual':np.r_[p[0].cpu().numpy(),Rotation.from_matrix(R[0].cpu().numpy()).as_euler('xyz')].tolist(),
    'tcp_goal':np.r_[final['target_p'],Rotation.from_matrix(final['target_R']).as_euler('xyz')].tolist(),
    'arm_goal':final['q'].tolist(),'arm_reference':self.ref[0].cpu().tolist(),'grip_actual':float(j[18:20].mean()/.035),'sim_time':self.live_t}
 saved=torch.load(args.checkpoint,map_location=args.device,weights_only=True)
 cfg=DuetCfg();cfg.scene.num_envs=1;cfg.sim.device=args.device;cfg.episode_length_s=3600
 cfg.coordinated_trot='coordinated_trot' in saved.get('manifest',{})
 cfg.phase_free_amp=saved.get('manifest',{}).get('phase_free_amp',False)
 cfg.directional_amp=saved.get('manifest',{}).get('directional_amp',False)
 cfg.robot.init_state.joint_pos.update(dict(zip(JOINT_NAMES,PREVIEW_DEFAULT_Q[7:].tolist())))
 env=LiveEnv(cfg);env.reset()
 from leg_action import LEG_ACTION_CONTROL
 if saved['manifest'].get('leg_action_controller')!=LEG_ACTION_CONTROL:raise ValueError('Checkpoint/controller mismatch; use the matching source package')
 if saved.get('format')!='roboduet_dualmode_v1':raise ValueError('This worker requires a new dual-policy checkpoint')
 env.training_stage=saved['training_stage'];env.level=saved['arm_level'];env.range_scale=saved['range_scale'];env.arm_enabled[:]=env.training_stage==2;env.init_live()
 loco=make_ppo(observations(env.get_loco_observation()),12,args.device);arm=make_ppo(observations(env.get_arm_observation()),8,args.device)
 loco.load(saved['loco'],load_cfg=None,strict=True);arm.load(saved['arm'],load_cfg=None,strict=True);loco.actor.eval();arm.actor.eval()
 controls={'seq':-1,'base':[0,0,0],'grip':.5,'paused':False,'arm_mode':'joint_position','arm_revision':0,'reset_revision':0}
 policy_home=saved.get('home_arm_q',saved.get('manifest',{}).get('home_arm_q'))
 home_matches=policy_home is not None and bool(np.allclose(np.asarray(policy_home),PREVIEW_DEFAULT_Q[19:25],atol=1e-6))
 home_message='折叠默认姿态，夹爪朝前' + ('；俯仰横滚自动协调' if home_matches else '；当前权重按展开姿态训练，新姿态行走尚未验收')
 last_seq=-2;arm_rev=0;reset_rev=0;failed=False;message=home_message;steps=0;wall=0.;limits=env.session.bridge.arm_limits.tolist()
 with torch.inference_mode():
  while True:
   try:incoming=json.loads((args.ipc/'command.json').read_text())
   except (OSError,ValueError):incoming={}
   if incoming.get('shutdown'):break
   if incoming.get('seq',-1)!=last_seq:
    controls.update(incoming);last_seq=controls['seq']
    if controls.get('reset_revision',0)!=reset_rev:
     reset_rev=controls['reset_revision'];env.reset();env.init_live();failed=False;message='已复位。'+home_message
    env.session.set_mode(controls['arm_mode'])
    if controls['arm_revision']!=arm_rev:
     arm_rev=controls['arm_revision']
     if not failed:
      try:
       if env.training_stage<2:raise ValueError('当前是第一阶段权重，尚未训练双模式配合；等待第二阶段权重')
       actual=env.snapshot();publish(dict(actual,status='planning',message='正在检查整机路径…',arm_limits=limits,arm_home=PREVIEW_DEFAULT_Q[19:25].tolist()))
       old=env.session.sample(env.live_t);env.session.bridge.configure(np.asarray(actual['qpos']))
       result=env.session.command(np.r_[controls['base'],controls['arm_target'],controls['grip']],env.live_t,controls['arm_mode'])
       # Preserve controller q/v/a when a newly accepted goal changes modes.
       q=torch.tensor(old['q'],device=env.device,dtype=torch.float32)[None];v=torch.tensor(old['qd'],device=env.device,dtype=torch.float32)[None];a=torch.tensor(old['qdd'],device=env.device,dtype=torch.float32)[None]
       env.reference_filter.reset(env.robot._ALL_INDICES,q,v,a);env.smooth_q[:]=q;env.smooth_v[:]=v;env.smooth_a[:]=a
       env.last_goal_sign.zero_();env.goal_crossings.zero_();message=f"目标已接受，路径 {result['duration_s']:.2f} 秒"
      except (ValueError,RuntimeError) as e:message='目标未接受：'+str(e)
   base=np.asarray(controls['base'],float)
   if base.shape!=(3,) or not np.isfinite(base).all():raise ValueError('Expected vx,vy,wz only')
   env.command_targets[0,:3]=apply_velocity_deadzone(torch.tensor(np.clip(base,[-1,-.6,-.8],[1,.6,.8]),device=env.device,dtype=torch.float32))
   env.grip_target[:]=float(np.clip(controls['grip'],0,1));paused=controls.get('paused',False) or failed
   if not paused:
    tic=time.monotonic();actual=env.snapshot();nextref=env.session.sample(env.live_t+.02)
    if not env.session.bridge.runtime_clear(actual['qpos'],nextref['x'],controls['grip']):
     failed=True;message='实际底盘／腿姿态下，下一步参考未通过碰撞检查，已暂停。'
    else:
     aa=arm.actor(observations(env.get_arm_observation())) if env.training_stage==2 else torch.zeros(1,8,device=env.device)
     env.prepare_arm(aa);la=loco.actor(observations(env.get_loco_observation()));env.step(torch.cat([la,aa[:,:6],torch.zeros(1,1,device=env.device)],-1));steps+=1
     if env.live_done:failed=True;message='触发物理终止条件，已暂停；具体原因见碰撞／姿态记录。'
    delay=env.step_dt-(time.monotonic()-tic)
    if delay>0:time.sleep(delay)
    wall+=time.monotonic()-tic
   else:time.sleep(.04)
   s=env.snapshot();s.update(status='failed_paused' if failed else ('paused' if paused else 'running'),message=message,
    arm_limits=limits,arm_home=PREVIEW_DEFAULT_Q[19:25].tolist(),home_pose='folded_forward',policy_home_matches=home_matches,arm_revision_ack=arm_rev,checkpoint_iteration=saved['iteration'],trained_range_scale=env.range_scale,
    training_stage=env.training_stage,arm_level=env.level,control_steps=steps,real_time_factor=steps*env.step_dt/max(.001,wall),physics_device=args.device,arm_mode=env.session.mode,failure_details=getattr(env,'failure_details',{}))
   publish(s)
   if args.test_steps and (steps>=args.test_steps or failed):
    (args.ipc/'inference_test.json').write_text(json.dumps({'steps':steps,'failed':failed,'finite':bool(torch.isfinite(env.robot.data.joint_pos).all()),'checkpoint':str(args.checkpoint)},indent=2));break
except BaseException as e:
 traceback.print_exc();publish({'status':'error','message':str(e)});os._exit(1)
os._exit(0)
