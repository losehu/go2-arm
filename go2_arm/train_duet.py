import argparse,json,os,sys,time,shutil,hashlib,traceback
from pathlib import Path
from datetime import datetime
from isaaclab.app import AppLauncher
from stage1_schedule import STAGE1_LEVELS,STAGE1_BUDGET,stage1_ready,next_stage1_level
p=argparse.ArgumentParser();p.add_argument('--num_envs',type=int,default=4096);p.add_argument('--stage1_iterations',type=int,default=STAGE1_BUDGET);p.add_argument('--stage2_iterations',type=int,default=40000);p.add_argument('--smoke',action='store_true');p.add_argument('--run_dir');p.add_argument('--resume_checkpoint',type=Path);p.add_argument('--allow_mount_rails_revision',action='store_true')
p.add_argument('--amp', action='store_true', help='Add the Go2 AMP style reward in stage one')
p.add_argument('--hip_splay_penalty', action='store_true', help='Penalize excessive outward hip angles')
gait_mode=p.add_mutually_exclusive_group()
gait_mode.add_argument('--coordinated_trot', action='store_true', help='Use cadence-matched diagonal gait and anti-slip rewards')
gait_mode.add_argument('--phase_free_amp', action='store_true', help='Learn style with AMP without gait phase observations or targets')
p.add_argument('--stop_after_updates', type=int, help='Stop after this many new updates and save a trial checkpoint')
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
if args.phase_free_amp and not args.amp:
 p.error('--phase_free_amp requires --amp')
if args.stop_after_updates is not None and args.stop_after_updates <= 0:
 p.error('--stop_after_updates must be positive')
R=Path(__file__).resolve().parent
run=Path(args.run_dir).resolve() if args.run_dir else R/'runs'/f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{'smoke' if args.smoke else 'roboduet_dualmode'}"
run.mkdir(parents=True,exist_ok=False);(run/'train.pid').write_text(str(os.getpid()));(R/('latest_smoke_run.txt' if args.smoke else 'active_run.txt')).write_text(str(run))
app=AppLauncher(args).app
try:
 import torch,numpy as np
 from duet_env import DuetEnv,DuetCfg
 from cooperative_ppo import make_ppo,observations,finite_models
 from robot_constants import DEFAULT_Q
 from torch.utils.tensorboard import SummaryWriter
 torch.set_num_threads(4);torch.manual_seed(61108)
 cfg=DuetCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=61108;cfg.sim.device=args.device
 cfg.hip_splay_penalty=args.hip_splay_penalty
 cfg.coordinated_trot=args.coordinated_trot
 cfg.phase_free_amp=args.phase_free_amp
 if not args.smoke:
  shutil.copytree(R/'robot',run/'robot');cfg.robot.spawn.usd_path=str(run/'robot/usd/robot.usd')
  (run/'source').mkdir()
  for f in [*R.glob('*.py'),*R.glob('*.json'),R/'path_bank.npz',R/'whole_body_bank.npz',R/'head_vertices.npy']:shutil.copy2(f,run/'source'/f.name)
 env=DuetEnv(cfg);env.reset()
 home=torch.tensor(DEFAULT_Q[19:25],device=env.device,dtype=torch.float32)
 assert torch.allclose(env.nominal[:,env.arm],home.expand(env.num_envs,-1),atol=1e-6)
 assert torch.allclose(env.bank[:,0],home.expand(len(env.bank),-1),atol=1e-6),'Path bank home differs from reset posture'
 assert torch.allclose(env.ref,home.expand(env.num_envs,-1),atol=1e-6),'Stage 1 is not holding folded home'
 assert not env.robot.is_fixed_base,'Locomotion training requires a free root'
 loco=make_ppo(observations(env.get_loco_observation()),12,args.device);arm=make_ppo(observations(env.get_arm_observation()),8,args.device)
 assert not ({id(p) for p in loco.actor.parameters()}&{id(p) for p in arm.actor.parameters()})
 if args.amp:
  from stage1_amp import Stage1AMP
  amp=Stage1AMP(R/'amp_reference',env)
 else:
  amp=None
 resume={}
 if args.resume_checkpoint:
  from checkpoint_resume import restore_stage1
  resume=restore_stage1(args.resume_checkpoint,env,loco,arm,home,args.stage1_iterations,args.allow_mount_rails_revision,amp=amp)
 first_iteration=resume.get('first_training_iteration',0)
 if args.stop_after_updates and first_iteration+args.stop_after_updates>args.stage1_iterations:
  raise ValueError('A bounded stage-one trial must end within the stage-one budget')
 budget=args.stage1_iterations+args.stage2_iterations
 limit=min(budget,first_iteration+args.stop_after_updates) if args.stop_after_updates else budget
 manifest={'run_dir':str(run),'pid':os.getpid(),'pid_start_ticks':int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]),'num_envs':args.num_envs,'stage1_iterations':args.stage1_iterations,'stage2_iterations':args.stage2_iterations,'total_budget':args.stage1_iterations+args.stage2_iterations,'user_continuous_commands':10,'automatic_pitch_roll':True,'loco_actions':12,'arm_actions':8,'loco_obs_dim':env.get_loco_observation().shape[-1],'arm_obs_dim':env.get_arm_observation().shape[-1],'separate_actors_critics_optimizers':True,'is_fixed_base':env.robot.is_fixed_base,'full_mass_kg':float(env.robot.root_physx_view.get_masses()[0].sum()),'from_scratch':True,'urdf_sha256':hashlib.sha256((R/'robot/robot.urdf').read_bytes()).hexdigest(),'source_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [*R.glob('*.py'),R/'whole_body_bank.npz',R/'controller_bench_report.json']},'acceptance_passed':False}
 manifest.update(home_pose='folded_forward',home_arm_q=DEFAULT_Q[19:25].tolist(),home_arm_deg=np.rad2deg(DEFAULT_Q[19:25]).tolist(),home_consistency_verified=True)
 manifest.update(optimization_revision='executed_entropy_posture_tracking_tail_v3',reward_revision='posture_and_tracking_log_tail_v1',tracking_reward_revision='tracking_log_tail_v1',entropy_regularizer='mixed-measure entropy of executed clamp(Normal,-1,1)',parent_run=json.loads((R/'repair_metadata.json').read_text())['parent_run'])
 manifest.update(curriculum_revision='decoupled_stage1_curriculum_v4',stage1_schedule=[{'range_scale':r,'guidance_cap_deg':g} for r,g in STAGE1_LEVELS],curriculum_min_dwell_iterations=500,stage1_budget_basis=json.loads((R/'repair_metadata.json').read_text())['stage1_budget_basis'])
 manifest.update(budget_revision='final_curriculum_time_reserve_v5',expected_final_checkpoint=f'model_{args.stage1_iterations+args.stage2_iterations-1}.pt')
 manifest.update(controller_revision='arm5_bandwidth_v7',arm_controller_omega_rad_s=env.omega.detach().cpu().tolist(),arm_controller_damping_ratio=env.zeta)
 manifest.update(episode_success_revision='timeout_without_physical_termination_v8')
 manifest.update(foot_sphere_radius_m=.022,foot_slip_metric_threshold_mps=.1)
 manifest['asset_hashes']={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [R/'robot/robot.urdf',R/'robot/robot_mjcf.xml',R/'robot/scene.xml',*sorted((R/'robot/usd').rglob('*.usd'))]}
 manifest.update(stage1_arm_hold_cost_weight=4.,stage1_arm_hold_cost_revision='mean_joint_error_log_tail_v1')
 manifest.update(robot_revision='box_mount_rails_v9',box_payload_mass_kg=.5,box_mount_rails_mass_kg=.08,box_mount_xyz_m=[-.1,0,.115])
 if args.phase_free_amp:
  from foot_gait import COORDINATED_TROT
  manifest.update(phase_free_amp=True,gait_reward_revision='amp_no_phase_contact_slip_v1',
   gait_phase_observation=False,gait_phase_reward=False,foot_slip_cost_weight=COORDINATED_TROT['slip_cost_weight'],
   foot_slip_scale_mps=COORDINATED_TROT['slip_scale_mps'],foot_contact_height_m=COORDINATED_TROT['contact_height_m'])
  assert manifest['loco_obs_dim']==69 and not hasattr(env,'phase')
 else:
  manifest.update(gait_reward_revision='diagonal_trot_footcenter_clearance_placement_v9',foot_swing_peak_clearance_m=.04,gait_name='diagonal_trot',gait_leg_order=['FL','FR','RL','RR'],gait_phase_offsets=[0.,.5,.5,0.],gait_frequency_hz=3.,gait_reward_weight=6.,gait_clearance_cost_weight=4.,gait_placement_cost_weight=25.,gait_standing_mode='four_feet_support')
  if args.coordinated_trot:
   from foot_gait import COORDINATED_TROT
   manifest.update(coordinated_trot=COORDINATED_TROT,gait_reward_revision=COORDINATED_TROT['revision'],
    gait_frequency_hz=COORDINATED_TROT['frequency_hz'],foot_swing_peak_clearance_m=COORDINATED_TROT['swing_peak_m'],
    gait_clearance_cost_weight=COORDINATED_TROT['clearance_cost_weight'])
 manifest.update(resume);manifest.setdefault('first_training_iteration',0)
 if args.hip_splay_penalty:
  from hip_splay import HIP_SPLAY_REWARD
  manifest['hip_splay_reward']=HIP_SPLAY_REWARD
 if amp is not None:
  if not resume:
   manifest['parent_run']=None
  manifest['stage1_amp']=amp.metadata
  if not args.smoke:
   shutil.copytree(R/'amp_reference',run/'source'/'amp_reference')
 manifest['trial_stop_after_updates']=args.stop_after_updates
 manifest['planned_stop_iteration']=limit-1
 (run/'manifest.json').write_text(json.dumps(manifest,indent=2));print('MODEL_VERIFIED',json.dumps(manifest),flush=True)
 writer=SummaryWriter(str(run));total=args.stage1_iterations+args.stage2_iterations;t0=time.time();last_advance=first_iteration;loco_updates=resume.get('inherited_loco_updates',0);arm_updates=resume.get('inherited_arm_updates',0)
 start_loco=next(loco.actor.parameters()).detach().clone();start_arm=next(arm.actor.parameters()).detach().clone()
 amp_updates=resume.get('inherited_amp_updates',0)
 start_discriminator=next(amp.discriminator.parameters()).detach().clone() if amp is not None and args.smoke else None
 def save(i,label=None):
  checkpoint=run/(label or f'model_{i}.pt');tmp=checkpoint.with_suffix('.tmp')
  payload={'format':'roboduet_dualmode_v1','iteration':i,'loco_updates':loco_updates,'arm_updates':arm_updates,'loco':loco.save(),'arm':arm.save(),'training_stage':env.training_stage,'range_scale':env.range_scale,'arm_level':env.level,'stage1_curriculum_level':env.stage1_curriculum_level,'guidance_cap_deg':env.guidance_cap_deg,'success_ema':env.success_ema,'precision_success_ema':env.precision_success_ema,'home_arm_q':DEFAULT_Q[19:25].tolist(),'manifest':manifest}
  if amp is not None:
   payload['amp']=amp.save()
   payload['amp_updates']=amp_updates
  torch.save(payload,tmp);tmp.replace(checkpoint);return checkpoint
 if resume:save(first_iteration-1,'resume_input.pt')
 status='stage1_trial_complete_pending_evaluation' if limit<total else 'iterations_complete_pending_acceptance'
 for i in range(first_iteration,limit):
  env.training_iteration=i
  if i==args.stage1_iterations:
   save(i-1,'stage1_final.pt')
   if not args.smoke and not stage1_ready(env.stage1_curriculum_level,env.range_scale,env.guidance_cap_deg,env.success_ema):
    status='stage1_budget_complete_gate_not_passed';print('STAGE1_GATE_NOT_PASSED',env.range_scale,env.success_ema,flush=True);break
   with torch.inference_mode():env.set_training_stage(2)
   print('COOPERATIVE_STAGE_STARTED',i,flush=True)
  if env.training_stage==1:
   next_level=next_stage1_level(env.stage1_curriculum_level,env.success_ema,i,last_advance)
   if next_level!=env.stage1_curriculum_level:
    gate_success=env.success_ema
    with torch.inference_mode():env.set_stage1_curriculum(next_level)
    last_advance=i
    print('STAGE1_CURRICULUM_ADVANCED',json.dumps({'iteration':i,'level':next_level,'range_scale':env.range_scale,'guidance_cap_deg':env.guidance_cap_deg,'previous_success_ema':gate_success}),flush=True)
  elif i-last_advance>=500 and env.success_ema>.7 and env.level<2 and env.precision_success_ema>.5:
   env.level+=1;env.success_ema=0.;env.precision_success_ema=0.;last_advance=i;env.last_curriculum_change_iteration=i
  tic=time.time();rew_l=[];rew_a=[];failures=0;loco_clips=[];arm_clips=[]
  active_amp=amp is not None and env.training_stage==1
  amp_rollout=[];amp_valid=[];amp_rewards=[]
  with torch.inference_mode():
   for step in range(24):
    ao=observations(env.get_arm_observation())
    aa=arm.act(ao) if env.training_stage==2 else torch.zeros(env.num_envs,8,device=env.device)
    env.prepare_arm(aa)
    lo=observations(env.get_loco_observation());la=loco.act(lo)
    loco_clips.append(float((la.abs()>1).float().mean()));arm_clips.append(float((aa.abs()>1).float().mean()))
    action=torch.cat([la,aa[:,:6],torch.zeros(env.num_envs,1,device=env.device)],-1)
    if active_amp:
     before=amp.observe(env)
    _,_,term,trunc,extras=env.step(action)
    dones=term|trunc;failures+=int(term.sum());extra={'time_outs':trunc&~term}
    next_lo=observations(env.get_loco_observation());next_ao=observations(env.get_arm_observation())
    leg_reward=env.loco_reward
    if active_amp:
     pair=amp.transition(before,amp.observe(env),env.commands[:,:3])
     style_reward=amp.reward(pair,dones,amp_updates+1,env.step_dt)
     leg_reward=leg_reward+style_reward
     amp_rollout.append(pair);amp_valid.append(~dones);amp_rewards.append(style_reward.mean())
    loco.process_env_step(next_lo,leg_reward,dones,extra)
    if env.training_stage==2:arm.process_env_step(next_ao,env.arm_reward,dones,extra)
    rew_l.append(float(env.loco_reward.mean()));rew_a.append(float(env.arm_reward.mean()))
   loco.compute_returns(next_lo)
   if env.training_stage==2:arm.compute_returns(next_ao)
  collected=time.time()-tic;learning=time.time()
  if args.smoke and active_amp and i==first_iteration:
   with torch.inference_mode():
    amp_foot_frame_error_m=amp.validate_foot_frames(env)
  if active_amp:
   amp_stats=amp.update(amp_rollout,amp_valid)
   amp_stats['mean_style_reward']=float(torch.stack(amp_rewards).mean())
   if not np.isfinite(list(amp_stats.values())).all():
    raise RuntimeError('Non-finite AMP optimization metrics')
   amp_updates+=1
  loss_l=loco.update();loco_updates+=1
  loss_a=arm.update() if env.training_stage==2 else {}
  if env.training_stage==2:arm_updates+=1
  assert finite_models(loco,arm),'Non-finite model parameters'
  state={'status':'running','iteration':i,'completed_iterations':i+1,'target_iterations':limit,'updated_at':datetime.now().astimezone().isoformat(),'elapsed_wall_s':time.time()-t0,'iteration_s':time.time()-tic,'collection_s':collected,'learning_s':time.time()-learning,'training_stage':env.training_stage,'range_scale':env.range_scale,'arm_level':env.level,'success_ema':env.success_ema,'precision_success_ema':env.precision_success_ema,'loco_updates':loco_updates,'arm_updates':arm_updates,'num_envs':env.num_envs,'pid':os.getpid(),'run_dir':str(run),'metrics':env.log_metrics,'termination_count':failures,'mean_leg_reward':float(np.mean(rew_l)),'mean_arm_reward':float(np.mean(rew_a))}
  state.update(stage1_curriculum_level=env.stage1_curriculum_level,guidance_cap_deg=env.guidance_cap_deg,last_curriculum_change_iteration=env.last_curriculum_change_iteration)
  if active_amp:
   state['amp']=amp_stats
   state['amp_updates']=amp_updates
   for key,value in amp_stats.items():
    writer.add_scalar('amp/'+key,value,i)
  state['optimization']={**{'loco/'+k:float(v) for k,v in loss_l.items()},**{'arm/'+k:float(v) for k,v in loss_a.items()}}
  state['policy_distribution']={'loco_std_min':float(loco.actor.distribution.log_std_param.exp().min()),'loco_std_max':float(loco.actor.distribution.log_std_param.exp().max()),'loco_std_mean':float(loco.actor.distribution.log_std_param.exp().mean()),'loco_raw_entropy':float(loco.actor.distribution.raw_entropy.mean()),'loco_action_clip_fraction':float(np.mean(loco_clips)),'arm_action_clip_fraction':float(np.mean(arm_clips))}
  tmp=run/'progress.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(run/'progress.json')
  for k,v in state['optimization'].items():writer.add_scalar('optimization/'+k,v,i)
  for k,v in state['policy_distribution'].items():writer.add_scalar('policy_distribution/'+k,v,i)
  for k,v in {**env.log_metrics,'range_scale':env.range_scale,'success_ema':env.success_ema,'stage':env.training_stage,'loco_reward':state['mean_leg_reward'],'arm_reward':state['mean_arm_reward']}.items():writer.add_scalar(k,v,i)
  if i%10==0 or args.smoke:print('TRAIN_PROGRESS',json.dumps(state),flush=True)
  if i%1000==0 or i==first_iteration:save(i)
 else:i=limit
 completed=i
 final_checkpoint=save(completed-1)
 final={'status':'smoke_passed' if args.smoke else status,'completed_iterations':completed,'checkpoint':str(final_checkpoint),'elapsed_wall_s':time.time()-t0,'training_stage':env.training_stage,'range_scale':env.range_scale,'arm_level':env.level,'success_ema':env.success_ema,'precision_success_ema':env.precision_success_ema,'loco_updates':loco_updates,'arm_updates':arm_updates,'loco_weights_changed':not torch.equal(start_loco,next(loco.actor.parameters())),'arm_weights_changed':not torch.equal(start_arm,next(arm.actor.parameters())),'acceptance_passed':False}
 final.update(resumed_from_iteration=resume.get('resumed_from_iteration'),first_training_iteration=first_iteration,updates_this_run=loco_updates-resume.get('inherited_loco_updates',0))
 final.update(stage1_curriculum_level=env.stage1_curriculum_level,guidance_cap_deg=env.guidance_cap_deg)
 if args.smoke:
  assert final['loco_weights_changed']
  if env.training_stage==2:
   assert final['arm_weights_changed'] and arm_updates>0
  else:
   assert not final['arm_weights_changed'] and arm_updates==0
  if amp is not None:
   assert amp_stats['valid_transitions']>0 and amp_stats['mean_style_reward']>0
   assert not torch.equal(start_discriminator,next(amp.discriminator.parameters()))
   final['amp']=amp_stats
   final['amp_foot_frame_max_error_m']=amp_foot_frame_error_m
  loaded=torch.load(final_checkpoint,map_location=args.device,weights_only=False)
  with torch.inference_mode():
   for name,alg,getter,dim in [('loco',loco,env.get_loco_observation,12),('arm',arm,env.get_arm_observation,8)]:
    obs=observations(getter());restored=make_ppo(obs,dim,args.device);restored.load(loaded[name],load_cfg=None,strict=True)
    expected=alg.actor(obs);actual=restored.actor(obs)
    assert torch.allclose(expected,actual,atol=1e-7),name+' restore mismatch'
  final['checkpoint_restore_actions_match']=True
 (run/'result.json').write_text(json.dumps(final,indent=2));writer.close();(run/'train.exitcode').write_text('0\n');print('RUN_FINISHED',json.dumps(final),flush=True)
except BaseException:
 (run/'train.exitcode').write_text('1\n');traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
sys.stdout.flush();sys.stderr.flush();os._exit(0)
