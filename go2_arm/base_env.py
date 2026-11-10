"""Whole-body reference tracking. Training collisions are failures, not safety guarantees.

Both public command modes resolve to the common checked joint trajectory in
motion.py. PPO learns balance and bounded residual corrections; it does not
replace IK, path feasibility checks, or the post-training acceptance tests.
"""
from pathlib import Path
import json,numpy as np,torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.envs import DirectRLEnv,DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat,quat_apply,quat_mul,subtract_frame_transforms
from robot_constants import DEFAULT_Q,JOINT_NAMES,EFFORT_LIMITS
from arm_fk import ArmFK
from curriculum import STAGES,sample_commands,tracking_terms,motion_gate
R=Path(__file__).resolve().parent
MOTOR_LIMITS={'[FR][LR]_(hip|thigh)_joint':23.7,'[FR][LR]_calf_joint':45.43,'joint[12]':20.,'joint3':15.,'joint4':7.,'joint[56]':5.,'finger_.*':10.}

@configclass
class DualModeCfg(DirectRLEnvCfg):
    episode_length_s=20.
    decimation=8
    action_space=19
    observation_space=300
    state_space=0
    sim=sim_utils.SimulationCfg(dt=.0025,render_interval=8,physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=.8,restitution=0.))
    scene=InteractiveSceneCfg(num_envs=4096,env_spacing=3.,replicate_physics=True)
    terrain=TerrainImporterCfg(prim_path='/World/ground',terrain_type='plane',collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=.8,restitution=0.))
    robot=ArticulationCfg(prim_path='/World/envs/env_.*/Robot',
        spawn=sim_utils.UsdFileCfg(usd_path=str(R/'robot/usd/robot.usd'),activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,solver_position_iteration_count=8,solver_velocity_iteration_count=2)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,.305),joint_pos=dict(zip(JOINT_NAMES,DEFAULT_Q[7:].tolist())),joint_vel={'.*':0.}),
        actuators={'motors':IdealPDActuatorCfg(joint_names_expr=['.*'],stiffness=0.,damping=0.,armature={'[FR][LR]_(hip|thigh|calf)_joint':.01,'joint[1-6]':.01,'finger_.*':.001},
            friction=.2,dynamic_friction=.2,viscous_friction={'[FR][LR]_(hip|thigh|calf)_joint':.1,'joint[1-6]':1.,'finger_.*':1.},
            effort_limit=MOTOR_LIMITS,effort_limit_sim=MOTOR_LIMITS,velocity_limit_sim={'[FR][LR]_(hip|thigh|calf)_joint':30.,'joint[1-6]':3.,'finger_.*':.2})})
    contact=ContactSensorCfg(prim_path='/World/envs/env_.*/Robot/.*',history_length=3,update_period=.0025,track_air_time=True)


class DualModeEnv(DirectRLEnv):
    cfg:DualModeCfg
    def _setup_scene(self):
        self.robot=Articulation(self.cfg.robot);self.scene.articulations['robot']=self.robot
        self.contact=ContactSensor(self.cfg.contact);self.scene.sensors['contact']=self.contact
        self.cfg.terrain.num_envs=self.scene.cfg.num_envs;self.cfg.terrain.env_spacing=self.scene.cfg.env_spacing
        self.terrain=self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device=='cpu':self.scene.filter_collisions(global_prim_paths=['/World/ground'])

    def __init__(self,cfg,**kwargs):
        super().__init__(cfg,**kwargs)
        dev=self.device;n=self.num_envs
        self.ids=torch.tensor([self.robot.joint_names.index(x) for x in JOINT_NAMES],device=dev)
        self.legs=self.ids[:12];self.arm=self.ids[12:18];self.fingers=self.ids[18:]
        self.mass_arm=self.arm+(0 if self.robot.is_fixed_base else 6)
        self.mass_fingers=self.fingers+(0 if self.robot.is_fixed_base else 6)
        expected_dofs=20+(0 if self.robot.is_fixed_base else 6)
        assert self.robot.root_physx_view.get_gravity_compensation_forces().shape[1]==expected_dofs
        assert self.robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces().shape[1]==expected_dofs
        self.limits=torch.tensor(EFFORT_LIMITS,device=dev,dtype=torch.float32)
        self.nominal=self.robot.data.default_joint_pos.clone()
        self.robot.write_joint_friction_coefficient_to_sim(.2,.2,1.,joint_ids=self.arm)
        self.robot.write_joint_friction_coefficient_to_sim(.1,.1,1.,joint_ids=self.fingers)
        self.robot.write_joint_friction_coefficient_to_sim(.2,.2,.1,joint_ids=self.legs)
        gate=json.loads((R/'controller_bench_report.json').read_text())
        if gate['selected'] is None:raise RuntimeError('Controller gate has not passed; refusing training')
        selected=gate['selected']
        self.omega=torch.as_tensor(selected.get('joint_omega_rad_s',[selected['omega_rad_s']]*6),device=self.device,dtype=torch.float32)
        assert self.omega.shape==(6,) and bool(torch.isfinite(self.omega).all()) and bool((self.omega>0).all()),'Invalid per-joint arm bandwidth'
        self.zeta=selected['damping_ratio']
        bank=np.load(R/'path_bank.npz')
        self.bank=torch.tensor(bank['q'],device=dev,dtype=torch.float32)
        self.bank_time=torch.tensor(bank['duration'],device=dev,dtype=torch.float32)
        self.bank_len=torch.tensor(bank['length'],device=dev,dtype=torch.long)
        self.bank_level=torch.tensor(bank['level'],device=dev,dtype=torch.long)
        self.bank_pose=torch.tensor(bank['goal_pose_xyzw'],device=dev,dtype=torch.float32)
        self.home_pose=torch.tensor(bank['home_pose_xyzw'],device=dev,dtype=torch.float32)
        self.path=torch.zeros(n,device=dev,dtype=torch.long);self.segment=torch.zeros_like(self.path)
        self.arm_enabled=torch.zeros(n,device=dev,dtype=torch.bool)
        self.reverse=torch.zeros(n,device=dev,dtype=torch.bool);self.clock=torch.zeros(n,device=dev)
        self.goal=torch.zeros(n,6,device=dev);self.ref=torch.zeros(n,6,device=dev);self.ref_v=torch.zeros_like(self.ref);self.ref_a=torch.zeros_like(self.ref)
        self.actions=torch.zeros(n,19,device=dev);self.last_actions=torch.zeros_like(self.actions);self.filtered=torch.zeros_like(self.actions)
        self.commands=torch.zeros(n,4,device=dev);self.command_targets=torch.zeros_like(self.commands)
        self.command_age=torch.zeros(n,device=dev)
        self.command_period=torch.full((n,),5.,device=dev)
        self.grip=torch.full((n,1),.5,device=dev);self.grip_target=self.grip.clone()
        self.mode=torch.zeros(n,1,device=dev);self.obs_history=torch.zeros(n,3,100,device=dev)
        self.forbidden=[i for i,x in enumerate(self.contact.body_names) if not x.endswith('_calf')]
        self.feet=[self.contact.body_names.index(f'{x}_calf') for x in ['FL','FR','RL','RR']]
        self.link6=self.robot.body_names.index('link6');self.tcp_offset=torch.tensor([0,0,.13],device=dev).expand(n,3)
        self.tcp_rot=torch.tensor([2**-.5,0,-2**-.5,0],device=dev).expand(n,4)
        self.head_vertices=torch.tensor(np.load(R/'head_vertices.npy'),device=dev,dtype=torch.float32)
        self.fk=ArmFK(dev)
        self.curriculum=0;self.range_scale=STAGES[0]['range_scale'];self.level=-1;self.success_ema=0.;self.precision_success_ema=0.;self.control_steps=0;self.last_advance=0
        self.track_sums=torch.zeros(n,3,device=dev);self.track_counts=torch.zeros(n,device=dev)
        self.gain_sums=torch.zeros(n,3,device=dev);self.active_counts=torch.zeros(n,3,device=dev)
        self.sums=torch.zeros(n,14,device=dev);self.steps=torch.zeros(n,device=dev)
        self.exceed=torch.zeros(n,8,device=dev)
        self.last_goal_sign=torch.zeros(n,6,device=dev);self.goal_crossings=torch.zeros(n,6,device=dev)
        self.oscillation_seen=torch.zeros(n,device=dev,dtype=torch.bool)
        self.overshoot_seen=torch.zeros(n,device=dev,dtype=torch.bool)
        self.last_tau=torch.zeros(n,20,device=dev);self.bad_contact=torch.zeros(n,device=dev,dtype=torch.bool)

    def _choose_path(self,env_ids):
        available=torch.where(self.bank_level<=max(0,self.level))[0]
        chosen=available[torch.randint(len(available),(len(env_ids),),device=self.device)]
        self.path[env_ids]=chosen;self.segment[env_ids]=0;self.reverse[env_ids]=False;self.clock[env_ids]=0
        self.arm_enabled[env_ids]=self.level>=0
        self.goal[env_ids]=self.bank[chosen,self.bank_len[chosen]-1]
        if self.level<0:self.goal[env_ids]=self.bank[chosen,0]

    def _reference(self):
        length=self.bank_len[self.path]
        seg=self.segment.clamp_min(0)
        i=torch.where(self.reverse,length-1-seg,seg);j=torch.where(self.reverse,i-1,i+1)
        q0=self.bank[self.path,i];q1=self.bank[self.path,j]
        home=self.bank[self.path,0]
        q0=torch.where(self.arm_enabled[:,None],q0,home);q1=torch.where(self.arm_enabled[:,None],q1,home)
        T=self.bank_time[self.path,torch.minimum(i,j)].clamp_min(.1)
        u=(self.clock/T).clamp(0,1);dq=q1-q0
        s=10*u**3-15*u**4+6*u**5
        sv=(30*u**2-60*u**3+30*u**4)/T
        sa=(60*u-180*u**2+120*u**3)/T**2
        self.ref=q0+dq*s[:,None];self.ref_v=dq*sv[:,None];self.ref_a=dq*sa[:,None]
        self.segment_goal=q1;self.segment_direction=torch.sign(dq);self.segment_duration=T
        self.segment_moved=dq.abs()>np.deg2rad(1.)
        return T

    def _pre_physics_step(self,actions):
        self.last_actions.copy_(self.actions);self.actions=actions.clamp(-1,1)
        self.filtered+=.2*(self.actions-self.filtered)
        self.command_age+=self.step_dt
        change=torch.where(self.command_age>self.command_period)[0]
        if len(change):self._sample_base_commands(change)
        slew=torch.tensor([1.,1.,1.,.3],device=self.device)*self.step_dt
        self.commands+=torch.clamp(self.command_targets-self.commands,-slew,slew)
        self.grip+=(self.grip_target-self.grip).clamp(-.01,.01)
        # A mode-only event never changes the planned reference.
        flip=torch.rand(self.num_envs,1,device=self.device)<.005
        self.mode=torch.where(flip,1-self.mode,self.mode)
        T=self._reference();done=torch.where(self.clock>T+1.25)[0]
        if len(done):
            self.last_goal_sign[done]=0;self.goal_crossings[done]=0
            self.clock[done]=0;self.segment[done]+=1
            end=done[self.segment[done]>=self.bank_len[self.path[done]]-1]
            returned=end[self.reverse[end]];outward=end[~self.reverse[end]]
            self.reverse[outward]=True;self.segment[outward]=0
            self.goal[outward]=self.bank[self.path[outward],0]
            if len(returned):self._choose_path(returned)
        self.control_steps+=1

    def _apply_action(self):
        self._reference();q=self.robot.data.joint_pos;v=self.robot.data.joint_vel
        M=self.robot.root_physx_view.get_generalized_mass_matrices()[:,self.mass_arm,:][:,:,self.mass_arm]
        M=M+torch.diag_embed(self.robot.data.joint_armature[:,self.arm])
        gravity=self.robot.root_physx_view.get_gravity_compensation_forces()
        coriolis=self.robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()
        effort=torch.zeros_like(q)
        target=self.nominal[:,self.legs]+.35*self.filtered[:,:12]
        effort[:,self.legs]=45*(target-q[:,self.legs])-1.5*v[:,self.legs]
        residual=(.003 if self.level>0 else 0.)*self.filtered[:,12:18]
        acc=self.ref_a+self.omega**2*(self.ref+residual-q[:,self.arm])+2*self.zeta*self.omega*(self.ref_v-v[:,self.arm])
        effort[:,self.arm]=torch.bmm(M,acc[:,:,None]).squeeze(-1)+gravity[:,self.mass_arm]+coriolis[:,self.mass_arm]+self.ref_v+.2*torch.tanh(self.ref_v/.02)
        gap=(.035*self.grip+.001*self.filtered[:,18:19]).clamp(0,.035)
        effort[:,self.fingers]=300*(gap-q[:,self.fingers])-5*v[:,self.fingers]+gravity[:,self.mass_fingers]
        self.last_tau=torch.clamp(effort[:,self.ids],-self.limits,self.limits)
        effort[:,self.ids]=self.last_tau;self.robot.set_joint_effort_target(effort)
        self.clock+=self.physics_dt

    def _metrics(self):
        q=self.robot.data.root_quat_w;Rmat=matrix_from_quat(q)
        pitch=torch.asin((-Rmat[:,2,0]).clamp(-1,1))
        err_v=self.robot.data.root_lin_vel_b[:,:2]-self.commands[:,:2]
        err_w=self.robot.data.root_ang_vel_b[:,2]-self.commands[:,2]
        err_p=pitch-self.commands[:,3]
        arm_e=self.robot.data.joint_pos[:,self.arm]-self.ref
        target_p,target_R=self.fk(self.ref)
        ee_R=matrix_from_quat(self.robot.data.body_quat_w[:,self.link6])
        ee_p=self.robot.data.body_pos_w[:,self.link6]+quat_apply(self.robot.data.body_quat_w[:,self.link6],self.tcp_offset)
        actual_p=torch.bmm(Rmat.transpose(1,2),(ee_p-self.robot.data.root_pos_w)[:,:,None]).squeeze(-1)
        actual_R=Rmat.transpose(1,2)@ee_R@self.fk.tcp_R
        self.tcp_pos_error=(actual_p-target_p).norm(dim=-1)
        relative=target_R.transpose(1,2)@actual_R
        self.tcp_rot_error=torch.acos(((relative.diagonal(dim1=1,dim2=2).sum(-1)-1)/2).clamp(-1,1))
        forces=self.contact.data.net_forces_w_history.norm(dim=-1).max(dim=1).values
        self.bad_contact=(forces[:,self.forbidden]>.5).any(-1)
        head=self.robot.data.root_pos_w[:,2]+(Rmat[:,2]@self.head_vertices.T).min(-1).values
        return err_v,err_w,err_p,arm_e,head,forces

    def _get_rewards(self):
        ev,ew,ep,ea,head,force=self._metrics()
        actual=torch.cat([self.robot.data.root_lin_vel_b[:,:2],self.robot.data.root_ang_vel_b[:,2:3]],-1)
        normalized,progress,stalled,active=tracking_terms(actual,self.commands[:,:3])
        r=4*torch.exp(-.5*normalized[:,:2].square().sum(-1))+2*torch.exp(-.5*normalized[:,2].square())
        r+=1.5*progress-2*stalled.float()+torch.exp(-ep.square()/np.deg2rad(3)**2)
        arm_weight=.10 if self.curriculum==0 else (0.5 if self.curriculum==1 else 1.)
        r+=arm_weight*(1.5*torch.exp(-ea.square().mean(-1)/np.deg2rad(1)**2)
            +torch.exp(-self.tcp_pos_error.square()/.01**2)+.5*torch.exp(-self.tcp_rot_error.square()/np.deg2rad(2)**2))
        overshoot=((self.robot.data.joint_pos[:,self.arm]-self.segment_goal)*self.segment_direction-np.deg2rad(.1)).clamp_min(0)*self.segment_moved
        r-=100*overshoot.square().sum(-1)
        self.overshoot_seen|=(overshoot>0).any(-1)
        goal_error=self.robot.data.joint_pos[:,self.arm]-self.segment_goal
        # Only count unwanted settled crossings; planned movement/unchanged axes
        # are not ringing. The trajectory now has a real post-settle hold window.
        eligible=self.segment_moved & (self.clock>self.segment_duration+.75)[:,None]
        sign=torch.where((goal_error.abs()>np.deg2rad(.05))&eligible,goal_error.sign(),0.)
        changed=(sign!=0)&(self.last_goal_sign!=0)&(sign!=self.last_goal_sign)
        self.goal_crossings+=changed.float();self.last_goal_sign=torch.where(sign!=0,sign,self.last_goal_sign)
        self.oscillation_seen|=(self.goal_crossings>1).any(-1)
        r-=.05*self.robot.data.joint_vel[:,self.arm].square().sum(-1)*(self.clock>self.segment_duration+.1)
        r+=.2
        r-=.5*self.robot.data.root_lin_vel_b[:,2].square()+.05*self.robot.data.root_ang_vel_b[:,:2].square().sum(-1)
        r-=.03*(self.actions-self.last_actions).square().sum(-1)
        r-=.00002*self.last_tau.square().sum(-1)
        r-=.8*self.robot.data.projected_gravity_b[:,1].square()
        r-=5*self.bad_contact.float()
        feet_contact=force[:,self.feet]>1
        first=self.contact.compute_first_contact(self.step_dt)[:,self.feet]
        air=self.contact.data.last_air_time[:,self.feet]
        r+=.5*((air-.08).clamp(0,.4)*first).sum(-1)*(self.commands[:,:3].norm(dim=-1)>.05)
        r-=.5*(self.robot.data.joint_vel[:,self.legs].square().sum(-1))*(self.commands[:,:3].norm(dim=-1)<.03)*.01
        metrics=torch.cat([ev.square(),ew.square()[:,None],ep.square()[:,None],ea.square(),self.bad_contact.float()[:,None],(head<.02).float()[:,None],self.tcp_pos_error.square()[:,None],self.tcp_rot_error.square()[:,None]],-1)
        self.sums+=metrics;self.steps+=1
        self.exceed+=torch.cat([(ea.abs()>np.deg2rad(1)).float(),(self.tcp_pos_error>.02).float()[:,None],(self.tcp_rot_error>np.deg2rad(3)).float()[:,None]],-1)
        valid=(self.command_age>1.)&(self.steps>50)
        self.track_sums+=normalized.square()*valid[:,None];self.track_counts+=valid.float()
        active_valid=active&valid[:,None]
        gain=actual*self.commands[:,:3]/(self.commands[:,:3].square()+1e-8)
        self.gain_sums+=gain*active_valid;self.active_counts+=active_valid.float()
        return r*self.step_dt

    def _get_dones(self):
        _,_,_,_,head,_=self._metrics()
        fallen=(self.robot.data.projected_gravity_b[:,2]>-.5)|(self.robot.data.root_pos_w[:,2]<.16)
        return fallen|self.bad_contact|(head<.02),self.episode_length_buf>=self.max_episode_length-1

    def _get_observations(self):
        pose=torch.where(self.reverse[:,None],self.home_pose,self.bank_pose[self.path])
        pose=torch.where(self.arm_enabled[:,None],pose,self.home_pose)
        obs=torch.cat([self.robot.data.root_lin_vel_b,self.robot.data.root_ang_vel_b,self.robot.data.projected_gravity_b,
            self.robot.data.joint_pos[:,self.ids]-self.nominal[:,self.ids],self.robot.data.joint_vel[:,self.ids]*.1,
            self.filtered,self.commands,self.ref-self.robot.data.joint_pos[:,self.arm],self.ref_v,
            self.goal-self.robot.data.joint_pos[:,self.arm],pose,self.grip,torch.ones_like(self.mode),(self.clock/5).clamp(0,1)[:,None]],-1)
        assert obs.shape[1]==100,obs.shape
        self.obs_history=torch.roll(self.obs_history,-1,dims=1);self.obs_history[:,-1]=obs
        return {'policy':self.obs_history.flatten(1)}

    def _reset_idx(self,env_ids):
        if env_ids is None:env_ids=self.robot._ALL_INDICES
        if len(env_ids)==0:return
        if hasattr(self,'sums'):
            count=self.steps[env_ids].clamp_min(1);rmse=(self.sums[env_ids,:10]/count[:,None]).sqrt()
            tcp_rmse=(self.sums[env_ids,12:14]/count[:,None]).sqrt()
            success=(count>=self.max_episode_length*.98)&(rmse[:,:3]<.1).all(-1)&(rmse[:,3]<np.deg2rad(3))&(rmse[:,4:]<np.deg2rad(.5)).all(-1)&(self.sums[env_ids,10:12]==0).all(-1)&(tcp_rmse[:,0]<.01)&(tcp_rmse[:,1]<np.deg2rad(2))
            success&=(self.exceed[env_ids]/count[:,None]<=.05).all(-1)&~self.oscillation_seen[env_ids]&~self.overshoot_seen[env_ids]
            normalized_rmse=(self.track_sums[env_ids]/self.track_counts[env_ids,None].clamp_min(1)).sqrt()
            directional_gain=self.gain_sums[env_ids]/self.active_counts[env_ids].clamp_min(1)
            response_ok=motion_gate(normalized_rmse,directional_gain,self.active_counts[env_ids],self.track_counts[env_ids])
            complete=count>=self.max_episode_length*.98
            contact_free=(self.sums[env_ids,10:12]==0).all(-1)
            promotion=complete&contact_free&response_ok&(rmse[:,3]<np.deg2rad(3))
            if self.curriculum>0:promotion&=(rmse[:,4:]<np.deg2rad(.5)).all(-1)&(tcp_rmse[:,0]<.01)
            # Course promotion is separate from final precision/no-oscillation acceptance.
            success&=response_ok
            decay=float(np.exp(-len(env_ids)/max(1,self.num_envs)))
            self.success_ema=decay*self.success_ema+(1-decay)*promotion.float().mean().item()
            self.precision_success_ema=decay*self.precision_success_ema+(1-decay)*success.float().mean().item()
            if self.control_steps-self.last_advance>12000 and self.success_ema>.70 and self.curriculum<len(STAGES)-1:
                self.curriculum+=1;stage=STAGES[self.curriculum];self.range_scale=stage['range_scale'];self.level=stage['arm_level'];self.last_advance=self.control_steps;self.success_ema=0.
                print('CURRICULUM_ADVANCE',self.curriculum,self.range_scale,self.level,flush=True)
            self.extras['log']={'Tracking/vx_rmse':rmse[:,0].mean(),'Tracking/vy_rmse':rmse[:,1].mean(),'Tracking/wz_rmse':rmse[:,2].mean(),'Tracking/pitch_rmse_deg':rmse[:,3].mean()*180/np.pi,'Curriculum/range_scale':self.range_scale,'Curriculum/arm_level':self.level,'Curriculum/success_ema':self.success_ema,'Failures/contact_fraction':self.sums[env_ids,10].gt(0).float().mean()}
            for j in range(6):self.extras['log'][f'Tracking/joint{j+1}_rmse_deg']=rmse[:,4+j].mean()*180/np.pi
            self.extras['log']['Tracking/tcp_position_rmse_m']=tcp_rmse[:,0].mean();self.extras['log']['Tracking/tcp_orientation_rmse_deg']=tcp_rmse[:,1].mean()*180/np.pi
            self.extras['log']['Failures/arm_ringing_fraction']=self.oscillation_seen[env_ids].float().mean();self.extras['log']['Failures/arm_overshoot_fraction']=self.overshoot_seen[env_ids].float().mean()
            self.extras['log'].update({'Curriculum/stage':self.curriculum,'Curriculum/precision_success_ema':self.precision_success_ema,'Tracking/normalized_rmse':normalized_rmse.mean(),'Tracking/response_pass_fraction':response_ok.float().mean(),'Tracking/moving_directional_gain':directional_gain[self.active_counts[env_ids]>=50].mean() if (self.active_counts[env_ids]>=50).any() else 0.})
        self.robot.reset(env_ids);super()._reset_idx(env_ids)
        root=self.robot.data.default_root_state[env_ids].clone();root[:,:3]+=self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root[:,:7],env_ids);self.robot.write_root_velocity_to_sim(root[:,7:],env_ids)
        self.robot.write_joint_state_to_sim(self.nominal[env_ids],torch.zeros_like(self.nominal[env_ids]),env_ids=env_ids)
        self._choose_path(env_ids);self.ref[env_ids]=self.bank[self.path[env_ids],0];self.ref_v[env_ids]=0;self.ref_a[env_ids]=0
        self.actions[env_ids]=0;self.last_actions[env_ids]=0;self.filtered[env_ids]=0;self.obs_history[env_ids]=0
        self._sample_base_commands(env_ids)
        self.commands[env_ids]=0;self.grip[env_ids]=.5;self.grip_target[env_ids]=torch.rand(len(env_ids),1,device=self.device)
        self.mode[env_ids]=torch.randint(0,2,(len(env_ids),1),device=self.device).float();self.sums[env_ids]=0;self.steps[env_ids]=0
        self.exceed[env_ids]=0;self.last_goal_sign[env_ids]=0;self.goal_crossings[env_ids]=0;self.oscillation_seen[env_ids]=False;self.overshoot_seen[env_ids]=False
        self.track_sums[env_ids]=0;self.track_counts[env_ids]=0;self.gain_sums[env_ids]=0;self.active_counts[env_ids]=0

    def _sample_base_commands(self,env_ids):
        self.command_targets[env_ids]=sample_commands(len(env_ids),self.curriculum,self.device)
        self.command_age[env_ids]=0;self.command_period[env_ids]=torch.rand(len(env_ids),device=self.device)*4+4
