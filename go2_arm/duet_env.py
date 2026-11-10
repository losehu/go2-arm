"""RoboDuet-inspired leg/arm cooperation on the validated Go2/Piper asset.

Physical dynamics and motor limits are inherited; separate policies/rewards,
heading-frame pose commands, gait phase, and internal attitude guidance are new.
Training contact termination is detection, not a collision-avoidance proof.
"""
import torch,numpy as np
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import matrix_from_quat
from base_env import DualModeEnv,DualModeCfg,R
from base_commands import apply_velocity_deadzone
from curriculum import tracking_terms,motion_gate
from stage1_schedule import STAGE1_LEVELS
from reference_filter import ReferenceFilter
from posture_objective import posture_objective
from tracking_objective import tracking_error_objective
from foot_gait import coordinated_trot_terms, gait_terms, planted_foot_cost, trot_frequency, trot_placement_time
from hip_splay import hip_splay
from air_time import AirTimeReward
from base_height_objective import BASE_HEIGHT_SPEC,low_base_cost
from directional_amp import guide_mixed_targets
from geometry import rpy_matrix,rot6d,rotation_error,base_to_heading,heading_to_base,damped_ik

@configclass
class DuetCfg(DualModeCfg):
    hip_splay_penalty: bool = False
    coordinated_trot: bool = False
    phase_free_amp: bool = False
    directional_amp: bool = False

class DuetEnv(DualModeEnv):
    def __init__(self,cfg,**kwargs):
        super().__init__(cfg,**kwargs)
        n=self.num_envs;d=self.device
        self.training_stage=1;self.training_iteration=0;self.range_scale=.5;self.level=-1;self.curriculum=0
        self.stage1_curriculum_level=0;self.guidance_cap_deg=3.;self.last_curriculum_change_iteration=0
        self.guidance=torch.zeros(n,2,device=d);self.guidance_target=torch.zeros_like(self.guidance)
        self.arm_action=torch.zeros(n,8,device=d);self.last_arm_action=torch.zeros_like(self.arm_action)
        if not cfg.phase_free_amp:
            self.phase=torch.zeros(n,device=d)
        self.walk_blend=torch.zeros(n,device=d)
        self.planned_attitude=torch.zeros(n,2,device=d)
        wb=np.load(R/'whole_body_bank.npz')
        for attribute,key,dtype in [('bank','q',torch.float32),('bank_time','duration',torch.float32),('bank_len','length',torch.long),('bank_level','level',torch.long),('bank_pose','goal_pose_xyzw',torch.float32)]:
            setattr(self,attribute,torch.tensor(wb[key],device=d,dtype=dtype))
        self.attitude_bank=torch.tensor(wb['attitude_waypoints'],device=d,dtype=torch.float32)
        self.ik_goal=self.nominal[:,self.arm].clone();self.smooth_q=self.ik_goal.clone();self.smooth_v=torch.zeros_like(self.ik_goal);self.smooth_a=torch.zeros_like(self.ik_goal)
        self.reference_filter=ReferenceFilter(self.ik_goal,self.physics_dt)
        self.arm_limits=self.robot.data.soft_joint_pos_limits[0,self.arm].clone()
        self.duet_sums=torch.zeros(n,9,device=d);self.roll_sums=torch.zeros(n,device=d)
        self.loco_reward=torch.zeros(n,device=d);self.arm_reward=torch.zeros(n,device=d)
        self.mode_switches=0;self.external_control=False;self.geometry_gate=None
        self.foot_stats=torch.zeros(n,9,device=d)
        if cfg.coordinated_trot or cfg.phase_free_amp:
            self.trot_stats=torch.zeros(n,6,device=d)
        if cfg.directional_amp:
            self.air_time_reward=AirTimeReward(n,d)
            self.air_time_sums=torch.zeros(n,device=d)
            self.base_height_sums=torch.zeros(n,2,device=d)
        if cfg.hip_splay_penalty:
            self.hip_splay_stats=torch.zeros(n,9,device=d)
        self.acceptance={'whole_body':False};self.stage1_pass_ema=0.;self.log_metrics={}
        self._trajectory_target()

    def set_stage1_curriculum(self,level):
        assert self.training_stage==1 and 0<=level<len(STAGE1_LEVELS)
        self.stage1_curriculum_level=level
        self.range_scale,self.guidance_cap_deg=STAGE1_LEVELS[level]
        self.last_curriculum_change_iteration=self.training_iteration
        self.reset()
        # The old episodes must not contribute to the new curriculum gate.
        self.success_ema=0.;self.precision_success_ema=0.;self.log_metrics={}

    def set_training_stage(self,stage):
        assert stage in (1,2)
        self.training_stage=stage;self.curriculum=0 if stage==1 else 1
        self.level=-1 if stage==1 else 0
        self.success_ema=0.;self.precision_success_ema=0.
        if stage==2:self.range_scale=1.;self.guidance_cap_deg=15.
        self.last_curriculum_change_iteration=self.training_iteration
        self.reset()

    def _choose_path(self,ids):
        super()._choose_path(ids)
        if hasattr(self,'planned_attitude'):
            self.planned_attitude[ids]=0
            self.mode[ids]=torch.randint(0,2,(len(ids),1),device=self.device).float()

    def _sample_base_commands(self,ids):
        super()._sample_base_commands(ids)
        if hasattr(self,'guidance_target'):
            # Balanced explicit moving commands; zero remains a separate command.
            caps=torch.tensor([1.,.6,.8],device=self.device)*self.range_scale
            axis=torch.randint(0,3,(len(ids),),device=self.device)
            c=torch.zeros(len(ids),3,device=self.device)
            low=.5*caps
            if self.cfg.directional_amp:low[1]=.12
            magnitude=low[axis]+(caps[axis]-low[axis])*torch.rand(len(ids),device=self.device)
            c[torch.arange(len(ids),device=self.device),axis]=magnitude*torch.where(torch.rand(len(ids),device=self.device)>.5,1.,-1.)
            mixed=torch.rand(len(ids),device=self.device)<.25
            c[mixed]+=(2*torch.rand(int(mixed.sum()),3,device=self.device)-1)*caps*.25
            c=c.clamp(-caps,caps);c[torch.rand(len(ids),device=self.device)<.1]=0
            c=apply_velocity_deadzone(c)
            if self.cfg.directional_amp and hasattr(self,'amp_command_bank'):
                scale, reference, signatures=self.amp_command_bank
                c=scale*guide_mixed_targets(c/scale,caps/scale,reference,signatures)
            self.command_targets[ids,:3]=c;self.command_targets[ids,3]=0
            if self.training_stage==1:
                cap=np.deg2rad(self.guidance_cap_deg)
                self.guidance_target[ids]=(2*torch.rand(len(ids),2,device=self.device)-1)*cap

    def prepare_arm(self,action):
        """Called before the leg observation; arm actions guide this same step."""
        self.last_arm_action.copy_(self.arm_action);self.arm_action=action.clamp(-1,1)
        if self.training_stage==2:
            cap=np.deg2rad(15.)
            self.guidance_target=self.arm_action[:,-2:]*cap
        self.guidance+=(self.guidance_target-self.guidance).clamp(-.3*self.step_dt,.3*self.step_dt)
        self.commands[:,3]=self.guidance[:,0]

    def _trajectory_target(self):
        super()._reference()
        self.path_ref=self.ref.clone();self.path_v=self.ref_v.clone();self.path_a=self.ref_a.clone()
        p,R=self.fk(self.path_ref)
        if hasattr(self,'attitude_bank'):
            length=self.bank_len[self.path];seg=self.segment.clamp_min(0)
            i=torch.where(self.reverse,length-1-seg,seg);j=torch.where(self.reverse,i-1,i+1)
            u=(self.clock/self.segment_duration).clamp(0,1);u=10*u**3-15*u**4+6*u**5
            self.planned_attitude=self.attitude_bank[self.path,i]+u[:,None]*(self.attitude_bank[self.path,j]-self.attitude_bank[self.path,i])
            if self.training_stage==1:self.planned_attitude.zero_()
            A=rpy_matrix(self.planned_attitude[:,1],self.planned_attitude[:,0],torch.zeros(self.num_envs,device=self.device))
            self.target_p=(A@p[:,:,None]).squeeze(-1);self.target_R=A@R
        else:self.target_p=p;self.target_R=R

    def _reference(self):
        if not hasattr(self,'smooth_q'):return super()._reference()
        self._trajectory_target()
        if self.training_stage==2:
            tcp=self.mode.bool()
            residual=.003*self.filtered[:,12:18]
            self.ref=torch.where(tcp,self.smooth_q,self.path_ref+residual)
            self.ref_v=torch.where(tcp,self.smooth_v,self.path_v)
            self.ref_a=torch.where(tcp,self.smooth_a,self.path_a)
        return self.segment_duration

    def _pre_physics_step(self,actions):
        # Preserve PPO's unclipped actions in its own buffer; clip at actuator interface.
        self.last_actions.copy_(self.actions);self.actions=actions.clamp(-1,1)
        self.filtered+=.2*(self.actions-self.filtered)
        self.command_age+=self.step_dt
        ids=torch.where(self.command_age>self.command_period)[0]
        if len(ids):self._sample_base_commands(ids)
        self._advance_base_command()
        self.commands[:,3]=self.guidance[:,0]
        self.grip+=(self.grip_target-self.grip).clamp(-.01,.01)
        self.walk_blend+=(self.commands[:,:3].norm(dim=-1).gt(.04).float()-self.walk_blend).clamp(-.04,.04)
        if not self.cfg.phase_free_amp:
            cadence=trot_frequency(self.commands) if self.cfg.coordinated_trot else 3.
            self.phase=(self.phase+cadence*self.step_dt*self.walk_blend)%1.
        T=self._reference()
        if not self.external_control:
            done=torch.where(self.clock>T+1.25)[0]
            if len(done):
                self.last_goal_sign[done]=0;self.goal_crossings[done]=0
                self.clock[done]=0;self.segment[done]+=1
                end=done[self.segment[done]>=self.bank_len[self.path[done]]-1]
                returned=end[self.reverse[end]];outward=end[~self.reverse[end]]
                self.reverse[outward]=True;self.segment[outward]=0;self.goal[outward]=self.bank[self.path[outward],0]
                if len(returned):self._choose_path(returned)
        self._reference()
        if self.training_stage==2:
            base_R=matrix_from_quat(self.robot.data.root_quat_w)
            p,R=heading_to_base(self.target_p,self.target_R,base_R,self.robot.data.root_pos_w[:,2])
            self.ik_goal=damped_ik(self.fk,self.smooth_q,p,R,self.arm_limits,steps=3)
            self.ik_goal=(self.ik_goal+.003*self.filtered[:,12:18]).clamp(self.arm_limits[:,0],self.arm_limits[:,1])
        self.control_steps+=1

    def _apply_action(self):
        if self.training_stage==2:
            # Bounded acceleration/jerk reference generator, identical in inference.
            dt=self.physics_dt
            self.smooth_q,self.smooth_v,self.smooth_a=self.reference_filter.advance(self.ik_goal)
        # Arm residual is already in the reference; avoid a second addition.
        saved_level=self.level;self.level=0
        saved_grip=self.filtered[:,18].clone();self.filtered[:,18]=0
        super()._apply_action()
        self.filtered[:,18]=saved_grip;self.level=saved_level

    def _metrics(self):
        ev,ew,ep,ea,head,force=super()._metrics()
        if hasattr(self,'guidance'):
            R=matrix_from_quat(self.robot.data.root_quat_w)
            self.actual_roll=torch.atan2(R[:,2,1],R[:,2,2]);self.actual_pitch=torch.asin((-R[:,2,0]).clamp(-1,1))
            self.roll_error=self.actual_roll-self.guidance[:,1]
            if self.training_stage==2:
                p,rot=self.fk(self.robot.data.joint_pos[:,self.arm])
                hp,hR=base_to_heading(p,rot,R,self.robot.data.root_pos_w[:,2])
                tcp=self.mode[:,0].bool()
                self.tcp_pos_error=torch.where(tcp,(hp-self.target_p).norm(dim=-1),self.tcp_pos_error)
                er=torch.acos((self.target_R.transpose(1,2)@hR).diagonal(dim1=1,dim2=2).sum(-1).sub(1).div(2).clamp(-1,1))
                self.tcp_rot_error=torch.where(tcp,er,self.tcp_rot_error)
        return ev,ew,ep,ea,head,force

    def get_loco_observation(self):
        terms=[self.robot.data.root_lin_vel_b,self.robot.data.root_ang_vel_b,self.robot.data.projected_gravity_b,
            self.robot.data.joint_pos[:,self.legs]-self.nominal[:,self.legs],.1*self.robot.data.joint_vel[:,self.legs],
            self.filtered[:,:12],self.commands[:,:3],self.guidance]
        if not self.cfg.phase_free_amp:
            phase=2*torch.pi*(self.phase[:,None]+self.phase.new_tensor([0.,.5,.5,0.]))
            terms.extend([phase.sin()*self.walk_blend[:,None],phase.cos()*self.walk_blend[:,None]])
        terms.extend([self.robot.data.joint_pos[:,self.arm],.1*self.robot.data.joint_vel[:,self.arm],
                      self.ref_v,self.walk_blend[:,None]])
        return torch.cat(terms,-1).clone()

    def get_arm_observation(self):
        self._reference()
        p,R=self.fk(self.robot.data.joint_pos[:,self.arm]);base_R=matrix_from_quat(self.robot.data.root_quat_w)
        hp,hR=base_to_heading(p,R,base_R,self.robot.data.root_pos_w[:,2])
        return torch.cat([self.robot.data.joint_pos[:,self.arm],.1*self.robot.data.joint_vel[:,self.arm],self.arm_action,
            self.robot.data.root_lin_vel_b,self.robot.data.root_ang_vel_b,self.robot.data.projected_gravity_b,
            self.commands[:,:3],self.guidance,self.target_p-hp,rot6d(self.target_R),rot6d(hR),
            self.path_ref-self.robot.data.joint_pos[:,self.arm],self.path_v,self.mode,self.grip],-1).clone()

    def _get_rewards(self):
        ev,ew,ep,ea,head,force=self._metrics()
        actual=torch.cat([self.robot.data.root_lin_vel_b[:,:2],self.robot.data.root_ang_vel_b[:,2:3]],-1)
        norm,progress,stalled,active=tracking_terms(actual,self.commands[:,:3])
        tracking=tracking_error_objective(norm)+1.5*progress-2*stalled.float()
        posture=posture_objective(ep,self.roll_error)
        foot_position,foot_velocity=self.foot_kinematics()
        foot_force=self.foot_contact_forces()
        foot_clearance=foot_position[:,:,2]-self.scene.env_origins[:,None,2]-.022
        walking=(self.walk_blend>.8)&~self.bad_contact
        if self.cfg.phase_free_amp:
            foot_contact, _, slip_cost=planted_foot_cost(foot_force,foot_clearance,foot_velocity)
            gait_reward=-slip_cost
            mid_swing=torch.zeros_like(foot_contact)
            diagonal_match=torch.zeros_like(walking)
        else:
            if self.cfg.coordinated_trot:
                gait,gait_cost,clearance_target,swing_phase,foot_contact=coordinated_trot_terms(
                    self.phase,self.walk_blend,foot_force,foot_clearance,foot_velocity)
            else:
                gait,clearance_cost,clearance_target,swing_phase=gait_terms(self.phase,self.walk_blend,foot_force,foot_clearance,foot_velocity)
                gait_cost=4.*clearance_cost
                foot_contact=foot_force>1.
            mid_swing=clearance_target>=.02
            expected_contact=~swing_phase.bool()
            diagonal_match=(foot_contact==expected_contact).all(-1)&walking
            # Keep the support polygon moving with the body; phase/command-driven foot placement.
            nominal_xy=foot_position.new_tensor([[0.24999998865751655, 0.19999997571297282], [0.24999998865751655, -0.19999997571297282], [-0.24999999846405746, 0.19999999666753684], [-0.24999999846405746, -0.19999999666753684]])
            root_R=matrix_from_quat(self.robot.data.root_quat_w)
            yaw=torch.atan2(root_R[:,1,0],root_R[:,0,0]);cos=yaw.cos()[:,None];sin=yaw.sin()[:,None]
            relative=foot_position[:,:,:2]-self.robot.data.root_pos_w[:,None,:2]
            heading_xy=torch.stack([cos*relative[:,:,0]+sin*relative[:,:,1],-sin*relative[:,:,0]+cos*relative[:,:,1]],dim=-1)
            if self.cfg.coordinated_trot:
                phase_offset=trot_placement_time(self.phase,self.commands)
            else:
                leg_phase=(self.phase[:,None]+self.phase.new_tensor([0.,.5,.5,0.]))%1.
                phase_offset=(torch.abs(1.-2.*leg_phase)-.5)*(0.5/3.)
            velocity_xy=self.commands[:,None,:2].expand(-1,4,-1).clone()
            velocity_xy[:,:,0]-=self.commands[:,2,None]*nominal_xy[None,:,1]
            velocity_xy[:,:,1]+=self.commands[:,2,None]*nominal_xy[None,:,0]
            desired_xy=nominal_xy[None]+phase_offset[:,:,None]*velocity_xy
            placement_cost=25.*(heading_xy-desired_xy).square().sum(dim=(1,2))*self.walk_blend
            gait_reward=6.*gait-gait_cost-placement_cost
        foot_sliding=foot_contact&(foot_velocity[:,:,:2].norm(dim=-1)>.1)
        pair_match=((foot_contact[:,0]==foot_contact[:,3])&(foot_contact[:,1]==foot_contact[:,2]))&walking
        if self.cfg.coordinated_trot or self.cfg.phase_free_amp:
            exclusive=pair_match&(foot_contact.sum(dim=-1)==2)
            self.trot_stats+=torch.cat([exclusive[:,None], (walking&(foot_contact.sum(dim=-1)==4))[:,None],
                                       foot_contact&walking[:,None]],dim=-1).float()
        self.foot_stats+=torch.stack([mid_swing.sum(-1),((foot_clearance<.01)&mid_swing).sum(-1),
            (foot_clearance*mid_swing).sum(-1),foot_contact.sum(-1),foot_sliding.sum(-1),
            self.robot.data.root_pos_w[:,2]-self.scene.env_origins[:,2],walking.float(),diagonal_match.float(),pair_match.float()],dim=-1)
        regular=-.03*(self.actions[:,:12]-self.last_actions[:,:12]).square().sum(-1)-.00002*self.last_tau[:,:12].square().sum(-1)
        regular-=.5*self.robot.data.root_lin_vel_b[:,2].square()+.05*self.robot.data.root_ang_vel_b[:,:2].square().sum(-1)
        collision=-8*self.bad_contact.float()-8*(head<.02).float()
        joint_tracking=torch.exp(-ea.square().mean(-1)/np.deg2rad(.5)**2)
        task=2*torch.exp(-self.tcp_pos_error.square()/.02**2)+torch.exp(-self.tcp_rot_error.square()/np.deg2rad(3)**2)
        planned_posture=torch.exp(-((self.actual_pitch-self.planned_attitude[:,0]).square()+(self.actual_roll-self.planned_attitude[:,1]).square())/np.deg2rad(4)**2)
        task=torch.where(self.mode[:,0].bool(),task,2*joint_tracking+task+.5*planned_posture)
        smooth=-.1*(self.arm_action-self.last_arm_action).square().sum(-1)-.00004*self.last_tau[:,12:18].square().sum(-1)
        overshoot=((self.robot.data.joint_pos[:,self.arm]-self.segment_goal)*self.segment_direction-np.deg2rad(.1)).clamp_min(0)*self.segment_moved*(~self.mode.bool())
        self.overshoot_seen|=(overshoot>0).any(-1)
        settled=(self.clock>self.segment_duration+.75)[:,None]
        err=self.robot.data.joint_pos[:,self.arm]-self.segment_goal
        eligible=self.segment_moved&settled&~self.mode.bool()
        sign=torch.where((err.abs()>np.deg2rad(.05))&eligible,err.sign(),0.)
        self.goal_crossings+=((sign!=0)&(self.last_goal_sign!=0)&(sign!=self.last_goal_sign)).float()
        self.last_goal_sign=torch.where(sign!=0,sign,self.last_goal_sign);self.oscillation_seen|=(self.goal_crossings>1).any(-1)
        smooth-=100*overshoot.square().sum(-1)+.05*self.robot.data.joint_vel[:,self.arm].square().sum(-1)*settled[:,0]
        # Stage-one locomotion must preserve the folded arm while learning to walk.
        arm_hold_cost=4.*torch.log1p((ea/np.deg2rad(.5)).square().mean(-1)) if self.training_stage==1 else 0.
        splay_cost=0.
        if self.cfg.hip_splay_penalty:
            splay_cost,hip_deg,hip_excess=hip_splay(self.robot.data.joint_pos[:,self.legs[::3]],self.commands)
            self.hip_splay_stats+=torch.cat([hip_deg,hip_excess.float(),splay_cost[:,None]],dim=-1)
        self.loco_reward=(tracking+posture+gait_reward-arm_hold_cost-splay_cost+regular+collision+.2+(.2*task if self.training_stage==2 else 0))*self.step_dt
        if self.cfg.directional_amp:
            move=(self.commands[:,:2].norm(dim=-1)>.06)&~self.bad_contact
            bonus=self.air_time_reward.step(foot_contact,foot_clearance,move,self.step_dt)
            height=self.robot.data.root_link_pos_w[:,2]-self.scene.env_origins[:,2]
            height_cost=low_base_cost(height)
            self.loco_reward+=bonus-height_cost*self.step_dt
            self.air_time_sums+=bonus
            self.base_height_sums+=torch.stack((height<BASE_HEIGHT_SPEC["lower_band_m"],height_cost),dim=-1)
        self.arm_reward=(task+posture+smooth+collision+.2*tracking)*self.step_dt
        self.steps+=1;self.roll_sums+=self.roll_error.square()
        self.sums+=torch.cat([ev.square(),ew.square()[:,None],ep.square()[:,None],ea.square(),self.bad_contact.float()[:,None],(head<.02).float()[:,None],self.tcp_pos_error.square()[:,None],self.tcp_rot_error.square()[:,None]],-1)
        valid=(self.command_age>1.)&(self.steps>50)
        self.track_sums+=norm.square()*valid[:,None];self.track_counts+=valid.float()
        act=active&valid[:,None];gain=actual*self.commands[:,:3]/(self.commands[:,:3].square()+1e-8)
        self.gain_sums+=gain*act;self.active_counts+=act.float()
        self.exceed+=torch.cat([(ea.abs()>np.deg2rad(1)).float(),(self.tcp_pos_error>.02).float()[:,None],(self.tcp_rot_error>np.deg2rad(3)).float()[:,None]],-1)
        return self.loco_reward

    def _reset_idx(self,ids):
        if ids is None:ids=self.robot._ALL_INDICES
        if len(ids)==0:return
        if hasattr(self,'guidance'):
            count=self.steps[ids].clamp_min(1);rmse=(self.sums[ids]/count[:,None]).sqrt();roll=(self.roll_sums[ids]/count).sqrt()
            normalized=(self.track_sums[ids]/self.track_counts[ids,None].clamp_min(1)).sqrt()
            gain=self.gain_sums[ids]/self.active_counts[ids].clamp_min(1)
            response=motion_gate(normalized,gain,self.active_counts[ids],self.track_counts[ids])
            # Only a normal time limit can complete an episode; physical failure wins ties.
            complete=(count>=self.max_episode_length*.98)&self.reset_time_outs[ids]&~self.reset_terminated[ids]
            passed=complete&response&(self.sums[ids,10:12]==0).all(-1)&(rmse[:,3]<np.deg2rad(3))&(roll<np.deg2rad(3))
            precision=passed&(rmse[:,4:10]<np.deg2rad(.5)).all(-1)&(rmse[:,12]<.01)&(rmse[:,13]<np.deg2rad(2))&~self.oscillation_seen[ids]&~self.overshoot_seen[ids]
            decay=float(np.exp(-len(ids)/self.num_envs));self.success_ema=decay*self.success_ema+(1-decay)*passed.float().mean().item()
            self.precision_success_ema=decay*self.precision_success_ema+(1-decay)*precision.float().mean().item()
            self.log_metrics={'vx_rmse':float(rmse[:,0].mean()),'vy_rmse':float(rmse[:,1].mean()),'wz_rmse':float(rmse[:,2].mean()),'pitch_rmse_deg':float(rmse[:,3].mean()*180/np.pi),'roll_rmse_deg':float(roll.mean()*180/np.pi),'tcp_rmse_m':float(rmse[:,12].mean()),'episode_steps':float(count.mean()),'response_pass_fraction':float(response.float().mean()),'contact_fraction':float((self.sums[ids,10]>0).float().mean()),'directional_gain':float(gain[self.active_counts[ids]>=50].mean()) if (self.active_counts[ids]>=50).any() else 0.}
            self.log_metrics.update(gate_complete_fraction=float(complete.float().mean()),gate_contact_free_fraction=float((self.sums[ids,10:12]==0).all(-1).float().mean()),gate_posture_pass_fraction=float(((rmse[:,3]<np.deg2rad(3))&(roll<np.deg2rad(3))).float().mean()),gate_pass_fraction=float(passed.float().mean()))
            self.log_metrics['calf_contact_episode_fraction']=float(self.calf_contact_seen[ids].float().mean())
            self.log_metrics.update(physical_termination_fraction=float(self.reset_terminated[ids].float().mean()),timeout_fraction=float(self.reset_time_outs[ids].float().mean()))
            foot=self.foot_stats[ids].sum(0)
            self.log_metrics.update(foot_contact_sliding_fraction=float(foot[4]/foot[3].clamp_min(1)),
                foot_contact_samples=int(foot[3]),root_height_mean_m=float(foot[5]/count.sum().clamp_min(1)))
            if not self.cfg.phase_free_amp:
                self.log_metrics.update(mid_swing_below_1cm_fraction=float(foot[1]/foot[0].clamp_min(1)),
                    mid_swing_clearance_mean_m=float(foot[2]/foot[0].clamp_min(1)),foot_mid_swing_samples=int(foot[0]),
                    diagonal_phase_contact_fraction=float(foot[7]/foot[6].clamp_min(1)))
            self.foot_stats[ids]=0
            self.log_metrics.update(tcp_orientation_rmse_deg=float(rmse[:,13].mean()*180/np.pi),arm_ringing_fraction=float(self.oscillation_seen[ids].float().mean()),arm_overshoot_fraction=float(self.overshoot_seen[ids].float().mean()),reset_batch_size=len(ids))
            for j in range(6):
                self.log_metrics[f'joint{j+1}_rmse_deg']=float(rmse[:,4+j].mean()*180/np.pi)
                self.log_metrics[f'joint{j+1}_over_1deg_fraction']=float((self.exceed[ids,j]/count).mean())
            self.log_metrics['tcp_over_2cm_fraction']=float((self.exceed[ids,6]/count).mean())
            self.log_metrics['orientation_over_3deg_fraction']=float((self.exceed[ids,7]/count).mean())
            self.log_metrics.update(diagonal_pair_sync_fraction=float(foot[8]/foot[6].clamp_min(1)),gait_walking_samples=int(foot[6]))
            if self.cfg.coordinated_trot or self.cfg.phase_free_amp:
                gait=(self.trot_stats[ids].sum(0)/foot[6].clamp_min(1)).tolist()
                self.log_metrics.update(exclusive_diagonal_support_fraction=gait[0],all_four_contact_fraction=gait[1])
                for j,leg in enumerate(('fl','fr','rl','rr')):
                    self.log_metrics[f'{leg}_walking_contact_fraction']=gait[2+j]
                self.trot_stats[ids]=0
            if self.cfg.hip_splay_penalty:
                hip=(self.hip_splay_stats[ids].sum(0)/count.sum()).tolist()
                for j,leg in enumerate(('fl','fr','rl','rr')):
                    self.log_metrics[f'{leg}_hip_outward_mean_deg']=hip[j]
                    self.log_metrics[f'{leg}_hip_excess_fraction']=hip[4+j]
                self.log_metrics['hip_splay_cost_mean']=hip[8]
                self.hip_splay_stats[ids]=0
            if self.cfg.directional_amp:
                self.log_metrics['air_time_reward_per_step']=float(self.air_time_sums[ids].sum()/count.sum().clamp_min(1))
                height_stats=(self.base_height_sums[ids].sum(0)/count.sum().clamp_min(1)).tolist()
                self.log_metrics.update(root_below_28p5cm_fraction=height_stats[0],root_height_cost_mean=height_stats[1])
                self.base_height_sums[ids]=0
                self.air_time_sums[ids]=0
                self.air_time_reward.reset(ids)
            self.extras['log']=self.log_metrics
        self.robot.reset(ids);DirectRLEnv._reset_idx(self,ids)
        self.calf_contact_seen[ids]=False
        root=self.robot.data.default_root_state[ids].clone();root[:,:3]+=self.scene.env_origins[ids]
        self.robot.write_root_pose_to_sim(root[:,:7],ids);self.robot.write_root_velocity_to_sim(root[:,7:],ids)
        self.robot.write_joint_state_to_sim(self.nominal[ids],torch.zeros_like(self.nominal[ids]),env_ids=ids)
        self._choose_path(ids);self._sample_base_commands(ids)
        for b in [self.actions,self.last_actions,self.filtered,self.commands,self.base_slew,self.obs_history,self.sums,self.steps,self.exceed,self.track_sums,self.track_counts,self.gain_sums,self.active_counts,self.last_goal_sign,self.goal_crossings]:b[ids]=0
        # Articulation.reset() clears actuator internals but leaves the effort target buffer.
        self.last_tau[ids]=0
        self.robot.set_joint_effort_target(self.last_tau[ids],env_ids=ids)
        self.grip[ids]=.5;self.grip_target[ids]=torch.rand(len(ids),1,device=self.device)
        self.oscillation_seen[ids]=False;self.overshoot_seen[ids]=False
        if hasattr(self,'guidance'):
            for b in [self.guidance,self.arm_action,self.last_arm_action,self.walk_blend,self.smooth_v,self.smooth_a,self.roll_sums]:b[ids]=0
            if not self.cfg.phase_free_amp:self.phase[ids]=0
            self.smooth_q[ids]=self.nominal[ids][:,self.arm];self.ik_goal[ids]=self.smooth_q[ids];self.reference_filter.reset(ids,self.smooth_q[ids])
        self.ref[ids]=self.nominal[ids][:,self.arm];self.ref_v[ids]=0;self.ref_a[ids]=0
