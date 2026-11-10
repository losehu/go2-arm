"""Sampled whole-body geometric feasibility, not a dynamic stability guarantee.

Eight planning coordinates: exact arm joints, body pitch, body roll. A stance
IK preserves the current four foot locations. Training and live PhysX still
must detect deviations; a sampled path is not a proof for all moving states.
"""
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from motion import CommandBridge,DEFAULT_Q
from robot_constants import LEG_NAMES

class WholeBodyBridge(CommandBridge):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.arm_limits=self.limits.copy()
        self.limits=np.vstack([self.arm_limits,np.deg2rad([[-15,15],[-15,15]])])
        joints=np.array([self.m.joint(n).id for n in LEG_NAMES])
        self.leg_qids=self.m.jnt_qposadr[joints];self.leg_dofs=self.m.jnt_dofadr[joints]
        self.leg_limits=self.m.jnt_range[joints]
        self.foot_ids=[self.m.geom(n).id for n in ['FL','FR','RL','RR']]
        self.head_vertices=np.load(__import__('pathlib').Path(__file__).parent/'head_vertices.npy')
        self.configure(DEFAULT_Q)

    def configure(self,configuration):
        self.base_q=np.array(configuration,dtype=float).copy()
        self.d.qpos[:]=self.base_q;mujoco.mj_forward(self.m,self.d)
        self.foot_targets=self.d.geom_xpos[self.foot_ids].copy()
        # Body height is a fixed planning reference; actual foot locations remain.
        self.base_q[2]=.305
        self.yaw=Rotation.from_quat(self.base_q[[4,5,6,3]]).as_euler('xyz')[2]
        self.heading_R=Rotation.from_euler('z',self.yaw).as_matrix()

    def state(self,configuration=None):
        q=self.base_q if configuration is None else np.asarray(configuration)
        roll,pitch,_=Rotation.from_quat(q[[4,5,6,3]]).as_euler('xyz')
        return np.r_[q[self.qids],pitch,roll]

    def place(self,x,aperture=.5,solve_stance=True):
        x=np.asarray(x);self.d.qpos[:]=self.base_q
        self.d.qpos[self.qids]=x[:6];self.d.qpos[-2:]=.035*aperture
        quat=Rotation.from_euler('xyz',[x[7],x[6],self.yaw]).as_quat()
        self.d.qpos[3:7]=quat[[3,0,1,2]]
        mujoco.mj_forward(self.m,self.d)
        if solve_stance:
            jp=np.zeros((3,self.m.nv));jr=np.zeros_like(jp)
            for _ in range(12):
                error=self.foot_targets-self.d.geom_xpos[self.foot_ids]
                if np.max(np.linalg.norm(error,axis=-1))<.0005:break
                for i,gid in enumerate(self.foot_ids):
                    cols=self.leg_dofs[3*i:3*i+3];qids=self.leg_qids[3*i:3*i+3]
                    mujoco.mj_jac(self.m,self.d,jp,jr,self.d.geom_xpos[gid],int(self.m.geom_bodyid[gid]))
                    J=jp[:,cols];delta=J.T@np.linalg.solve(J@J.T+1e-6*np.eye(3),error[i])
                    self.d.qpos[qids]=np.clip(self.d.qpos[qids]+np.clip(delta,-.15,.15),self.leg_limits[3*i:3*i+3,0],self.leg_limits[3*i:3*i+3,1])
                mujoco.mj_forward(self.m,self.d)
            self.stance_error=float(np.max(np.linalg.norm(self.foot_targets-self.d.geom_xpos[self.foot_ids],axis=-1)))
        else:self.stance_error=0.
        return self.d.qpos.copy()

    def pose(self,x):
        self.place(x,solve_stance=False)
        origin=np.r_[self.base_q[:2],.305]
        return self.heading_R.T@(self.d.xpos[self.ee]-origin),Rotation.from_matrix(self.heading_R.T@self.d.xmat[self.ee].reshape(3,3))

    def collision_free(self,x,apertures=(0.,.5,1.)):
        x=np.asarray(x)
        if x.shape!=(8,) or not np.isfinite(x).all() or np.any(x<self.limits[:,0]) or np.any(x>self.limits[:,1]):return False
        q=self.place(x)
        if self.stance_error>.002:return False
        R=self.d.xmat[self.base].reshape(3,3)
        if self.d.xpos[self.base,2]+np.min(self.head_vertices@R[2])<.02:return False
        for aperture in apertures:
            self.d.qpos[:]=q;self.d.qpos[-2:]=.035*aperture;mujoco.mj_forward(self.m,self.d)
            for c in self.d.contact:
                a,b=int(c.geom1),int(c.geom2)
                if self.floor in (a,b) and ({a,b}-{self.floor})<=self.feet:continue
                if {int(self.m.geom_bodyid[a]),int(self.m.geom_bodyid[b])}==self.finger_bodies:continue
                if c.dist<.001:return False
        return True

    def inverse(self,pose,current_q,attempts=8):
        pose=np.asarray(pose,float)
        if pose.shape!=(6,) or not np.isfinite(pose).all():raise ValueError('Expected finite xyz/rpy in heading frame')
        target=Rotation.from_euler('xyz',pose[3:]);solutions=[]
        def residual(x):
            p,R=self.pose(x)
            return np.r_[p-pose[:3],.2*(target*R.inv()).as_rotvec()]
        for i in range(attempts):
            seed=current_q if i==0 else np.clip(current_q+self.rng.normal(0,.3,8),self.limits[:,0]+1e-6,self.limits[:,1]-1e-6)
            r=least_squares(residual,seed,bounds=(self.limits[:,0],self.limits[:,1]),max_nfev=150)
            e=residual(r.x)
            if np.linalg.norm(e[:3])<.002 and np.linalg.norm(e[3:])/.2<np.deg2rad(.5) and self.collision_free(r.x):solutions.append(r.x)
        if not solutions:raise ValueError('No collision-checked whole-body IK solution within budget')
        return min(solutions,key=lambda x:np.linalg.norm(x-current_q))

    def target(self,mode,command,current_q):
        command=np.asarray(command,float)
        if mode=='end_effector_pose':goal=self.inverse(command,current_q)
        elif mode=='joint_position':
            if command.shape!=(6,):raise ValueError('Six exact joint targets required')
            candidates=[np.r_[command,current_q[6:]],np.r_[command,0.,0.]]
            candidates += [np.r_[command,np.deg2rad(p),np.deg2rad(r)] for p in [-10,10] for r in [-5,0,5]]
            goals=[x for x in candidates if self.collision_free(x)]
            if not goals:raise ValueError('Joint target has no verified whole-body configuration')
            goal=min(goals,key=lambda x:np.linalg.norm(x-current_q))
        else:raise ValueError('Unknown arm mode')
        return self.plan(current_q,goal,max_nodes=1200)

    def runtime_clear(self,actual_configuration,reference,aperture):
        # Actual root/legs, not the planned stance: check the immediate arm reference.
        old=self.base_q.copy();self.base_q=np.asarray(actual_configuration).copy()
        try:
            self.d.qpos[:]=self.base_q;self.d.qpos[self.qids]=reference[:6];self.d.qpos[-2:]=.035*aperture
            mujoco.mj_forward(self.m,self.d)
            R=self.d.xmat[self.base].reshape(3,3)
            if self.d.xpos[self.base,2]+np.min(self.head_vertices@R[2])<.02:return False
            for c in self.d.contact:
                a,b=int(c.geom1),int(c.geom2)
                if self.floor in (a,b) and ({a,b}-{self.floor})<=self.feet:continue
                if {int(self.m.geom_bodyid[a]),int(self.m.geom_bodyid[b])}==self.finger_bodies:continue
                if c.dist<.001:return False
            return True
        finally:self.base_q=old
