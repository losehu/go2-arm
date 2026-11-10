"""Shared joint/TCP command adapter and C2 trajectory generation (CPU planning).

Collision checks are sampled at <=0.01 rad joint spacing, not a continuous proof.
This planner assumes the supplied base/leg configuration during a trajectory.
The runtime must revalidate when the base, legs, payload or environment changes.
"""
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
ARM_NAMES = [f"joint{i}" for i in range(1, 7)]
from robot_constants import DEFAULT_Q


class SmoothSegment:
    """Quintic position reference with continuous velocity and acceleration.

    A new target starts at the current reference q/v/a. Duration is stretched
    until sampled speed/acceleration/jerk bounds pass; never clip derivatives.
    """
    def __init__(self, q, goal, v=None, a=None, duration=.6,
                 vmax=.6, amax=2., jmax=15., limits=None):
        q=np.asarray(q); goal=np.asarray(goal)
        v=np.zeros_like(q) if v is None else np.asarray(v)
        a=np.zeros_like(q) if a is None else np.asarray(a)
        for _ in range(55):
            T=duration
            c=np.zeros((6,len(q)));c[0]=q;c[1]=v;c[2]=a/2
            A=np.array([[T**3,T**4,T**5],[3*T*T,4*T**3,5*T**4],[6*T,12*T*T,20*T**3]])
            b=np.stack([goal-q-v*T-a*T*T/2,-v-a*T,-a])
            c[3:]=np.linalg.solve(A,b);self.c=c;self.duration=T
            x,xd,xdd,xddd=self.sample(np.linspace(0,T,301))
            if max(np.max(abs(xd))/vmax,np.max(abs(xdd))/amax,np.max(abs(xddd))/jmax)<=1.000001:
                if limits is not None and (np.any(x<limits[:,0]-1e-9) or np.any(x>limits[:,1]+1e-9)):
                    raise ValueError("A continuous retarget would leave joint limits; decelerate before replanning")
                return
            duration*=1.12
        raise ValueError("Could not satisfy reference derivative limits")

    def sample(self,t):
        t=np.clip(np.asarray(t),0,self.duration)[...,None]
        q=sum(self.c[i]*t**i for i in range(6))
        v=sum(i*self.c[i]*t**(i-1) for i in range(1,6))
        a=sum(i*(i-1)*self.c[i]*t**(i-2) for i in range(2,6))
        j=sum(i*(i-1)*(i-2)*self.c[i]*t**(i-3) for i in range(3,6))
        return q,v,a,j


class CommandBridge:
    def __init__(self, model_path=None, seed=20261106):
        self.m=mujoco.MjModel.from_xml_path(str(model_path or ROOT/'robot/scene.xml'))
        self.d=mujoco.MjData(self.m);self.base_q=DEFAULT_Q.copy()
        self.ids=np.array([self.m.joint(n).id for n in ARM_NAMES])
        self.qids=self.m.jnt_qposadr[self.ids];self.limits=self.m.jnt_range[self.ids].copy()
        self.ee=self.m.body('end_effector').id;self.base=self.m.body('base_link').id
        self.floor=self.m.geom('floor').id;self.rng=np.random.default_rng(seed)
        self.finger_bodies={self.m.body('link7').id,self.m.body('link8').id}
        self.feet={self.m.geom(i).id for i in range(self.m.ngeom)
                   if self.m.geom(i).name in ['FL','FR','RL','RR']}
        # Contact margin also rejects tiny clearances. Foot-ground contacts are allowed.
        active=(self.m.geom_contype!=0)|(self.m.geom_conaffinity!=0)
        self.m.geom_margin[active]=.001

    def forward(self,q,aperture=.5):
        self.d.qpos[:]=self.base_q;self.d.qpos[self.qids]=q
        self.d.qpos[-2:]=.035*aperture;mujoco.mj_forward(self.m,self.d)
        Rb=self.d.xmat[self.base].reshape(3,3)
        p=Rb.T@(self.d.xpos[self.ee]-self.d.xpos[self.base])
        R=Rb.T@self.d.xmat[self.ee].reshape(3,3)
        return p,Rotation.from_matrix(R)

    def collision_free(self,q,apertures=(0.,.5,1.)):
        if np.any(q<self.limits[:,0]) or np.any(q>self.limits[:,1]):return False
        for aperture in apertures:
            self.forward(q,aperture)
            for c in self.d.contact:
                a,b=int(c.geom1),int(c.geom2)
                if self.floor in (a,b) and ({a,b}-{self.floor}) <= self.feet:continue
                if {int(self.m.geom_bodyid[a]),int(self.m.geom_bodyid[b])}==self.finger_bodies:continue
                if c.dist<.001:return False
        return True

    def edge_free(self,a,b,step=.01):
        n=max(2,int(np.ceil(np.max(abs(b-a))/step))+1)
        return all(self.collision_free(q) for q in np.linspace(a,b,n))

    def inverse(self,pose,current_q,attempts=12):
        pose=np.asarray(pose,float)
        if pose.shape!=(6,) or not np.isfinite(pose).all():raise ValueError('Expected finite xyz/rpy in base_link frame')
        pos=pose[:3];rot=Rotation.from_euler('xyz',pose[3:])
        def residual(q):
            p,R=self.forward(q)
            return np.r_[p-pos,.20*(rot*R.inv()).as_rotvec()]
        solutions=[]
        for i in range(attempts):
            seed=current_q if i==0 else self.rng.uniform(self.limits[:,0]+1e-7,self.limits[:,1]-1e-7)
            result=least_squares(residual,np.clip(seed,self.limits[:,0]+1e-7,self.limits[:,1]-1e-7),bounds=(self.limits[:,0],self.limits[:,1]),max_nfev=120,gtol=1e-9,xtol=1e-9,ftol=1e-9)
            e=residual(result.x)
            if np.linalg.norm(e[:3])<.002 and np.linalg.norm(e[3:])/.2<np.deg2rad(.5) and self.collision_free(result.x):solutions.append(result.x)
        if not solutions:raise ValueError('No verified IK solution for this pose; command rejected')
        return min(solutions,key=lambda q:np.linalg.norm(q-current_q))

    def plan(self,start,goal,max_nodes=4500):
        start=np.asarray(start);goal=np.asarray(goal)
        if not self.collision_free(start) or not self.collision_free(goal):raise ValueError('Start/goal is in collision or outside limits')
        if self.edge_free(start,goal):return np.stack([start,goal])
        trees=[([start.copy()],[-1]),([goal.copy()],[-1])]
        def extend(tree,target):
            pts,parents=tree;i=int(np.argmin(np.linalg.norm(np.asarray(pts)-target,axis=1)))
            delta=target-pts[i];dist=np.linalg.norm(delta);q=pts[i]+delta*min(1.,.20/max(dist,1e-10))
            if not self.edge_free(pts[i],q):return None,False
            pts.append(q);parents.append(i);return len(pts)-1,dist<=.20
        def trace(tree,i):
            path=[]
            while i>=0:path.append(tree[0][i]);i=tree[1][i]
            return path[::-1]
        for n in range(max_nodes):
            side=n%2;ta,tb=trees[side],trees[1-side]
            target=tb[0][0] if self.rng.random()<.15 else self.rng.uniform(self.limits[:,0],self.limits[:,1])
            ia,_=extend(ta,target)
            if ia is None:continue
            for _ in range(60):
                ib,done=extend(tb,ta[0][ia])
                if ib is None:break
                if done:
                    a=trace(ta,ia);b=trace(tb,ib)
                    path=a+b[::-1][1:]
                    if side:path=path[::-1]
                    # Greedy shortcut remains collision checked.
                    out=[path[0]];i=0
                    while i<len(path)-1:
                        j=len(path)-1
                        while j>i+1 and not self.edge_free(path[i],path[j]):j-=1
                        out.append(path[j]);i=j
                    return np.asarray(out)
        raise ValueError('No collision-free path found within the planning budget; command deferred')

    def target(self,mode,command,current_q):
        if mode=='joint_position':goal=np.asarray(command,float)
        elif mode=='end_effector_pose':goal=self.inverse(command,current_q)
        else:raise ValueError('Unknown arm control mode')
        return self.plan(current_q,goal)
