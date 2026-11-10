"""Ten continuous user inputs; pitch and roll are never user commands."""
import numpy as np
from motion import SmoothSegment,DEFAULT_Q
from whole_body import WholeBodyBridge

class CommandSession:
    def __init__(self,configuration=None):
        self.bridge=WholeBodyBridge();self.bridge.configure(DEFAULT_Q if configuration is None else configuration)
        self.hold=self.bridge.state();self.segments=[];self.start_time=0.
        self.base=np.zeros(3);self.aperture=.5;self.mode='joint_position';self.active_mode=self.mode
    def set_mode(self,mode):
        if mode not in ('joint_position','end_effector_pose'):raise ValueError('Unknown mode')
        # A mode-only toggle changes the next input interpretation, not q/v/a.
        self.mode=mode
    def sample(self,t):
        elapsed=max(0.,t-self.start_time)
        for segment in self.segments:
            if elapsed<=segment.duration:
                x,v,a,_=segment.sample(elapsed);break
            elapsed-=segment.duration
        else:
            x=self.segments[-1].sample(self.segments[-1].duration)[0] if self.segments else self.hold
            v=np.zeros(8);a=np.zeros(8)
        p,R=self.bridge.pose(x)
        return {'x':x.copy(),'q':x[:6].copy(),'qd':v[:6].copy(),'qdd':a[:6].copy(),'xd':v.copy(),'xdd':a.copy(),
                'target_p':p,'target_R':R.as_matrix(),'base':self.base.copy(),'aperture':self.aperture,'active_mode':self.active_mode}
    def command(self,values,t,mode=None):
        values=np.asarray(values,float)
        if values.shape!=(10,) or not np.isfinite(values).all():raise ValueError('Expected ten values: vx,vy,wz,arm[6],aperture')
        if np.any(abs(values[:3])>np.array([1.,.6,.8])) or not 0<=values[9]<=1:raise ValueError('Command outside limits')
        selected=self.mode if mode is None else mode;state=self.sample(t)
        path=self.bridge.target(selected,values[3:9],state['x']);segments=[]
        for i,(a,b) in enumerate(zip(path,path[1:])):
            segment=SmoothSegment(a,b,v=state['xd'] if i==0 else None,a=state['xdd'] if i==0 else None,limits=self.bridge.limits)
            samples=segment.sample(np.linspace(0,segment.duration,max(101,int(np.ceil(segment.duration*.6/.01))+1)))[0]
            if not all(self.bridge.collision_free(x) for x in samples):raise ValueError('Retarget path rejected; current trajectory retained')
            segments.append(segment)
        self.base=values[:3].copy();self.aperture=float(values[9]);self.mode=selected;self.active_mode=selected
        self.segments=segments;self.start_time=float(t)
        return {'accepted':True,'duration_s':sum(s.duration for s in segments),'waypoints':len(path),'geometry_only':True}
