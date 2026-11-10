"""Finite-duration monotone filter: rate limit + three 0.35 s moving averages.

For constant targets all kernels are positive, hence no reference overshoot.
Velocity <=0.4 rad/s; continuous acceleration <=1.72 rad/s² and jerk
<=13.07 rad/s³. A stopped input finishes after 1.05 s, avoiding an asymptotic
creep tail. Physical tracking and changing-target paths need separate checks.
"""
import torch
KERNEL_S=.35;VMAX=.4
class ReferenceFilter:
 def __init__(self,q,dt=.0025):
  self.dt=dt;self.n=round(KERNEL_S/dt);self.index=0;self.dtype=q.dtype
  self.q=q.double().clone();self.v=torch.zeros_like(self.q);self.input=self.q.clone()
  self.buffers=[self.q[None].repeat(self.n,1,1) for _ in range(3)]
  self.sums=[self.q*self.n for _ in range(3)]
 def reset(self,ids,q,v=None,a=None):
  q=q.double();v=torch.zeros_like(q) if v is None else v.double();a=torch.zeros_like(q) if a is None else a.double()
  mu=(self.n-1)*self.dt/2;var=(self.n*self.n-1)*self.dt*self.dt/12
  m=3*mu;variance=3*var
  c=q+m*v+.5*a*(m*m-variance);b=v+m*a
  # Reconstruct polynomial input history so output begins at current q/v/a.
  age=torch.arange(self.n-1,-1,-1,device=q.device,dtype=torch.float64)*-self.dt
  for stage,buf in enumerate(self.buffers):
   u=stage*mu;w=stage*var
   values=(c-u*b+.5*a*(u*u+w))[None]+(b-u*a)[None]*age[:,None,None]+.5*a[None]*age[:,None,None].square()
   order=(torch.arange(self.n,device=q.device)+self.index)%self.n
   buf[order[:,None],ids[None,:]]=values
   self.sums[stage][ids]=values.sum(0)
  self.input[ids]=c;self.q[ids]=q;self.v[ids]=v-.5*a*self.dt
 def advance(self,goal):
  self.input+=(goal.double()-self.input).clamp(-VMAX*self.dt,VMAX*self.dt)
  value=self.input
  for buf,total in zip(self.buffers,self.sums):
   total+=value-buf[self.index];buf[self.index]=value;value=total/self.n
  self.index=(self.index+1)%self.n
  v=(value-self.q)/self.dt;a=(v-self.v)/self.dt
  self.q=value.clone();self.v=v.clone()
  return self.q.to(self.dtype),v.to(self.dtype),a.to(self.dtype)
