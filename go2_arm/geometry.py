"""Heading-frame poses and differentiable geometric Jacobians; metres/radians."""
import torch

def rpy_matrix(roll,pitch,yaw):
    c,s=roll.cos(),roll.sin();cp,sp=pitch.cos(),pitch.sin();cy,sy=yaw.cos(),yaw.sin()
    return torch.stack([cy*cp,cy*sp*s-sy*c,cy*sp*c+sy*s,sy*cp,sy*sp*s+cy*c,sy*sp*c-cy*s,-sp,cp*s,cp*c],-1).reshape(-1,3,3)

def rotation_error(target,actual):
    return .5*torch.cross(actual.transpose(1,2),target.transpose(1,2),dim=-1).sum(1)

def rot6d(R):return R[:,:,:2].transpose(1,2).reshape(-1,6)

def heading_rotation(R):
    yaw=torch.atan2(R[:,1,0],R[:,0,0]);z=torch.zeros_like(yaw)
    return rpy_matrix(z,z,yaw)

def base_to_heading(p,R,base_R,base_z,reference_z=.305):
    relative=heading_rotation(base_R).transpose(1,2)@base_R
    offset=torch.zeros_like(p);offset[:,2]=base_z-reference_z
    return (relative@p[:,:,None]).squeeze(-1)+offset,relative@R

def heading_to_base(p,R,base_R,base_z,reference_z=.305):
    relative=heading_rotation(base_R).transpose(1,2)@base_R
    offset=torch.zeros_like(p);offset[:,2]=base_z-reference_z
    return (relative.transpose(1,2)@(p-offset)[:,:,None]).squeeze(-1),relative.transpose(1,2)@R

def fk_jacobian(fk,q):
    n=q.shape[0];p=fk.base_p.expand(n,3).clone();R=fk.base_R.expand(n,3,3).clone();points=[];axes=[]
    for i,(op,oR) in enumerate(fk.origins):
        p=p+(R@op.expand(n,3)[:,:,None]).squeeze(-1);R=R@oR
        points.append(p.clone());axes.append(R[:,:,2].clone())
        c,s=q[:,i].cos(),q[:,i].sin();z=torch.zeros_like(c);one=torch.ones_like(c)
        R=R@torch.stack([c,-s,z,s,c,z,z,z,one],-1).reshape(n,3,3)
    tcp=p+(R@fk.tcp_p.expand(n,3)[:,:,None]).squeeze(-1)
    axes=torch.stack(axes,2);points=torch.stack(points,2)
    linear=torch.cross(axes,tcp[:,:,None]-points,dim=1)
    return tcp,R@fk.tcp_R,torch.cat([linear,axes],1)

def damped_ik(fk,q,p,R,limits,steps=3):
    q=q.clone();eye=torch.eye(6,device=q.device)[None]
    for _ in range(steps):
        pos,rot,J=fk_jacobian(fk,q)
        error=torch.cat([p-pos,.2*rotation_error(R,rot)],-1)
        J=J.clone();J[:,3:]*=.2
        dq=(J.transpose(1,2)@torch.linalg.solve(J@J.transpose(1,2)+.0025*eye,error[:,:,None])).squeeze(-1)
        q=(q+dq.clamp(-.12,.12)).clamp(limits[:,0],limits[:,1])
    return q
