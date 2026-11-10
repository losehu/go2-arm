from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch

class ArmFK:
    def __init__(self,device):
        root=ET.parse(Path(__file__).resolve().parent/'robot/robot.urdf').getroot()
        def origin(joint):
            o=joint.find('origin');p=np.fromstring(o.get('xyz','0 0 0'),sep=' ');a=np.fromstring(o.get('rpy','0 0 0'),sep=' ')
            cx,cy,cz=np.cos(a);sx,sy,sz=np.sin(a)
            R=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
            return torch.tensor(p,device=device,dtype=torch.float32),torch.tensor(R,device=device,dtype=torch.float32)
        self.base_p,self.base_R=origin(root.find("joint[@name='Piper_fixed_joint']"))
        self.origins=[origin(root.find(f"joint[@name='joint{i}']")) for i in range(1,7)]
        self.tcp_p=torch.tensor([0,0,.13],device=device,dtype=torch.float32)
        self.tcp_R=torch.tensor([[0,0,-1],[0,1,0],[1,0,0]],device=device,dtype=torch.float32)
    def __call__(self,q):
        n=q.shape[0];p=self.base_p.expand(n,3).clone();R=self.base_R.expand(n,3,3).clone()
        for i,(op,oR) in enumerate(self.origins):
            p=p+torch.einsum('bij,j->bi',R,op);R=R@oR
            c=q[:,i].cos();s=q[:,i].sin();z=torch.zeros_like(c);one=torch.ones_like(c)
            rot=torch.stack([c,-s,z,s,c,z,z,z,one],-1).reshape(n,3,3);R=R@rot
        return p+torch.einsum('bij,j->bi',R,self.tcp_p),R@self.tcp_R
