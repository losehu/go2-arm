"""Command stages and dimensionless tracking gates, independently testable."""
import math
import torch

# vx, vy, wz, pitch. Arm levels: -1 holds home; 0/1/2 use checked paths.
STAGES = [
    dict(name='learn_locomotion', envelope=[.50,.30,.40,0.], arm_level=-1, range_scale=.5),
    dict(name='add_pitch_and_small_arm', envelope=[.50,.30,.40,math.radians(5)], arm_level=0, range_scale=.5),
    dict(name='expand_combined_motion', envelope=[.75,.45,.60,math.radians(10)], arm_level=1, range_scale=.75),
    dict(name='full_requested_envelope', envelope=[1.,.60,.80,math.radians(15)], arm_level=2, range_scale=1.),
]

def tolerances(command):
    """At low commands, enforce relative tracking instead of a permissive 0.1 cap."""
    return torch.where(command.abs()>.025,(.25*command.abs()).clamp(.02,.10),torch.full_like(command,.10))

def tracking_terms(actual,command):
    error=actual-command
    normalized=error/tolerances(command)
    active=command.abs()>.025
    ratio=torch.where(active,actual*command/(command.square()+1e-8),torch.zeros_like(actual))
    progress=(ratio.clamp(-1,1)*active).sum(-1)/active.sum(-1).clamp_min(1)
    moving=active.any(-1)
    stalled=moving & (progress<.2)
    return normalized,progress,stalled,active

def motion_gate(normalized_rmse,directional_gain,active_counts,valid_count):
    enough=valid_count>=100
    active=active_counts>=50
    # Every axis that was commanded must respond in the correct direction.
    gain_ok=((directional_gain>=.7)&(directional_gain<=1.3))|~active
    return enough & active.any(-1) & (normalized_rmse<=1.).all(-1) & gain_ok.all(-1)

def sample_commands(count,stage,device):
    cap=torch.tensor(STAGES[stage]['envelope'],device=device,dtype=torch.float32)
    c=torch.zeros(count,4,device=device)
    category=torch.randint(0,3,(count,),device=device)
    magnitude=(.5+.5*torch.rand(count,device=device))*cap[category]
    sign=torch.where(torch.rand(count,device=device)>.5,1.,-1.)
    c[torch.arange(count,device=device),category]=magnitude*sign
    # Introduce combined directional commands gradually; zero is a distinct command.
    mixed=torch.rand(count,device=device)<(.1 if stage==0 else .35)
    c[mixed,:3]=(2*torch.rand(int(mixed.sum()),3,device=device)-1)*cap[:3]
    # Preserve a clear active axis even in the mixed group.
    c[torch.arange(count,device=device),category]=magnitude*sign
    stand=torch.rand(count,device=device)<.1
    c[stand,:3]=0
    if stage>0:c[:,3]=(2*torch.rand(count,device=device)-1)*cap[3]
    return c

def self_check():
    cmd=torch.tensor([[.25,0,0],[-.25,0,0],[0,.15,0],[0,0,.2],[.5,.3,.4]])
    for fraction,expected in [(0.,False),(-1.,False),(1.,True),(.9,True)]:
        n,_,_,active=tracking_terms(cmd*fraction,cmd)
        gain=torch.where(active,torch.full_like(cmd,fraction),torch.zeros_like(cmd))
        result=motion_gate(n.abs(),gain,active.float()*200,torch.full((len(cmd),),200.))
        assert bool((result==expected).all()),(fraction,result)
    c=sample_commands(20000,0,'cpu')
    nonzero=c[:,:3].abs().max(-1).values>0
    assert bool((c[nonzero,:3].abs().max(-1).values>=.15-1e-6).all())
    assert bool((c[:,3]==0).all())
    return {'zero_response_rejected_all_axes':True,'wrong_direction_rejected':True,
            'correct_tracking_accepted':True,'initial_sample_count':len(c),
            'initial_nonzero_command_min':float(c[nonzero,:3].abs().max(-1).values.min())}

if __name__=='__main__':
    import json
    from pathlib import Path
    r=self_check();Path(__file__).with_name('curriculum_validation.json').write_text(json.dumps(r,indent=2));print(r)
