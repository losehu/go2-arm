import numpy as np
ARM_NAMES=[f'joint{i}' for i in range(1,7)]
LEG_NAMES=[f'{leg}_{j}_joint' for leg in ['FL','FR','RL','RR'] for j in ['hip','thigh','calf']]
JOINT_NAMES=LEG_NAMES+ARM_NAMES+['finger_left_joint','finger_right_joint']
EFFORT_LIMITS=np.array([23.7,23.7,45.43]*4+[20,20,15,7,5,5]+[10,10])
DEFAULT_Q=np.array([0,0,.305,1,0,0,0,
 .1951981669,.5621964749,-1.4967812363,
 -.1951981669,.5621964749,-1.4967812363,
 .1951982020,.9252357145,-1.4967808377,
 -.1951982020,.9252357145,-1.4967808377,
 .0337745298,1.9050440016,-1.6156231878,-.2730932641,.1348640956,.0493991727,.0175,.0175])

# User-selected home shared by training, trajectory banks and preview/reset.
FOLDED_HOME_ARM_Q=np.deg2rad([0.,40.,-3.,0.,-32.,0.])
DEFAULT_Q[19:25]=FOLDED_HOME_ARM_Q
PREVIEW_DEFAULT_Q=DEFAULT_Q.copy()
