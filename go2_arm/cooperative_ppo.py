"""Two independent RSL-RL PPO models, optimizers and rollout buffers."""
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

def observations(x):return TensorDict({'policy':x},batch_size=[len(x)])

def make_ppo(obs,action_dim,device):
    actor=MLPModel(obs,{'actor':['policy']},'actor',action_dim,hidden_dims=[512,256,128],activation='elu',obs_normalization=True,
        distribution_cfg={'class_name':'executed_entropy:ExecutedEntropyGaussian','init_std':.6,'std_type':'log'})
    critic=MLPModel(obs,{'critic':['policy']},'critic',1,hidden_dims=[512,256,128],activation='elu',obs_normalization=True)
    storage=RolloutStorage('rl',obs.batch_size[0],24,obs,[action_dim],device=device)
    return PPO(actor,critic,storage,num_learning_epochs=5,num_mini_batches=4,learning_rate=3e-4,schedule='adaptive',desired_kl=.01,
        gamma=.99,lam=.95,entropy_coef=.005,max_grad_norm=1.,device=device)

def finite_models(*algorithms):
    return all(torch.isfinite(p).all() for a in algorithms for m in [a.actor,a.critic] for p in m.parameters())
