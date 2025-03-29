
import gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from typing import Union, Callable, Sequence, Tuple
from collections import OrderedDict
from torch import Tensor, FloatTensor
from numpy import ndarray
from torchvision import transforms as T
import random


#==start train=======
import gym
import gym_multi_car_racing
import sys
#---------Agent-------
import importlib.util
#===================actor /crititc network=====================
LOG_SIG_MAX = 1
LOG_SIG_MIN = -1
epsilon = 1e-6
DEVICE="cpu"
from typing import Union, Callable, Sequence, Tuple
from collections import OrderedDict
from pathlib import Path
#env = gym.make("CarRacing-v0")
seed=69
torch.manual_seed(seed)
np.random.seed(seed)
# Type for str and Path inputs
PathOrStr = Union[str, Path]
# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super().__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)
        
    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super().__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        mean[:, 0] = torch.clamp(mean[:, 0], -1, 1)  # mean[0] 限制在 -1 到 1 之间
        mean[:, 1] = torch.tanh(mean[:, 1])   # mean[1] 限制在 0 到 1 之间
        mean[:, 2] = torch.tanh(mean[:, 2])   # mean[2] 限制在 0 到 1 之间
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean
class ConvBetaVAE(nn.Module):
    def __init__(self, z_dim: int = 32):
        super().__init__()
        # encoder
        self.encoder = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 32, 4, stride=2, padding=1)),
            ("relu1", nn.LeakyReLU(0.2)), #32x32x32
            ("block1", ConvBlock(32, 64, 4, stride=2, padding=1, slope=0.2)), # 64x16x16
            ("block2", ConvBlock(64, 128, 4, stride=2, padding=1, slope=0.2)), # 128x8x8
            ("block3", ConvBlock(128, 256, 4, stride=2, padding=1, slope=0.2)), # 256x4x4
        ]))

        ## Latent representation of mean and std
        # 256x4x4 = 4096
        self.fc1 = nn.Linear(4096, z_dim)
        self.fc2 = nn.Linear(4096, z_dim)
        self.fc3 = nn.Linear(z_dim, 4096)

        # decoder
        self.decoder = nn.Sequential(OrderedDict([
            ("deconv1", DeConvBlock(4096, 256, 4, stride=1, padding=0, slope=0.2)),
            ("deconv2", DeConvBlock(256, 128, 4, stride=2, padding=1, slope=0.2)),
            ("deconv3", DeConvBlock(128, 64, 4, stride=2, padding=1, slope=0.2)),
            ("deconv4", DeConvBlock(64, 32, 4, stride=2, padding=1, slope=0.2)),
            ("convt1", nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1))
        ]))


    def encode(self, x):
        x = self.encoder(x)
        x = x.view(-1, 4096)
        return self.fc1(x), self.fc2(x)

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        z = self.fc3(z).view(-1, 256 * 4 * 4, 1, 1)
        z = self.decoder(z)
        return torch.sigmoid(z)
    
    def sample(self, x):
        # encode x
        x = self.encoder(x).view(-1, 4096)

        # get mu and logvar from input
        mu, logvar = self.fc1(x), self.fc2(x)

        # generate and return sample
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)*0.001
        z = eps * std + mu
        z = z.detach().cpu()
        return z.squeeze().numpy()


    def forward(self, x, encode: bool = False, mean: bool = True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if encode:
            if mean:
                return mu
            return z
        return self.decode(z), mu, logvar
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, 
                        out_channels: int,
                        kernel_size: int, 
                        stride: int = 2, 
                        padding: int = 1, 
                        slope: float = 0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=slope)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
class DeConvBlock(nn.Module):
    def __init__(self, in_channels: int, 
                        out_channels: int,
                        kernel_size: int, 
                        stride: int = 2, 
                        padding: int = 1, 
                        slope: float = 0.2):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=slope)
    
    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))
#===================Memory buffer=========================================
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    def memory_reset(self):
        self.buffer=[]
        self.priorities=np.zeros((self.capacity,),dtype=np.float32)
    def push(self, batch_state, batch_action, batch_reward, batch_next_state,batch_td_error):
        self.buffer.append((batch_state, batch_action, batch_reward, batch_next_state))
        self.priorities[len(self.buffer)-1] = batch_td_error
    def sample(self):
        priorities = list(self.priorities[:len(self.buffer)])
        max_index = priorities.index(max(priorities))
        state, action, reward, next_state= self.buffer.pop(max_index)
        #state, action, reward, next_state= self.buffer.pop(len(self.buffer)-1)
        return state, action, reward, next_state
#======================================================================
class Agent:
    def __init__(self):
		
        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters()
        self.alpha=0.2
        self.target_entropy = -torch.prod(torch.Tensor(3).to("cpu")).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device="cpu")
        self.alpha_optim = Adam([self.log_alpha], lr=self.lr)
        self.obs_buffer=[]
        self.i=0
        # Extract environment information
        self.env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
        use_random_direction=True, backwards_flag=True, h_ratio=0.25,
        use_ego_color=False)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.actor=GaussianPolicy(32, 3, 256).to("cpu")
        self.tau=0.005                                                # ALG STEP 1
        self.critic = QNetwork(32, 3, 256).to("cpu")
        self.critic_target = QNetwork(32, 3, 256).to("cpu")
        self.hard_update(self.critic_target, self.critic)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        self.encoder= ConvBetaVAE()
        self.encoder.load_state_dict(torch.load("env.pt", map_location="cpu"))
        self.memory=ReplayMemory(1000000)
    def apply(self,func: Callable[[Tensor], Tensor], M: Tensor, d: int = 0) -> Tensor:
        tList = [func(m) for m in torch.unbind(M, dim=d) ]
        res = torch.stack(tList, dim=d)
        return res 
    def soft_update(self,target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self,target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    def process_observation(self,obs: ndarray) -> FloatTensor:
        cropper = T.Compose([T.ToPILImage(),
                                T.CenterCrop((64,64)),
                                T.ToTensor()])
        converted = torch.from_numpy(obs.copy())
        converted = torch.einsum("nhwc -> nchw", converted)
        return self.apply(cropper, converted).to("cpu")
    def update_parameters(self,total_reward):
        #reward_batch = torch.FloatTensor(reward_batch).to(DEVICE).unsqueeze(1)
        #===========compute next_state_batch======================
        state_batch,action_batch,reward_batch,next_state_batch=self.memory.sample()
        #===========compute next q value===========================
        for i in range(1,5):
            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + self.gamma * (min_qf_next_target)
        #===================critic loss=============
            qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        #============policy loss====================
            with torch.no_grad():
                pi, log_pi, _ = self.actor.sample(state_batch)
                qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi).detach()
            policy_loss = ((0.02* log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        #============env update=======================
            # alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            # self.alpha_optim.zero_grad()
            # alpha_loss.backward()
            # self.alpha_optim.step()
            # self.alpha = self.log_alpha.exp()
        #=============critic update====================
        if total_reward>500:
            self.critic_optim.zero_grad()
            qf1_loss.backward()
            self.critic_optim.step()
            self.critic_optim.zero_grad()
            qf2_loss.backward()
            self.critic_optim.step()
        #===============policy update=======================
            self.actor_optim.zero_grad()
            torch.set_grad_enabled(True) 
            #print(policy_loss.grad_fn)
            policy_loss.requires_grad = True
            policy_loss.backward()
            self.actor_optim.step()
        #==========critic _target update=====================
        if total_reward>600:
            self.soft_update(self.critic_target, self.critic, self.tau)

    def learn(self):
        obs = self.env.reset()
        total_reward=0
        #print("needshape",obs.shape)
        obs,batch_obs, batch_acts, batch_rtgs,done,batch_td_error = self.rollout(obs)
        max_reward=total_reward
        for t in range(20000):  
            #==============catch memory=============================
            obs,next_batch_obs, batch_acts, batch_rtgs,batch_td_error,done = self.rollout(obs)  
            #self.env.render()
            total_reward+= np.sum(np.array(batch_rtgs))
            self.memory.push(batch_obs,batch_acts,batch_rtgs,next_batch_obs,batch_td_error)
            batch_obs=next_batch_obs
            if done:
                #===============update parameter===========================
                if len(self.memory.buffer)>10 and total_reward<1000:
                    for t in range(10):
                        self.update_parameters(total_reward)
                #==========save dict=======================
                if total_reward>=600:
                    torch.save(self.actor.state_dict(), '112061588_hw3_data')
                    torch.save(self.critic.state_dict(), 'critic.pth')
                    torch.save(self.encoder.state_dict(), 'env.pt')
                    max_reward=total_reward
                #===========reset======================
                done=False
                print("total reward:"+str(total_reward))
                total_reward=0
                self.memory.memory_reset()
                obs=self.env.reset()
    def act(self,obs):

        obs = self.process_observation(obs)
        state = self.encoder.sample(obs)
        state = torch.FloatTensor(state).to(DEVICE).unsqueeze(0)
        _, _, action = self.actor.sample(state)
        #self.obs_buffer.append(state)
        action=action.detach().cpu().numpy()
        return action[0]

    def rollout(self,obs):
        batch_obs = []
        batch_acts = []
        batch_rews = []
        batch_rtgs = []

        t = 0 
        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch: # 200 per patch
            #obs = self.env.reset()
            done = False
            t += 1
            # Track observations in this batch
            #self.env.render()     
            action = self.act(obs) # log prob_=192
            #print(log_prob.shape)
            obs, rew, done, _ = self.env.step(action)
            state = self.process_observation(obs)
            state = self.encoder.sample(state)
            state = torch.FloatTensor(state).to(DEVICE).unsqueeze(0)
            batch_obs.append(state)
            batch_acts.append(action)
            batch_rews.append(rew)
            if done:
                break
        while(len(batch_obs)<self.timesteps_per_batch):
            batch_obs.append(torch.zeros_like(torch.tensor(state)))
        while(len(batch_acts)<self.timesteps_per_batch):
            batch_acts.append(torch.zeros_like(torch.tensor(action)))
        while(len(batch_rews)<self.timesteps_per_batch):
            batch_rews.append(torch.full_like(torch.tensor(rew), -0.1))
        last_obs=obs
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float).squeeze()
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_rtgs=torch.tensor(np.array(batch_rews).squeeze(),dtype=torch.float)
        #print(batch_obs.shape)
        with torch.no_grad():
            #print(batch_obs.shape)
            next_state_action, next_state_log_pi, _ = self.actor.sample(batch_obs)
            #print(next_state_action.shape)
            print(next_state_action.shape)
            print(next_state_log_pi.shape)
            qf1_next_target, qf2_next_target = self.critic_target(batch_obs, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = batch_rtgs + self.gamma * (min_qf_next_target)
        current_q_value = torch.min(*self.critic(batch_obs, batch_acts))
        td_error = torch.abs(current_q_value - next_q_value)
        batch_td_error = np.sum(td_error.detach().cpu().numpy())
        return last_obs,batch_obs, batch_acts, batch_rtgs,batch_td_error,done
    def _init_hyperparameters(self):
        self.timesteps_per_batch = 32    # Number of timesteps to run per batch
        self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
        self.lr = 0.0001                             # Learning rate of actor optimizer
        self.gamma = 0.3                             # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.save_freq = 10                             # How often we save in number of iterations
        self.avg_ep_lens=0
        self.avg_ep_rews=0
        self.actor_losses=0
#==start train=======
import gym
import gym_multi_car_racing
import gym
import sys
import torch
import importlib.util
import numpy as np

#-----------------start training---------------------
env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
        use_random_direction=True, backwards_flag=True, h_ratio=0.25,
        use_ego_color=False)
agent=Agent()
#path_to_actor: str = "./hw3_data/sac_actor_carracer_klein_6_24_18.pt"
path_to_actor: str = "sac_data"
DEVICE="cpu"
agent.actor.load_state_dict(torch.load(path_to_actor, map_location=DEVICE))
#path_to_critic: str = "./hw3_data/sac_critic_carracer_klein_6_24_18.pt"
path_to_critic: str = "critic.pth"
agent.critic.load_state_dict(torch.load(path_to_critic, map_location=DEVICE))

agent.learn()
env.close()

