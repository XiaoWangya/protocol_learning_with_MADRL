import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

torch.manual_seed(123)
random.seed(123)

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim,hidden_dim = 256):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(Critic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.cov1 = nn.Conv2d(1,4,5,5)#1,1,180,30
        # self.cov2 = nn.Conv2d(4,2,3,3)
        # self.s1 = nn.Linear(self.state_dim, 64)#1,2,12,3
        self.s1 = nn.Linear(self.state_dim, 16)
        self.a1 = nn.Linear(action_dim, 16)
        self.fc1 = nn.Linear(32,hidden_dim)
        # self.fc1.weight.data.uniform_(-1/64,1/64)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        # self.fc2.weight.data.uniform_(-1/16,1/16)
        self.fc3 = nn.Linear(hidden_dim,self.action_dim)
        # self.fc3.weight.data.uniform_(-3e-1, 3e-1)
    
    def forward(self, state, action,batch = False):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        # state = torch.unsqueeze(torch.unsqueeze(state,0),0)
        state = state.view(int(state.numel()/self.state_dim),self.state_dim).float()
        # s1 = nn.ReLU()(self.cov1(state))
        # s1 = self.cov2(s1)
        # s1 = s1.view(s1.size(0),-1)
        s1 = self.s1(state)
        # s1 = self.s2(s1)
        a1 = self.a1(action)
        x = torch.cat((s1,a1),dim=1)
        # x = torch.cat((s1,action),dim=1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return x

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim,hidden_dim = 256):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of output action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :return:
        """
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(self.state_dim, hidden_dim)#1,2,12,20
        
        # self.fc1.weight.data.uniform_(-1/64,1/64)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc2.weight.data.uniform_(-1/np.sqrt(16), 1/np.sqrt(16))
        self.fc3 = nn.Linear(hidden_dim, self.action_dim)
        # self.fc3.weight.data.uniform_(-3e-2, 3e-2)


    def forward(self, state,batch = False):
        """
        returns policy function Pi(s) obtained from actor network
        this function is a gaussian prob distribution for all actions
        with mean lying in (-1,1) and sigma lying in (0,1)
        The sampled action can , then later be rescaled
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,action_dim] )
        """
        # state = torch.squeeze(torch.squeeze(state,0),0)
        state = state.view(int(state.numel()/self.state_dim),self.state_dim).float()
        x = self.fc1(state)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        action = torch.tanh(x)
        return action

class TD3(nn.Module):
    def __init__(self,state_dim, action_dim,hidden_dim,memory_size,batch_size, tau = 0.01,gamma = 0.8,actor_lr = 1e-6, critic_lr = 1e-6):
        super(TD3, self).__init__()
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.target_actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim)
        self.memory = torch.zeros((memory_size, state_dim * 2 + 1+action_dim))
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_opt = torch.optim.Adam(self.actor.parameters())
        self.critic_opt = torch.optim.Adam(self.critic.parameters())
        self.critic_opt2 = torch.optim.Adam(self.critic2.parameters())
        # self.schedule_actor = torch.optim.lr_scheduler.StepLR(self.actor_opt, step_size=schedule_step, gamma=0.1)
        # self.schedule_critic = torch.optim.lr_scheduler.StepLR(self.critic_opt, step_size=schedule_step, gamma=0.1)
        self.memory_counter = 0
        self.critic_update_counter = 0
        self.actor_update_counter = 0
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.loss_a = []
        self.loss_c = []

    def store(self, s, a, r, s_):

        transition = torch.cat((s, a, r, s_),-1)
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition.clone()
        self.memory_counter += 1

    def choose_action(self,state):
        return self.actor(state).detach().numpy()
        
    def update(self, noise_std: float = 1e-3):
        
        sample_index = np.random.choice(self.memory_size, self.batch_size)
        b_memory = self.memory[sample_index, :]

        b_s = b_memory[:, :self.state_dim].float()
        b_a = b_memory[:, self.state_dim:self.state_dim+self.action_dim].float()
        b_r = b_memory[:, self.state_dim+self.action_dim:self.state_dim+self.action_dim+1].float()#假设reward是1维的
        b_s_ = b_memory[:, -self.state_dim:].float()

        target_Q = self.target_critic(b_s_, (torch.normal(0,std = torch.tensor(noise_std)).clamp(-1,1)+self.target_actor(b_s_)).clamp(-1,1).detach())      
        target_Q2 = self.target_critic2(b_s_, (torch.normal(0,std = torch.tensor(noise_std)).clamp(-1,1)+self.target_actor(b_s_)).clamp(-1,1).detach())
        target_Q = b_r + (self.gamma * torch.min(target_Q,target_Q2)).detach()

        # Get current Q estimate
        current_Q = self.critic(b_s, b_a)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q).float()
        # Optimize the critic
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        current_Q2 = self.critic2(b_s, b_a)
        # Compute critic loss
        critic_loss2 = F.mse_loss(current_Q2, target_Q).float()
        # Optimize the critic
        self.critic_opt2.zero_grad()
        critic_loss2.backward()
        self.critic_opt2.step()

        if critic_loss.is_cuda:
            self.loss_c.append(torch.min(critic_loss,critic_loss2).cpu().detach().numpy())
        else:
            self.loss_c.append(torch.min(critic_loss,critic_loss2).detach().numpy())
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        actor_loss = -self.critic(b_s, self.actor(b_s,batch = True)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        if actor_loss.is_cuda:
            self.loss_a.append(actor_loss.cpu().detach().numpy())
        else:
            self.loss_a.append(actor_loss.detach().numpy())

        #soft update
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        self.actor_update_counter += 1
        # self.schedule_critic.step()
        # self.critic_lr = max(self.critic_lr*0.9999,1e-4)
        # self.critic_opt = torch.optim.Adam(self.critic.parameters(),lr = self.critic_lr)