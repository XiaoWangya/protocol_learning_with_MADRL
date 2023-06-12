import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(3401)

def G_soft_max(x, temperature : float= 1.0):
    gumbel_softmax = F.gumbel_softmax(x, tau=temperature)
    return gumbel_softmax

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout = 0.1, alpha = 0.01, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = h@self.W # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh).sum(1)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = Wh@self.a[:self.out_features, :]
        Wh2 = Wh@self.a[self.out_features:, :]
        # broadcast add
        e = Wh1 + Wh2.transpose(1,2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class Actor(nn.Module):
    def __init__(self, d_input, d_output, d_hidden : int = 128):
        super(Actor, self).__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.d_hidden = d_hidden
        self.l1 = nn.Linear(self.d_input, self.d_hidden)
        # nn.init.uniform_(self.l1.weight, -1e-3, 1e-3)
        self.l2 = nn.Linear(self.d_hidden, self.d_output)
        # nn.init.uniform_(self.l2.weight, -1e-3, 1e-3)

    def forward(self, x):
        x = x.float()
        x = torch.relu(self.l1(x))
        x = torch.sigmoid(self.l2(x)).squeeze()
        return x
    
class Critic(nn.Module):
    def __init__(self, d_input, d_output, d_hidden : int = 64, n_hidden_layer : int = 2):
        super(Critic, self).__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.d_hidden = d_hidden
        self.l1 = nn.Linear(self.d_input, self.d_hidden)
        self.l2 = nn.Linear(self.d_hidden, self.d_output)
        self.gru = nn.GRU(self.d_output, self.d_output, n_hidden_layer, batch_first=True)
    
    def forward(self, s, a, hidden_x : torch.tensor = None):
        x = torch.relu(self.l1(torch.cat((s, a), -1)))
        x = torch.sigmoid(self.l2(x))
        embedding, x_hidden = self.gru(x, hidden_x)# embedding.shape : d_input, batch, d_output | x_hidden.shape : n_hidden_layer, batch, d_output
        return embedding, x_hidden

class HetGAT_critic(nn.Module):
    def __init__(self, d_input_ue, d_output_ue, d_input_bs, d_output_bs, d_hidden : int = 64, n_agent : int = 10, n_hidden_layer : int = 2):
        super(HetGAT_critic, self).__init__()
        self.d_input_ue = d_input_ue
        self.d_output_ue = d_output_ue
        self.d_input_bs = d_input_bs
        self.d_output_bs = d_output_bs
        self.d_hidden = d_hidden
        self.n_agent = n_agent
        self.n_hidden_layer = n_hidden_layer
        self.critic_share_ue = Critic(d_input_ue + d_output_ue, d_hidden, d_hidden)
        self.critic_bs= Critic(d_input_bs + d_output_bs,d_hidden, d_hidden)
        self.GAT = GraphAttentionLayer(d_hidden, d_hidden, concat=True)
        self.adj = nn.Parameter(self.generate_adjacent_matrix(self.n_agent+1))
        self.evaluator = nn.Sequential(nn.Linear(self.d_hidden, d_hidden), nn.ReLU(), nn.Linear(self.d_hidden, 1))

    def forward(self, s, a):
        batch, _ = s.shape
        s_ue, s_bs = s[:, :self.d_input_ue*self.n_agent].view(batch, self.n_agent, -1), s[:, self.d_input_ue*self.n_agent:].view(batch, 1, -1)
        a_ue, a_bs = a[:, :self.d_output_ue*self.n_agent].view(batch, self.n_agent, self.d_output_ue), a[:, -self.d_output_bs:].view(batch, 1, self.d_output_bs)
        embedding_ue, _ = self.critic_share_ue(s_ue, a_ue)
        embedding_bs, _ = self.critic_bs(s_bs, a_bs)
        embedding = torch.cat((embedding_ue, embedding_bs), 1)
        h_prime = self.GAT(embedding, self.adj)
        evaluation = self.evaluator(h_prime)
        return evaluation
    
    def generate_adjacent_matrix(self, N):
        A = torch.zeros([N, N])
        A[1:, 0] = 1
        A[0, 1:] = 1
        return A 

class HybridActor(nn.Module):
    def __init__(self,  d_input_ue, d_output_ue, d_input_bs, d_output_bs, num_ue):
        super(HybridActor, self).__init__()
        self.d_input_ue = d_input_ue
        self.d_output_ue = d_output_ue
        self.d_input_bs = d_input_bs
        self.d_output_bs = d_output_bs
        self.actor_ue = Actor(d_input_ue, d_output_ue)
        self.actor_bs = Actor(d_input_bs, d_output_bs)
        self.n_client = num_ue

    def forward(self, x):
        batch, _ = x.shape
        s_ue, s_bs = x[:, :self.d_input_ue*self.n_client].view(batch, self.n_client, -1), x[:, self.d_input_ue*self.n_client:].view(batch, 1, -1)
        a_ue = self.actor_ue(s_ue)
        a_bs = self.actor_bs(s_bs)
        return torch.cat((a_ue, a_bs), -1)


class HetGAT_MADRL_PL(nn.Module):
    def __init__(self, d_input_ue, d_output_ue, d_input_bs, d_output_bs, num_ue, d_memory, device, d_batch : int = 32, tau : float = 0.9, gamma : float = 0.9, n_hidden_layer : int = 2):
        super(HetGAT_MADRL_PL, self).__init__()
        self.d_input_ue = d_input_ue
        self.d_output_ue = d_output_ue
        self.d_input_bs = d_input_bs
        self.d_output_bs = d_output_bs
        self.n_client = num_ue
        self.d_memory = d_memory
        self.d_batch = d_batch
        self.n_hidden_layer = n_hidden_layer
        self.memory_counter = 0
        self.memory_full = False
        self.tau = tau
        self.gamma = gamma
        self.num_actor_update_iteration = 0
        self.num_critic_update_iteration = 0
        self.actor_est = HybridActor(d_input_ue, d_output_ue, d_input_bs, d_output_bs, num_ue= self.n_client).to(device)
        self.actor_tar = HybridActor(d_input_ue, d_output_ue, d_input_bs, d_output_bs, num_ue= self.n_client).to(device)
        # self.actor_shared_est_ue = Actor(d_input_ue, d_output_ue).to(device)
        # self.actor_shared_tar_ue = Actor(d_input_ue, d_output_ue).to(device)
        # self.actor_shared_est_bs = Actor(d_input_bs, d_output_bs).to(device)
        # self.actor_shared_tar_bs = Actor(d_input_bs, d_output_bs).to(device)
        self.critic_est = HetGAT_critic(d_input_ue, d_output_ue, d_input_bs, d_output_bs, n_agent= self.n_client, n_hidden_layer= n_hidden_layer).to(device)
        self.critic_tar = HetGAT_critic(d_input_ue, d_output_ue, d_input_bs, d_output_bs, n_agent= self.n_client, n_hidden_layer= n_hidden_layer).to(device)
        self.critic_est2 = HetGAT_critic(d_input_ue, d_output_ue, d_input_bs, d_output_bs, n_agent= self.n_client, n_hidden_layer= n_hidden_layer).to(device)
        self.critic_tar2 = HetGAT_critic(d_input_ue, d_output_ue, d_input_bs, d_output_bs, n_agent= self.n_client, n_hidden_layer= n_hidden_layer).to(device)
        self.optimizer_actor = torch.optim.Adam(self.actor_est.parameters())
        # self.optimizer_actor = torch.optim.RMSprop(self.actor_shared_est_ue.parameters(), lr=1e-4)
        # self.optimizer_actor.add_param_group({"params": self.actor_shared_est_bs.parameters(), "lr":1e-4})
        self.optimizer_critic = torch.optim.Adam(self.critic_est.parameters(), lr = 3e-3)
        self.optimizer_critic2 = torch.optim.Adam(self.critic_est2.parameters(), lr = 3e-3)
        # self.loss_f = nn.MSELoss()
        self.memory = torch.zeros(d_memory, (d_input_ue*self.n_client + d_input_bs)*2 + d_output_ue*self.n_client + d_output_bs + 1).to(device)


    def store(self, s, a, r, s_):
        temp_tensor  = torch.cat((s, a, r, s_), 0)
        if temp_tensor.device != self.memory.device:
            temp_tensor.to(device=self.memory.device)
        self.memory[self.memory_counter % self.d_memory, :] = temp_tensor
        self.memory_counter += 1
        if self.memory_counter >= self.d_memory:
            self.memory_full = True
        return self.memory_full

    def update(self, hidden_x : torch.tensor = None):
        self.optimizer_critic.zero_grad()
        self.optimizer_actor.zero_grad()
        sample_idx = np.random.choice(self.d_memory, self.d_batch, replace=False)
        samples = self.memory[sample_idx, :]
        s = samples[:, :self.d_input_ue*self.n_client + self.d_input_bs]
        # s_ue, s_bs = s[:, :self.d_input_ue*self.n_client].view(self.d_batch, self.n_client, -1), s[:, self.d_input_ue*self.n_client:].view(self.d_batch, -1)
        a = samples[:, self.d_input_ue*self.n_client + self.d_input_bs:(self.d_input_ue  + self.d_output_ue)*self.n_client+self.d_output_bs+ self.d_input_bs]
        r = samples[:, (self.d_input_ue  + self.d_output_ue)*self.n_client+self.d_output_bs+ self.d_input_bs:(self.d_input_ue  + self.d_output_ue)*self.n_client+self.d_output_bs+ self.d_input_bs+1]
        s_ = samples[:, -(self.d_input_ue*self.n_client + self.d_input_bs):]
        # s_ue_, s_bs_ = s_[:, :self.d_input_ue*self.n_client].view(self.d_batch, self.n_client, -1), s_[:, self.d_input_ue*self.n_client:].view(self.d_batch, -1)
        
        #calculate target Q
        
        target_evaluation = r  + self.gamma*torch.min(self.critic_tar(s_, self.actor_tar(s_).detach()), self.critic_tar2(s_, self.actor_tar(s_).detach()))

        evaluation = self.critic_est(s, a.detach())
        loss1 = F.mse_loss(evaluation, target_evaluation.detach())
        
        loss1.backward()
        self.optimizer_critic.step()

        evaluation2 = self.critic_est2(s, a.detach())
        loss2 = F.mse_loss(evaluation2, target_evaluation.detach())
        
        loss2.backward()
        self.optimizer_critic2.step()
        
        #update actor
        loss3 = -self.critic_est(s, self.actor_est(s)).mean()  
        
        loss3.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_est.parameters(), max_norm=1.0)
        # print(self.critic_est.critic_bs.l1.weight.grad)
        self.optimizer_actor.step()

        for param, target_param in zip(self.critic_est.parameters(), self.critic_tar.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_est2.parameters(), self.critic_tar2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        for param, target_param in zip(self.actor_est.parameters(), self.actor_tar.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        return [loss1, loss3]