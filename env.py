""" 
    author : Zixin Wang
    shanghaitech unitersity
    protocol learning environment
"""
# import 
import numpy as np
from utils import flatten
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
#
class protocol_learning():
    def __init__(self, 
                 n_client: int = 2, 
                 n_subcarrier : int = 10, 
                 n_antenna_bs : int = 1,
                 n_antenna_ue : int = 1,
                 bandwidth:int = 1, 
                 channel_model : str= 'guassian',
                 noise_level : int = -100, 
                 record_length:int = 5
                 ):
        self.n_client = n_client
        self.n_subcarrier = n_subcarrier
        self.n_antenna_bs = n_antenna_bs
        self.n_antenna_ue = n_antenna_ue
        self.bandwidth = bandwidth
        self.channel_model = channel_model
        self.noise_in_watt = 10**(noise_level/10 - 3)
        self.energy_enc = 1e-1
        self.record_length = record_length
        self.channel = np.random.rand(n_client, n_antenna_bs, n_antenna_ue) + 1j*np.random.rand(n_client, n_antenna_bs, n_antenna_ue)
        self.ue_energy_list = [2 for _ in range(n_client)]
        self.control_status_list, self.stability_constant = self.control_status_initialization()
        # print(self.stability_constant)
        self.request_record = [[0 for _ in range(record_length)] for k in range(self.n_client)]
        self.grant_record = [[0 for _ in range(record_length)] for k in range(self.n_client)]
        self.drift_plus_penalty_paraneter = 1
        self.statbility_info = {'stability':[0 for _ in range(self.n_client)], 'indicator': [0 for _ in range(self.n_client)]}
        self.done = False
        self.rounds = 0
        self.reward_list = []
    
    def step(self, bs_grants, bs_RB_allocation):
        self.rounds += 1
        
        # calculate rate, delay, and energy consumption
        transmit_power_list, bandwidth_list = bs_RB_allocation
        rate_list = [self.cal_rate(self.channel[k, : , : ], bandwidth_list[k]*self.bandwidth, transmit_power_list[k]) for k in range(self.n_client)]
        delay_list = [1/(rate+1e-50) for rate in rate_list]
        energy_comsup_list = flatten([min(self.ue_energy_list[k], bs_grants[k]*transmit_power_list[k]*delay_list[k] + self.request_record[k][-1]*self.energy_enc)  for k in range(self.n_client)])
        self.ue_energy_list = [max(0, self.ue_energy_list[k] -( energy_comsup_list[k] + self.request_record[k][-1]*self.energy_enc) )for k in range(self.n_client)]
        
        # evolution of control state and check the stability of system
        self.check_done(bs_grants)
        self.evolution_control_status(bs_grants) 
        #update the reward
        reward = -sum(flatten([energy_comsup_list[k] for k in range(self.n_client)])) - self.drift_plus_penalty_paraneter*sum(flatten([energy_comsup_list[k]*(energy_comsup_list[k] - 2*self.ue_energy_list[k]) for k in range(self.n_client)])) + bool(self.rounds*sum(bs_grants))- self.done + bool(sum(self.ue_energy_list)==0) -0.1*sum([(rate_list[k])<2*bs_grants[k] for k in range(self.n_client)])#14(a)

        #update channel
        if self.channel_model == 'guassian':
            self.channel = np.random.normal(size=(self.n_client, self.n_antenna_bs, self.n_antenna_ue)) + 1j*np.random.normal(size=(self.n_client, self.n_antenna_bs, self.n_antenna_ue))
        elif self.channel_model == 'Rayleigh':
            variantion = np.random.normal(size=(self.n_client, self.n_antenna_bs, self.n_antenna_ue)) + 1j*np.random.normal(size=(self.n_client, self.n_antenna_bs, self.n_antenna_ue))
            self.channel = 0.9*self.channel + np.sqrt(1-0.9**2) * variantion

        # update request & grants record
        for k in range(self.n_client):
            self.grant_record[k][:self.record_length -1] = self.grant_record[k][1:]
            self.grant_record[k][-1] = bs_grants[k]
        self.reward_list.append(reward)
        #update state
        
        return reward, self.done

    def check_done(self, scheduling = None):
        #print(sum(self.statbility_info['stability']))
        temp1 = []
        temp2 = []
        for k in range(self.n_client):
            A, B, K, P, x = self.control_status_list[k]
            # dvdt = (A-scheduling[k]*B@K).dot(x).T.dot(Z).dot(A-scheduling[k]*B@K).dot(x) - x.T.dot(Z).dot(x)
            # v_x = x.T@P@x
            # dvdt = x.T@((A-scheduling[k]*B@K).T@P + P@(A-scheduling[k]*B@K))@x
            # dvdt = x.T@P@x -np.dot(np.dot(-(B@K@x).T, -(B@K@x)), 1)*scheduling[k]
            # temp.append((dvdt>0)*1)
            temp1.append((((A@x -scheduling[k]*B@K@x).T@P@(A@x -scheduling[k]*B@K@x) - self.stability_constant[k])>0)*1)
            temp2.append((self.ue_energy_list[k]==0)*1)
        self.done = (sum(temp1)>0)*5 + (sum(temp2)>0)*1

    def get_state_ue(self):
        state_ue = [
            {
            'channel':[abs(self.channel[k].squeeze())],
            'energy_remain':[self.ue_energy_list[k]],
            'request_history':self.request_record[k],
            'grants':self.grant_record[k],
            'control status':self.statbility_info['indicator'][k]
            } for k in range(self.n_client)]
        # #print(state_ue, '\n')
        return state_ue
    
    def get_state_bs(self, ue_requests = None):
        if ue_requests is not None:
            # #print(ue_requests, '\n')
            for k in range(self.n_client):
                self.request_record[k][:self.record_length -1] = self.request_record[k][1:]
                self.request_record[k][-1] = ue_requests[k]
                self.ue_energy_list[k] -= ue_requests[k]*self.energy_enc
        # #print(self.channel)
        state_bs = [
            {
            'channel':abs(self.channel.squeeze()).tolist(),
            'stability_indicator':self.statbility_info['indicator'],
            'energy_remain':self.ue_energy_list, 
            'request_history':self.request_record,
            'grants':self.grant_record, 
            'control status':[self.control_status_list[k][-1].squeeze().tolist() for k in range(self.n_client)]
            }]
        # #print(state_bs, '\n')
        return state_bs

    def cal_rate(self, channel, bandwidth, power):
        return bandwidth*np.log(1+ power*abs(channel)**2/self.noise_in_watt)
    
    def control_status_initialization(self, shape :int = 5):
        control_status_list = []
        stability_constant = []
        max_x = np.ones((shape, 1))
        for i in range(self.n_client):
            x = np.ones((shape, 1))*0.5
            A, B, K, P = self.build_linear_system(shape)
            control_status_list.append([A, B, K, P, x])
            stability_constant.append(max_x.T@P@max_x)
        return control_status_list, stability_constant
    
    def evolution_control_status(self, scheduling):
        
        for i in range(self.n_client):
            state_transition_matrix, control_action_matrix, feedback_gain_matrix, lyapunov_matrix, control_status = self.control_status_list[i]
            # n, n = state_transition_matrix.shape
            # self.control_status_list[i][-1] = state_transition_matrix @control_status -scheduling[i]*control_action_matrix@ feedback_gain_matrix@control_status 
            self.control_status_list[i][-1] = state_transition_matrix @control_status - scheduling[i]*control_action_matrix@ feedback_gain_matrix@control_status + np.random.normal(scale= 1e-4, size= control_status.shape)

        # check stability
            stability_indicator = (state_transition_matrix@self.control_status_list[i][-1]).T@lyapunov_matrix@state_transition_matrix@self.control_status_list[i][-1]
            # stability_indicator = 1

            # self.statbility_info['stability'][i] = ((self.control_status_list[i][-1].T@lyapunov_matrix@self.control_status_list[i][-1] - self.stability_constant[i])>0)*1
            self.statbility_info['indicator'][i] = stability_indicator

    def build_linear_system(self, shape):
        """
        Builds a linear system with the given shape.
        Calculates the state transition matrix A, control action matrix B,
        optimal LQR gain K, and Lyapunov matrix P.

        Args:
            shape (tuple): A tuple containing the number of states and number of inputs.

        Returns:
            tuple: A tuple containing the state transition matrix A, control action matrix B,
            optimal LQR gain K, and Lyapunov matrix P.
        """
        spectral_radius = 1.05
        # Generate random matrices for A and B with spectral radius greater than unity
        A = np.eye(shape)*spectral_radius
        B = np.eye(shape)
        # Calculate the optimal LQR gain K using the Lyapunov equation
        Q = np.eye(shape)  # State cost matrix
        R = np.eye(shape)  # Control cost matrix
        P = solve_discrete_are(A, B, Q, R)  # Solve the continuous-time algebraic Riccati equation
        K = np.linalg.inv(R) @ B.T @ P  # Calculate the optimal LQR gain
        return A, B, K, P