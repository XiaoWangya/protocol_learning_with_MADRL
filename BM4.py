"""
Round
"""
import torch, os                       # Importing the torch library for PyTorch deep learning framework
import numpy as np                    # Importing NumPy library 
from tensorboardX import SummaryWriter # Importing SummaryWrite from TensorBoard, which is used to write events for displaying them in TensorBoard visualization tool
from env import protocol_learning as plenv   # Importing a module 'protocol_learning' located within the same directory, using an alias 'plenv'
from TD3 import TD3  # Importing a class named 'HetGAT_MADRL_PL' located within 'HetGAT_MADRL' module/package
from utils import *                     # Importing all content from a module 'utils' located within the same directory
from get_args import args                # Importing argument values from a module 'get_args' located within the same directory

all_random(args.seeds)  
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
parallel_available, _ = gpu_setting()   # Assigning two variables 'parallel_available' and 'num_devices' with the values returned by the function 'gpu_setting'. This function is not defined in this code snippet. It might be defined in one of the imported modules.

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'   # Checking if GPU is available or not. If available, assign it to the device variable, otherwise assign CPU.

# Setting environment parameters
n_client = args.n_client                  # Number of clients
n_subcarrier = min(args.K, n_client)   # Maximum number of subcarriers, as calculated based on user input 'SCR'

# Setting RL (Reinforcement Learning) parameters
d_input_ue = (args.record_length+1)*2+5   # Input dimensionality of the UE (User Equipment). It is based on 'record_length' and energy records. 
d_output_ue = 1   # Output dimensionality of the UE, which is a single number representing local requests only.
d_input_bs = (args.record_length*2+3+5)*n_client  # Input dimensionality of BS (Base Station), based on 'record_length', energy records, and n_clients.
d_output_bs = n_client*2   # Output dimensionality of the BS, which includes power, bandwidth, and grants stored in a 1 * num_clients array
d_memory = args.d_memory   # Memory capacity for storing experiences while training

def main():
    Bernard = SummaryWriter(comment='_BM4')    # Assigning an instance of SummaryWriter class to Bernard variable, to be used to write summaries to TensorBoard log files.

    Agent = TD3(d_input_bs, d_output_bs, args.n_hidden, d_memory, args.batch, tau= 5e-3).to(device)   # Creating an instance of HetGAT_MADRL_PL class with given inputs as parameters
    Agent.memory = Agent.memory.to(device)
    epoch = 0
        # initial epoch setting
    for loops in range(args.loops):   # Running the loop for given number of iterations
        environment = plenv(n_client, n_subcarrier, record_length=args.record_length, channel_model= 'Rayleigh')  # Creating an instance of plenv class with given inputs as parameters.
        grants_his = [0 for _ in range(n_client)]
        access_his = [0 for _ in range(n_client)]
        request_eng = [0 for _ in range(n_client)]
        transmit_eng = [0 for _ in range(n_client)]
        eff_request_eng = [0 for _ in range(n_client)]
        eff_transmit_eng = [0 for _ in range(n_client)]
        while not environment.done:   # While the environment has not been completed (i.e., done=False)
            epoch += 1    # Incrementing the value of epoch by 1

            requests = torch.ones(d_output_ue*n_client).to(device)
            state_bs = environment.get_state_bs(requests.tolist())  # Getting new state of BS, based on requests made by UE
            state_bs = state_dic2tensor(state_bs).to(device).view(1, d_input_bs)  # Converting the dictionary object of BS state into a tensor object       
            action_bs = torch.normal(Agent.actor(state_bs).detach(), args.action_noise).clip(0, 1)

            action_bs_r = action_bs.view(2, n_client)  # Reshaping the output of actor network into a specific shape of 3 * num_clients
            round_robin = [0 for _ in range(n_client)]
            for i in range(n_client):
                if i < 3:
                    round_robin[(i + epoch)%n_client] = 1
                else:
                    round_robin[(i + epoch)%n_client] = 0
            action_bs_r = torch.cat((torch.tensor(round_robin).to(device).unsqueeze(0), action_bs_r), 0)  # Getting predicted action for BS from actor network, which takes current BS state as input

            grants = torch.zeros_like(requests)  # Computing 'grants' based on predicted action by binarizing one of the actions and transposing the other.
            grants[torch.topk(action_bs_r[0, :].masked_fill(requests == 0, float('-inf')), n_subcarrier)[-1]] = requests[torch.topk(action_bs_r[0, :].masked_fill(requests == 0, float('-inf')), n_subcarrier)[-1]]
            power = action_bs_r[1, :]*args.pmax*grants   # Computing the power allocation to each client
            bandwidth = normalized_bandwidth(action_bs_r[2, :], grants)  # Computing the bandwidth allocated to each client after normalizing the input received from actor network.
            RB_blocks = (shrink(power), shrink(bandwidth))   # Shrinking power and bandwidth allocations using 'shrink' function
            
            reward, done = environment.step(shrink(grants), RB_blocks)   # Computing the reward and new status of environment based on current allocations
            
            state_bs_ = environment.get_state_bs()  # Getting the new state of BS 
            state_bs_ = state_dic2tensor(state_bs_).to(device).view(1, d_input_bs)   # Converting the dictionary object of BS state into a tensor object
        
            Agent.store(state_bs.view(-1), action_bs.view(-1), torch.tensor(reward).view(-1).to(device), state_bs_.view(-1))  # Adding experience tuple to memory buffer
            
            # Writing the summaries to the log file using Summary Writer instance
            
            if Agent.memory_counter>d_memory:
                [loss1, loss2] = Agent.update(args.action_noise)
                Bernard.add_scalar('loss of critic', loss1, global_step=epoch)
                Bernard.add_scalar('loss of actor', loss2, global_step=epoch)
                # args.action_noise *= (1-5e-4)
            access_his = [requests[_] + access_his[_] for _ in range(n_client)]    
            grants_his = [grants[_] + grants_his[_] for _ in range(n_client)]    
            request_eng = [environment.energy_comsup_list_req[_] + request_eng[_] for _ in range(n_client)]
            transmit_eng = [environment.energy_comsup_list_tns[_] + transmit_eng[_] for _ in range(n_client)]
            eff_request_eng = [environment.energy_comsup_list_req[_] + request_eng[_] for _ in range(n_client) if environment.grant_record[_][-1]]
            eff_transmit_eng = [environment.energy_comsup_list_tns[_] + transmit_eng[_] for _ in range(n_client) if environment.bs_grants_[_]]
        args.action_noise *= (1-args.decay_factor)
        access_efficiency = [grants_his[_]/max(1, access_his[_]) for _ in range(n_client)]   
        Bernard.add_scalar('rounds exist', environment.rounds, global_step= loops)
        Bernard.add_scalar('Reward', sum(environment.reward_list), global_step=loops)
        Bernard.add_scalar('Normalized average access efficiency', torch.tensor(access_efficiency).mean()*environment.rounds, global_step=loops)
        Bernard.add_scalar('Normalized average energy efficiency', (2*n_client - sum(environment.ue_energy_list)) / environment.rounds, global_step=loops)
        Bernard.add_scalar('Energy consumption in requesting', sum(request_eng), global_step=loops)
        Bernard.add_scalar('Efficient energy consumption in requesting', sum(eff_request_eng), global_step=loops)
        Bernard.add_scalar('Energy consumption in transmission', sum(transmit_eng), global_step=loops)
        Bernard.add_scalar('Efficient energy consumption in transmission', sum(eff_transmit_eng), global_step=loops)
    Bernard.add_scalar('number of edge devices', n_client)
    Bernard.close()   # Closing the TensorBoard Summary Writer instance
    
if __name__ == "__main__":
    main()   # Calling the main function