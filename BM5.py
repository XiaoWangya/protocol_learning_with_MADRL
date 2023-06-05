"""
Only smart edge server
"""
import torch, os                       # Importing the torch library for PyTorch deep learning framework
import numpy as np                    # Importing NumPy library 
from tensorboardX import SummaryWriter # Importing SummaryWrite from TensorBoard, which is used to write events for displaying them in TensorBoard visualization tool
from env import protocol_learning as plenv   # Importing a module 'protocol_learning' located within the same directory, using an alias 'plenv'
from TD3 import TD3  # Importing a class named 'HetGAT_MADRL_PL' located within 'HetGAT_MADRL' module/package
from utils import *                     # Importing all content from a module 'utils' located within the same directory
from get_args import args                # Importing argument values from a module 'get_args' located within the same directory

all_random(3417)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
parallel_available, num_devices = gpu_setting()   # Assigning two variables 'parallel_available' and 'num_devices' with the values returned by the function 'gpu_setting'. This function is not defined in this code snippet. It might be defined in one of the imported modules.

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'   # Checking if GPU is available or not. If available, assign it to the device variable, otherwise assign CPU.

# Setting environment parameters
n_client = args.n_client                  # Number of clients
n_subcarrier = max(1, np.floor(args.SCR*n_client))   # Maximum number of subcarriers, as calculated based on user input 'SCR'

# Setting RL (Reinforcement Learning) parameters
d_input_ue = (args.record_length+1)*2+5   # Input dimensionality of the UE (User Equipment). It is based on 'record_length' and energy records. 
d_output_ue = 1   # Output dimensionality of the UE, which is a single number representing local requests only.
d_input_bs = (args.record_length*2+3+5)*n_client  # Input dimensionality of BS (Base Station), based on 'record_length', energy records, and n_clients.
d_output_bs = n_client*3   # Output dimensionality of the BS, which includes power, bandwidth, and grants stored in a 1 * num_clients array
d_memory = args.d_memory   # Memory capacity for storing experiences while training

def main():
    Bernard = SummaryWriter(comment='_BM5')    # Assigning an instance of SummaryWriter class to Bernard variable, to be used to write summaries to TensorBoard log files.

    Agent = TD3(d_input_bs, d_output_bs, args.n_hidden, d_memory, args.batch, tau= 5e-3).to(device)   # Creating an instance of HetGAT_MADRL_PL class with given inputs as parameters
    Agent.memory = Agent.memory.to(device)
    epoch = 0
        # initial epoch setting
    for loops in range(args.loops):   # Running the loop for given number of iterations
        environment = plenv(n_client, n_subcarrier, record_length=args.record_length, channel_model= 'Rayleigh')  # Creating an instance of plenv class with given inputs as parameters.
        grants_his = [0 for _ in range(n_client)]
        access_his = [0 for _ in range(n_client)]
        while not environment.done:   # While the environment has not been completed (i.e., done=False)
            epoch += 1    # Incrementing the value of epoch by 1
            requests = torch.ones(d_output_ue*n_client).to(device)
            for i in range(n_client):
                if environment.statbility_info['indicator'][i] < 0.5* environment.stability_constant[i]:
                    requests[i] *= 0
            
            
            state_bs = environment.get_state_bs(requests.tolist())  # Getting new state of BS, based on requests made by UE
            state_bs = state_dic2tensor(state_bs).to(device).view(1, d_input_bs)  # Converting the dictionary object of BS state into a tensor object       
            action_bs = torch.normal(Agent.actor(state_bs).detach(), args.action_noise).clip(0, 1)  # Getting predicted action for BS from actor network, which takes current BS state as input
            action_bs_r = action_bs.view(3, n_client)  # Reshaping the output of actor network into a specific shape of 3 * num_clients

            grants = torch.zeros_like(requests )  # Computing 'grants' based on predicted action by binarizing one of the actions and transposing the other.
            grants[torch.topk(action_bs_r[0, :].masked_fill(requests == 0, float('-inf')), int(args.SCR*args.n_client))[-1]]= requests[torch.topk(action_bs_r[0, :].masked_fill(requests == 0, float('-inf')), int(args.SCR*args.n_client))[-1]]
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
                args.action_noise *= (1-5e-4)
            access_his = [requests[_] + access_his[_] for _ in range(n_client)]    
            grants_his = [grants[_] + grants_his[_] for _ in range(n_client)]    
            
            Bernard.add_scalars('agent requests', {'ue%d'%index: value for index, value in enumerate(requests)}, global_step=epoch ) 
            Bernard.add_scalars('BS grants', {'ue%d'%index: value for index, value in enumerate(grants)}, global_step=epoch ) 
            Bernard.add_scalars('channel', {'ue%d'%index: value for index, value in enumerate(abs(environment.channel))}, global_step=epoch ) 
        # environment = plenv(n_client, n_subcarrier, record_length=args.record_length)   # Creating a new instance of plenv at the end of each loop.
        access_efficiency = [grants_his[_]/max(1, access_his[_]) for _ in range(n_client)]   
        Bernard.add_scalar('rounds exist', environment.rounds, global_step= loops)
        Bernard.add_scalar('Reward', sum(environment.reward_list), global_step=loops)
        Bernard.add_scalars('energy remain', {'ue%d'%index: value for index, value in enumerate(environment.ue_energy_list)}, global_step=loops )
        Bernard.add_scalars('Access efficiency', {'ue%d'%index: value for index, value in enumerate(access_efficiency)}, global_step=loops) 
    Bernard.close()   # Closing the TensorBoard Summary Writer instance
    
if __name__ == "__main__":
    main()   # Calling the main function
