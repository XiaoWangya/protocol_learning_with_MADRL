import argparse
args = argparse.ArgumentParser(description='Description of your program')
# Add arguments
args.add_argument('--n_client', type=int, default = 5, help='Number of clients')
args.add_argument('--SCR', type=float, default = 0.6, help='subcarrier ratio')
args.add_argument('--record_length', type=int, default = 5, help='length of access record')
args.add_argument('--n_hidden', type=int, default = 256, help='dimension of hidden layer')
args.add_argument('--d_memory', type=int, default = 2500, help='size of memory buffer')
args.add_argument('--batch', type=int, default = 64, help='Batch Size')
args.add_argument('--loops', type=int, default = 2000, help='total iterations')
args.add_argument('--pmax', type=float, default = 1.0, help='Description of pmax')
args.add_argument('--action_noise', type= float, default= 3e-1)
args.add_argument('--threshold', type= float, default= 3e-1)

args = args.parse_args()