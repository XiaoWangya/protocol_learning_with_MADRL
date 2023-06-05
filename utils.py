import numpy as np                  # Importing the NumPy module and aliased it as np
import scipy as scp                 # Importing the SciPy module and aliased it as scp
import torch                         # Importing the PyTorch module
import torch.nn.functional as F     # Importing the functional interface of PyTorch's neural network module and aliased it as F
import random                        # Importing the random module
import itertools                    

def all_random(seed: int = 1) -> None:
    """Set the seed for random operations.

    Args:
        seed (int, optional): Seed value. Defaults to 1.
    """
    random.seed(1)                   # Set the seed value for Python's built-in random module
    torch.manual_seed(1)             # Set the seed value for PyTorch's random generator
    
def gpu_setting(idx: int = 0) -> tuple[bool, int]:
    """Check the availability of GPUs and print their details.

    Args:
        idx (int, optional): Index value. Defaults to 0.

    Returns:
        Tuple[bool, int]: A tuple containing information about the availability of parallel processing units (GPUs), and the number of available devices.
    """
    parallel_available = False
    num_devices = torch.cuda.device_count()   # Get the number of available GPUs       # Initialize a boolean flag 
    if not idx:                      # If idx is not provided or is equal to 0
        if torch.cuda.is_available():# Check if CUDA (GPU) is available on the system
            if num_devices > 1:      # Check if there's more than one GPU available
                parallel_available = True     # Set the flag as True if multiple GPUs are available
            for i in range(num_devices):  # Iterate over the available GPUs and print their names
                print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available.")   # Print a message if no GPU is available on the system
            
    return parallel_available, num_devices   # Return information about the availability of parallel processing units (GPUs) and the number of available devices

def state_dic2tensor(state_list) -> torch.Tensor:
    """Convert a list of dictionaries to a flattened tensor.

    Args:
        state_list (list): A list of dictionary objects containing states.

    Returns:
        torch.Tensor: A flattened tensor object that contains all state values.
    """
    state_list_1 = [list(state_list[_].values()) for _ in range(len(state_list))]  # Create a nested list by getting the values from each dictionary in state_list
    state_list_2 = flatten(state_list_1)     # Flatten the nested list using the custom function `flatten()`
    return torch.tensor(state_list_2)    # Convert the flattened list into a PyTorch tensor object and return it

def normalized_bandwidth(bandwidth_list, grants) -> np.ndarray:
    """Normalize non-zero values in a numpy array.

    Args:
        bandwidth_list (np.ndarray): An array containing bandwidth values.

    Returns:
        np.ndarray: The normalized bandwidth array.
    """
    x_nonzero = bandwidth_list[grants!= 0]  # Create a new array containing only the non-zero values from the input array
    bandwidth_list[grants!= 0] = F.softmax(x_nonzero.float(), dim = 0).float()  # Normalize the non-zero values using the softmax function, which returns a probability distribution along the provided axis (dim=0)
    bandwidth_list[grants == 0] = 0
    return bandwidth_list   # Return the normalized bandwidth array

def flatten(lst) -> list:
    """Flatten a nested list.

    Args:
        lst (list): A nested list.

    Returns:
        list: The flattened list object.
    """
    flat_list = []  # Initialize an empty list
    for item in lst:   # Iterate over each element of the input list
        if isinstance(item, list):   # If the current element is a list
            flat_list.extend(flatten(item))   # Recursively call this function to flatten the current element
        elif isinstance(item, np.ndarray):  # If the current element is a NumPy array
            flat_list.extend(item.flatten())  # Append the flattened version of the NumPy array to the output list
        else:
            flat_list.append(item)  # If the current element is neither a list nor a NumPy array, simply append it to the output list
    return flat_list   # Return the flattened list object

def shrink(x) -> list:
    """Convert a PyTorch tensor object to a flattened list.

    Args:
        x (torch.Tensor): A PyTorch tensor object.

    Returns:
        list: The flattened list representation of the input tensor.
    """
    return x.cpu().detach().tolist()  # Convert the tensor to a numpy array, remove all dimensions with size 1 (i.e., squeeze), then convert the numpy array to a flattened list. 
