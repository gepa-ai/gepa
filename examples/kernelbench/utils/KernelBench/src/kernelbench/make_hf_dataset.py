from datasets import Dataset, DatasetDict
import os


# dataset_example_1 = {
#     "code": """
# import torch
# import torch.nn as nn

# class Model(nn.Module):
#     \"\"\"
#     Simple model that performs a single square matrix multiplication (C = A * B)
#     \"\"\"
#     def __init__(self):
#         super(Model, self).__init__()
    
#     def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
#         \"\"\"
#         Performs the matrix multiplication.

#         Args:
#             A (torch.Tensor): Input matrix A of shape (N, N).
#             B (torch.Tensor): Input matrix B of shape (N, N).

#         Returns:
#             torch.Tensor: Output matrix C of shape (N, N).
#         \"\"\"
#         return torch.matmul(A, B)

# N = 2048

# def get_inputs():
#     A = torch.randn(N, N)
#     B = torch.randn(N, N)
#     return [A, B]

# def get_init_inputs():
#     return []  # No special initialization inputs needed
# """,
#     "level": 1,
#     "name": "1_Square_matrix_multiplication_.py"
# }

# dataset_example_2 = {
#     "code": """
# import torch
# import torch.nn as nn

# class Model(nn.Module):
#     \"\"\"
#     Simple model that performs a convolution, applies ReLU, and adds a bias term.
#     \"\"\"
#     def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
#         super(Model, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
#         self.bias = nn.Parameter(torch.randn(bias_shape)) 

#     def forward(self, x):
#         x = self.conv(x)
#         x = torch.relu(x)
#         x = x + self.bias
#         return x

# batch_size = 128
# in_channels = 3
# out_channels = 16
# height, width = 32, 32
# kernel_size = 3
# bias_shape = (out_channels, 1, 1)

# def get_inputs():
#     return [torch.randn(batch_size, in_channels, height, width)]

# def get_init_inputs():
#     return [in_channels, out_channels, kernel_size, bias_shape]
# """,
#     "level": 2,
#     "name": "1_Conv2D_ReLU_BiasAdd.py"
# }

# dataset_example_3 = {
#     "code":"""
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Model(nn.Module):
#     def __init__(self, input_size, layer_sizes, output_size):
#         \"\"\"
#         :param input_size: The number of input features
#         :param layer_sizes: A list of ints containing the sizes of each hidden layer
#         :param output_size: The number of output features
#         \"\"\"
#         super(Model, self).__init__()
        
#         layers = []
#         current_input_size = input_size
        
#         for layer_size in layer_sizes:
#             layers.append(nn.Linear(current_input_size, layer_size))
#             layers.append(nn.ReLU())
#             current_input_size = layer_size
        
#         layers.append(nn.Linear(current_input_size, output_size))
        
#         self.network = nn.Sequential(*layers)
    
#     def forward(self, x):
#         \"\"\"
#         :param x: The input tensor, shape (batch_size, input_size)
#         :return: The output tensor, shape (batch_size, output_size)
#         \"\"\"
#         return self.network(x)

# # Test code
# batch_size = 1
# input_size = 1000
# layer_sizes = [400, 800]
# output_size = 500

# def get_inputs():
#     return [torch.randn(batch_size, input_size)]

# def get_init_inputs():
#     return [input_size, layer_sizes, output_size]
# """,
#     "level": 3,
#     "name": "1_MLP.py"
# }

# dataset_list = [
#     dataset_example_1,
#     dataset_example_2,
#     dataset_example_3
#]

dataset_list = []

def make_dataset_examples(dir_path, level):
    global dataset_list
    # list all files in the directory
    file_list = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    file_list = sorted(file_list)
    # count = 0
    for f in file_list:
        # if count > 3:
        #     continue
        if f.endswith(".py"):
            file_path = os.path.join(dir_path, f)
            code = open(file_path, "r").read()
            name = f.split(".")[0]
            problem_id = int(name.split("_")[0])
            json_object = {
                "code": "",
                "level": 0,
                "name": "",
                "problem_id": 0
            }
            json_object["code"] = code
            json_object["level"] = level
            json_object["name"] = name
            json_object["problem_id"] = problem_id
            dataset_list.append(json_object)
            # count += 1

make_dataset_examples("../KernelBench/level1", 1)
make_dataset_examples("../KernelBench/level2", 2)
make_dataset_examples("../KernelBench/level3", 3)
make_dataset_examples("../KernelBench/level4", 4)

level_1 = [ex for ex in dataset_list if ex["level"] == 1]
level_2 = [ex for ex in dataset_list if ex["level"] == 2]
level_3 = [ex for ex in dataset_list if ex["level"] == 3]
level_4 = [ex for ex in dataset_list if ex["level"] == 4]


hf_level_1 = Dataset.from_list(level_1)
hf_level_2 = Dataset.from_list(level_2)
hf_level_3 = Dataset.from_list(level_3)
hf_level_4 = Dataset.from_list(level_4)

dataset_dict = DatasetDict({
    "level_1": hf_level_1,
    "level_2": hf_level_2,
    "level_3": hf_level_3,
    "level_4": hf_level_4
})

dataset_dict.push_to_hub("ScalingIntelligence/KernelBench")