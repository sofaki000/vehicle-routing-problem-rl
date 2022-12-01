import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
thessaloniki_coordinates = [40.629269,  22.947412 ]

seed = 12345
test = False
num_nodes = 5
actor_lr = 5e-4
critic_lr = 5e-4
max_grad_norm = 2.
batch_size = 256
hidden_size = 128
dropout = 0.1
layers = 1
train_size = 1 #1000
valid_size = 1 #100

data_path = "C:\\Users\\Lenovo\\Desktop\\Διπλωματική\\vehicle-routing-problem-rl\\rl_for_solving_the_vrp\\data\\vrp_data.xlsx"
num_epochs = 2 #1000
# print('NOTE: SETTTING CHECKPOINT: ')
# args.checkpoint = os.path.join('vrp', '10', '12_59_47.350165' + os.path.sep)
# print(args.checkpoint)

STATIC_SIZE = 2  # (x, y)
DYNAMIC_SIZE = 2  # (load, demand)

test_dir = 'test'