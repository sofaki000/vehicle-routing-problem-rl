import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
thessaloniki_coordinates = [40.629269,  22.947412 ]
embedding_dim = 128
seed = 12345
test = False
num_nodes = 3
actor_lr = 5e-4
critic_lr = 5e-4
max_grad_norm = 2.
batch_size = 64
hidden_size = 128
dropout = 0.1
layers = 1
train_size = 100
valid_size = 100

data_path = "C:\\Users\\Lenovo\\Desktop\\Διπλωματική\\vehicle-routing-problem-rl\\rl_for_solving_the_vrp\\data\\vrp_data.xlsx"
num_epochs = 10 #00
# print('NOTE: SETTTING CHECKPOINT: ')
# args.checkpoint = os.path.join('vrp', '10', '12_59_47.350165' + os.path.sep)
# print(args.checkpoint)

# for load and demand only problem:
#DYNAMIC_SIZE = 2  # (load, demand)
STATIC_SIZE = 2  # (x, y)
DYNAMIC_SIZE = 3

test_dir = 'test'


# for electric vehicle routing
capacity = 60
t_limit = 11
num_afs = 3
velocity = 40
cons_rate =0.2
num_nodes_including_depots_afs= num_nodes+num_afs+1 # 1 is for the depot


