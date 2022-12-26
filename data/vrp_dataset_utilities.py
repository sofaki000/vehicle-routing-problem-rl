import numpy as np
import pandas as pd
import torch

# this utilities are for the demands-loads problem.
from rl_for_solving_the_vrp.implementation_1 import config


def get_loads_and_demands(use_test_data=True):
    if use_test_data:
        loads, demands = get_random_loads_and_demands(config.train_size, config.num_nodes)
    else:
        locations, loads, demands = get_excel_data(file_path=config.data_path)
        loads, demands = get_loads_and_demands_from_file(loads, demands)
    return loads, demands

def get_loads_and_demands_from_file(loads, demands):
    depot_loads = np.max(loads)
    loads = [depot_loads/depot_loads for i in range(len(demands))]

    demands = torch.tensor(demands) / float(depot_loads)
    demands = demands[None, None, :] # prepei na exei shape: (num_samples, 1 , num_nodes)
    demands[:, 0, 0] = 0  # depot starts with a demand of 0
    loads = torch.tensor(loads, dtype=torch.float32)[None, None, :]
    return loads, demands

# loads, demands = get_random_loads_and_demands(config.train_size, config.num_nodes)

def get_random_loads_and_demands(num_samples, input_size):
    max_load = 20
    max_demand = 9
    if max_load < max_demand:
        raise ValueError(':param max_load: must be > max_demand')

    # All states will broadcast the drivers current load
    # Note that we only use a load between [0, 1] to prevent large
    # numbers entering the neural network
    # dynamic_shape = (num_samples, 1, input_size + 1) why +1?
    dynamic_shape = (num_samples, 1, input_size)
    loads = torch.full(dynamic_shape, 1.)
    # All states will have their own intrinsic demand in [1, max_demand),
    # then scaled by the maximum load. E.g. if load=10 and max_demand=30,
    # demands will be scaled to the range (0, 3)
    demands = torch.randint(1, max_demand + 1, dynamic_shape)
    demands = demands / float(max_load)
    demands[:, 0, 0] = 0  # depot starts with a demand of 0

    return loads, demands

def get_excel_data(file_path):
    df = pd.read_excel(file_path)
    df = pd.DataFrame(df)
    latitude_x = df.iloc[:, 1]
    longtitude_y = df.iloc[:, 2]
    demands = df.iloc[:, 3]
    loads = df.iloc[:, 4][0]
    return [latitude_x.tolist(), longtitude_y.tolist()], loads,demands.tolist()


#locations, loads, demands = get_excel_data(file_path=config.data_path)

# map = Map(center=config.thessaloniki_coordinates, zoom_start=13)
# map.add_markers_at_points(locations)
# map.show_map("initial_map_meaningful", open_in_browser=True, save_png=True)
#
#
