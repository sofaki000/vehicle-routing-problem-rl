import torch

from rl_for_solving_the_vrp.implementation_1 import config
from rl_for_solving_the_vrp.implementation_1.map import Map
from rl_for_solving_the_vrp.implementation_1.train import train_vrp
from rl_for_solving_the_vrp.implementation_1.utilities.vrp_dataset_utilities import get_loads_and_demands_from_file, \
    get_random_loads_and_demands, get_excel_data
from rl_for_solving_the_vrp.implementation_1.vrp import VehicleRoutingDataset

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    locations, loads, demands = get_excel_data(file_path=config.data_path)
    locations = torch.FloatTensor(locations)[None,:,:]
    loads, demands = get_loads_and_demands_from_file(loads, demands)
    # loads, demands = get_random_loads_and_demands(config.train_size, config.num_nodes)

    train_data = VehicleRoutingDataset(num_samples=config.train_size,
                                       nodes_number=config.num_nodes,
                                       loads=loads,
                                       demands=demands,
                                       locations=locations)

    result_tour_indixes = train_vrp(train_data=train_data,
                                    valid_data=train_data,
                                    seed=config.seed,
                                    num_nodes=config.num_nodes,
                                    actor_lr=config.actor_lr,
                                    critic_lr=config.critic_lr,
                                    max_grad_norm=config.max_grad_norm,
                                    batch_size=config.batch_size,
                                    hidden_size=config.hidden_size,
                                    dropout=config.dropout,
                                    num_layers=config.layers,
                                    num_epochs=config.num_epochs)

    locations = locations.squeeze(0).numpy()
    xs = [locations[0][i] for i in result_tour_indixes][0][0]
    ys = [locations[1][i] for i in result_tour_indixes][0][0]

    print(f'Resulting tour is : {result_tour_indixes}')
    number_of_stops = len(xs)
    points_visiting = []

    for i in range(number_of_stops):
        x = xs[i]
        y = ys[i]
        points_visiting.append([x, y])

    map = Map(center=config.thessaloniki_coordinates, zoom_start=13)
    map.plot_lines_between_points(points_visiting, show_map=True)


    # # Determines the maximum amount of load for a vehicle based on num nodes
    # LOAD_DICT = {4: 20, 10: 20, 20: 30, 50: 40, 100: 50}
    # MAX_DEMAND = 9
    #
    # max_load = LOAD_DICT[config.num_nodes]
    #
    # train_data = VehicleRoutingDataset(num_samples=config.train_size,
    #                                    nodes_number=config.num_nodes,
    #                                    max_load= max_load,
    #                                    max_demand= MAX_DEMAND,
    #                                    seed= config.seed)
    #
    # print('Train data: {}'.format(train_data))
    # valid_data = VehicleRoutingDataset(config.valid_size,
    #                                    config.num_nodes,
    #                                    max_load,
    #                                    MAX_DEMAND,
    #                                    config.seed + 1)
    # # test_data = VehicleRoutingDataset(valid_size,  num_nodes,  max_load,  MAX_DEMAND,  seed + 2)
    # train_vrp(seed=config.seed,
    #           num_nodes=config.num_nodes,
    #           actor_lr=config.actor_lr,
    #           critic_lr=config.critic_lr,
    #           max_grad_norm=config.max_grad_norm,
    #           batch_size=config.batch_size,
    #           hidden_size=config.hidden_size,
    #           dropout=config.dropout,
    #           num_layers=config.layers,
    #           num_epochs=config.num_epochs)

