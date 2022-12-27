import torch

from rl_for_solving_the_vrp.problem import get_problem
from rl_for_solving_the_vrp.src import config
from maps.folium_map_utilities.folium_map import plot_locations_visited_map
from rl_for_solving_the_vrp.src.agent import get_agent
from rl_for_solving_the_vrp.src.problem_variations.ELECTRIC_VRP_PROBLEM_WITH_TIME_AND_DEMANDS import \
    EVRP_WITH_TIME_AND_DEMANDS
from rl_for_solving_the_vrp.src.train import train_vrp
from data.vrp_dataset_utilities import get_excel_data,  get_loads_and_demands
from rl_for_solving_the_vrp.src.problem_variations.VRP_PROBLEM_DEMANDS_LOADS import VehicleRoutingDataset

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    # gets random data for training
    loads, demands = get_loads_and_demands(use_test_data=True)
    train_data = get_problem(evrp=True ,
                             with_demands=False,
                             loads=loads,
                             demands=demands)

    # loads excel data
    loads, demands = get_loads_and_demands(use_test_data=False)
    validation_data = get_problem(evrp=True ,
                                  with_demands=False,
                                  loads=loads,
                                  demands=demands)


    evrp = True  # solve evrp problem
    agent = get_agent(evrp, train_data, config.hidden_size, config.layers, config.dropout)

    result_tour_indixes = train_vrp(agent=agent,
                                    train_data=train_data,
                                    valid_data=train_data,
                                    num_nodes=config.num_nodes,
                                    actor_lr=config.actor_lr,
                                    critic_lr=config.critic_lr,
                                    max_grad_norm=config.max_grad_norm,
                                    batch_size=config.batch_size,
                                    hidden_size=config.hidden_size,
                                    dropout=config.dropout,
                                    num_layers=config.layers,
                                    num_epochs=config.num_epochs)

    # locations, loads, demands = get_excel_data(file_path=config.data_path)
    # locations = torch.FloatTensor(locations)[None, :, :]
    # plot_locations_visited_map(locations, result_tour_indixes)



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

