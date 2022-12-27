from rl_for_solving_the_vrp.src import config
from rl_for_solving_the_vrp.src.problem_variations.ELECTRIC_VRP_PROBLEM_WITH_TIME_AND_DEMANDS import \
    EVRP_WITH_TIME_AND_DEMANDS
from rl_for_solving_the_vrp.src.problem_variations.VRP_PROBLEM_DEMANDS_LOADS import VehicleRoutingDataset
from rl_for_solving_the_vrp.src.problem_variations.EVRP import EVRP

DYNAMIC_SIZE = 0

def get_problem(evrp=True,with_demands=True, loads=None, demands=None):
    if evrp and with_demands:
        train_data = EVRP_WITH_TIME_AND_DEMANDS(train_size=config.train_size,
                                            num_nodes=config.num_nodes,
                                            t_limit=config.t_limit,
                                            capacity=config.capacity, num_afs=config.num_afs,
                                            velocity=config.velocity,
                                            seed=config.seed)

        DYNAMIC_SIZE = 3
    elif evrp:
        train_data = EVRP(train_size=config.train_size,
                                            num_nodes=config.num_nodes,
                                            capacity=config.capacity,
                                            num_afs=config.num_afs,
                                            seed=config.seed)
        DYNAMIC_SIZE = 1
    else:
        train_data = VehicleRoutingDataset(num_samples=config.train_size,
                                           nodes_number=config.num_nodes,
                                           loads=loads,
                                           demands=demands,
                                           locations=None)
        DYNAMIC_SIZE = 2
    return train_data

def get_dynamic_size():
    return DYNAMIC_SIZE