from rl_for_solving_the_vrp.src import config
from rl_for_solving_the_vrp.src.models_impl_1.RLAgent_implt1 import RLAgent
from rl_for_solving_the_vrp.src.models_impl_2.RLAgent_implt2 import RLAgent_implt2

device = config.device

def get_agent(EVRP, train_data,hidden_size,num_layers, dropout):
    if EVRP:
        # pros to paron mono autos o agent dinei meaningful apotelesmata gia to evrp
        agent = RLAgent_implt2(static_features=config.STATIC_SIZE, dynamic_features=config.DYNAMIC_SIZE,
                               embedding_dim=config.embedding_dim, hidden_size=config.hidden_size,
                               exploring_c=10,
                               n_processing=3,
                               update_fn=train_data.update_dynamic,
                               beam_width=1,
                               capacity=60, velocity=40,
                               cons_rate=0.2, t_limit=config.t_limit, num_afs=config.num_afs)
    else:
        if hasattr(train_data, 'initial_update_mask'):
            # einai evr problem
            agent = RLAgent(hidden_size=hidden_size, update_dynamic=train_data.update_dynamic,
                            update_mask=train_data.update_mask, num_layers=num_layers, dropout=dropout,
                            initialize_mask_fn= train_data.initial_update_mask).to(device)
        else:
            agent = RLAgent(hidden_size=hidden_size, update_dynamic=train_data.update_dynamic,
                            update_mask=train_data.update_mask, num_layers=num_layers, dropout=dropout,
                            initialize_mask_fn=None).to(device)

    return agent