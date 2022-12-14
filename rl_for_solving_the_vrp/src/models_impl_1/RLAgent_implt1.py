from torch import nn
import torch

from rl_for_solving_the_vrp.src import config
from rl_for_solving_the_vrp.src.models_impl_1.actor import DRL4TSP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from  rl_for_solving_the_vrp.src.models_impl_1.critic import StateCritic

class RLAgent(nn.Module):
    def __init__(self, hidden_size, update_dynamic, update_mask, num_layers, dropout, initialize_mask_fn):
        super().__init__()
        self.ptnet = DRL4TSP(config.STATIC_SIZE,
                        config.DYNAMIC_SIZE,
                        hidden_size,
                        update_fn= update_dynamic ,
                        mask_fn=update_mask,
                        initialize_mask_fn=initialize_mask_fn,
                        num_layers=num_layers, dropout=dropout).to(device)

        self.critic =  StateCritic(static_size=config.STATIC_SIZE, dynamic_size=config.DYNAMIC_SIZE, hidden_size=hidden_size).to(device)

    def forward(self, static, dynamic,decoder_input=None, distances=None):
        """
        :param static: raw data, tensor, (batch, 2, seq_len)
        :param dynamic: (batch, 2, seq_len)
        :param distances: tensor, (batch, seq_len, seq_len)
        :return tours: tensor, (batch, seq_len-1)
        :return prob_log: (batch)
        :return vals: (batch)
        """
        # Full forward pass through the dataset 
        tour_indices, tour_logp = self.ptnet(static, dynamic,decoder_input, distances=distances)
        # Query the critic for an estimate of the reward
        vals = self.critic(static, dynamic).view(-1)
        return tour_indices, tour_logp, vals