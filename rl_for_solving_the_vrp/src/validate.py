import os
import numpy as np
import torch
from rl_for_solving_the_vrp.src import config

device = config.device


def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.', num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""
    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    result_tour_indixes = []
    for batch_idx, batch in enumerate(data_loader):
        static, dynamic, x0 , _ = batch
        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)
            result_tour_indixes.append(tour_indices)

        reward = reward_fn(static, tour_indices).mean().item()
        rewards.append(reward)

        if render_fn is not None and batch_idx < num_plot:
            name = f'batch{batch_idx}_{reward:.4f}.png'
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices, path)

    actor.train()
    return np.mean(rewards),result_tour_indixes