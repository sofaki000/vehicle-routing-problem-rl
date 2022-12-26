import os
import time
import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from Models.actor import DRL4TSP
from rl_for_solving_the_vrp.implementation_1.problem_variations import VRP_PROBLEM_DEMANDS_LOADS
from plot_utilities import save_plot_with_multiple_functions_in_same_figure
from rl_for_solving_the_vrp.implementation_1 import config
from rl_for_solving_the_vrp.implementation_1.validate import validate
from  rl_for_solving_the_vrp.implementation_1.Models.critic import StateCritic

device = config.device


def train(actor, critic, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm, num_epochs):
    """Constructs the main actor & critic networks, and performs all training."""

    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join("vrp", '%d' % num_nodes, now)

    print('Starting training')

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)
    best_reward = np.inf

    mean_actor_rewards_per_epoch =  []
    mean_critic_rewards_per_epoch = []
    for epoch in range(num_epochs):
        actor.train()
        critic.train()

        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start

        for batch_idx, batch in enumerate(train_loader):
            static, dynamic, x0 = batch

            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None

            # Full forward pass through the dataset
            tour_indices, tour_logp = actor(static, dynamic, x0)

            # Sum the log probabilities for each city in the tour
            reward = reward_fn(static, tour_indices)

            # Query the critic for an estimate of the reward
            critic_est = critic(static, dynamic).view(-1)

            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            if (batch_idx + 1) % 10 == 0:
                end = time.time()
                times.append(end - start)
                start = end
                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])
                print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(train_loader), mean_reward, mean_loss,
                       times[-1]))

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards) # for actor
        mean_critic_reward = np.mean(critic_rewards)

        mean_actor_rewards_per_epoch.append(mean_reward)
        mean_critic_rewards_per_epoch.append(mean_critic_reward)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        # Save rendering of validation set tours
        valid_dir = os.path.join(save_dir, '%s' % epoch)
        mean_valid, result_tour_indixes= validate(valid_loader, actor, reward_fn, render_fn, valid_dir, num_plot=5)

        # Save best model parameters
        if mean_valid < best_reward:
            best_reward = mean_valid

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

        time_taken =  time.time() - epoch_start
        print(f'Mean epoch loss/reward: {mean_loss:.4f}, {mean_reward:.4f}, {mean_valid:.4f}, took: {time_taken:.4f}s ({ np.mean(times):.4f}s / 100 batches)\n')

    # all epochs finished
    # we plot mean_actor_rewards_per_epoch, mean_critic_rewards_per_epoch
    results = [mean_critic_rewards_per_epoch, mean_actor_rewards_per_epoch]
    labels = ["critic rewards", "actor rewards"]
    file_name="results/actor_critic_rewards"
    title=f"E:{num_epochs} actor_lr:{actor_lr}, critic_lr:{critic_lr}, num_nodes:{num_nodes}"
    save_plot_with_multiple_functions_in_same_figure(results, labels, file_name, title)

def train_vrp(train_data, valid_data,   num_nodes, hidden_size, num_layers, dropout,
              batch_size,  actor_lr,  critic_lr, max_grad_norm, num_epochs):
    # Goals from paper:
    # VRP10, Capacity 20:  4.84  (Greedy)
    # VRP20, Capacity 30:  6.59  (Greedy)
    # VRP50, Capacity 40:  11.39 (Greedy)
    # VRP100, Capacity 50: 17.23  (Greedy)

    print('Starting VRP training...')
    actor = DRL4TSP(config.STATIC_SIZE,
                    config.DYNAMIC_SIZE,
                    hidden_size,
                    train_data.update_dynamic, train_data.update_mask,
                    num_layers, dropout).to(device)

    critic = StateCritic(config.STATIC_SIZE, config.DYNAMIC_SIZE, hidden_size).to(device)


    train(num_epochs=num_epochs,
          actor=actor,
          critic=critic, num_nodes=num_nodes,
          train_data=train_data, valid_data= valid_data,
          reward_fn= VRP_PROBLEM_DEMANDS_LOADS.reward,
          render_fn=VRP_PROBLEM_DEMANDS_LOADS.render,
          batch_size=batch_size, actor_lr=actor_lr, critic_lr=critic_lr,
          max_grad_norm=max_grad_norm)

    test_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    out, result_tour_indixes = validate(test_loader, actor,
                                        VRP_PROBLEM_DEMANDS_LOADS.reward,
                                        VRP_PROBLEM_DEMANDS_LOADS.render,
                                        config.test_dir,
                                        num_plot=5)

    print('Average tour length: ', out)
    return result_tour_indixes
