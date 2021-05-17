import os

import gym, csv, time
import copy
import numpy as np
import torch
import torch.nn as nn

# Definition of Parameters
MUTATION_STRENGTH = 0.02
POPULATION_SIZE = 50
N_PARENTS = 10
GOAL_REWARD = 199
ENV_NAME = "CartPole-v1"
IS_CONTINIOUS = False
MAX_LENGTH_OF_EPISODE = 500


class Network(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(Network, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_outputs),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.nn(x)


def fitness_function(env, nn, device):
    obs = env.reset()
    total_reward = 0.0
    for i in range(MAX_LENGTH_OF_EPISODE):
        obs_v = torch.FloatTensor([obs]).to(device)
        act_prob = nn(obs_v)
        if IS_CONTINIOUS:
            action = torch.tanh(act_prob)
            action = action.cpu().detach().numpy()[0]
            action = action.clip(env.action_space.low[0], env.action_space.high[0])
            obs, reward, done, _ = env.step(action)
        else:
            acts = act_prob.max(dim=1)[1]
            obs, reward, done, _ = env.step(acts.data.numpy()[0])
        total_reward += reward
        if done:
            break
    return total_reward


def mutate(nn):
    child = copy.deepcopy(nn)
    for param in child.parameters():
        param.data += MUTATION_STRENGTH * torch.randn_like(param)
    return child


def main_loop(save_path="Data/" + ENV_NAME + "/"):
    save_path += "Test"
    while os.path.exists(save_path+"/"):
        save_path += "I"
    save_path+="/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = torch.device("cpu")
    with open(save_path + "params.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter="\t")
        writer.writerow(["MUTATION_STRENGTH", "POPULATION_SIZE", "N_PARENTS", "GOAL_REWARD"])
        writer.writerow([MUTATION_STRENGTH, POPULATION_SIZE, N_PARENTS, GOAL_REWARD])

    start_time = time.time()

    env = gym.make(ENV_NAME)
    gen_counter = 0
    if IS_CONTINIOUS:
        population = [[Network(env.observation_space.shape[0], env.action_space.shape[0]).to(device), 0] for _ in
                      range(POPULATION_SIZE)]
    else:
        population = [[Network(env.observation_space.shape[0], env.action_space.n).to(device), 0] for _ in
                      range(POPULATION_SIZE)]
    for individual in population:
        individual[1] = fitness_function(env, individual[0], device)

    # ----------------Training Loop--------------------------------------------------------
    with open(save_path + "log.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter="\t")
        writer.writerow(["reward_max", "reward_mean", "time"])
        while True:
            population.sort(key=lambda p: p[1], reverse=True)
            rewards = [p[1] for p in population[:N_PARENTS]]

            avg_reward = np.mean(rewards)
            max_reward = np.max(rewards)
            writer.writerow([max_reward, avg_reward, time.time() - start_time])
            print(f"gen: {gen_counter} \t max_reward: {max_reward} \t avg_reward: {avg_reward}")

            if avg_reward > 199:
                population.sort(key=lambda p: p[1], reverse=True)
                torch.save(population[0][0], f"{save_path}net.pth")
                break

            prev_population = population
            population = [population[0]]
            for _ in range(POPULATION_SIZE - 1):
                parent_idx = np.random.randint(0, N_PARENTS)
                parent = prev_population[parent_idx][0]
                child = mutate(parent)
                fitness = fitness_function(env, child, device)
                population.append((child, fitness))
            gen_counter += 1
