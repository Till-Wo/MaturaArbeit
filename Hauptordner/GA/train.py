"""
This file contains the code used to train the GA algorithm and the algorithm itself
"""
import os
import random
from scipy.special import softmax
import gym, csv, time
import copy
import numpy as np
import torch
import torch.nn as nn
from parameters import *
import platform, psutil

# Definition of Parameters
MUTATION_STRENGTH = 0.02
POPULATION_SIZE = 20
GOAL_REWARD = 199
max_generation = 2000
CROSSOVER = True


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


def fitness_function(env, nn):
    obs = env.reset()
    total_reward = 0.0
    for i in range(max_ep_len):
        obs_v = torch.FloatTensor([obs])
        act_prob = nn(obs_v)
        if has_continuous_action_space:
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


def main_loop(save_path="GA/Data/" + ENV_NAME + "/"):
    # ---------------------Logging------------------------------------------
    save_path += "Test"
    while os.path.exists(save_path + "/"):
        save_path += "I"
    save_path += "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    memory = str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB"
    uname = platform.uname()
    cpufreq = psutil.cpu_freq()
    with open(save_path + 'sysinfo.txt', 'w') as f:
        f.write(
            f"System: {uname.system}\nProcessor: {uname.processor}\nCurrent Frequency: {cpufreq.current:.2f}Mhz" +
            f"\nMemory: {memory}\nPercentage: {psutil.virtual_memory().percent}%")
    with open(save_path + "params.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter="\t")
        writer.writerow(["MUTATION_STRENGTH", "POPULATION_SIZE", "GOAL_REWARD"])
        writer.writerow([MUTATION_STRENGTH, POPULATION_SIZE, GOAL_REWARD])

    start_time = time.time()
    # =================================================================================
    env = gym.make(ENV_NAME)
    gen_counter = 0
    if has_continuous_action_space:
        population = [[Network(env.observation_space.shape[0], env.action_space.shape[0]), 0] for _ in range(POPULATION_SIZE)]
    else:
        population = [[Network(env.observation_space.shape[0], env.action_space.n), 0] for _ in
                      range(POPULATION_SIZE)]
    for individual in population:
        individual[1] = fitness_function(env, individual[0])

    # ----------------Training Loop--------------------------------------------------------
    with open(save_path + "log.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter="\t")
        writer.writerow(["reward_max", "reward_mean", "time"])
        population.sort(key=lambda p: p[1], reverse=True)
        while True:
            prev_population = population
            population = [population[0]]

            if CROSSOVER:
                parent_idx = np.random.choice(range(len(prev_population)), (POPULATION_SIZE - 1) * 2,
                                              p=softmax([indiv[1] for indiv in prev_population]))
                for i in range(POPULATION_SIZE - 1):
                    child = copy.deepcopy(prev_population[parent_idx[i * 2]][0])
                    father_data = [param.data for param in prev_population[parent_idx[i * 2 + 1]][0].parameters()]
                    for i, param in enumerate(child.parameters()):
                        if random.getrandbits(1):
                            param.data = father_data[i]
                    child = mutate(child)
                    fitness = fitness_function(env, child)
                    population.append((child, fitness))

            else:
                parent_idx = np.random.choice(range(len(prev_population)), (POPULATION_SIZE - 1),
                                              p=softmax([indiv[1] for indiv in prev_population]))
                for _ in range(POPULATION_SIZE - 1):
                    parent = prev_population[parent_idx[_]][0]
                    child = mutate(parent)
                    fitness = fitness_function(env, child)
                    population.append((child, fitness))

            # ====================Logging=============================
            rewards = [p[1] for p in population]
            avg_reward = np.mean(rewards)
            max_reward = np.max(rewards)
            writer.writerow([max_reward, avg_reward, time.time() - start_time])
            print(f"gen: {gen_counter} \t max_reward: {max_reward} \t avg_reward: {avg_reward}")
            population.sort(key=lambda p: p[1], reverse=True)
            try:
                    torch.save(population[0][0], f"{save_path}net.pth")
            except:
                print("ERROR!!!!---NET COULD NOT BE SAVED")
            # =====================================================================

            if avg_reward > reward_bound or gen_counter >= max_generation or time.time() - start_time > time_length * 60:
                break
            gen_counter += 1
