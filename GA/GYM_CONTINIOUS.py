import gym
import copy
import numpy as np
import torch
import torch.nn as nn
import TCF

MUTATION_STRENGTH = 0.01
POPULATION_SIZE = 50
N_PARENTS = 10
GOAL_REWARD = 199


class Network(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(Network, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(n_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, n_outputs)

        )

    def forward(self, x):
        return self.nn(x)


def fitness_function(env, nn, device, render= False):
    obs = env.reset()
    total_reward = 0.0
    while True:
        if render:
            env.render()

        obs_v = torch.FloatTensor([obs]).to(device)
        action = nn(obs_v)
        action = torch.tanh(action)
        action = action.cpu().detach().numpy()[0]
        action = action.clip(env.action_space.low[0], env.action_space.high[0])
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    if render:
        env.close()
    return total_reward


def mutate(nn):
    child = copy.deepcopy(nn)
    for param in child.parameters():
        param.data += MUTATION_STRENGTH * torch.randn_like(param)
    return child


def main_loop(name="BipedalWalker-v3", excel_add="", net_add=""):
    Timer = TCF.TimeIt()
    Writer = TCF.Writer(name+excel_add)
    device = torch.device("cpu") #set on cuda if you want to run it on the GPU

    env = gym.make(name)
    gen_counter = 0
    population = [[Network(env.observation_space.shape[0], env.action_space.shape[0]).to(device), 0] for _ in range(POPULATION_SIZE)]
    for individual in population:
        individual[1] = fitness_function(env, individual[0], device)


    while True:
        population.sort(key=lambda p: p[1], reverse=True)
        rewards = [p[1] for p in population[:N_PARENTS]]

        avg_reward = np.mean(rewards)
        max_reward = population[0][1]
        Writer.save(str(gen_counter), "reward_max", max_reward)
        Writer.save(str(gen_counter), "reward_mean", avg_reward)
        Writer.save(str(gen_counter), "time", Timer.update_and_reset())
        print(f"gen: {gen_counter} | max_reward: {max_reward} | avg_reward: {avg_reward}")

        """
        env2 = gym.wrappers.Monitor(env, directory=f"mon\\{gen_counter}", force=True)
        fitness_function(env2, population[0][0], True)
        env2.close()
        """

        if avg_reward > 100:
            torch.save(population[-1][0], f"Saved_Nets\\{name+net_add}.pth")
            Writer.save(str(0), "MUTATION_STRENGTH", MUTATION_STRENGTH)
            Writer.save(str(0), "POPULATION_SIZE", POPULATION_SIZE)
            Writer.save(str(0), "N_PARENTS", N_PARENTS)
            Writer.save(str(0), "total time since start", Timer.time_since_start())
            Writer.close()
            break



        # TODO Other algorithms
        prev_population = population
        population = [population[0]]
        for _ in range(POPULATION_SIZE-1):
            parent_idx = np.random.randint(0, N_PARENTS)
            parent = prev_population[parent_idx][0]
            child = mutate(parent)
            fitness = fitness_function(env, child, device)
            population.append((child, fitness))
        gen_counter += 1
