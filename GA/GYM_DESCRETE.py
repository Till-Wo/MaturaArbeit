import gym
import copy
import numpy as np
import torch
import torch.nn as nn
import TCF

MUTATION_STRENGTH = 0.02
POPULATION_SIZE = 50
N_PARENTS = 10
GOAL_REWARD = 199


class Network(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(Network, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(n_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, n_outputs),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.nn(x)


def fitness_function(env, nn):
    obs = env.reset()
    total_reward = 0.0
    while True:
        obs_v = torch.FloatTensor([obs])
        act_prob = nn(obs_v)
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


def main_loop(name="CartPole-v0", excel_add="", net_add="", path=""):
    Timer = TCF.TimeIt()
    Writer = TCF.Writer(path + name + excel_add)

    env = gym.make(name)
    gen_counter = 0
    population = [[Network(env.observation_space.shape[0], env.action_space.n), 0] for _ in range(POPULATION_SIZE)]
    for individual in population:
        individual[1] = fitness_function(env, individual[0])


    while True:
        population.sort(key=lambda p: p[1], reverse=True)
        rewards = [p[1] for p in population[:N_PARENTS]]

        avg_reward = np.mean(rewards)
        max_reward = np.max(rewards)
        Writer.save(str(gen_counter), "reward_max", max_reward)
        Writer.save(str(gen_counter), "reward_mean", avg_reward)
        Writer.save(str(gen_counter), "time", Timer.update_and_reset())
        print(f"gen: {gen_counter} | max_reward: {max_reward} | avg_reward: {avg_reward}")




        if avg_reward > 199:
            torch.save(population[-1][0], f"Saved_Nets/{name+net_add}.pth")
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
            fitness = fitness_function(env, child)
            population.append((child, fitness))
        gen_counter += 1
