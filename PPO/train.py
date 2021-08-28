import time, csv, gym, os
from PPO.PPO import PPO
from parameters import *
import numpy as np
import platform, psutil


max_training_timesteps = int(8e8)  # break training loop if timeteps > max_training_timesteps
log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)

action_std = 0.6  # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2e5)  # action_std decay frequency (in num timesteps)

update_timestep = max_ep_len * 2  # update policy every n timesteps
K_epochs = 10  # update policy for K epochs in one PPO update

eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor

lr_actor = 0.0003  # learning rate for actor network
lr_critic = 0.001  # learning rate for critic network




def train(save_path="PPO/Data/"+ENV_NAME+"/"):
    save_path += "Test"
    while os.path.exists(save_path+"/"):
        save_path += "I"
    save_path+="/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path+"params.csv", "w") as params_file:
        writer = csv.writer(params_file, delimiter="\t")
        writer.writerow(["lr_actor", "lr_critic", "gamma", "K_epochs", "eps_clip", "has_continuous_action_space",
                    "action_std"])
        writer.writerow([lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std])

    env = gym.make(ENV_NAME)
    state_dim = np.prod(list(env.observation_space.shape))
    action_dim = env.action_space.shape[0] if has_continuous_action_space else env.action_space.n

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = time.time()

    log_running_reward = 0
    log_running_episodes = 0
    time_step = 0
    i_episode = 0
    avg_reward = 0

    memory = str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB"
    uname = platform.uname()
    cpufreq = psutil.cpu_freq()
    with open(save_path + 'sysinfo.txt', 'w') as f:
        f.write(
            f"System: {uname.system}\nProcessor: {uname.processor}\nCurrent Frequency: {cpufreq.current:.2f}Mhz\nMemory: {memory}\nPercentage: {psutil.virtual_memory().percent}%")

    # training loop
    with open(save_path+"log.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter="\t")
        writer.writerow(["time_step", "reward_avg", "time"])
        while time_step <= max_training_timesteps:

            state = env.reset()
            current_ep_reward = 0

            for t in range(1, max_ep_len + 1):

                # select action with policy
                action = ppo_agent.select_action(state.flatten())
                state, reward, done, _ = env.step(action)

                # saving reward and is_terminals
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                time_step += 1
                current_ep_reward += reward

                # update PPO agent
                if time_step % update_timestep == 0:
                    ppo_agent.update()

                # if continuous action space; then decay action std of ouput action distribution
                if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                    ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

                # log in logging file
                if time_step % log_freq == 0:
                    # log average reward till last episode
                    avg_reward = log_running_reward / log_running_episodes
                    avg_reward = round(avg_reward, 4)
                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                            avg_reward))
                    writer.writerow([time_step, avg_reward, round(time.time()-start_time, 4)])

                    log_running_reward = 0
                    log_running_episodes = 0

                # break; if the episode is over
                if done:
                    break


            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1

            try:
                ppo_agent.save(save_path + "net.pth")
            except:
                print("ERROR!!!!-NET COULD NOT BE SAVED")
            if avg_reward > reward_bound or time.time()-start_time>time_length*60:
                break
        env.close()


def go():
    train()

if __name__ == '__main__':
   for i in range(8):
        train()
