"""
This file contains code to make animate gifs of the training results
This is a modified version of https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/make_gif.py

The licence:

MIT License

Copyright (c) 2018 Nikhil Barhate

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import glob
from PIL import Image
import numpy as np
from parameters import *
import gym

from PPO import PPO

def save_gif_images(env_name, has_continuous_action_space, max_ep_len, action_std, path):
    print("============================================================================================")

    total_test_episodes = 1  # save gif for only one episode

    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor
    lr_critic = 0.001  # learning rate for critic

    env = gym.make(env_name)

    # state space dimension
    state_dim = np.prod(list(env.observation_space.shape))

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # make directory for saving gif images
    gif_images_dir = path+"gif_images"
    if not os.path.exists(gif_images_dir):
        os.makedirs(gif_images_dir)



    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)


    ppo_agent.load(path+"net.pth")

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes + 1):

        ep_reward = 0
        env.render()
        state = env.reset()

        for t in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(state.flatten())
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            img = env.render(mode='rgb_array')

            img = Image.fromarray(img)
            img.save(gif_images_dir + '/' + str(t).zfill(6) + '.jpg')

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
    env.close()

    print("============================================================================================")

    print("total number of frames / timesteps / images saved : ", t)

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


def save_gif(env_name, path):
    print("============================================================================================")

    gif_num = 0  #### change this to prevent overwriting gifs in same env_name folder

    # adjust following parameters to get desired duration, size (bytes) and smoothness of gif
    total_timesteps = 300
    step = 10
    frame_duration = 150

    # input images
    gif_images_dir = path + "gif_images/" +  '/*.jpg'

    gif_path = path + 'PPO_' + env_name + '_gif_' + str(gif_num) + '.gif'

    img_paths = sorted(glob.glob(gif_images_dir))
    img_paths = img_paths[:total_timesteps]
    img_paths = img_paths[::step]

    print("total frames in gif : ", len(img_paths))
    print("total duration of gif : " + str(round(len(img_paths) * frame_duration / 1000, 2)) + " seconds")

    # save gif
    img, *imgs = [Image.open(f) for f in img_paths]
    img.save(fp=gif_path, format='GIF', append_images=imgs, save_all=True, optimize=True, duration=frame_duration,
             loop=0)

    print("saved gif at : ", gif_path)

    print("============================================================================================")

if __name__ == '__main__':
    for i in range(10):
        env_name = ENV_NAME
        path = "Data/" + env_name + "/Test" + "I"*i + "/"
        action_std = 0.1           # set same std for action distribution which was used while saving
        save_gif_images(env_name, has_continuous_action_space, max_ep_len, action_std, path)

        save_gif(env_name, path)
