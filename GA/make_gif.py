import os
import glob
from PIL import Image
from parameters import *
import torch
import gym
device = torch.device("cpu")

def save_gif_images(env_name, has_continuous_action_space, max_ep_len, path):
    print("============================================================================================")

    total_test_episodes = 1  # save gif for only one episode

    env = gym.make(env_name)

    # make directory for saving gif images
    gif_images_dir = path+"gif_images"
    if not os.path.exists(gif_images_dir):
        os.makedirs(gif_images_dir)
    print(path+"net.pth")
    nn = torch.load(path+"net.pth")
    print(path+"net.pth", nn)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes + 1):

        ep_reward = 0
        env.render()
        obs = env.reset()

        for t in range(1, max_ep_len + 1):
            obs_v = torch.FloatTensor([obs]).to(device)
            act_prob = nn(obs_v)
            if has_continuous_action_space:
                action = torch.tanh(act_prob)
                action = action.cpu().detach().numpy()[0]
                action = action.clip(env.action_space.low[0], env.action_space.high[0])
                obs, reward, done, _ = env.step(action)
            else:
                acts = act_prob.max(dim=1)[1]
                obs, reward, done, _ = env.step(acts.data.numpy()[0])
            ep_reward+=reward
            img = env.render(mode='rgb_array')

            img = Image.fromarray(img)
            img.save(gif_images_dir + '/' + str(t).zfill(6) + '.jpg')

            if done:
                break


        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
    env.close()


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
    for i in range(20):
        env_name = ENV_NAME
        path = "Data/" + env_name + "/Test" + "I"*i + "/"
        gif_images_dir = path + "net.pth"
        if not os.path.exists(gif_images_dir):
            continue
        save_gif_images(env_name, has_continuous_action_space, max_ep_len, path)

        save_gif(env_name, path)
