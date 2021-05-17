import GA.train
import PPO.train
GA.train.ENV_NAME = PPO.train.ENV_NAME = "BipedalWalker-v2"
GA.train.IS_CONTINIOUS = PPO.train.has_continuous_action_space = True
GA.train.MAX_LENGTH_OF_EPISODE = PPO.train.max_ep_len = 1000

for i in range(10):


    GA.train.main_loop()
    PPO.train.train()

