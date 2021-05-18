import GA.train
import PPO.train


for i in range(10):


    GA.train.main_loop()
    PPO.train.train()

