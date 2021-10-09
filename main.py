"""
This file manages the training
"""
def train_ga():
    import GA.train
    GA.train.main_loop()

def train_ppo():
    import PPO.train
    PPO.train.train()


for i in range(10):
    train_ppo()
    train_ga()

