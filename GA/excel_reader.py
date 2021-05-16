import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

mean_reward_list = []
x = np.asarray([i for i in range(20)])
for i in range(25):
    df = pd.read_excel (r'G:\Hauptordner\Schule\MATURA_PROJEKT\Code\GA\GA_v2\Data\Mutation_Strength\0.15\CartPole-v0_'+str(i)+".xlsx")
    a = df["reward_mean"].to_numpy()
    mean_reward_list.append(a)
    plt.plot(a)
plt.show()
print(mean_reward_list)
