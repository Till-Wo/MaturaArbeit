"""
This file contains code - used to create the graphs showing the training
"""

import csv
import matplotlib.pyplot as plt
import os
import random




def moving_avg(data, window_size):
    i = 0
    moving_averages = []
    while i < len(data) - window_size + 1:
        this_window = data[i: i + window_size]
        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1
    return moving_averages


def get_data(ENV_NAME):
    list_rewards = list()
    list_time = list()
    for i in range(10):
        list_rewards.append([])
        list_time.append([])
        path = ENV_NAME + "\\Test" + "I"*i + "\\"
        with open(path + "log.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    if line_count % 4 == 0:
                        if Algorithm=="GA":
                            list_rewards[i].append(float(row[0])) # Max Average
                        else:
                            list_rewards[i].append(float(row[1]))
                        list_time[i].append(float(row[2])/60)
                    line_count += 1
    return [list_time, list_rewards]


if __name__ == '__main__':
    for Algorithm in ["GA", "PPO"]:
        subfolders = [f.path for f in os.scandir(f"{Algorithm}\\Data") if f.is_dir()]
        print(subfolders)
        for subfolder in subfolders:
            datax, datay = get_data(subfolder)
            len_shortest_episode = len(min(datax, key=lambda x: len(x)))
            average_time = [[] for i in range(len_shortest_episode)]
            average_data = [[] for i in range(len_shortest_episode)]
            for i in range(10):
                y = datay[i]
                x = datax[i]
                plt.plot(x[-len(y):], y, alpha = 0.3)
                for j in range(len(x)-len_shortest_episode):
                    x.pop(random.randint(1,len(x)-2))
                    y.pop(random.randint(1,len(y)-2))
                for k in range(len(x)):
                    average_time[k].append(x[k])
                    average_data[k].append(y[k])

            print(average_time)
            average_time = [sum(i)/len(i) for i in average_time]
            average_data = [sum(i)/len(i) for i in average_data]
            yamah = moving_avg(average_data, 4)
            plt.plot(average_time[3:], yamah)
            plt.xlabel('Zeit in Minuten')
            plt.ylabel('Reward')
            if Algorithm == "GA":
                plt.title(Algorithm+" "+subfolder[8:-3])
                plt.savefig("Pics/"+Algorithm+"_"+subfolder[8:])
            else:
                plt.title(Algorithm+" "+subfolder[9:-3])
                plt.savefig("Pics/"+Algorithm+"_"+subfolder[9:])
            plt.clf()
