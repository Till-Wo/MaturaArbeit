from parameters import *
import csv
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import random



if __name__ == '__main__':
    list_rewards = list()
    list_time = list()
    for i in range(10):
        list_rewards.append([])
        list_time.append([])
        path = "GA/Data/" + ENV_NAME + "/Test" + "I"*i + "/"
        with open(path + "log.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    if line_count % 4 == 0:
                        list_rewards[i].append(float(row[1]))
                        list_time[i].append(float(row[2])/60)
                    line_count += 1

    for i in range(10):
        y = list_rewards[i]
        x = list_time[i]
        yhat = savgol_filter(y, 21, 5)
        plt.plot(x, yhat, color=(0, 0.5, 1, 0.4))




    list_rewards = list()
    list_time = list()
    for i in range(10):
        list_rewards.append([])
        list_time.append([])
        path = "PPO/Data/" + ENV_NAME + "/Test" + "I"*i + "/"
        with open(path + "log.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    if line_count % 4 == 0:
                        list_rewards[i].append(float(row[1]))
                        list_time[i].append(float(row[2])/60)
                    line_count += 1

    for i in range(10):
        y = list_rewards[i]
        x = list_time[i]
        yhat = savgol_filter(y, 21, 5)
        plt.plot(x, yhat, color=(1, 0.5, 0, 0.4))




    plt.show()