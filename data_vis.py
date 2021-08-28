import csv
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import savgol_filter
import os
import pandas
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
            for i in range(10):
                y = datay[i]
                x = datax[i]
                yhat = moving_avg(y, 5)
                plt.plot(x[-len(yhat):], yhat)
            plt.xlabel('Zeit in Minuten')
            plt.ylabel('Reward')
            if Algorithm == "GA":
                plt.title(Algorithm+" "+subfolder[8:-3])
                plt.savefig("Pics/"+Algorithm+"_"+subfolder[8:])
            else:
                plt.title(Algorithm+" "+subfolder[9:-3])
                plt.savefig("Pics/"+Algorithm+"_"+subfolder[9:])

            plt.clf()
