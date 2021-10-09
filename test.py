"""
This file contains code used to print out information about the collected information of the runs
"""
import csv
import os




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
    print(f"Name der Umgebung\tBestes wÃ¤hrend dem Testen erreichtes Ergebnis\tBestes Endergebnis\tDurchschnittliches Endergebnis")
    rst = list()
    for Algorithm in ["PPO"]:
        subfolders = [f.path for f in os.scandir(f"{Algorithm}\\Data") if f.is_dir()]
        for subfolder in subfolders:

            datax, datay = get_data(subfolder)
            if Algorithm == "GA":
                name = (Algorithm+" "+subfolder[8:-3])
            else:
                name = (Algorithm+" "+subfolder[9:-3])
            flat_list = [item for sublist in datay for item in sublist]
            average = sum(i[-1] for i in datay)/len(datay)
            bestes_endereb = max(i[-1] for i in datay)

            rst.append(f"{name}\t{max(flat_list)}\t{bestes_endereb}\t{average}")

    for i in range(len(rst)):
        print(rst[i])