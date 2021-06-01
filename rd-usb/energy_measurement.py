import csv
import numpy as np
import pandas as pd

data_filename = "C:\\Users\\saira\\OneDrive\\Desktop\\rd-usb\\data.csv"
df = pd.read_csv(data_filename, sep=",")
timestamp = df[['timestamp']].copy()
power_measurements = df[['power']].copy()

time_stamp_filename = "C:\\Users\\saira\\OneDrive\\Desktop\\rd-usb\\timestamp.csv"

start_timestamp = []
end_timestamp = []
rows = []
energy_measurement = []

with open(time_stamp_filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    _ = next(csvreader)
    for row in csvreader:
        if(len(row)==0):
            continue
        start_timestamp.append(row[0])
        end_timestamp.append(row[1])

for i in range(0, len(start_timestamp)):
    start_time = float(start_timestamp[i])
    end_time = float(end_timestamp[i])
    start_idx = int((timestamp.iloc[(np.searchsorted(df.timestamp.values, start_time) - 1).clip(0)] - 1).name)
    end_idx = int((timestamp.iloc[(np.searchsorted(df.timestamp.values, end_time) - 1).clip(0)]).name)
    print(start_idx, end_idx)
    energy = 0
    for idx in range(start_idx, end_idx):
        time_interval = timestamp['timestamp'][idx+1] - timestamp['timestamp'][idx]
        energy = energy + (power_measurements['power'][idx] * time_interval)
    print("Energy: " + str(energy))
    energy_measurement.append(energy)

print("Energy Consumption: " + str(sum(energy_measurement)))
print("Number of Measurements: " + str(len(energy_measurement)))
print("Average Energy Consumption: " + str(sum(energy_measurement) / len(energy_measurement)))



