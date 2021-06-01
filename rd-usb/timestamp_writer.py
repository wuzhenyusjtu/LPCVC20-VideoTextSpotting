import csv
import time
import random

filename = "C:\\Users\\saira\\OneDrive\\Desktop\\rd-usb\\timestamp.csv"
headers = ["start_timestamp", "end_timestamp"]
with open(filename, "w") as outfile:
    csvwriter = csv.writer(outfile)
    csvwriter.writerow(headers)

cnt = 0
for i in range(0, 10):
    cnt = cnt + 1
    print(cnt)
    start_time = time.time()
    time.sleep(random.randint(1, 11))
    end_time = time.time()
    values = [start_time, end_time]
    with open(filename, "a") as outfile:
        csvwriter = csv.writer(outfile)
        csvwriter.writerow(values)

