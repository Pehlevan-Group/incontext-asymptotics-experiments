import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns

mydir = sys.argv[1]
length = int(sys.argv[2])
# name = int(sys.argv[2])
experimentdata = []
for i in range(length):
    file_path = f'./{mydir}/error-{i}.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file if line.strip()]
    experimentdata.append(numbers)

means = [np.mean(experimentdata[i]) for i in range(len(experimentdata))]
print(means)