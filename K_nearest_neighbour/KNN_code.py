import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


data = np.load('D:\ML_basics_Python\____Machinelearning___\datasets\knn\data.npy')
print(data.shape)
label = np.load('D:\ML_basics_Python\____Machinelearning___\datasets\knn\labels.npy')
print(label.shape)

for i in range(len(label)):
    if label[i] == 'Infected':
        label[i] = int(1)
    elif label[i] == 'Normal':
        label[i] = int(0)
        
print(label)
#print(type(label[1]))
labels = np.asarray(label, dtype=int)
#b = np.asarray(a, dtype=float)
print(type(labels[1]))
print(type(data[1][1][1]))
