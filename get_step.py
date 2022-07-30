
import os
import pickle

import numpy as np


result_dir = "."
result_file_name = "results.pkl"

with open(os.path.join(result_dir, result_file_name), "rb") as f:
    results = pickle.load(f)
    f.close()


min_train_cost = {}
min_val_cost = {}
max_train_acc = {}
max_val_acc = {}

for step_length in results["train_acc"]:
    for optimizer in results["train_acc"][step_length]:
        if optimizer not in max_train_acc:
            max_train_acc[optimizer] = (-np.inf, 0, 0)
        max_acc = np.max(results["train_acc"][step_length][optimizer])
        pos = np.argmax(results["train_acc"][step_length][optimizer])
        if max_acc > max_train_acc[optimizer][0]:
            max_train_acc[optimizer] = (max_acc, step_length, pos)

for step_length in results["val_acc"]:
    for optimizer in results["val_acc"][step_length]:
        if optimizer not in max_val_acc:
            max_val_acc[optimizer] = (-np.inf, 0, 0)
        max_acc = np.max(results["val_acc"][step_length][optimizer])
        pos = np.argmax(results["val_acc"][step_length][optimizer])
        if max_acc > max_val_acc[optimizer][0]:
            max_val_acc[optimizer] = (max_acc, step_length, pos)

for step_length in results["train_cost"]:
    for optimizer in results["train_cost"][step_length]:
        if optimizer not in min_train_cost:
            min_train_cost[optimizer] = (np.inf, 0, 0)
        min_cost = np.min(results["train_cost"][step_length][optimizer])
        pos = np.argmin(results["train_cost"][step_length][optimizer])
        if min_cost < min_train_cost[optimizer][0]:
            min_train_cost[optimizer] = (min_cost, step_length, pos)

for step_length in results["val_cost"]:
    for optimizer in results["val_cost"][step_length]:
        if optimizer not in min_val_cost:
            min_val_cost[optimizer] = (np.inf, 0)
        min_cost = np.min(results["val_cost"][step_length][optimizer])
        pos = np.argmin(results["train_cost"][step_length][optimizer])
        if min_cost < min_val_cost[optimizer][0]:
            min_val_cost[optimizer] = (min_cost, step_length, pos)


#print("Max train acc: {}, Max val acc: {}, Min train cost: {}, Min val cost: {}".format(max_train_acc, max_val_acc, min_train_cost, min_val_cost))
print(max_train_acc, max_val_acc, min_train_cost, min_val_cost)

print(max(max_train_acc.items(), key=lambda x: x[1][0]), max(max_val_acc.items(), key=lambda x: x[1][0]), min(min_train_cost.items(), key=lambda x: x[1][0]), min(min_val_cost.items(), key=lambda x: x[1][0]))