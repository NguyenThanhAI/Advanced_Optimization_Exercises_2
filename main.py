import os

import argparse
import time
import copy
import pickle

from math import ceil
from typing import List, Tuple

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from utils import AverageMeter, sigmoid, predict, sigmoid_cross_entropy_with_logits, sigmoid_cross_entropy_with_x_w, sigmoid_cross_entropy_truncated, derivative_cost_wrt_params, backtracking_line_search, check_wolfe_II, check_goldstein, line_search, adjust_step_length, adam_step, momentum_step, adagrad_step, rmsprop_step, adadelta_step, adamax_step, nadam_step, amsgrad_step, adabelief_step


np.random.seed(1000)

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, default=r"C:\Users\Thanh\Downloads\voice_gender\voice.csv")

    args = parser.parse_args()

    return args


def create_data(csv_path: str, normalize: str="minmax", use_bias: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)

    df['label']=df['label'].replace({'male':1,'female':0})

    df2 = df[["meanfreq", "sd", "median", "Q25", "IQR", "sp.ent", "sfm", "mode", "meanfun", "minfun", "maxfun", "meandom", "mindom", "maxdom", "label"]]

    df2['meanfreq'] = df2['meanfreq'].apply(lambda x:x*2)
    df2['median'] = df2['meanfreq'] + df2['mode']
    df2['median'] = df2['median'].apply(lambda x: x/3)
    df2['pear_skew'] = df2['meanfreq']-df2['mode']
    df2['pear_skew'] = df2['pear_skew']/df2['sd']

    x = df2.drop("label", axis=1).to_numpy(dtype=np.float32)
    y = df2["label"].values.astype(np.float32)

    if normalize == "minmax":

        x = (x - np.min(x, axis=0, keepdims=True))/(np.max(x, axis=0, keepdims=True) - np.min(x, axis=0, keepdims=True))

    elif normalize == "standardize":
        
        x = (x-np.mean(x, axis=0, keepdims=True))/np.std(x, axis=0, keepdims=True)

    else:
        raise ValueError("No normalizing initializer name {}".format(normalize))

    if use_bias:
        ones = np.ones(shape=[x.shape[0], 1], dtype=np.float32)
        x = np.append(x, ones, axis=1)

    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(x, y)

    for train_index, val_index in skf.split(x, y):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

    return x_train, y_train, x_val, y_val


def init_weights(x: np.ndarray, use_bias: bool=True, initializer: str="xavier") -> np.ndarray:

    if use_bias:
        if initializer == "xavier":
            weights = np.random.normal(loc=0, scale=np.sqrt(2/(x.shape[1])), size=x.shape[1]-1)
        else:
            weights = np.random.rand(x.shape[1]-1)

        weights = np.append(weights, 0.)

    else:
        if initializer == "xavier":
            weights = np.random.normal(loc=0, scale=np.sqrt(2/(x.shape[1])), size=x.shape[1])
        else:
            weights = np.random.rand(x.shape[1])

    return weights


def train_gradient_descent(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, init_weights: np.ndarray, optimizer: str="gd", threshold: float=0.6, num_epochs: int=100000, c_1: float=1e-4, c_2: float=0.9, c: float=0.25, rho: float=0.5, init_alpha: float=2, epsilon_1: float=0.001, epsilon_2: float=0.001, epsilon_3: float=0.001, stop_condition: bool=False):
    max_val_acc = - np.inf
    max_val_acc_weights = None

    min_val_cost = np.inf
    min_val_cost_weights = None
    
    train_cost_list = []
    train_acc_list = []

    val_cost_list = []
    val_acc_list = []

    time_epoch_list = []

    timestamp_epoch_list = []

    wolfe_II_list = []
    goldstein_list = []

    weights = init_weights
    prev_weights = copy.deepcopy(weights)
    prev_train_cost = 0

    t = 1
    m = np.zeros_like(weights, dtype=np.float32)
    v = np.zeros_like(weights, dtype=np.float32)
    v_hat = np.zeros_like(weights, dtype=np.float32)
    d = np.zeros_like(weights, dtype=np.float32)
    u = 0.

    start_timestamp = time.time()

    for epoch in tqdm(range(num_epochs)):

        epoch_start = time.time()

        dweights = derivative_cost_wrt_params(x=x_train, w=weights, y=y_train)

        if optimizer.lower() == "gd":
            alpha = backtracking_line_search(x=x_train, w=weights, y=y_train, p=-dweights, rho=rho, alpha=init_alpha, c=c_1)
            #alpha = init_alpha*(1 - epoch / num_epochs)
            wolfe_II_list.append(check_wolfe_II(x=x_train, w=weights, y=y_train, alpha=alpha, p=-dweights, c_2=c_2))
            goldstein_list.append(check_goldstein(x=x_train, w=weights, y=y_train, alpha=alpha, p=-dweights, c=c))
            weights -= alpha * dweights
        elif optimizer.lower() == "adam":
            weights, m, v = adam_step(weights=weights, dweights=dweights, m=m, v=v, alpha=init_alpha, t=t, beta_1=0.5,
                                      beta_2=0.9, epsilon=1e-8)
        elif optimizer.lower() == "momentum":
            weights, m = momentum_step(weights=weights, dweights=dweights, m=m, alpha=init_alpha,
                                       beta_1=0.9)
        elif optimizer.lower() == "adagrad":
            weights, m = adagrad_step(weights=weights, dweights=dweights, v=v, alpha=init_alpha, epsilon=1e-8)
        elif optimizer.lower() == "rmsprop":
            weights, v = rmsprop_step(weights=weights, dweights=dweights, v=v, alpha=init_alpha, beta_2=0.9, epsilon=1e-8)
        elif optimizer.lower() == "adadelta":
            weights, v, d = adadelta_step(weights=weights, dweights=dweights, v=v, d=d, 
                                          alpha=init_alpha, beta_2=0.9, epsilon=1e-8)
        elif optimizer.lower() == "adamax":
            weights, m, u = adamax_step(weights=weights, dweights=dweights, m=m, u=u, 
                                        alpha=init_alpha, t=t, beta_1=0.9,
                                        beta_2=0.99, epsilon=1e-8)
        elif optimizer.lower() == "nadam":
            weights, m, v = nadam_step(weights=weights, dweights=dweights, m=m, v=v,
                                       alpha=init_alpha, t=t, beta_1=0.5, beta_2=0.9, epsilon=1e-8)
        elif optimizer.lower() == "amsgrad":
            weights, m, v, v_hat = amsgrad_step(weights=weights, dweights=dweights, m=m, v=v, v_hat=v_hat,
                                                alpha=init_alpha, t=t, beta_1=0.5, beta_2=0.9, epsilon=1e-8)
        elif optimizer.lower() == "adabelief":
            weights, m, v = adabelief_step(weights=weights, dweights=dweights, m=m, v=v, alpha=init_alpha,
                                           t=t, beta_1=0.5, beta_2=0.9, epsilon=1e-8)
        else:
            raise ValueError("No optimizer name {}".format(optimizer))
        t += 1

        epoch_end = time.time()

        time_epoch_list.append(epoch_end - epoch_start)
        timestamp_epoch_list.append(epoch_end - start_timestamp)

        train_cost = sigmoid_cross_entropy_with_x_w(x=x_train, w=weights, y=y_train)
        val_cost = sigmoid_cross_entropy_with_x_w(x=x_val, w=weights, y=y_val)

        train_output = predict(x=x_train, w=weights, threshold=threshold).astype(np.float32)
        train_acc = (train_output == y_train).sum()/y_train.shape[0]

        val_output = predict(x=x_val, w=weights, threshold=threshold).astype(np.float32)
        val_acc = (val_output == y_val).sum()/y_val.shape[0]

        train_cost_list.append(train_cost)
        train_acc_list.append(train_acc)

        val_cost_list.append(val_cost)
        val_acc_list.append(val_acc)

        if val_cost < min_val_cost:
            min_val_cost = copy.deepcopy(val_cost)
            min_val_cost_weights = copy.deepcopy(weights)

        if val_acc > max_val_acc:
            max_val_acc = copy.deepcopy(val_acc)
            max_val_acc_weights = copy.deepcopy(weights)

        if stop_condition:
            dweights = derivative_cost_wrt_params(x=x_train, w=weights, y=y_train)
            if np.linalg.norm(weights - prev_weights) < epsilon_1 and np.abs(prev_train_cost - train_cost) < epsilon_2 and np.linalg.norm(dweights) < epsilon_3:
                print("Satisfy stop condition. Stop training")
                break

        if (epoch + 1) % 10000 == 0:
            #print(epoch, alpha, train_cost, val_cost, train_acc, val_acc, np.linalg.norm(weights - prev_weights), np.abs(prev_train_cost - train_cost), np.linalg.norm(dweights))
            print(epoch, train_cost, val_cost, train_acc, val_acc, np.linalg.norm(weights - prev_weights), np.abs(prev_train_cost - train_cost), np.linalg.norm(dweights))

        prev_weights = copy.deepcopy(weights)
        prev_train_cost = copy.deepcopy(train_cost)

    if optimizer.lower() == "gd":
        return weights, min_val_cost_weights, max_val_acc_weights, train_cost_list, train_acc_list, val_cost_list, val_acc_list, time_epoch_list, timestamp_epoch_list, wolfe_II_list, goldstein_list
    else:
        return weights, min_val_cost_weights, max_val_acc_weights, train_cost_list, train_acc_list, val_cost_list, val_acc_list, time_epoch_list, timestamp_epoch_list


def train_batch_gradient_descent(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, init_weights: np.ndarray, batch_size: int, optimizer: str="gd", threshold: float=0.6, num_epochs: int=100000, c_1: float=1e-4, c_2: float=0.9, c: float=0.25, rho: float=0.5, init_alpha: float=2, epsilon_1: float=0.001, epsilon_2: float=0.001, epsilon_3: float=0.001, stop_condition: bool=False):
    max_val_acc = - np.inf
    max_val_acc_weights = None

    min_val_cost = np.inf
    min_val_cost_weights = None
    
    train_cost_list = []
    train_acc_list = []

    #train_cost_step_list = []
    #train_acc_step_list = []

    val_cost_list = []
    val_acc_list = []

    time_epoch_list = []
    #time_step_list = []

    timestamp_epoch_list = []
    #timestamp_step_list = []

    wolfe_II_list = []
    goldstein_list = []

    weights = init_weights
    prev_weights = copy.deepcopy(weights)
    prev_train_cost = 0

    t = 1
    m = np.zeros_like(weights, dtype=np.float32)
    v = np.zeros_like(weights, dtype=np.float32)
    v_hat = np.zeros_like(weights, dtype=np.float32)
    d = np.zeros_like(weights, dtype=np.float32)
    u = 0.

    start_timestamp = time.time()

    for epoch in tqdm(range(num_epochs)):
        index = np.arange(x_train.shape[0])
        np.random.shuffle(index)
        epoch_start = time.time()
        #for j in range(ceil(x_train.shape[0]/batch_size)):
        #    index_batch = index[j * batch_size: j * batch_size + batch_size]
        index_batch = index[:batch_size]
        x_batch = x_train[index_batch]
        y_batch = y_train[index_batch]

        #step_start = time.time()

        dweights = derivative_cost_wrt_params(x=x_batch, w=weights, y=y_batch)

        if optimizer.lower() == "gd":
            alpha = backtracking_line_search(x=x_batch, w=weights, y=y_batch, p=-dweights, rho=rho, alpha=init_alpha, c=c_1)

            wolfe_II_list.append(check_wolfe_II(x=x_batch, w=weights, y=y_batch, alpha=alpha, p=-dweights, c_2=c_2))
            goldstein_list.append(check_goldstein(x=x_batch, w=weights, y=y_batch, alpha=alpha, p=-dweights, c=c))
            weights -= alpha * dweights
        elif optimizer.lower() == "adam":
            weights, m, v = adam_step(weights=weights, dweights=dweights, m=m, v=v, alpha=init_alpha, t=t, beta_1=0.5,
                                      beta_2=0.9, epsilon=1e-8)
        elif optimizer.lower() == "momentum":
            weights, m = momentum_step(weights=weights, dweights=dweights, m=m, alpha=init_alpha,
                                       beta_1=0.9)
        elif optimizer.lower() == "adagrad":
            weights, m = adagrad_step(weights=weights, dweights=dweights, v=v, alpha=init_alpha, epsilon=1e-8)
        elif optimizer.lower() == "rmsprop":
            weights, v = rmsprop_step(weights=weights, dweights=dweights, v=v, alpha=init_alpha, beta_2=0.9, epsilon=1e-8)
        elif optimizer.lower() == "adadelta":
            weights, v, d = adadelta_step(weights=weights, dweights=dweights, v=v, d=d, 
                                          alpha=init_alpha, beta_2=0.9, epsilon=1e-8)
        elif optimizer.lower() == "adamax":
            weights, m, u = adamax_step(weights=weights, dweights=dweights, m=m, u=u, 
                                        alpha=init_alpha, t=t, beta_1=0.9,
                                        beta_2=0.99, epsilon=1e-8)
        elif optimizer.lower() == "nadam":
            weights, m, v = nadam_step(weights=weights, dweights=dweights, m=m, v=v,
                                       alpha=init_alpha, t=t, beta_1=0.5, beta_2=0.9, epsilon=1e-8)
        elif optimizer.lower() == "amsgrad":
            weights, m, v, v_hat = amsgrad_step(weights=weights, dweights=dweights, m=m, v=v, v_hat=v_hat,
                                                alpha=init_alpha, t=t, beta_1=0.5, beta_2=0.99, epsilon=1e-8)
        elif optimizer.lower() == "adabelief":
            weights, m, v = adabelief_step(weights=weights, dweights=dweights, m=m, v=v, alpha=init_alpha,
                                           t=t, beta_1=0.5, beta_2=0.9, epsilon=1e-8)
        else:
            raise ValueError("No optimizer name {}".format(optimizer))

        t += 1

        #step_end = time.time()
#
        #time_step_list.append(step_end - step_start)
        #timestamp_step_list.append(step_end - start_timestamp)

        #cost = sigmoid_cross_entropy_with_x_w(x=x_batch, w=weights, y=y_batch)
        #output = predict(x=x_batch, w=weights, threshold=threshold).astype(np.float32)
        #acc = (output == y_batch).sum()/y_batch.shape[0]
#
        #train_cost_step_list.append(cost)
        #train_acc_step_list.append(acc)

        epoch_end = time.time()

        time_epoch_list.append(epoch_end - epoch_start)
        timestamp_epoch_list.append(epoch_end - start_timestamp)

        train_cost = sigmoid_cross_entropy_with_x_w(x=x_train, w=weights, y=y_train)
        val_cost = sigmoid_cross_entropy_with_x_w(x=x_val, w=weights, y=y_val)

        train_output = predict(x=x_train, w=weights, threshold=threshold).astype(np.float32)
        train_acc = (train_output == y_train).sum()/y_train.shape[0]

        val_output = predict(x=x_val, w=weights, threshold=threshold).astype(np.float32)
        val_acc = (val_output == y_val).sum()/y_val.shape[0]

        train_cost_list.append(train_cost)
        train_acc_list.append(train_acc)

        val_cost_list.append(val_cost)
        val_acc_list.append(val_acc)

        if val_cost < min_val_cost:
            min_val_cost = copy.deepcopy(val_cost)
            min_val_cost_weights = copy.deepcopy(weights)

        if val_acc > max_val_acc:
            max_val_acc = copy.deepcopy(val_acc)
            max_val_acc_weights = copy.deepcopy(weights)
        
        if stop_condition:
            dweights = derivative_cost_wrt_params(x=x_train, w=weights, y=y_train)
            if np.linalg.norm(weights - prev_weights) < epsilon_1 and np.abs(prev_train_cost - train_cost) < epsilon_2 and np.linalg.norm(dweights) < epsilon_3:
                print("Satisfy stop condition. Stop training")
                break

        if (epoch + 1) % 10000 == 0:
            #print(epoch, alpha, train_cost, val_cost, train_acc, val_acc, np.linalg.norm(weights - prev_weights), np.abs(prev_train_cost - train_cost), np.linalg.norm(dweights))
            print(epoch, train_cost, val_cost, train_acc, val_acc, np.linalg.norm(weights - prev_weights), np.abs(prev_train_cost - train_cost), np.linalg.norm(dweights))

        prev_weights = copy.deepcopy(weights)
        prev_train_cost = copy.deepcopy(train_cost)
    
    #if optimizer.lower() == "gd":
    #    return weights, min_val_cost_weights, max_val_acc_weights, train_cost_list, train_acc_list, val_cost_list, val_acc_list, train_cost_step_list, train_acc_step_list, time_epoch_list, time_step_list, timestamp_epoch_list, timestamp_step_list, wolfe_II_list, goldstein_list
    #else:
    #    return weights, min_val_cost_weights, max_val_acc_weights, train_cost_list, train_acc_list, val_cost_list, val_acc_list, train_cost_step_list, train_acc_step_list, time_epoch_list, time_step_list, timestamp_epoch_list, timestamp_step_list

    if optimizer.lower() == "gd":
        return weights, min_val_cost_weights, max_val_acc_weights, train_cost_list, train_acc_list, val_cost_list, val_acc_list, time_epoch_list, timestamp_epoch_list, wolfe_II_list, goldstein_list
    else:
        return weights, min_val_cost_weights, max_val_acc_weights, train_cost_list, train_acc_list, val_cost_list, val_acc_list, time_epoch_list, timestamp_epoch_list


if __name__ == "__main__":

    args = get_args()

    csv_path = args.csv_path
    normalize = "minmax"
    use_bias = True
    initializer = "xavier"
    num_epochs = 20000
    c_1 = 1e-2
    c_2 = 0.9
    c = 0.25
    threshold = 0.6
    rho = 0.5
    batch_size = 64
    stop_condition = False
    save_dir = "."

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    optimizer_list = ["gd", "Adam", "Momentum", "Adagrad", "RMSProp", "Adadelta", "Adamax", "Nadam", "AMSGrad", "AdaBelief"]


    x_train, y_train, x_val, y_val = create_data(csv_path=csv_path, normalize=normalize, use_bias=use_bias)

    #print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    start_weights = init_weights(x=x_train, use_bias=use_bias, initializer=initializer)
    #weights = {"gradient_descent": copy.deepcopy(weights),
    #           "batch_gradient_descent": copy.deepcopy(weights),
    #           "stochastic_gradient_descent": copy.deepcopy(weights)}
    #weights = np.array([0.87, 2.89, -2.13, -4.36, 3.51, 8.56, 2.85, -7.92, 9.78, -9.57, 0.87, 0.78, -30.24, 7.25, -0.2, 0.17, -0.2, -0.17, 0.07, -3.02, 7.19])

    print(start_weights)

    result_weights = {}
    result_min_val_cost_weights = {}
    result_max_val_acc_weights = {}
    result_train_cost_list = {} 
    result_train_acc_list = {} 
    result_val_cost_list = {} 
    result_val_acc_list = {} 
    #result_train_cost_step_list = {} 
    #result_train_acc_step_list = {} 
    result_time_epoch_list = {} 
    #result_time_step_list = {}
    result_timestamp_epoch_list = {}
    #result_timestamp_step_list = {} 
    result_wolfe_II_list = {} 
    result_goldstein_list = {}

    for optimizer in optimizer_list:

        print("Optimizer {}".format(optimizer))

        if optimizer.lower() == "gd":

            weights, min_val_cost_weights, max_val_acc_weights, train_cost_list, train_acc_list, val_cost_list, val_acc_list, time_epoch_list, timestamp_epoch_list, wolfe_II_list, goldstein_list = train_gradient_descent(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, init_weights=copy.deepcopy(start_weights),
                                                                                                                                                                                                                            optimizer=optimizer,
                                                                                                                                                                                                                            threshold=threshold, num_epochs=num_epochs, c_1=c_1, c_2=c_2, c=c, rho=rho,
                                                                                                                                                                                                                            init_alpha=1, stop_condition=stop_condition)
            result_wolfe_II_list[optimizer] = wolfe_II_list
            result_goldstein_list[optimizer] = goldstein_list
        else:
            weights, min_val_cost_weights, max_val_acc_weights, train_cost_list, train_acc_list, val_cost_list, val_acc_list, time_epoch_list, timestamp_epoch_list = train_gradient_descent(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, init_weights=copy.deepcopy(start_weights),
                                                                                                                                                                                             optimizer=optimizer,
                                                                                                                                                                                             threshold=threshold, num_epochs=num_epochs, c_1=c_1, c_2=c_2, c=c, rho=rho,
                                                                                                                                                                                             init_alpha=1, stop_condition=stop_condition)
        result_weights[optimizer] = weights
        result_min_val_cost_weights[optimizer] = min_val_cost_weights
        result_max_val_acc_weights[optimizer] = max_val_acc_weights
        result_train_cost_list[optimizer] = train_cost_list
        result_train_acc_list[optimizer] = train_acc_list
        result_val_cost_list[optimizer] = val_cost_list
        result_val_acc_list[optimizer] = val_acc_list
        result_time_epoch_list[optimizer] = time_epoch_list
        result_timestamp_epoch_list[optimizer] = timestamp_epoch_list

    
    for optimizer in optimizer_list:

        print("Optimizer Batch {}".format(optimizer))

        if optimizer.lower() == "gd":

            weights, min_val_cost_weights, max_val_acc_weights, train_cost_list, train_acc_list, val_cost_list, val_acc_list, time_epoch_list, timestamp_epoch_list, wolfe_II_list, goldstein_list = train_batch_gradient_descent(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, init_weights=copy.deepcopy(start_weights),
                                                                                                                                                                                                                                  optimizer=optimizer,
                                                                                                                                                                                                                                  batch_size=batch_size, threshold=threshold, num_epochs=num_epochs, c_1=c_1, c_2=c_2, c=c,
                                                                                                                                                                                                                                  rho=rho, init_alpha=1e-1,
                                                                                                                                                                                                                                  stop_condition=stop_condition)
            result_wolfe_II_list["batch_{}".format(optimizer)] = wolfe_II_list
            result_goldstein_list["batch_{}".format(optimizer)] = goldstein_list
        else:
            weights, min_val_cost_weights, max_val_acc_weights, train_cost_list, train_acc_list, val_cost_list, val_acc_list, time_epoch_list, timestamp_epoch_list = train_batch_gradient_descent(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, init_weights=copy.deepcopy(start_weights),
                                                                                                                                                                                                   optimizer=optimizer,
                                                                                                                                                                                                   batch_size=batch_size, threshold=threshold, num_epochs=num_epochs, c_1=c_1, c_2=c_2, c=c,
                                                                                                                                                                                                   rho=rho, init_alpha=1e-1,
                                                                                                                                                                                                   stop_condition=stop_condition)

        result_weights["batch_{}".format(optimizer)] = weights
        result_min_val_cost_weights["batch_{}".format(optimizer)] = min_val_cost_weights
        result_max_val_acc_weights["batch_{}".format(optimizer)] = max_val_acc_weights
        result_train_cost_list["batch_{}".format(optimizer)] = train_cost_list
        result_train_acc_list["batch_{}".format(optimizer)] = train_acc_list
        result_val_cost_list["batch_{}".format(optimizer)] = val_cost_list
        result_val_acc_list["batch_{}".format(optimizer)] = val_acc_list
        #result_train_cost_step_list["batch_{}".format(optimizer)] = train_cost_step_list
        #result_train_acc_step_list["batch_{}".format(optimizer)] = train_acc_step_list
        result_time_epoch_list["batch_{}".format(optimizer)] = time_epoch_list
        #result_time_step_list["batch_{}".format(optimizer)] = time_step_list
        result_timestamp_epoch_list["batch_{}".format(optimizer)] = timestamp_epoch_list
        #result_timestamp_step_list["batch_{}".format(optimizer)] = timestamp_step_list                                                                                                                                                                           

    for optimizer in optimizer_list:

        print("Optimizer Stochastic {}".format(optimizer))

        if optimizer.lower() == "gd":

            weights, min_val_cost_weights, max_val_acc_weights, train_cost_list, train_acc_list, val_cost_list, val_acc_list, time_epoch_list, timestamp_epoch_list, wolfe_II_list, goldstein_list = train_batch_gradient_descent(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, init_weights=copy.deepcopy(start_weights),
                                                                                                                                                                                                                                  optimizer=optimizer,
                                                                                                                                                                                                                                  batch_size=1, threshold=threshold, num_epochs=num_epochs, c_1=c_1, c_2=c_2, c=c,
                                                                                                                                                                                                                                  rho=rho, init_alpha=1e-1,
                                                                                                                                                                                                                                  stop_condition=stop_condition)

            result_wolfe_II_list["stochastic_{}".format(optimizer)] = wolfe_II_list
            result_goldstein_list["stochastic_{}".format(optimizer)] = goldstein_list

        else:
            weights, min_val_cost_weights, max_val_acc_weights, train_cost_list, train_acc_list, val_cost_list, val_acc_list, time_epoch_list, timestamp_epoch_list = train_batch_gradient_descent(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, init_weights=copy.deepcopy(start_weights),
                                                                                                                                                                                                   optimizer=optimizer,
                                                                                                                                                                                                   batch_size=1, threshold=threshold, num_epochs=num_epochs, c_1=c_1, c_2=c_2, c=c,
                                                                                                                                                                                                   rho=rho, init_alpha=1e-1,
                                                                                                                                                                                                   stop_condition=stop_condition)
        result_weights["stochastic_{}".format(optimizer)] = weights
        result_min_val_cost_weights["stochastic_{}".format(optimizer)] = min_val_cost_weights
        result_max_val_acc_weights["stochastic_{}".format(optimizer)] = max_val_acc_weights
        result_train_cost_list["stochastic_{}".format(optimizer)] = train_cost_list
        result_train_acc_list["stochastic_{}".format(optimizer)] = train_acc_list
        result_val_cost_list["stochastic_{}".format(optimizer)] = val_cost_list
        result_val_acc_list["stochastic_{}".format(optimizer)] = val_acc_list
        #result_train_cost_step_list["stochastic_{}".format(optimizer)] = train_cost_step_list
        #result_train_acc_step_list["stochastic_{}".format(optimizer)] = train_acc_step_list
        result_time_epoch_list["stochastic_{}".format(optimizer)] = time_epoch_list
        #result_time_step_list["stochastic_{}".format(optimizer)] = time_step_list
        result_timestamp_epoch_list["stochastic_{}".format(optimizer)] = timestamp_epoch_list
        #result_timestamp_step_list["stochastic_{}".format(optimizer)] = timestamp_step_list

    
    ''''results = {"weights": result_weights, "min_val_cost_weights": result_min_val_cost_weights,
               "max_val_acc_weights": max_val_acc_weights, "train_cost": result_train_cost_list,
               "train_acc": result_train_acc_list, "val_cost": result_val_cost_list,
               "val_acc": result_val_acc_list, "train_cost_step": result_train_cost_step_list,
               "train_acc_step": result_train_acc_step_list, "time_epoch": result_time_epoch_list,
               "time_step": result_time_step_list, "timestamp_epoch": result_timestamp_epoch_list,
               "timestamp_step": result_time_step_list,
               "wolf_II": result_wolfe_II_list, "goldstein": result_goldstein_list}'''

    results = {"weights": result_weights, "min_val_cost_weights": result_min_val_cost_weights,
               "max_val_acc_weights": max_val_acc_weights, "train_cost": result_train_cost_list,
               "train_acc": result_train_acc_list, "val_cost": result_val_cost_list,
               "val_acc": result_val_acc_list, "time_epoch": result_time_epoch_list,
               "timestamp_epoch": result_timestamp_epoch_list,
               "wolf_II": result_wolfe_II_list, "goldstein": result_goldstein_list}


    with open(os.path.join(save_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)
        f.close()
