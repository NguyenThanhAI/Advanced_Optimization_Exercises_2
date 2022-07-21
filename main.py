import os

import argparse
import time
import copy

from math import ceil
from typing import List, Tuple

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from utils import AverageMeter, sigmoid, predict, sigmoid_cross_entropy_with_logits, sigmoid_cross_entropy_with_x_w, sigmoid_cross_entropy_truncated, derivative_cost_wrt_params, backtracking_line_search, check_wolfe_II, check_goldstein, line_search, adjust_step_length, adam_step, momentum_step, adagrad_step, rmsprop_step, adadelta_step, adamax_step, nadam_step, amsgrad_step, adabelief_step


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, default=r"C:\Users\Thanh\Downloads\voice_gender\voice.csv")

    args = parser.parse_args()

    return args


def create_data(csv_path: str, normalize: str="minmax", use_bias: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)

    df['label']=df['label'].replace({'male':1,'female':0})

    x = df.drop("label", axis=1).to_numpy(dtype=np.float)
    y = df["label"].values.astype(np.float)

    if normalize == "minmax":

        x = (x - np.min(x, axis=0, keepdims=True))/(np.max(x, axis=0, keepdims=True) - np.min(x, axis=0, keepdims=True))

    elif normalize == "standardize":
        
        x = (x-np.mean(x, axis=0, keepdims=True))/np.std(x, axis=0, keepdims=True)

    else:
        raise ValueError("No normalizing initializer name {}".format(normalize))

    if use_bias:
        ones = np.ones(shape=[x.shape[0], 1], dtype=np.float)
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


def train_gradient_descent(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, init_weights: np.ndarray, optimizer: str="gd", threshold: float=0.6, num_epochs: int=100000, c_1: float=1e-4, c_2: float=0.9, c: float=0.25, rho: float=0.5, init_alpha: float=2, epsilon_1: float=0.01, epsilon_2: float=0.01, epsilon_3: float=0.01):
    train_cost_list = []
    train_acc_list = []

    val_cost_list = []
    val_acc_list = []

    wolfe_II_list = []
    goldstein_list = []

    weights = init_weights
    prev_weights = copy.deepcopy(weights)
    prev_train_cost = 0

    t = 1
    m = np.zeros_like(weights, dtype=np.float)
    v = np.zeros_like(weights, dtype=np.float)
    v_hat = np.zeros_like(weights, dtype=np.float)
    d = np.zeros_like(weights, dtype=np.float)
    u = 0.
    for epoch in tqdm(range(num_epochs)):

        dweights = derivative_cost_wrt_params(x=x_train, w=weights, y=y_train)

        alpha = backtracking_line_search(x=x_train, w=weights, y=y_train, p=-dweights, rho=rho, alpha=init_alpha, c=c_1)
        #alpha = init_alpha*(1 - epoch / num_epochs)
        wolfe_II_list.append(check_wolfe_II(x=x_train, w=weights, y=y_train, alpha=alpha, p=-dweights, c_2=c_2))
        goldstein_list.append(check_goldstein(x=x_train, w=weights, y=y_train, alpha=alpha, p=-dweights, c=c))

        if optimizer.lower() == "gd":
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
                                        beta_2=0.99)
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

        train_cost = sigmoid_cross_entropy_with_x_w(x=x_train, w=weights, y=y_train)
        val_cost = sigmoid_cross_entropy_with_x_w(x=x_val, w=weights, y=y_val)

        train_output = predict(x=x_train, w=weights, threshold=threshold).astype(np.float)
        train_acc = (train_output == y_train).sum()/y_train.shape[0]

        val_output = predict(x=x_val, w=weights, threshold=threshold).astype(np.float)
        val_acc = (val_output == y_val).sum()/y_val.shape[0]

        train_cost_list.append(train_cost)
        train_acc_list.append(train_acc)

        val_cost_list.append(val_cost)
        val_acc_list.append(val_acc)

        if (epoch + 1) % 100 == 0:
            print(epoch, alpha, train_cost, val_cost, train_acc, val_acc, np.linalg.norm(weights - prev_weights), np.abs(prev_train_cost - train_cost), np.linalg.norm(dweights))

        prev_weights = copy.deepcopy(weights)
        prev_train_cost = copy.deepcopy(train_cost)


def train_batch_gradient_descent(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, init_weights: np.ndarray, batch_size: int, optimizer: str="gd", threshold: float=0.6, num_epochs: int=100000, c_1: float=1e-4, c_2: float=0.9, c: float=0.25, rho: float=0.5, init_alpha: float=2, epsilon_1: float=0.01, epsilon_2: float=0.01, epsilon_3: float=0.01):
    train_cost_list = []
    train_acc_list = []

    val_cost_list = []
    val_acc_list = []

    wolfe_II_list = []
    goldstein_list = []

    weights = init_weights
    prev_weights = copy.deepcopy(weights)
    prev_train_cost = 0

    t = 1
    m = np.zeros_like(weights, dtype=np.float)
    v = np.zeros_like(weights, dtype=np.float)
    v_hat = np.zeros_like(weights, dtype=np.float)
    d = np.zeros_like(weights, dtype=np.float)
    u = 0.
    for epoch in tqdm(range(num_epochs)):
        index = np.arange(x_train.shape[0])
        np.random.shuffle(index)
        for j in range(ceil(x_train.shape[0]/batch_size)):
            index_batch = index[j * batch_size: j * batch_size + batch_size]
            x_batch = x_train[index_batch]
            y_batch = y_train[index_batch]

            dweights = derivative_cost_wrt_params(x=x_batch, w=weights, y=y_batch)
            alpha = backtracking_line_search(x=x_batch, w=weights, y=y_batch, p=-dweights, rho=rho, alpha=init_alpha, c=c_1)

            wolfe_II_list.append(check_wolfe_II(x=x_batch, w=weights, y=y_batch, alpha=alpha, p=-dweights, c_2=c_2))
            goldstein_list.append(check_goldstein(x=x_batch, w=weights, y=y_batch, alpha=alpha, p=-dweights, c=c))

            if optimizer.lower() == "gd":
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
                                            beta_2=0.99)
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

        train_cost = sigmoid_cross_entropy_with_x_w(x=x_train, w=weights, y=y_train)
        val_cost = sigmoid_cross_entropy_with_x_w(x=x_val, w=weights, y=y_val)

        train_output = predict(x=x_train, w=weights, threshold=threshold).astype(np.float)
        train_acc = (train_output == y_train).sum()/y_train.shape[0]

        val_output = predict(x=x_val, w=weights, threshold=threshold).astype(np.float)
        val_acc = (val_output == y_val).sum()/y_val.shape[0]

        train_cost_list.append(train_cost)
        train_acc_list.append(train_acc)

        val_cost_list.append(val_cost)
        val_acc_list.append(val_acc)

        if (epoch + 1) % 100 == 0:
            print(epoch, alpha, train_cost, val_cost, train_acc, val_acc, np.linalg.norm(weights - prev_weights), np.abs(prev_train_cost - train_cost), np.linalg.norm(dweights))

        prev_weights = copy.deepcopy(weights)
        prev_train_cost = copy.deepcopy(train_cost)

    

if __name__ == "__main__":

    args = get_args()

    csv_path = args.csv_path
    normalize = "minmax"
    use_bias = True
    initializer = "xavier"
    c_1 = 1e-2
    c_2 = 0.9
    c = 0.25
    threshold = 0.6
    rho = 0.5
    batch_size = 64


    x_train, y_train, x_val, y_val = create_data(csv_path=csv_path, normalize=normalize, use_bias=use_bias)

    #print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    weights = init_weights(x=x_train, use_bias=use_bias, initializer=initializer)
    #weights = {"gradient_descent": copy.deepcopy(weights),
    #           "batch_gradient_descent": copy.deepcopy(weights),
    #           "stochastic_gradient_descent": copy.deepcopy(weights)}
    #weights = np.array([0.87, 2.89, -2.13, -4.36, 3.51, 8.56, 2.85, -7.92, 9.78, -9.57, 0.87, 0.78, -30.24, 7.25, -0.2, 0.17, -0.2, -0.17, 0.07, -3.02, 7.19])

    print(weights)

    train_gradient_descent(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, init_weights=weights,
                           optimizer="AdaBelief",
                           threshold=threshold, num_epochs=100000, c_1=c_1, c_2=c_2, c=c, rho=rho,
                           init_alpha=0.1)

    train_batch_gradient_descent(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, init_weights=weights,
                                optimizer="AdaBelief",
                                batch_size=batch_size, threshold=threshold, num_epochs=100000, c_1=c_1, c_2=c_2, c=c,
                                rho=rho, init_alpha=1e-1)