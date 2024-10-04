import pandas as pd
import numpy as np


def data_process(X, Y):
    X = np.delete(X, [23, 55, 4, 9, 52, 44, 45, 33, 43, 20, 1, 50], axis=0)
    Y = np.delete(Y, [23, 55, 4, 9, 52, 44, 45, 33, 43, 20, 1, 50], axis=0)
    print("Shape of input is = {} and the output is = {}".format(X.shape, Y.shape))
    return X, Y


def load_data(single_task=False, task_name=None):
    if single_task == True and task_name == None:
        raise ValueError("Argument Task Name is required when Single Task = True")
    if single_task == True:
        X = np.load("dataset/" + str(task_name) + "_FEATURES_60.npy")
        Y = np.load("dataset/" + str(task_name) + "_LABELS_60.npy")
        # dataset= np.load("Final_N95_FEATURES_60.npy")
        # X = np.array(dataset[:,1:-2], dtype=float)
        # if task_name == "FVC":
        #     Y = np.array(dataset[:,-1], dtype=float)
        # elif task_name == "FEV1":
        #     Y = np.array(dataset[:,-2], dtype=float)

    else:
        dataset= np.load("Final_N95_FEATURES_60.npy")
        X = np.array(dataset[:,1:-2], dtype=float)
        Y1 = np.array(dataset[:,-1], dtype=float).reshape(-1, 1)
        Y2 = np.array(dataset[:,-2], dtype=float).reshape(-1, 1)
        Y = np.concatenate((Y1, Y2), axis=1)
        # X1 = np.load("dataset/FVC_FEATURES_60.npy")
        # X2 = np.load("dataset/FEV1_FEATURES_60.npy")
        # # X3 = np.load("dataset/PEF_FEATURES_60.npy")

        # Y1 = np.load("dataset/FVC_LABELS_60.npy").reshape(-1, 1)
        # Y2 = np.load("dataset/FEV1_LABELS_60.npy").reshape(-1, 1)
        # Y3 = np.load("dataset/PEF_LABELS_60.npy").reshape(-1, 1)
        # X = np.concatenate((X1, X2), axis=1)
        

    # X, Y = data_process(X, Y)

    return X, Y


def train_pool_split(X, Y, train_idx, test_idx):
    pool_idx = [i for i in range(0, 48) if i not in train_idx + test_idx]
    X_train, X_test, X_pool = X[train_idx], X[test_idx], X[pool_idx]
    Y_train, Y_test, Y_pool = Y[train_idx], Y[test_idx], Y[pool_idx]
    return X_train, X_test, X_pool, Y_train, Y_test, Y_pool
