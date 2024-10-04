import numpy as np
from sklearn.metrics import mean_squared_error as mse
from .models import *


def random_sampling(X_pool):
    query_idx = np.random.choice(
        range(X_pool.shape[0]), size=1, replace=False
    )  # Querying a random index from the pool
    return query_idx

def rank_sampling(rank_array):
    sorted_arr = rank_array.copy()
    sorted_arr = sorted(sorted_arr, reverse=True)
    ranks_dict = {}
    for i, num in enumerate(sorted_arr):
        if num not in ranks_dict:
            ranks_dict[num] = i + 1
    ranks_arr = [ranks_dict[num] for num in rank_array]
    return ranks_arr

def calculate_std_devn(X_pool,model_list):
    std_devn = np.zeros(shape=(X_pool.shape[0], len(model_list)))
    for model_idx, model in enumerate(model_list):
        std_devn[:, model_idx] = model.get_std_dev(X_pool)[1]
    return std_devn

def query_using_std_dev(X_pool, model_list, w=None):
    std_devn = calculate_std_devn(X_pool, model_list)
    for i in range(len(model_list)):
        std_devn[:, i] = (std_devn[:, i] * w[i]).reshape(
            -1,
        )
    final_std = np.sum(std_devn, axis=1)
    q = np.argmax(final_std)
    return q, final_std

def query_using_rank(X_pool, model_list):
    std_devn = calculate_std_devn(X_pool, model_list)
    rank = np.zeros(shape=(X_pool.shape[0], len(model_list)))
    for model_idx, model in enumerate(model_list):
        rank[:, model_idx] = rank_sampling(std_devn[:, model_idx])
    final_rank = np.sum(rank, axis=1)
    q = np.argmin(final_rank)
    return q, final_rank

def query_using_round_robin(X_pool, model_list,X_train_shape,X_train_orig):
    std_devn = calculate_std_devn(X_pool, model_list)
    iter = X_train_shape - X_train_orig
    print(iter)
    q = np.argmax(std_devn[:, iter % len(model_list)])
    return q, std_devn[:, iter % len(model_list)]

def mape_score(y_true, y_pred):
    return 100* np.mean(np.abs((y_true - y_pred) / y_true)) 

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))