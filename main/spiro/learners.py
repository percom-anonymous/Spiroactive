import random
import numpy as np
from .models import *
from .utils import *


class Learner:
    def __init__(self, UncertainityModel, acq_function: str, X_pool, y_pool):
        self.model = UncertainityModel
        self.acq_function = acq_function
        self.X_train = None
        self.Y_train = None
        self.X_pool = X_pool
        self.Y_pool = y_pool

    def fit(self, X, y,verbose=0):
        self.X_train = X
        self.Y_train = y
        if verbose==0:
            print("Fitting the model on initial train data")
        self.model.fit(X, y)

    def score(self, X, actual, metrics="mape",print_score=False):
        pred, _ = self.model.predict(X)
        if metrics == "mape":
            score = mape_score(actual, pred)
            if print_score:
                print("MAPE score: ",score)
        elif metrics=='rmse':
            score = rmse(actual, pred)
            if print_score:
                print("RMSE score: ",score)
        else:
            raise ValueError("Invalid metrics")
        return score, pred

    def query(self,print_query_idx=False):
        if self.acq_function == "std_dev":
            q, std_devn = self.model.get_std_dev(self.X_pool)
            if print_query_idx:
                print("Queried index is",q)
            return q, std_devn
        elif self.acq_function == "random":
            # return self.model.random_sampling(self.X_pool), None
            q = random_sampling(self.X_pool)
            if print_query_idx:
                print("Queried index is",q)
            return q, None
        
    def teach(self, q):
        self.X_train = np.append(self.X_train, self.X_pool[q].reshape(1, -1), axis=0)
        self.Y_train = np.append(self.Y_train, self.Y_pool[q])
        self.X_pool = np.delete(self.X_pool, q, axis=0)
        self.Y_pool = np.delete(self.Y_pool, q)
        self.fit(self.X_train, self.Y_train,verbose=1)

    # def teach(self, q):
    #     # print(q)
    #     # print(self.X_pool[q].shape)
    #     self.X_train = np.append(self.X_train, self.X_pool[q], axis=0)
    #     print(self.X_train.shape)
    #     self.Y_train = np.append(self.Y_train, self.Y_pool[q])
    #     self.X_pool = np.delete(self.X_pool, q, axis=0)
    #     self.Y_pool = np.delete(self.Y_pool, q)
    #     self.fit(self.X_train, self.Y_train,verbose=1)


class MultiLearner:
    def __init__(self, model_list, acq_func: str, X_pool, Y_pool: list):
        self.model_list = model_list
        self.acq_func = acq_func
        self.X_pool = X_pool
        self.Y_pool = {}
        self.Y_train = {}
        for i in range(len(Y_pool)):
            self.Y_pool[str(i + 1)] = Y_pool[i]
            self.Y_train[str(i + 1)] = None
        self.X_train = None
        self.X_train_shape = None

    def fit(self, X, Y: list,verbose=0):
        self.X_train = X
        self.X_train_shape = X.shape[0]
        if verbose==0:
            print("Fitting the model on initial train data")
        for model_idx, model in enumerate(self.model_list):
            self.Y_train[str(model_idx + 1)] = Y[model_idx]
            model.fit(X, Y[model_idx])

    def query(self, w=None,print_query_idx=False):
        self.q = None
        if self.acq_func == "std_dev" and w == None:
            raise ValueError("Weight vector is required for std_dev")

        if self.acq_func == "std_dev":
            self.q,final_std = query_using_std_dev(self.X_pool, self.model_list, w)
            if print_query_idx:
                print("Queried index is",self.q)
            return self.q, final_std
        
        elif self.acq_func == "random":
            self.q = random_sampling(self.X_pool)
            if print_query_idx:
                print("Queried index is",self.q)
            return self.q, None

        elif self.acq_func == "rank":
            self.q,final_rank = query_using_rank(self.X_pool, self.model_list)
            if print_query_idx:
                print("Queried index is",self.q)
            return self.q, final_rank

        elif self.acq_func == "round_robin":
            self.q,final_std = query_using_round_robin(self.X_pool, self.model_list,self.X_train.shape[0],self.X_train_shape)
            if print_query_idx:
                print("Queried index is",self.q)
            return self.q, final_std
        
        

    def teach(self, q):
        self.X_train = np.append(self.X_train, self.X_pool[q].reshape(1, -1), axis=0)
        for model_idx, model in enumerate(self.model_list):
            self.Y_train[str(model_idx + 1)] = np.append(
                self.Y_train[str(model_idx + 1)], self.Y_pool[str(model_idx + 1)][q]
            )
            model.fit(self.X_train, self.Y_train[str(model_idx + 1)])

    def score(self, X, actual, metrics="mape",print_score=False):
        scores = []
        for i in range(len(self.model_list)):
            pred, _ = self.model_list[i].predict(X)
            if metrics == "mape":
                score = mape_score(actual[:, i], pred)
            elif metrics=='rmse':
                score = rmse(actual[:,i], pred)
            else:
                raise ValueError("Invalid metrics")
            scores.append(score)
        if print_score:
            print("Score for the multiple tasks are", scores)

        return scores, None


class NN_Learner:
    def __init__(self, model, acq_function: str, X_pool, y_pool):
        self.model = model
        self.acq_function = acq_function
        self.X_train = None
        self.Y_train = None
        self.X_pool = X_pool
        self.Y_pool = y_pool
        self.epochs = 10

    def fit(self, X, y,epochs,callback=None,verbose=0):
        self.X_train = X
        self.Y_train = y
        self.epochs = epochs
        if verbose==0:
            print("Fitting the model on initial train data")
        self.model.fit(X, y,callbacks=callback,epochs=epochs)

    def score(self, X, actual, metrics="mape",print_score=False):
        pred = self.model.predict(X)
        if metrics == "mape":
            score = mape_score(actual, pred)
            if print_score:
                print("MAPE score: ",score)
        elif metrics=='rmse':
            score = rmse(actual, pred)
            if print_score:
                print("RMSE score: ",score)
        else:
            raise ValueError("Invalid metrics")
        return score, pred

    def query(self):
        if self.acq_function == "std_dev":
            q, std_devn = self.model.get_std_dev(self.X_pool)
            return q, std_devn
        if self.acq_function == "random":
            return random_sampling(self.X_pool), None

    def teach(self, q):
        self.X_train = np.append(self.X_train, self.X_pool[q].reshape(1, -1), axis=0)
        self.Y_train = np.append(self.Y_train, self.Y_pool[q])
        self.X_pool = np.delete(self.X_pool, q, axis=0)
        self.Y_pool = np.delete(self.Y_pool, q)
        self.fit(self.X_train, self.Y_train,self.epochs,verbose= 1)



