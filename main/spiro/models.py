import numpy as np
import random
from sklearn.metrics import mutual_info_score

class Models:
    def __init__(self):
        self.model = None

class SklearnEnsemble(Models):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        predn = [tree.predict(X) for tree in self.model.estimators_]

        return self.model.predict(X), predn

    def get_std_dev(self, X):
        predn = [tree.predict(X) for tree in self.model.estimators_]
        std_devn = np.std(predn, axis=0)
        # indices = sorted(range(len(std_devn)), key=lambda i: std_devn[i], reverse=True)
        # query_idx = indices[:2]
        query_idx = np.argmax(std_devn)
        return query_idx, std_devn

    def teach(self, X, y):
        self.model.teach(X, y)

class CommitteeRegressor(Models):
    def __init__(self, model_list):
        self.model_list = model_list

    def fit(self, X, y):
        for learner_idx, learner in enumerate(self.model_list):
            learner.fit(X, y)

    def predict(self, X):
        prediction = np.zeros(shape=(len(X), len(self.model_list)))
        for learner_idx, learner in enumerate(self.model_list):
            prediction[:, learner_idx] = learner.predict(X).reshape(
                -1,
            )
        mean_pred = np.mean(prediction, axis=1)
        return mean_pred, prediction

    # def get_std_dev(self, X):
    #     _, prediction = self.predict(X)
    #     std_devn = np.std(prediction, axis=1)
    #     indices = sorted(range(len(std_devn)), key=lambda i: std_devn[i], reverse=True)
    #     query_idx = indices[:2]
    #     return query_idx, std_devn
    
    def get_std_dev(self, X):
        _, prediction = self.predict(X)
        std_devn = np.std(prediction, axis=1)
        query_idx = np.argmax(std_devn)
        return query_idx, std_devn


class NeuralNetwork(Models):
    def __init__(self, model, loss, optimizer, learning_rate,device):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
    
    def fit(self,train_dataset,epochs):
        for epoch in range(epochs):
            for data,labels in train_dataset:
                self.optimizer.zero_grad()
                output = self.model(data.to(self.device))
                loss_fn = self.loss(output,labels.to(self.device))
                loss_fn.backward()
                self.optimizer.step()
        
    def predict(self,test_dataset):
        outputs = []
        self.model.eval()
        for data,labels in test_dataset:
            output = self.model(data.to(self.device))
            outputs.append(output)
        return outputs.detach().numpy()
            


