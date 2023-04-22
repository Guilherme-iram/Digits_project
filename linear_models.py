import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
import numpy.linalg as LA
import random
from random import sample 
from tqdm import tqdm


class LinearRegression():

    def fit(self, X, y):
        h = np.dot(X.T, X)
        g = np.dot(X.T, y)
        self.w = np.dot((inv(h)), g)

    def predict(self, X):
        return np.sign(np.dot(self.w.T, X.T))
    
    def get_w(self):
        return self.w
    

class PocketPLA():

    def __init__(self, iter = 1000, Nmin=50, Nmax=200):
        self.iter = iter
        self.Nmin = Nmin
        self.Nmax = Nmax

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.w = np.zeros(X.shape[1])

        bestError = len(y)
        bestW = self.w
        
        for j in tqdm(range(self.iter)):
            
            N = random.randint(self.Nmin, self.Nmax)
            indexes = np.random.randint(len(X), size=N)
            X_ = X[indexes]
            y_ = y[indexes]

            for i in range(N):
                if(np.sign(self.w @ X_[i]) != y_[i]):
                    self.w = self.w + (X_[i] * y_[i])
                    eIN = self.errorIN(X_, y_)
                    if(bestError > eIN):
                        bestError = eIN
                        bestW = self.w

        self.w = bestW

    def get_w(self):
        return self.w
    
    def set_w(self, w):
        self.w = w
    
    def h(self, x):
        return np.sign(np.dot(self.w, x))
    
    def errorIN(self, X, y):
        return np.mean(np.sign(np.dot(self.w, X.T)) != y)

    def predict(self, X):
        return [self.h(x) for x in X]
        # return np.sign(np.dot(self.w, X.T))


class LogisticRegression:
    
    def __init__(self, eta=0.1, tmax=1000, batch_size=2048):
        self.eta = eta
        self.tmax = tmax
        self.batch_size = batch_size
        self.w = None
    
    def fit(self, X, y, lamb=1e-6):
        N, d = X.shape
        X = np.array(X)
        y = np.array(y)# .reshape(-1, 1)
        w = np.zeros(d)

        for t in tqdm(range(self.tmax)):
            if self.batch_size < N:
                rand_indexes = np.random.choice(N, self.batch_size, replace=False)
                X_batch, y_batch = X[rand_indexes], y[rand_indexes]
            else:
                X_batch, y_batch = X, y

            sigm = 1 / (1 + np.exp(y_batch.reshape(-1, 1) * np.dot(w, X_batch.T).reshape(-1, 1)))
            gt = - 1 / N * np.sum(X_batch * y_batch.reshape(-1, 1) * sigm, axis=0) 
            gt += lamb * np.linalg.norm(w)

            if np.linalg.norm(gt) < 1e-8:
                break
            
            w -= self.eta * gt

        self.w = w 
    

    def predict_prob(self, X):
        return 1 / (1 + np.exp(-np.dot(X, self.w)))

    def predict(self, X):
        pred = self.predict_prob(X)
        y = np.where(pred >= 0.5, 1, -1)
        return y 


    def get_w(self):
        return self.w
    
    def set_w(self, w):
        self.w = w



class Um_contra_todos:

    def __init__(self, model, digitos):
        self.model = model
        self.digitos = digitos
        self.all_w = []


    def execute(self, X_train, y_train):

        for i, d in enumerate(self.digitos[:-1]):
            # atribua a classe i como positiva e as outras como negativas
            if i == 0:
                y_train_i = np.where(y_train == d, 1, -1)
                
                self.model.fit(X_train, y_train_i)
                self.all_w.append(self.model.get_w())
                d_anterior = d

            else:
                X_train = np.delete(X_train, np.where(y_train == d_anterior), axis=0)
                y_train = np.delete(y_train, np.where(y_train == d_anterior))
                y_train_i = np.where(y_train == d, 1, -1)
                
                self.model.fit(X_train, y_train_i)
                self.all_w.append(self.model.get_w())
                d_anterior = d
        
    def predict_digit(self, X):
        predictions = []
        for i, x in enumerate(X):
            for j, d in enumerate(self.digitos[:-1]):
                if np.sign(self.all_w[j] @ x) == 1:
                    predictions.append(d)
                    break

            if len(predictions) < i+1:
                predictions.append(self.digitos[-1])

        return np.array(predictions)

    def get_all_w(self):
        return self.all_w