import random

import numpy as np


class passiveAgressive:

    def __init__(self ,train_x = "", train_y = "",test = "",test_y=""):
        self.weights = np.random.uniform(-0.08, 0.08, [3, 10])
        self.train_x=np.array(train_x)
        self.train_y = np.array(train_y)
    def training(self):
        """

        training pa.

        """
        epohes = 20
        w = self.weights
        counter = 0
        for e in range(epohes):
            for x, i_y in zip(self.train_x, self.train_y):
                y_hat = np.argmax(np.dot(self.weights, x))
                if (i_y != y_hat):
                    loss = max(0, 1 - np.dot(self.weights[i_y,:], x) + np.dot(self.weights[y_hat,:], x))
                    denom =  (2 * (np.power(np.linalg.norm(x), 2)))
                    if denom != 0:
                        tau = loss / denom
                    else:
                        tau = loss / 2
                    self.weights[i_y, :] += tau* x
                    self.weights[y_hat, :] -= tau* x
                    w = w + self.weights
                    counter = counter+1
        self.weights = w/counter

    def pa_test(self,test_x,test_y):
        """

        :param test_x:
        :param test_y:
        :return:
        """
        m = len(test_x)
        M_per = 0
        for x, i_y in zip(test_x, test_y):
            y_hat = np.argmax(np.dot(self.weights, x))
            if (i_y == int(y_hat)):
                M_per = M_per + 1
        return  float(M_per) / m
    def answer(self,xi):
        """

        :param xi:
        :return:
        """
        return np.argmax(np.dot(self.weights,xi))
