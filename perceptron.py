import random

import numpy as np


class Perceptron:

    def __init__(self ,train_x = "", train_y = ""):
        """

        :param train_x:
        :param train_y:
        """
        self.weights = np.random.uniform(-0.08, 0.08, [3, 10])
        self.train_x = np.array(train_x)
        self.train_y = train_y

    def perceptron_training(self):
        """

        :return:
        """
        epochs = 30
        etha =1/ epochs
        for e in range(epochs):
            mapIndexPosition = list(zip(self.train_x, self.train_y))
            random.shuffle(mapIndexPosition)
            self.train_x, self.train_y = zip(*mapIndexPosition)
            for x,i_y in zip(self.train_x,self.train_y):
                y_hat = np.argmax(np.dot(self.weights,x))
                if (i_y != y_hat):
                    self.weights[i_y,:] = self.weights[i_y,:] + etha* x
                    self.weights[y_hat,:] = self.weights[y_hat,:] - etha*x
            etha /= (e+1)
    def answer(self,xi):
        """

        :param xi:
        :return:
        """
        return np.argmax(np.dot(self.weights,xi))
    def perceptron_test(self,test_x,test_y):
        """

        :param te:
        :param test_y:
        :return:
        """
        m = len(test_x)
        M_per = 0
        for x, i_y in zip(test_x, test_y):
            y_hat = np.argmax(np.dot(self.weights, x))
            if (i_y == y_hat):
                M_per = M_per + 1
        return (float(M_per)/m)

