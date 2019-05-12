from random import shuffle

import numpy as np
from scipy.stats import stats


class Perceptron:

    def __init__(self ,train_x = "", train_y = "",test = "",test_y=""):
        self.weights = np.zeros((3,8))
        self.train_x = stats.zscore(train_x)
        self.train_y = train_y
        self.test = stats.zscore(test)
        self.test_y = test_y
    def perceptron_training(self):
        etha = 0.01
        epochs = 100
        for e in range(epochs):
            #self.train_x, self.train_y = shuffle(self.train_x, self.train_y)
            for x,y in zip(self.train_x,self.train_y):
                y_hat = np.argmax(np.dot(self.weights,x))
                i_y = float(y[0])
                i_y = int(i_y)
                if (i_y != int(y_hat)):
                    y_hat = float(y_hat)
                    y_hat = int(y_hat)
                    self.weights[i_y,:] = self.weights[i_y,:] + np.dot(etha, x)
                    self.weights[y_hat,:] = self.weights[y_hat,:] - np.dot(etha ,x)

    def perceptron_test(self):
        m = len(self.test)
        M_per = 0
        for x, y in zip(self.test, self.test_y):
            y_hat = np.argmax(np.dot(self.weights, x))
            i_y = float(y[0])
            i_y = int(i_y)
            if (i_y != int(y_hat)):
                M_per = M_per + 1
            print (y_hat)
            #hiiiiiiiiiii
        print ("perceptron err" , float(M_per)/m)

