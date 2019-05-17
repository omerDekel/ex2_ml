import random

import numpy as np


class passiveAgressive:

    def zscore_normalization(self,data):
        """

        :param data:
        :return:
        """
        #array = np.array(data)
        attributes_len = data.shape[1]

        for i in range(attributes_len):

            mean = np.mean(data[:, i])
            std_dev = np.std(data[:, i])
            if (std_dev != 0 and std_dev != None and mean != None):
                data[:, i] = (data[:, i] - mean) / std_dev
        return data
    def min_max_norm(self,data):
        """

        :param data:
        :return:
        """
        #array = np.array(data)
        num_of_columns = data.shape[1]
        for i in range(0, num_of_columns):
            v = data[:, i]
            if (v.max() - v.min() != 0):
                data[:, i] = (v - v.min()) / (v.max() - v.min())
        return data
    def __init__(self ,train_x = "", train_y = "",test = "",test_y=""):
        #self.weights = np.zeros((3, 10))
        self.weights = np.random.uniform(-0.08, 0.08, [3, 10])
        #self.train_x = stats.zscore(train_x)
        #self.train_x = self.min_max_norm(train_x)
        self.train_x=np.array(train_x)
        #self.train_x = self.zscore_normalization(train_x)
        # self.train_x = np.array(train_x)
        # self.train_x = self.min_max_norm(train_x)
        self.train_y = np.array(train_y)
        # self.test = stats.zscore(test)
        #test = np.array(test)
        #self.test =  self.min_max_norm(test)
        #self.test = self.zscore_normalization(test)
        #self.test = test
        #self.test_y = test_y
    def training(self):
        """

        :return:

        """
        epohes = 30
        for e in range(epohes):
            # mapIndexPosition = list(zip(self.train_x, self.train_y))
            # random.shuffle(mapIndexPosition)
            # self.train_x, self.train_y = zip(*mapIndexPosition)
            for x, i_y in zip(self.train_x, self.train_y):
                y_hat = np.argmax(np.dot(self.weights, x))
                if (i_y != y_hat):
                    loss = max(0, 1 - np.dot(self.weights[i_y,:], x) + np.dot(self.weights[y_hat,:], x))
                    tau = loss / (2 * (np.power(np.linalg.norm(x, ord=2), 2)))
                    self.weights[i_y, :] += tau* x
                    self.weights[y_hat, :] -= tau* x
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
                #print (y_hat)
        #print("pa err", float(M_per) / m)
        return  float(M_per) / m
    def answer(self,xi):
        return np.argmax(np.dot(self.weights,xi))
