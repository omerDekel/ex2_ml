import random
from scipy import stats

import numpy as np
#from scipy.stats import stats


class Perceptron:
    """

    """
    def zscore_normalization(self,data):
        #array = np.array(data)
        attributes_len = data.shape[1]

        for i in range(attributes_len):

            mean = np.mean(data[:, i])
            std_dev = np.std(data[:, i])
            if (std_dev != 0 and std_dev != None and mean != None):
                data[:, i] = (data[:, i] - mean) / std_dev
        return data
    def min_max_norm(self,data):
        #array = np.array(data)
        num_of_columns = data.shape[1]  # shape[1] = num of columns.
        for i in range(0, num_of_columns):  # normlize per all values in column, not by values per rows.
            v = data[:, i]
            if (v.max() - v.min() != 0):
                data[:, i] = (v - v.min()) / (v.max() - v.min())
        return data

        """
                for i in range(data):
            for i,x in enumerate(line):
                v = data[i]
                base= min(v)
                range_ = max(v) - base
                if range_ != 0:
                   line[i] = (x - base)/range
        """


    def __init__(self ,train_x = "", train_y = "",test = "",test_y=""):
        self.weights = np.zeros((3,8))
        #self.train_x = stats.zscore(train_x)
        #self.train_x = self.min_max_norm(train_x)
        train_x = np.array(train_x)
        self.train_x = self.zscore_normalization(train_x)
        #self.train_x = np.array(train_x)
        #self.train_x = self.min_max_norm(train_x)
        #self.train_y = np.array(train_y)
        self.train_y = train_y
        #self.test = stats.zscore(test)
        test = np.array(test)
        #self.test =  self.min_max_norm(test)
        self.test = self.zscore_normalization(test)
        #self.test = test
        self.test_y = test_y
    def perceptron_training(self):
        etha = 0.01
        epochs = 30
        for e in range(epochs):
            #self.train_x, self.train_y = shuffle(self.train_x, self.train_y)
            # mapIndexPosition = list(zip(self.train_x, self.train_y))
            # random.shuffle(mapIndexPosition)
            # self.train_x, self.train_y = zip(*mapIndexPosition)

            for x,i_y in zip(self.train_x,self.train_y):
                y_hat = np.argmax(np.dot(self.weights,x))
                # i_y = float(y[0])
                # i_y = int(i_y)
                if (i_y != y_hat):
                    # y_hat = float(y_hat)
                    # y_hat = int(y_hat)
                    self.weights[i_y,:] = self.weights[i_y,:] + etha* x
                    self.weights[y_hat,:] = self.weights[y_hat,:] - etha*x
            etha /= (e+1)
    def answer(self,xi):
        return np.argmax(np.dot(self.weights,xi))
    def perceptron_test(self):
        m = len(self.test)
        result = []
        M_per = 0
        for x, i_y in zip(self.test, self.test_y):
            y_hat = np.argmax(np.dot(self.weights, x))
            # i_y = float(y[0])
            # i_y = int(i_y)
            if (i_y != y_hat):
                M_per = M_per + 1
            print (y_hat)
            #result.append(y_hat)
        print ("perceptron err" , float(M_per)/m)
        #return result
