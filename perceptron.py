import random
from scipy import stats

import numpy as np
#from scipy.stats import stats


class Perceptron:
    """

    """
    def zscore_normalization(self,data):
        array = np.array(data)
        attributes_len = array.shape[1]

        for i in range(attributes_len):

            mean = np.mean(array[:, i])
            std_dev = np.std(array[:, i])

            array[:, i] = (array[:, i] - mean) / std_dev
        return array
    def min_max_norm(self,data):
        array = np.array(data)
        num_of_columns = array.shape[1]  # shape[1] = num of columns.
        for i in range(0, num_of_columns):  # normlize per all values in column, not by values per rows.
            v = array[:, i]
            if (v.max() - v.min() != 0):
                array[:, i] = (v - v.min()) / (v.max() - v.min())
        return array
        
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
        self.train_x = self.zscore_normalization(train_x)
        #self.train_x = np.array(train_x)
        #self.train_x = self.min_max_norm(train_x)
        self.train_y = np.array(train_y)
        #self.test = stats.zscore(test)
        #self.test = np.array(test)
        #self.test =  self.min_max_norm(test)
        self.test = self.zscore_normalization(test)
        #self.test = test
        self.test_y = np.array(test_y)
    def perceptron_training(self):
        etha = 0.001
        epochs = 1000
        for e in range(epochs):
            #self.train_x, self.train_y = shuffle(self.train_x, self.train_y)
            # mapIndexPosition = list(zip(self.train_x, self.train_y))
            # random.shuffle(mapIndexPosition)
            # self.train_x, self.train_y = zip(*mapIndexPosition)
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
            #print (y_hat)
            #hiiiiiiiiiii aaaa
        print ("perceptron err" , float(M_per)/m)

