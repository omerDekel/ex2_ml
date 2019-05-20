import random

import numpy as np
import sys

import SVM
import perceptron
import PassiveAgressive
#

def zscore_normalization(data):
    """
    zscore normalizaition for train
    :param data:  train
    :return: norm of data .
    """
    #data = np.array(data)
    attributes_len = data.shape[1]

    for i in range(attributes_len):

        mean = np.mean(data[:, i])
        std_dev = np.std(data[:, i])
        if (std_dev != 0 and std_dev != None and mean != None):
            data[:, i] = (data[:, i] - mean) / std_dev
    return data
def zscore_test(test,std_dev_vec, mean_vec):
    """
    zscore normalizaition for test
    :param test:
    :param std_dev_vec: vector
    :param mean_vec: vector
    :return: norm of test data.
    """
    test= np.array(test)
    attributes_len = test.shape[1]
    for i in range(attributes_len):
        std_dev = std_dev_vec[i]
        mean = mean_vec[i]
        if (std_dev != 0 and std_dev != None and mean != None):
            test[:, i] = (test[:, i] - mean) / std_dev
    return test
def min_max_norm_test(data,mini,maxi):
    """

    :param data: min max normalization of test .
    :param mini: vector
    :param maxi: vector
    :return:
    """
    mini = np.array(mini)
    max = np.array(maxi)
    data = np.array(data)
    cols = data.shape[1]
    for i in range(0,cols):
        v = data[:, i]
        if (maxi[i] - mini[i] != 0):
            data[:, i] = (v -mini[i]) / (max[i] - mini[i])
        else:
            data[:,i] = 1
    return data
def min_max_norm(data):
    """

    :param data:
    :return:
    """
    mini_vec = []
    maxi_vec = []
    data = np.array(data)
    num_of_columns = data.shape[1]
    for i in range(0, num_of_columns):
        v = data[:, i]
        max_v = v.max()
        min_v= v.min()
        mini_vec.append(min_v)
        maxi_vec.append(max_v)
        if (max_v - min_v != 0):
            data[:, i] = (v - min_v) / (max_v - min_v)
        else:
            data[:,i] = 1
    return data,mini_vec,maxi_vec



def parse_data(file_name_x='trainx.txt'):
    """
    Parse data from file.
    :param file_name:  file path to parse
    :return:  a pair of attributes and data.
    """
    train_x = []
    with open(file_name_x, 'r') as f:
        for line in f:
            new_line = edit_line(line)
            train_x.append(new_line)
    train_x = np.array(train_x)
    train_x = train_x.astype(dtype = 'float')
    return train_x
def edit_line(line):
    """
    edit line . translate the categorial feature to vector and concatenate it.
    :param line:
    :return: features in line in vector.
    """
    sex_class = {"I": np.asarray([0, 1, 0]), "M": np.asarray([0, 0, 1]), "F": np.asarray([1, 0, 0])}
    sex = line[0]

    new_l = line.strip('\n').split(',')
    new_l.remove(sex)
    sex_vec = sex_class[sex]
    new_l = np.concatenate((sex_vec, new_l))
    return new_l
def calculate_stddev_mean(data):
    """

    :param data:
    :return:
    """
    std_vec = []
    mean_vec = []
    attributes_len = data.shape[1]

    for i in range(attributes_len):

        mean = np.mean(data[:, i])
        std_dev = np.std(data[:, i])
        mean_vec.append(mean)
        std_vec.append(std_dev)
    return mean_vec,std_vec
def calculate_minmax_vec(data):
    """

    :param data:
    :return:
    """
    mini = []
    max = []
    for i in range(0, 10):
        v = data[:, i]
        mini.append(v.min())
        max.append(v.max())
    return mini,max
def main():
    """

    :return:
    """
    # parsing the data from user
    train_x = parse_data(sys.argv[1])
    test_x = parse_data(sys.argv[3])
    train_y = np.genfromtxt(sys.argv[2], dtype="str")
    train_y = train_y.astype(np.float)
    train_y = train_y.astype(np.int)
    #shuffling
    mapIndexPosition = list(zip(train_x, train_y))
    random.shuffle(mapIndexPosition)
    train_x, train_y = zip(*mapIndexPosition)
    #train_x = np.array(train_x)
    #train_y = np.array(train_y)
    #cross_validation(train_x,train_y,5)
    # normalize the train and return its minimum and maximum vector.
    mtrain_x,mini, maxi = min_max_norm(train_x)
    mtest_x = min_max_norm_test(test_x,mini,maxi)
    #creating and training machine learning algorithms on provided data
    per = perceptron.Perceptron(mtrain_x, train_y)
    per.perceptron_training()
    svm = SVM.svm(mtrain_x, train_y)
    svm.svm_training()
    pa = PassiveAgressive.passiveAgressive(mtrain_x, train_y)
    pa.training()

    # printing prediction of all three algorithms over received example of data
    for x in mtest_x:
        print("perceptron: {0}, svm: {1}, pa: {2}".format(per.answer(x), svm.answer(x), pa.answer(x)))



def cross_validation(X_train, Y_train, K):
    """

    :param X_train:
    :param Y_train:
    :param K:
    :return:
    """
    per_success = 0
    svm_success = 0
    pa_success = 0
    splitted_X = np.array_split(X_train, K, axis=0)
    splitted_Y = np.array_split(Y_train, K, axis=0)
    for i in range(0, K):
        X_tr =  np.concatenate([splitted_X[i] for i in range(K) if K != i], axis=0)
        Y_tr = np.concatenate([splitted_Y[i] for i in range(K) if K != i], axis=0)
        X_tst = splitted_X[i]
        #X_tst = zscore_test(X_tst, std_vec, mean_vec)
        Y_tst = splitted_Y[i]
        # = calculate_minmax_vec(X_tr)
        X_tr,mini, maxi= min_max_norm(X_tr)
        X_tst = min_max_norm_test(X_tst, mini, maxi)
        per = perceptron.Perceptron(X_tr, Y_tr)
        pa = PassiveAgressive.passiveAgressive(X_tr, Y_tr)
        svm = SVM.svm(X_tr,Y_tr)
        svm.svm_training()
        #
        per.perceptron_training()
        per_success += per.perceptron_test(X_tst, Y_tst)

        svm_success += svm.svm_test(X_tst, Y_tst)

        pa.training()
        pa_success +=  pa.pa_test(X_tst, Y_tst)

    # print avg. success
    print("perceptron success:" + str(per_success/K))
    print("svm success:" + str(svm_success / K))
    print("pa success:" + str(pa_success / K))

if __name__ == '__main__':
    main()
