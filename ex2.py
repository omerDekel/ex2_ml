import numpy as np

import SVM
import perceptron
import PassiveAgressive


def zscore_normalization(data):
    data = np.array(data)
    attributes_len = data.shape[1]

    for i in range(attributes_len):

        mean = np.mean(data[:, i])
        std_dev = np.std(data[:, i])
        if (std_dev != 0 and std_dev != None and mean != None):
            data[:, i] = (data[:, i] - mean) / std_dev
    return data


def min_max_norm(data):
    data = np.array(data)
    num_of_columns = data.shape[1]  # shape[1] = num of columns.
    for i in range(0, num_of_columns):  # normlize per all values in column, not by values per rows.
        v = data[:, i]
        if (v.max() - v.min() != 0):
            data[:, i] = (v - v.min()) / (v.max() - v.min())
    return data

def parse_data(file_name_x='trainx.txt', file_name_y ='trainy.txt'):
    """
    Parse data from file.
    :param file_name:  file path to parse
    :return:  a pair of attributes and data.
    """
    sex_class = {"I":0, "M":1, "F":2}
    train_x = []
    test = []
    test_y = []
    train_y = []
    with open(file_name_x, 'r') as file:
        train_x += [line.strip('\n').split(',') for line in file]
    # with open(file_name_y,'r') as file:
    #     train_y += [line.strip('\n').split(',') for line in file]
    train_y = np.genfromtxt(file_name_y,dtype="int" )
    for line in train_x:
        line[0] = sex_class[line[0]]
        for i,att in enumerate(line):
            line[i] = float(att)
    return train_x, train_y
def main():
    train_x = []
    train_y = []
    train_x, train_y = parse_data()
    test_x, test_y = parse_data(file_name_x='test_x.txt',file_name_y= 'test_y.txt')
    # test_x = np.array(test_x)
    test_x = zscore_normalization(test_x)
    # test_y = test_y
    per = perceptron.Perceptron(train_x, train_y, test_x,test_y)
    per.perceptron_training()
    #per.perceptron_test()
    pa = PassiveAgressive.passiveAgressive(train_x,train_y,test_x,test_y)
    pa.training()
    #pa.pa_test()

    for x in test_x:
        print("perceptron: {0}, svm: {1}, pa: {2}".format(per.answer(x), SVM.answer(x), pa.answer(x)))
    #print()

if __name__ == '__main__':
    main()