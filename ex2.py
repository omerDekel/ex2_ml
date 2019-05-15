import numpy as np
import sys

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


def parse_data(file_name_x='trainx.txt'):
    """
    Parse data from file.
    :param file_name:  file path to parse
    :return:  a pair of attributes and data.
    """
    train_x = []
    # with open(file_name_x, 'r') as file:
    #     train_x += [line.strip('\n').split(',') for line in file]
    with open(file_name_x, 'r') as f:
        for line in f:
            new_line = edit_line(line)
            train_x.append(new_line)
    return train_x
def edit_line(line):
    sex_class = {"I": np.asarray([0, 1, 0]), "M": np.asarray([0, 0, 1]), "F": np.asarray([1, 0, 0])}
    sex = line[0]
    new_l = line.strip('\n').split(',')
    new_l.remove(sex)
    sex_vec = sex_class[sex]
    new_l = np.concatenate((sex_vec, new_l))
    new_l = [float(attr) for attr in new_l]
    return new_l


def transfer_sex(data):
    sex_class = {"I": 0, "M": 1, "F": 2}
    for line in data:
        line[0] = sex_class[line[0]]
        for i, att in enumerate(line):
            line[i] = float(att)
def main():
    train_x = parse_data(sys.argv[1])
    test_x = parse_data(sys.argv[3])
    train_y = np.genfromtxt(sys.argv[2], dtype="int")
    test_x = np.array(test_x)
    test_x = zscore_normalization(test_x)
    test_y = np.genfromtxt('test_y.txt', dtype="int")


    per = perceptron.Perceptron(train_x, train_y)
    per.perceptron_training()
    per.perceptron_test(test_x,test_y)
    # pa = PassiveAgressive.passiveAgressive(train_x, train_y)
    # pa.training()
    # pa.pa_test(test_x, test_y)
    #
    # for x in test_x:
    #     print("perceptron: {0}, svm: {1}, pa: {2}".format(per.answer(x), SVM.answer(x), pa.answer(x)))


if __name__ == '__main__':
    main()
