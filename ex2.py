import numpy as np
import perceptron
def parse_data(file_name_x='train_x.txt', file_name_y ='train_y.txt'):
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
    with open(file_name_y,'r') as file:
        train_y += [line.strip('\n').split(',') for line in file]
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
    per = perceptron.Perceptron(train_x, train_y, test_x,test_y)
    per.perceptron_training()
    per.perceptron_test()
    #print()

if __name__ == '__main__':
    main()