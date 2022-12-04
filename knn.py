import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

k = 5


def distance(x, y):
    dist = 0
    for i in range(len(x)):
        dist += pow(x[i]-y[i], 2)
    return round(sqrt(dist), 2)


def gedata():
    data_csv = pd.read_csv("knn_data.csv")
    data_csv.dropna(axis=0, subset=["AUST"])
    data = np.array(data_csv)
    data = np.delete(data, 0, 1)
    data = np.delete(data, 0, 1)
    inputs = data[:, :-1]
    target = data[:, -1]
    return inputs, target


def preparedata(input, target):
    return train_test_split(input, target, test_size=0.3, random_state=42)


def knearsetneighbours(x_train, p):
    distances = {}
    for i in range(len(x_train)):
        dist = distance(x_train[i], p)
        distances[i] = dist
    knearest_enighbours = sorted(distances.items(), key=lambda x: x[1])
    return knearest_enighbours[:k]


def knn(x_train, y_train, p):
    k_nearset_neighbours = knearsetneighbours(x_train, p)

    classes = {}
    max_val = 0
    maxclass = ""

    for neightbour in k_nearset_neighbours:
        index = neightbour[0]
        key_class = y_train[index]
        if key_class in classes:
            classes[key_class] += 1
        else:
            classes[key_class] = 1

        if (classes[key_class] > max_val):
            max_val = classes[key_class]
            maxclass = key_class

    return maxclass


def main():
    data, target = gedata()
    x_train, x_test, y_train, y_test = preparedata(data, target)

    test_actual = []
    test_predicted = []
    for i in range(len(x_test)):
        test_actual.append(y_test[i])
        knn_predicted = knn(x_train, y_train, x_test[i])
        test_predicted.append(knn_predicted)
    cf_matrix = confusion_matrix(test_actual, test_predicted)
    print(cf_matrix)


main()
