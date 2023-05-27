#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 20:14:11 2022
@author: msa
"""
import csv
import math
import random
import operator


class KNN:
    def __init__(self, k):
        self.k = k

    def load_dataset(self, filename, split):
        training_set = []
        test_set = []
        with open(filename, 'rt') as csvfile:
            lines = csv.reader(csvfile)
            dataset = list(lines)
            traning_split = 0
            for x in range(len(dataset)-1):
                traning_split += 0.01
                for y in range(4):
                    dataset[x][y] = float(dataset[x][y])
                if random.random() < split:
                    training_set.append(dataset[x])
                else:
                    test_set.append(dataset[x])
        return training_set, test_set

    def euclidean_distance(self, instance1, instance2, length):
        distance = 0
        for x in range(length):
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)

    def get_neighbors(self, training_set, testInstance, k):
        distances = []
        length = len(testInstance)-1
        for x in range(len(training_set)):
            dist = self.euclidean_distance(
                testInstance, training_set[x], length)
            distances.append((training_set[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    def get_response(self, neighbors):
        class_votes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in class_votes:
                class_votes[response] += 1
            else:
                class_votes[response] = 1
        sorted_votes = sorted(class_votes.items(),
                              key=operator.itemgetter(1), reverse=True)
        return sorted_votes[0][0]

    def get_accuracy(self, test_set, predictions):
        correct = 0
        for x in range(len(test_set)):
            if test_set[x][-1] == predictions[x]:
                correct += 1
        return (correct/float(len(test_set))) * 100.0


if __name__ == "__main__":
    knn = KNN(k=3)
    training_set, test_set = knn.load_dataset('iris.txt', split=0.70)
    print('Train set: ' + repr(len(training_set)))
    print('Test set: ' + repr(len(test_set)))
    predictions = []
    k = 3
    for x in range(len(test_set)):
        neighbors = knn.get_neighbors(training_set, test_set[x], k)
        result = knn.get_response(neighbors)
        predictions.append(result)
        print('actual=' + repr(test_set[x][-1]
                               ) + '-> predicted=' + repr(result))
    accuracy = knn.get_accuracy(test_set, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
