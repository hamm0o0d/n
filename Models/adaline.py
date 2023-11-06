import numpy as np
import random
from HelperFunctions.metrics import confusion_matrix
from HelperFunctions.metrics import unique_elements
class Adaline:
    def __init__(self, mseThreshold,num_features, learning_rate=0.01, num_epochs=100, hasBias = True):
        self.mseThreshold = mseThreshold
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.hasBias = hasBias
        if hasBias:
            self.weights=[random.randint(0, 10) for i in range(num_features + 1)]
        else:
            self.weights=[random.randint(0, 10) for i in range(num_features)]


    def __addBiasToX(self, X):
        # add a column of 1's at the begining
        return np.c_[np.ones((X.shape[0], 1)), X]

    def train(self, training_data, labels):

        if self.hasBias:
            training_data = self.__addBiasToX(training_data)

        for _ in range(self.num_epochs):
            for features, label in zip(training_data, labels):
                prediction = np.dot(features, self.weights)
                error = label - prediction
                update = self.learning_rate * error
                self.weights += update * features
            meanSquareError =0
            for features, label in zip(training_data, labels):
                prediction = np.dot(features, self.weights)
                error = label - prediction
                meanSquareError += 0.5 * (error**2)

            meanSquareError = meanSquareError / len(training_data)
            # print(meanSquareError)
            if meanSquareError < self.mseThreshold:
               break

    def predict(self, features):
    #    if self.hasBias:
    #         features = self.__addBiasToX(features)
       prediction = np.dot(features, self.weights)
       return 1 if prediction >= 0 else 0

# evaluating will not be in this file

    def evaluate(self, test_data, labels):
        if self.hasBias:
            test_data = self.__addBiasToX(test_data)
        y_hat=[]
        correct = 0
        for features, label in zip(test_data, labels):
            prediction = self.predict(features)
            y_hat.append(prediction)
            if prediction == label:
                correct += 1
        accuracy = correct / len(test_data)
        unique_element = unique_elements(labels, y_hat)
        confusion_matrix(labels , y_hat, unique_element )
        return accuracy




