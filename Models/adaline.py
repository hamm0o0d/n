import numpy as np
import random
class Adaline:
    def __init__(self, mseThreshold,num_features, learning_rate=0.01, num_epochs=100):
        """
        Initialize the Adaline.

        Parameters:
        - mseThreshold (float): Number of min mean square error.
        - num_features (int): Number of input features.
        - learning_rate (float): Learning rate or step size for weight updates.
        - num_epochs (int): Number of training epochs.
        - use_bias (bool): Flag to indicate whether to use the bias term.
        """
        self.mseThreshold = mseThreshold
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights=[random.randint(0, 10) for i in range(num_features)]

    def predict(self, features):
        """
        Predict the output label based on the input features.

        Parameters:
        - features (array-like): Input features.

        Returns:
        - adaline (int): Predicted label .
        """
        activation = np.dot(features, self.weights)
        return 1 if activation >= 0 else 0

    def train(self, training_data, labels):
        """
        Train the Adaline using the training data and labels.

        Parameters:
        - training_data (array-like): Training data.
        - labels (array-like): Labels corresponding to the training data.
        """
        maxIteration = self.num_epochs * 100   # avoid if threshold is too small to reach
        for _ in range(maxIteration):
            for features, label in zip(training_data, labels):
                prediction = self.predict(features)
                error = label - prediction
                update = self.learning_rate * error
                self.weights += update * features
            meanSquareError =0
            for features, label in zip(training_data, labels):
                prediction = self.predict(features)
                error = label - prediction
                meanSquareError += (0.5*error*error)

            meanSquareError *= 1 / len(training_data)
            print(meanSquareError)
            if meanSquareError < self.mseThreshold:
               break

    def evaluate(self, test_data, labels):
        """
        Evaluate the accuracy of the Adaline on the test data.

        Parameters:
        - test_data (array-like): Test data.
        - labels (array-like): Labels corresponding to the test data.

        Returns:
        - accuracy (float): Accuracy of the Perceptron on the test data.
        """
        correct = 0
        for features, label in zip(test_data, labels):
            prediction = self.predict(features)
            if prediction == label:
                correct += 1
        accuracy = correct / len(test_data)
        return accuracy




