import numpy as np
from HelperFunctions.metrics import confusion_matrix
from HelperFunctions.metrics import unique_elements

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, num_epochs=100, use_bias=True):
        """
        Initialize the Perceptron.

        Parameters:
        - num_features (int): Number of input features.
        - learning_rate (float): Learning rate or step size for weight updates.
        - num_epochs (int): Number of training epochs.
        - use_bias (bool): Flag to indicate whether to use the bias term.
        """
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.use_bias = use_bias
        self.weights = np.zeros(num_features + int(use_bias))  # Additional weight for the bias term

    def predict(self, features):
        """
        Predict the output label based on the input features.

        Parameters:
        - features (array-like): Input features.

        Returns:
        - prediction (int): Predicted label (0 or 1).
        """
        activation = np.dot(features, self.weights[int(self.use_bias):]) + self.weights[0] * self.use_bias
        return 1 if activation >= 0 else 0

    def train(self, training_data, labels):
        """
        Train the Perceptron using the training data and labels.

        Parameters:
        - training_data (array-like): Training data.
        - labels (array-like): Labels corresponding to the training data.
        """
        for _ in range(self.num_epochs):
            for features, label in zip(training_data, labels):
                prediction = self.predict(features)
                update = self.learning_rate * (label - prediction)
                self.weights[int(self.use_bias):] += update * features
                self.weights[0] += update * self.use_bias

    def evaluate(self, test_data, labels):
        """
        Evaluate the accuracy of the Perceptron on the test data.

        Parameters:
        - test_data (array-like): Test data.
        - labels (array-like): Labels corresponding to the test data.

        Returns:
        - accuracy (float): Accuracy of the Perceptron on the test data.
        """
        correct = 0
        y_hat = []
        for features, label in zip(test_data, labels):
            prediction = self.predict(features)
            y_hat.append(prediction)
            if prediction == label:
                correct += 1
        accuracy = correct / len(test_data)
        unique_element = unique_elements(labels, y_hat)
        confusion_matrix(labels, y_hat, unique_element)
        return accuracy




