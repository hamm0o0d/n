import numpy as np
import pandas as pd
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
        for features, label in zip(test_data, labels):
            prediction = self.predict(features)
            if prediction == label:
                correct += 1
        accuracy = correct / len(test_data)
        return accuracy


def train_test_split(x, y, test_size=0.3, random_state=7):
    num_samples = len(y)
    num_test_samples = int(test_size * len(y))

    # Create an array of indices and shuffle it
    indices = np.arange(num_samples)
    np.random.seed(random_state)
    np.random.shuffle(indices)

    # Split the shuffled indices into training and test sets
    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]

    # Use the indices to select data for training and testing
    X_train = x[train_indices]
    y_train = y[train_indices]
    X_test = x[test_indices]
    y_test = y[test_indices]

    # Reset the indices and convert DataFrames and Series to NumPy arrays
    X_train = pd.DataFrame(X_train).reset_index(drop=True).values
    X_test = pd.DataFrame(X_test).reset_index(drop=True).values
    y_train = pd.Series(y_train).reset_index(drop=True).values
    y_test = pd.Series(y_test).reset_index(drop=True).values

    return X_train, y_train, X_test, y_test

