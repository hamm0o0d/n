import pandas as pd
import numpy as np

def train_test_split(x, y, test_size=0.3, random_state=7):

    x = np.array(x)
    y = np.array(y)
    
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