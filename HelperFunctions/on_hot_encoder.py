import numpy as np

def one_hot_encode(labels, num_classes):
    # Utilize NumPy's eye function for efficient one-hot encoding
    one_hot_encoded = np.eye(num_classes)[labels]
    return one_hot_encoded