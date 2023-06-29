import numpy as np

def compute_cost(y_hat, y):
    return np.mean(np.square(y_hat - y))


def compute_accuracy(Y_hat, Y):
    correct_count = sum((Y[i] == Y_hat[i]).all() for i in range(len(Y)))
    accuracy = correct_count / len(Y)
    return accuracy