import numpy as np
import matplotlib.pyplot as plt 

def compute_cost(y_hat, y):
    return np.mean(np.square(y_hat - y))


def compute_accuracy(Y_hat, Y):
    correct_count = sum((Y[i] == Y_hat[i]).all() for i in range(len(Y)))
    accuracy = correct_count / len(Y)
    return accuracy

def plot_analysis(title, y_label, x_data, y_data, color, x_label="Radius"):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_data, y_data, color)
    plt.show()

def plot_full_analysis(title, y_label, x_data, y1_data, y2_data, y3_data, color1, color2, color3, label1, label2, label3, x_label="Radius"):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_data, y1_data, color=color1, label=label1)
    plt.plot(x_data, y2_data, color=color2, label=label2)
    plt.plot(x_data, y3_data, color=color3, label=label3)
    plt.legend()
    plt.show()