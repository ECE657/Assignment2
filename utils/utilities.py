import numpy as np
import matplotlib.pyplot as plt 

def compute_cost(y_hat, y):
    return np.mean(np.square(y_hat - y))


def compute_accuracy(Y_hat, Y):
    correct_count = sum((Y[i] == Y_hat[i]).all() for i in range(len(Y)))
    accuracy = correct_count / len(Y)
    return accuracy

def plot_data(X, y, ax, pos_label="y=1", neg_label="y=-1", s=80, loc='best'):
    postive = y == 1
    negative = y == -1
    pos = postive.reshape(-1,) #convert to 1D array if needed
    neg = negative.reshape(-1,) #convert to 1D array if needed
    # print(f"pos = {pos}")
    # Plot examples
    ax.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c = 'red', label=pos_label)
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, label=neg_label, facecolors='none', edgecolors='green', lw=3)
    ax.legend(loc=loc)

    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False

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