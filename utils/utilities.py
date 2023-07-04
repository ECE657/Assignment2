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

def plot_data(X, y, ax, s=80, positive_label="y=1", negative_label="y=-1"):
    pos = y == 1
    neg = y == -1
    pos = pos.reshape(-1) #convert to 1D array if needed
    neg = neg.reshape(-1) #convert to 1D array if needed

    ax.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c = 'red', label=positive_label)
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, c = 'blue', label=negative_label, facecolors='none', lw=3)
    ax.legend(loc='best')

def plot_data_in_comparison(X_train, y_train, X_test, y_test, X_pred, y_pred):
    fig,axs = plt.subplots(1,3,figsize=(12,4))

    for ax in axs:
        ax.axis([-2.5, 2.5, -2.5, 2.5])
        ax.set_ylabel('$x_2$', fontsize=12)
        ax.set_xlabel('$x_1$', fontsize=12)

    axs[0].set_title('Training data')
    axs[1].set_title('Testing data (expectation)')
    axs[2].set_title('Prediction')
    plot_data(X_train, y_train, axs[0], s=50)
    plot_data(X_test, y_test, axs[1], s=50)
    plot_data(X_pred, y_pred, axs[2], s=50)
    plt.tight_layout()
    plt.show()

