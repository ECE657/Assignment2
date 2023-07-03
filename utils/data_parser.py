import numpy as np
import pandas as pd
import os

def data_split_train_test(df_X, df_Y, train_portion:float=0.8, seed:int=42):
    """
    Usage: df_X_train, df_Y_train, df_X_test, df_Y_test = data_split_train_test(df_X, df_Y, 0.8)
    Params:
        df_X dataframe features (X)
        df_Y dataframe labels (Y)
        train_portion float
        random seed
    Returns:
        Dataframes df_X_train, df_Y_train, df_X_test, df_Y_test
    """
    df_merge = df_X.join(df_Y)
    df_train=df_merge.sample(frac=train_portion,random_state=seed)
    df_test=df_merge.drop(df_train.index)
    df_X_train = df_train[df_X.columns]
    df_Y_train = df_train[df_Y.columns]
    df_X_test = df_test[df_X.columns]
    df_Y_test = df_test[df_Y.columns]
    return df_X_train, df_Y_train, df_X_test, df_Y_test

def prepare_datasource():
    data_exists = os.path.exists('training_data.csv')
    label_exists = os.path.exists('training_labels.csv')
    if data_exists and label_exists:
        data = np.loadtxt('training_data.csv', delimiter=',')
        labels = np.loadtxt('training_labels.csv', delimiter=',')
        return data, labels
    
    i = np.arange(0,21)  #i = 0..20
    j = np.arange(0,21)  #j = 0..20
    xi, xj = np.meshgrid(-2.0 + 0.2 * i, -2.0 + 0.2*j)  #get all possible combination (xi,xj)

    xi_flattened = xi.reshape(-1) #convert xi 2d to 1d array
    xj_flattened = xj.reshape(-1) #convert xj 2d to 1d array

    samples = np.column_stack((xi_flattened, xj_flattened))  #get pair (xi,xj)
    training_indices = np.random.choice(samples.shape[0], size=441, replace=False) #random 441 indices
    X = samples[training_indices, :] #shape (441,2)
    y = []
    for point in X:
        y.append(f(point[0], point[1]))
    np.savetxt('training_data.csv', X, delimiter=',')
    np.savetxt('training_labels.csv', y, delimiter=',')
    return X, y

def f(x1, x2):
    return 1 if x1*x1 + x2*x2 <= 1 else -1