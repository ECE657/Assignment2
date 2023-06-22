import numpy as np
import pandas as pd

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