import numpy as np
import pandas as pd

def check_missing_data_percentage(df):
    """
    Check for missing data
    Params:
        df dataframe
    Returns:
        None
    """
    # get the number of missing data points per column
    missing_values_count = df.isnull().sum()

    # how many total missing values do we have?
    total_cells = np.product(df.shape)
    total_missing = missing_values_count.sum()

    # percentage of data that is missing
    percent_missing = (total_missing/total_cells)
    print("Percentage Missing:", "{:.2%}".format(percent_missing))
    return "{:.2%}".format(percent_missing)