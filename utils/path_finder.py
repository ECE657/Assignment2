import pandas as pd
import os

def get_file_path(filename):
    """
    search file across whole repo and return abspath
    Params:: 
        filename
    Returns:
        absolute path of file
    """
    for root, dirs, files in os.walk(r'.'):
        for name in files:
            if name == filename:
                return os.path.abspath(os.path.join(root, name))
    raise FileNotFoundError(filename, "not found.")