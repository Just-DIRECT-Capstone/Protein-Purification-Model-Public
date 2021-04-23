import pandas as pd
import os

def load_data(filename, filepath = ['just-private','data']):
    path = os.getcwd()
    p2f = os.path.join(path,*filepath,filename) 
    return pd.read_csv(p2f, index_col = 0)