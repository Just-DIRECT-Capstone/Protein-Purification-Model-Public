import pandas as pd
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(parentdir, filename, filepath = ['just-private','data']):
    """Load data from a .csv file into a dataframe. 

    Parameters
    ----------
    parentdir : str
        Root projec directory
    filename : str

    filepath : list, default = ['just-private','data']
        List of folders pointing to the file from the root projec directory.

    Returns
    -------
    data : pd.DataFrame

    """
    p2f = os.path.join(parentdir,*filepath,filename)
    try:
        return pd.read_csv(p2f, index_col = 0)
    except Exception as e:
        raise e


# custom train test split to ensure cuts from the same run are all in the same size
def chroma_train_test_split(data : pd.DataFrame, x : list, y : list, keep_cuts = True, test_size=0.20, random_seed=None):
    """Split dataframe into train and test sets. If keep_cuts is True, then all data from the same run
    is maintained in either train or test. Otherwise, different cuts from the same run can be split.

    Parameters
    ----------
    data : pd.DataFrame

    x : list
        List of dependent variable names in the dataframe
    
    y : list
        List of independent variable names in the dataframe
    
    keep_cuts : bool, default = True
        Specifies if all the cuts from the same run are kept together
    
    test_size : float, default=0.2
        Fraction of the dataset to include in the test split. 

    random_seed : int, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    train_x : pd.DataFrame
    test_x : pd.DataFrame
    train_y : pd.DataFrame
    test_y : pd.DataFrame

    """
    if random_seed: np.random.seed(random_seed)

    try:
        indexes = [0,*np.where(np.diff(data['cut 1'])<0)[0]+1, len(data)]
        # TODO: add here other option, like data['id']
    except:
        raise Exception("No variable 'cut 1' in the dataframe!")
    
    if keep_cuts:
        runs = []
        for i in range(len(indexes)-1):
            runs.append(data[indexes[i]:indexes[i+1]].sample(frac=1))

        np.random.shuffle(runs)

        test_idx = int(np.ceil(len(runs)*test_size))
        test = pd.concat(runs[:test_idx])
        train = pd.concat(runs[test_idx:])

    else:
        train, test = train_test_split(data, test_size=test_size, random_state=random_seed)

    if not np.all([var in data.columns for var in [*x,*y]]):
        raise Exception("x or y labels not in the dataframe")

    return train[x], test[x], train[y], test[y]


def preprocessing(arrays, standarize = False, bounds = dict(), skip = None):
    """Preprocess the data before doing any modeling. 

    Parameters
    ----------
    *arrays : list
        List with pd.DataFrames. First element should be train set. 
    
    standarize : bool, default = True
        Standardize features by removing the mean and scaling to unit variance

    bounds : dict, default = {}
        Dicitionary spcecifying cutoff min and max values for variables

    skip : list, default = None
        Variables to ommit from the preprocessing

    Returns
    -------
    preprocessed: List with pd.DataFrames

    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    
    if standarize: scaler = StandardScaler().fit(arrays[0])
    preprocessed = []
    for i in arrays:

        # Cutoff or filter
        if len(bounds) > 0:
            for k, b in bounds.items():
                i = i.loc[(i[k] >= b[0]) & (i[k] <= b[1])]
        # Scaling
        if standarize: preprocessed.append(pd.DataFrame(scaler.transform(i), index=i.index, columns=i.columns))
        else: preprocessed.append(i)

        if skip:
            for k in skip:
                preprocessed[-1][k] = i[k]

    if standarize: preprocessed.append(scaler)

    return preprocessed


def count_parameters(model):
    total_parameters = 0 
    #iterating over all variables 
    for variable in model.trainable_variables:   
        local_parameters=1 
        shape = variable.get_shape()  #getting shape of a variable 
        for i in shape: 
            local_parameters*=i  #mutiplying dimension values 
        total_parameters+=local_parameters 
    return total_parameters

def get_model_name(model,dataset):
    return model.name[:-(1+len(dataset[:-4]))]