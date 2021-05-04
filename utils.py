import pandas as pd
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filename, filepath = ['just-private','data']):
    path = os.getcwd()
    p2f = os.path.join(path,*filepath,filename)
    try:
        return pd.read_csv(p2f, index_col = 0)
    except Exception as e:
        raise e


# custom train test split to ensure cuts from the same run are all in the same size
def chroma_train_test_split(data : pd.DataFrame, x : list, y : list, keep_cuts = True, test_size=0.20, random_state=None):
    if random_state: np.random.seed(random_state)

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
        train, test = train_test_split(data, test_size=test_size, random_state=random_state)

    if not np.all([var in data.columns for var in [*x,*y]]):
        raise Exception("x or y labels not in the dataframe")

    return train[x], test[x], train[y], test[y]







# from keras. could use sklearn too 
def get_train_and_test_splits(x,y, train_size, batch_size=1):
    # We prefetch with a buffer the same size as the dataset because th dataset
    # is very small and fits into memory.
    dataset = (
        tf.data.Dataset.from_tensor_slices((x.to_dict('list'), y.to_dict('list')))
    )
    # We shuffle with a buffer the same size as the dataset.
    train_dataset = (
        dataset.take(train_size).shuffle(buffer_size=train_size).batch(batch_size)
    )
    test_dataset = dataset.skip(train_size).batch(batch_size)

    return train_dataset, test_dataset


# make sure option for keeping all runs in either test or train
# normalized or standarized 