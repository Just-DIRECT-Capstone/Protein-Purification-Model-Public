import pandas as pd
import tensorflow as tf
import os

def load_data(filename, filepath = ['just-private','data']):
    path = os.getcwd()
    p2f = os.path.join(path,*filepath,filename) 
    return pd.read_csv(p2f, index_col = 0)

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