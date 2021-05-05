import numpy as np
import matplotlib.pyplot as plt

def histograms(data,x,y):
    n_outputs = len(y)
    n_inputs = len(x)
    n = int(np.ceil(n_inputs**0.5))

    fig, ax = plt.subplots(n, n, figsize = (12,12))
    fig.tight_layout()

    for a,var in zip(ax.flat,x):
        data[var].hist(ax = a)
        a.set_title(var)

    fig, ax = plt.subplots(n_outputs, 1, figsize = (3,6))
    fig.tight_layout()

    for a,var in zip(ax.flat,y):
        data[var].hist(ax = a)
        a.set_title(var)

    return fig, ax