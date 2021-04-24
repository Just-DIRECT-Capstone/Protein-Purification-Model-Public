import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *

filename = 'mol_res_scan_results.csv'
data = load_data(filename)

n_outputs = 2
n_inputs = (data.shape[1]-4)
n = int(np.ceil(n_inputs**0.5))

fig, ax = plt.subplots(n, n, figsize = (12,12))
fig.tight_layout()

for a,var in zip(ax.flat,data.columns[4:]):
    data[var].hist(ax = a)
    a.set_title(var)

fig, ax = plt.subplots(n_outputs, 1, figsize = (3,6))
fig.tight_layout()

for a,var in zip(ax.flat,data.columns[2:4]):
    data[var].hist(ax = a)
    a.set_title(var)