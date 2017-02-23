'''Preprocessing code for the Auto-MPG data: https://archive.ics.uci.edu/ml/datasets/Auto+MPG'''
import numpy as np
import pandas as pd
from utils import standardize, unitize, create_folds, save_details

names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']
df = pd.read_table('experiments/uci/data/auto-mpg.data.txt', header=None, delim_whitespace=True, names=names)

# Preprocess the features
unitize(df, ['cylinders', 'model year', 'origin'])
standardize(df, ['displacement','horsepower', 'weight', 'acceleration'])
del df['car name']

# Convert from one significant decimal place to discrete integers
df['mpg'] = (df['mpg'] * 10).apply(np.int32)
df['mpg'] -= df['mpg'].min()

# Reorder columns to put the target column at the end
cols = df.columns.tolist()
cols = cols[1:] + cols[0:1]
df = df[cols]

print df.describe()

create_folds('experiments/uci/data/splits/auto_mpg', df)

save_details('auto_mpg', len(df), df.shape[1]-1, df['mpg'].max()+1)
