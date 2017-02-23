'''Preprocessing code for the Energy Efficiency data: https://archive.ics.uci.edu/ml/datasets/Energy+efficiency'''
import numpy as np
import pandas as pd
from utils import standardize, unitize, create_folds, save_details

df = pd.read_table('experiments/uci/data/ENB2012_data.csv',  header=0, sep=',')

# # Preprocess the features
standardize(df, ['X1', 'X2', 'X3', 'X4'])
unitize(df, ['X5', 'X6', 'X7', 'X8'])

# # Convert from one significant decimal place to discrete integers
df['Y1'] = (df['Y1'].round()).apply(np.int32)
df['Y1'] -= df['Y1'].min()
df['Y2'] = (df['Y2'].round()).apply(np.int32)
df['Y2'] -= df['Y2'].min()

print df.describe()

create_folds('experiments/uci/data/splits/energy_efficiency', df)

save_details('energy_efficiency', len(df), df.shape[1]-2, (df['Y1'].max()+1, df['Y2'].max()+1))
