'''Preprocessing code for the Housing data: https://archive.ics.uci.edu/ml/datasets/Housing'''
import numpy as np
import pandas as pd
from utils import standardize, unitize, create_folds, save_details

names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_table('experiments/uci/data/housing.data.txt', header=None, delim_whitespace=True, names=names)

# Preprocess the features
standardize(df, ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
unitize(df, ['ZN', 'RAD'])

# Convert from one significant decimal place to discrete integers
df['MEDV'] = (df['MEDV'] * 10).apply(np.int32)
df['MEDV'] -= df['MEDV'].min()

print df.describe()

create_folds('experiments/uci/data/splits/housing', df)

save_details('housing', len(df), df.shape[1]-1, df['MEDV'].max()+1)
