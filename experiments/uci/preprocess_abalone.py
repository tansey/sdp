'''Preprocessing code for the Abalone data: https://archive.ics.uci.edu/ml/datasets/Abalone'''
import numpy as np
import pandas as pd
from utils import standardize, unitize, onehot, create_folds, save_details

names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
df = pd.read_table('experiments/uci/data/abalone.data.txt', header=None, sep=',', names=names)

# Preprocess the features
onehot(df, ['Sex'])
standardize(df, ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight'])

# Reorder columns to put the target column at the end
cols = df.columns.tolist()
cols = cols[:-4] + cols[-3:] + cols[-4:-3]
df = df[cols]

# # Convert from one significant decimal place to discrete integers
df['Rings'] = df['Rings'].apply(np.int32)
df['Rings'] -= df['Rings'].min()

print df.describe()

create_folds('experiments/uci/data/splits/abalone', df)

save_details('abalone', len(df), df.shape[1]-1, df['Rings'].max()+1)
