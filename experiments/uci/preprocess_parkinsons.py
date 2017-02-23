'''Preprocessing code for the Parkinsons Telemonitoring data: https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring'''
import numpy as np
import pandas as pd
from utils import standardize, unitize, onehot, create_folds, save_details

df = pd.read_table('experiments/uci/data/parkinsons_updrs.data.txt',  header=0, sep=',')

# Create a one-hot encoding subject ID
onehot(df, ['subject#'])

# Move the target columns to the end
cols = df.columns.tolist()
cols = cols[:3] + cols[5:] + cols[3:5]
df = df[cols]

# Preprocess the features
unitize(df, ['age'])
standardize(df, cols[2:-2])

# Create discrete labels
df['motor_UPDRS'] = (df['motor_UPDRS'].round()).apply(np.int32)
df['motor_UPDRS'] -= df['motor_UPDRS'].min()
df['total_UPDRS'] = (df['total_UPDRS'].round()).apply(np.int32)
df['total_UPDRS'] -= df['total_UPDRS'].min()

print df.describe()

create_folds('experiments/uci/data/splits/parkinsons', df)

save_details('parkinsons', len(df), df.shape[1]-2, (df['motor_UPDRS'].max()+1, df['total_UPDRS'].max()+1))
