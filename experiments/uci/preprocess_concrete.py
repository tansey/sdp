'''Preprocessing code for the Concrete Slump Test Data data: https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test'''
import numpy as np
import pandas as pd
from utils import standardize, unitize, onehot, create_folds, save_details

df = pd.read_table('experiments/uci/data/slump_test.data.txt',  header=0, sep=',')

# Remove the ID column
del df['No']

# # Preprocess the features
# unitize(df, ['age'])
standardize(df, ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr.', 'Fine Aggr.'])

# Create discrete labels
df['SLUMP(cm)'] = (df['SLUMP(cm)'].round()).apply(np.int32)
df['SLUMP(cm)'] -= df['SLUMP(cm)'].min()
df['FLOW(cm)'] = (df['FLOW(cm)'].round()).apply(np.int32)
df['FLOW(cm)'] -= df['FLOW(cm)'].min()
df['Compressive Strength (28-day)(Mpa)'] = (df['Compressive Strength (28-day)(Mpa)'].round()).apply(np.int32)
df['Compressive Strength (28-day)(Mpa)'] -= df['Compressive Strength (28-day)(Mpa)'].min()

print df.describe()

create_folds('experiments/uci/data/splits/concrete', df)

save_details('concrete', len(df), df.shape[1]-3, (df['SLUMP(cm)'].max()+1, df['FLOW(cm)'].max()+1, df['Compressive Strength (28-day)(Mpa)'].max()+1))
