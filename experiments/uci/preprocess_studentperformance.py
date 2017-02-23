'''Preprocessing code for the Student Performance data: https://archive.ics.uci.edu/ml/datasets/Student+Performance'''
import numpy as np
import pandas as pd
from utils import standardize, unitize, onehot, create_folds, save_details

df = pd.read_table('experiments/uci/data/student-mat.csv',  header=0, sep=';')
df_por = pd.read_table('experiments/uci/data/student-por.csv',  header=0, sep=';')

# Merge the two tables and take the subject-specific data
df['absences_mat'] = df['absences']
df['absences_por'] = df_por['absences']
df['failures_mat'] = df['failures']
df['failures_por'] = df_por['failures']
df['paid_mat'] = df['paid']
df['paid_por'] = df_por['paid']

del df['absences']
del df['failures']
del df['paid']

# Create a one-hot encoding subject ID
onehot(df, ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian'])
unitize(df, ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures_mat', 'failures_por',
             'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health'])
standardize(df, ['absences_mat', 'absences_por'])

# Make binary yes/no into binary 1/0
df.replace('yes', 1, inplace=True)
df.replace('no', 0, inplace=True)

# Create the target columns
df['G3_mat'] = df['G3']
df['G3_por'] = df_por['G3']
del df['G3']
del df['G2']
del df['G1']

with pd.option_context('display.max_columns', 1000):
    print df.describe()

create_folds('experiments/uci/data/splits/student_performance', df)

save_details('student_performance', len(df), df.shape[1]-2, (df['G3_mat'].max()+1, df['G3_por'].max()+1))
