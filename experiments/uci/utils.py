import os
import json
import numpy as np
import pandas as pd
import collections
from tfcde.datasets import DataSet

############### Preprocessing tools ################################
def standardize(data, cols):
    if not hasattr(cols, "__len__") or isinstance(cols, basestring):
        cols = [cols]
    for col in cols:
        data[col] = (data[col] - data[col].mean()) / data[col].std()

def unitize(data, cols):
    if not hasattr(cols, "__len__") or isinstance(cols, basestring):
        cols = [cols]
    for col in cols:
        min_val = data[col].min()
        max_val = data[col].max()
        data[col] = (data[col] - min_val) / float(max_val - min_val)

def onehot(data, cols):
    '''Converts each column into a series of one-hot columns'''
    if not hasattr(cols, "__len__") or isinstance(cols, basestring):
        cols = [cols]
    for col in cols:
        uniques = data[col].unique()
        for u in uniques:
            colname = '{0}_{1}'.format(col, u)
            data[colname] = [1 if c == u else 0 for c in data[col]]
        del data[col]

def create_folds(outfile, data, k=10):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    split_size = int(np.floor(len(data) / k))
    start = 0
    for i in xrange(k):
        end = start + split_size
        if (split_size % k) < i:
            end += 1
        if len(data) < end:
            end = len(data)
        hold_out = indices[start:end]
        train = data[-data.index.isin(hold_out)]
        test = data[data.index.isin(hold_out)]
        train.to_csv('{0}_train{1}.csv'.format(outfile,i), index=False)
        test.to_csv('{0}_test{1}.csv'.format(outfile,i), index=False)
        start = end
    data.to_csv('{0}_all.csv'.format(outfile), index=False)

def save_details(dataset_name, nsamples, nfeatures, nbins):
    details_filename = 'experiments/uci/data/details.json'
    if os.path.exists(details_filename):
        with open(details_filename, 'rb') as f:
            details = json.load(f)
    else:
        details = {}
    details[dataset_name] = {'nsamples': nsamples, 'nfeatures': nfeatures, 'nbins': nbins}
    with open(details_filename, 'wb') as f:
        json.dump(details, f)

############### Data loading tools ################################
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

def all_dataset_details(inputdir):
    with open(os.path.join(inputdir, 'details.json'), 'rb') as f:
        details = json.load(f)
    return details

def dataset_details(dataset, inputdir):
    with open(os.path.join(inputdir, 'details.json'), 'rb') as f:
        details = json.load(f)
    if not hasattr(details[dataset]['nbins'], "__len__"):
        details[dataset]['nbins'] = [details[dataset]['nbins']]
    return details[dataset]['nfeatures'], tuple(details[dataset]['nbins']), details[dataset]['nsamples']

def load_dataset(dataset=None, train_id=0, inputdir='experiments/uci/data/',
                 batchsize=50,
                 validation_pct=0.2, **kwargs):
    x_shape, num_classes, nsamples = dataset_details(dataset, inputdir)
    ndims = len(num_classes)
    df_train = pd.read_csv(os.path.join(inputdir, 'splits/{0}_train{1}.csv'.format(dataset, train_id)))
    df_test = pd.read_csv(os.path.join(inputdir, 'splits/{0}_test{1}.csv'.format(dataset, train_id)))
    validation_size = int(np.round(len(df_train) * validation_pct))
    df_train, df_validation = df_train[validation_size:], df_train[:validation_size]
    cols = df_train.columns.tolist()
    train_features = df_train[cols[:-ndims]].as_matrix()
    train_labels = df_train[cols[-ndims:]].as_matrix()
    validation_features = df_validation[cols[:-ndims]].as_matrix()
    validation_labels = df_validation[cols[-ndims:]].as_matrix()
    test_features = df_test[cols[:-ndims]].as_matrix()
    test_labels = df_test[cols[-ndims:]].as_matrix()
    return Datasets(train=DataSet(train_features, train_labels, batch_size=batchsize),
                    validation=DataSet(validation_features, validation_labels, batch_size=batchsize),
                    test=DataSet(test_features, test_labels, batch_size=batchsize))








