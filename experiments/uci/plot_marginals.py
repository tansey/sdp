import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import os
from tfcde.utils import pretty_str
from utils import all_dataset_details

def plot_1d(dataset, nbins):
    data = np.loadtxt('experiments/uci/data/splits/{0}_all.csv'.format(dataset), skiprows=1, delimiter=',')[:,-1]
    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=2)
        plt.rc('lines', lw=3)
        plt.figure(1)
        plt.hist(data, bins=np.arange(nbins+1), color='blue')
        plt.ylabel('Count', weight='bold', fontsize=24)
        xticks = list(plt.gca().get_xticks())
        while (nbins-1) / float(xticks[-1]) < 1.1:
            xticks = xticks[:-1]
        while xticks[0] < 0:
            xticks = xticks[1:]
        xticks.append(nbins-1)
        xticks = list(sorted(xticks))
        plt.gca().set_xticks(xticks)
        plt.xlim([int(np.ceil(-0.05*nbins)),int(np.ceil(nbins*1.05))])
        plt.legend(loc='upper right')
        plt.savefig('plots/marginals-{0}.pdf'.format(dataset.replace('_','-')), bbox_inches='tight')
        plt.clf()
        plt.close()

def plot_2d(dataset, nbins, data=None, extra=None):
    if data is None:
        data = np.loadtxt('experiments/uci/data/splits/{0}_all.csv'.format(dataset), skiprows=1, delimiter=',')[:,-2:]
    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=2)
        plt.rc('lines', lw=2)
        rows, cols = nbins
        im = np.zeros(nbins)
        for i in xrange(rows):
            for j in xrange(cols):
                im[i,j] = ((data[:,0] == i) & (data[:,1] == j)).sum()
        plt.imshow(im, cmap='gray_r', interpolation='none')
        if extra is not None:
            dataset += extra
        plt.savefig('plots/marginals-{0}.pdf'.format(dataset.replace('_','-')), bbox_inches='tight')
        plt.clf()
        plt.close()

def plot_3d(dataset, nbins):
    data = np.loadtxt('experiments/uci/data/splits/{0}_all.csv'.format(dataset.replace('_','-')), skiprows=1, delimiter=',')[:,-3:]
    plot_2d(dataset, (nbins[0], nbins[1]), data=data[:,[0,1]], extra='-xy')
    plot_2d(dataset, (nbins[0], nbins[2]), data=data[:,[0,2]], extra='-xz')
    plot_2d(dataset, (nbins[1], nbins[2]), data=data[:,[1,2]], extra='-yz')

if __name__ == '__main__':
    datasets = all_dataset_details('experiments/uci/data')
    for dataset,vals in datasets.iteritems():
        nbins = vals['nbins']
        
        if isinstance(nbins, int):
            plot_1d(dataset, nbins)
        elif len(nbins) == 2:
            plot_2d(dataset, nbins)
        elif len(nbins) == 3:
            plot_3d(dataset, nbins)




