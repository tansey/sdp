import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import os

def plot_1d(dataset, nbins, data):
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

def plot_2d(dataset, nbins, data, extra=None):
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

def mnist_data():
    from mnist_utils import DataLoader
    nbins = 256
    train_data = DataLoader('experiments/pixels/data', 'train', 50000, 50, seed=0, dimsize=256)
    plot_1d('mnist', nbins, train_data.labels)

def cifar_data():
    from cifar_utils import DataLoader
    nbins = (64,64,64)
    train_data = DataLoader('experiments/pixels/data', 'train', 50000, 50, seed=0, dimsize=64)
    plot_2d('cifar', (nbins[0], nbins[1]), train_data.labels[:,[0,1]], extra='-rg')
    plot_2d('cifar', (nbins[0], nbins[2]), train_data.labels[:,[0,2]], extra='-rb')
    plot_2d('cifar', (nbins[1], nbins[2]), train_data.labels[:,[1,2]], extra='-gb')

if __name__ == '__main__':
    mnist_data()
    cifar_data()




