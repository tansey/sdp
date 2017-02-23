import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import os
from pygfl.utils import pretty_str

def get_score(filename):
    if os.path.exists(filename):
        arr = np.loadtxt(filename, delimiter=',')
        if len(arr) >= 2 and arr[1] != 0:
            return arr[1]
    return np.nan

if __name__ == '__main__':
    dargs = {}
    dargs['model'] = 'skd'
    dargs['dist_type'] = 'edge_biased'
    dargs['nbins'] = 2500
    dargs['train_samples'] = 2500
    
    neighbor_radius = [1,3,5,10,25,50]

    ntrials = 10
    scores = np.zeros((len(neighbor_radius)+2,ntrials))
    for train_id in xrange(ntrials):
        dargs['train_id'] = train_id
        filename = 'experiments/mnist/results/{model}_{dist_type}_{nbins}_{train_samples}_{train_id}_score.csv'
        
        dargs['model'] = 'multinomial'
        scores[0,train_id] = get_score(filename.format(**dargs))
        
        dargs['model'] = 'skd'
        scores[1,train_id] = get_score(filename.format(**dargs))
        
        filename = 'experiments/mnist/results/{model}_{dist_type}_{nbins}_{train_samples}_{neighbor_radius}_{train_id}_score.csv'
        for i,r in enumerate(neighbor_radius):
            dargs['neighbor_radius'] = r
            scores[i+2,train_id] = get_score(filename.format(**dargs))

    print 'Multinomial score: ', pretty_str(scores[0])
    for i,n in enumerate([0]+neighbor_radius):
        print 'Radius {0} = {1}'.format(n,pretty_str(scores[i+1]))

    scores = np.nanmean(scores,axis=1)
    print 'Means:'
    print pretty_str(scores)

    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=2)
        plt.rc('lines', lw=3)
        
        plt.figure(1)
        plt.axhline(scores[0], lw=4, ls='--', color='gray', label='Multinomial')
        plt.plot([0]+neighbor_radius, scores[1:], lw=4, ls='solid', color='orange', label='SDT')
        plt.ylabel('TV Error', weight='bold', fontsize=24)
        plt.xlabel('Neighborhood Radius', weight='bold', fontsize=24)
        plt.legend(loc='upper right')
        plt.savefig('plots/mnist-neighborhood-scores.pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

